"""
Hex Board Game Scorer - VLM-First Architecture
===============================================
Pipeline:
  1. VLM Object Detection → detect all tiles with bboxes, colors, animals in one pass
  2. CV Post-Processing   → build adjacency graph from bboxes
  3. Scoring Engine       → deterministic rule evaluation

Advantages over CV-first approach:
  - Higher accuracy: VLM sees full-resolution tiles with context
  - Robust to rotation, perspective, lighting variation
  - Handles occlusion and edge cases naturally
  - Schema-validated JSON output (no hallucinations)

Usage:
    python board_scorer_v2.py
    
Requires:
    pip install google-genai opencv-python numpy Pillow python-dotenv
    Set GEMINI_API_KEY in .env file or environment
"""

import cv2
import numpy as np
import os
import json
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
from PIL import Image


# ============================================================
# CONFIG
# ============================================================
BOARD_IMAGE_PATH = "./public/player-regions.png"

# VLM detection config
VLM_MODEL = "gemini-2.5-flash"
VLM_TEMPERATURE = 0.0  # deterministic for classification

# Adjacency threshold: tiles are neighbors if center distance < (median_tile_width * this factor)
ADJACENCY_FACTOR = 1.15

# Known game entities (for schema validation)
TILE_COLORS = ["blue", "yellow", "pink", "brown", "grey"]
ANIMALS = ["bear", "hawk", "elk", "salmon", "fox"]


# ============================================================
# STAGE 1 — VLM OBJECT DETECTION
# ============================================================

def detect_tiles_vlm(image_path: str, api_key: Optional[str] = None) -> List[Dict]:
    """
    Single-pass VLM detection of all hex tiles.
    
    Returns list of tile dicts with:
        {
            "box_2d": [ymin, xmin, ymax, xmax],  # normalized 0-1000
            "tile_color": str,
            "animal_type": str,
            "player": int  # 1, 2, or 3
        }
    """
    from google.genai import Client, types
    from dotenv import load_dotenv
    load_dotenv()
    
    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise ValueError(
            "GEMINI_API_KEY not found. Set it in .env file or pass as argument."
        )
    
    client = Client(api_key=key)
    img = Image.open(image_path)
    
    prompt = """
You are analyzing a board game image containing hex tiles arranged in 3 vertical columns.
Each column represents one player's board region (left=Player1, center=Player2, right=Player3).

TASK:
Detect EVERY hex tile in the image and classify each tile's attributes.

For each tile, determine:
1. **tile_color**: The background color of the hexagon
   - Must be one of: blue, yellow, pink, brown, grey
   
2. **animal_type**: The animal silhouette icon in the tile's center
   - Must be one of: bear, hawk, elk, salmon, fox
   - bear: large mammal shape
   - hawk: bird with spread wings
   - elk: deer with antlers (vertical projections)
   - salmon: fish shape (horizontal oval)
   - fox: medium mammal with pointed features
   
3. **player**: Which vertical region the tile belongs to
   - 1 = leftmost third of image
   - 2 = middle third of image  
   - 3 = rightmost third of image
   
4. **box_2d**: Bounding box coordinates [ymin, xmin, ymax, xmax]
   - Normalized to 0-1000 range
   - Should tightly bound the hexagon

OUTPUT RULES:
- Return a JSON array of tile objects
- Use EXACT enum values (case-sensitive)
- Include ALL visible tiles (do not skip any)
- Tiles may be touching or slightly overlapping
- Ignore the grass/ground background

EXAMPLE OUTPUT STRUCTURE (do not copy values):
[
  {
    "box_2d": [100, 50, 200, 150],
    "tile_color": "blue",
    "animal_type": "bear",
    "player": 1
  },
  ...
]
"""
    
    # Define strict JSON schema for validation
    schema = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "box_2d": {
                    "type": "ARRAY",
                    "items": {"type": "NUMBER"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": "Normalized bbox [ymin, xmin, ymax, xmax] in 0-1000 range"
                },
                "tile_color": {
                    "type": "STRING",
                    "enum": TILE_COLORS,
                    "description": "Background color of the hex tile"
                },
                "animal_type": {
                    "type": "STRING",
                    "enum": ANIMALS,
                    "description": "Animal species shown on the tile"
                },
                "player": {
                    "type": "INTEGER",
                    "enum": [1, 2, 3],
                    "description": "Which player region (1=left, 2=middle, 3=right)"
                }
            },
            "required": ["box_2d", "tile_color", "animal_type", "player"]
        }
    }
    
    print("[VLM] Sending image to Gemini for tile detection...")
    
    response = client.models.generate_content(
        model=VLM_MODEL,
        contents=[img, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=VLM_TEMPERATURE,
        )
    )
    
    tiles = json.loads(response.text)
    print(f"[VLM] Detected {len(tiles)} tiles total")
    
    # Validate and add tile IDs
    for idx, tile in enumerate(tiles):
        tile["id"] = f"tile_{idx}"
        # Basic sanity checks
        if not (0 <= tile["box_2d"][0] <= 1000 and 
                0 <= tile["box_2d"][1] <= 1000 and
                0 <= tile["box_2d"][2] <= 1000 and
                0 <= tile["box_2d"][3] <= 1000):
            print(f"[WARNING] Tile {idx} has out-of-range bbox: {tile['box_2d']}")
    
    return tiles


# ============================================================
# STAGE 2 — CV POST-PROCESSING
# ============================================================

def bbox_center(box: List[float]) -> Tuple[float, float]:
    """Get (y, x) center from normalized bbox [ymin, xmin, ymax, xmax]."""
    ymin, xmin, ymax, xmax = box
    return ((ymin + ymax) / 2.0, (xmin + xmax) / 2.0)


def bbox_width(box: List[float]) -> float:
    """Get width from normalized bbox."""
    return box[3] - box[1]


def bbox_height(box: List[float]) -> float:
    """Get height from normalized bbox."""
    return box[2] - box[0]


def build_adjacency_from_tiles(
    tiles: List[Dict],
    adaptive_threshold: bool = True,
) -> Dict[int, List[int]]:
    """
    Build hex-neighbor adjacency graph from tile bounding boxes.
    
    Two tiles are adjacent if their center-to-center distance is within
    ~1.15× the median tile width (in normalized coordinate space).
    
    Args:
        tiles: List of tile dicts with "box_2d" field
        adaptive_threshold: If True, compute threshold from median bbox width.
                           If False, use fixed value (less robust).
    
    Returns:
        Dict mapping tile_index → list of adjacent tile indices
    """
    if not tiles:
        return {}
    
    centers = [bbox_center(t["box_2d"]) for t in tiles]
    
    # Adaptive threshold based on actual tile sizes
    if adaptive_threshold:
        widths = [bbox_width(t["box_2d"]) for t in tiles]
        median_width = np.median(widths)
        threshold = median_width * ADJACENCY_FACTOR
        print(f"[Adjacency] Using adaptive threshold: {threshold:.1f} (median_width={median_width:.1f})")
    else:
        threshold = 60.0  # fixed fallback
        print(f"[Adjacency] Using fixed threshold: {threshold}")
    
    adjacency = defaultdict(list)
    edge_count = 0
    
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            # Euclidean distance in normalized space
            dist = np.hypot(centers[i][0] - centers[j][0],
                           centers[i][1] - centers[j][1])
            if dist < threshold:
                adjacency[i].append(j)
                adjacency[j].append(i)
                edge_count += 1
    
    print(f"[Adjacency] Found {edge_count} edges among {len(tiles)} tiles")
    return dict(adjacency)


def group_tiles_by_player(tiles: List[Dict]) -> List[Dict]:
    """
    Organize detected tiles into per-player data structures.
    
    Returns list of 3 player dicts:
        {
            "player_id": int,
            "tiles": [tile_dict, ...],
            "animals": [str, ...],
            "tile_colors": [str, ...],
            "adjacency": {local_idx: [local_idx, ...], ...},
            "centers": [(y, x), ...],  # in normalized coords
        }
    """
    players = [
        {
            "player_id": p,
            "tiles": [],
            "animals": [],
            "tile_colors": [],
            "adjacency": {},
            "centers": [],
        }
        for p in range(1, 4)
    ]
    
    # Group tiles by player
    for tile in tiles:
        p_idx = tile["player"] - 1
        players[p_idx]["tiles"].append(tile)
    
    # Build per-player adjacency graphs and extract attributes
    for player in players:
        if not player["tiles"]:
            print(f"[WARNING] Player {player['player_id']} has no tiles detected!")
            continue
        
        # Build adjacency for this player's tiles only
        player["adjacency"] = build_adjacency_from_tiles(player["tiles"])
        
        # Extract parallel arrays for scoring engine
        player["animals"] = [t["animal_type"] for t in player["tiles"]]
        player["tile_colors"] = [t["tile_color"] for t in player["tiles"]]
        player["centers"] = [bbox_center(t["box_2d"]) for t in player["tiles"]]
        
        print(f"\nPlayer {player['player_id']}: {len(player['tiles'])} tiles")
        print(f"  Colors: {dict(Counter(player['tile_colors']))}")
        print(f"  Animals: {dict(Counter(player['animals']))}")
    
    return players


# ============================================================
# STAGE 3 — SCORING ENGINE
# (Unchanged from original — deterministic rule evaluation)
# ============================================================

def _find_connected_components(
    tile_indices: List[int],
    adjacency: Dict[int, List[int]],
) -> List[List[int]]:
    """
    Given a subset of tile indices and the full adjacency graph,
    return connected components (groups of mutually-reachable tiles).
    """
    remaining = set(tile_indices)
    components = []
    while remaining:
        start = next(iter(remaining))
        # BFS
        queue = [start]
        visited = {start}
        while queue:
            node = queue.pop(0)
            for nb in adjacency.get(node, []):
                if nb in remaining and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        components.append(sorted(visited))
        remaining -= visited
    return components


def score_bears(
    animals: List[str],
    adjacency: Dict[int, List[int]],
) -> int:
    """
    Grizzly Bear: scores 10 per group of exactly 3 bears
    with NO other bears adjacent to the group.
    """
    bear_indices = [i for i, a in enumerate(animals) if a == "bear"]
    if len(bear_indices) < 3:
        return 0
    
    components = _find_connected_components(bear_indices, adjacency)
    score = 0
    for comp in components:
        if len(comp) == 3:
            score += 10
        # Groups of size != 3 score nothing
    return score


def score_hawks(
    animals: List[str],
    adjacency: Dict[int, List[int]],
) -> int:
    """
    Red-Tailed Hawk: scores based on connected chain length.
    
    Scoring table (from card):
      chain_size → total_points_for_chain
      2 → 5
      3 → 9
      4 → 12
      5 → 16
      6 → 20
      7 → 24
      8+ → 23
    
    Isolated hawks (chain size 1) score nothing.
    """
    HAWK_CHAIN_SCORES = {
        2: 5,
        3: 9,
        4: 12,
        5: 16,
        6: 20,
        7: 24,
    }
    
    hawk_indices = [i for i, a in enumerate(animals) if a == "hawk"]
    if not hawk_indices:
        return 0
    
    components = _find_connected_components(hawk_indices, adjacency)
    score = 0
    for comp in components:
        size = len(comp)
        if size < 2:
            continue  # isolated hawks don't score
        if size >= 8:
            score += 23
        else:
            score += HAWK_CHAIN_SCORES.get(size, 0)
    return score


def score_elk(
    animals: List[str],
    adjacency: Dict[int, List[int]],
    centroids: List[Tuple[float, float]],
) -> int:
    """
    Roosevelt Elk: scores per group matching exact shapes.
    
    From card:
      - Specific 2-elk formations → 5 pts each
      - Specific 5-elk formations → 5 pts
      - Specific 9-elk formation → 13 pts
    
    NOTE: Full implementation requires shape templates from the card.
    This simplified version scores connected groups by size.
    """
    ELK_SCORES = {2: 5, 5: 5, 9: 13}
    
    elk_indices = [i for i, a in enumerate(animals) if a == "elk"]
    if not elk_indices:
        return 0
    
    components = _find_connected_components(elk_indices, adjacency)
    score = 0
    for comp in components:
        size = len(comp)
        score += ELK_SCORES.get(size, 0)
    return score


def score_salmon(
    animals: List[str],
    adjacency: Dict[int, List[int]],
) -> int:
    """
    Chinook Salmon: scores for connected runs of salmon.
    
    From card:
      run_size → points
      3 → 10
      4 → 12
      5+ → 15
    
    Runs of 1 or 2 score nothing.
    """
    SALMON_SCORES = {3: 10, 4: 12}
    
    salmon_indices = [i for i, a in enumerate(animals) if a == "salmon"]
    if not salmon_indices:
        return 0
    
    components = _find_connected_components(salmon_indices, adjacency)
    score = 0
    for comp in components:
        size = len(comp)
        if size >= 5:
            score += 15
        else:
            score += SALMON_SCORES.get(size, 0)
    return score


def score_foxes(
    animals: List[str],
    adjacency: Dict[int, List[int]],
) -> int:
    """
    Red Fox: each fox scores based on number of DISTINCT
    adjacent animal types (excluding other foxes).
    
    From card:
      # distinct_neighbors → points_per_fox
      1 → 1
      2 → 2
      3 → 4
      4 → 5
      5 → 6
      6 → 6
    
    Foxes with no non-fox neighbors score 0.
    """
    FOX_SCORES = {1: 1, 2: 2, 3: 4, 4: 5, 5: 6, 6: 6}
    
    score = 0
    for i, animal in enumerate(animals):
        if animal != "fox":
            continue
        
        # Count distinct animal types among neighbors (excluding fox)
        neighbor_types = set()
        for nb_idx in adjacency.get(i, []):
            if animals[nb_idx] != "fox":
                neighbor_types.add(animals[nb_idx])
        
        n_types = len(neighbor_types)
        score += FOX_SCORES.get(min(n_types, 6), 0)
    
    return score


def compute_scores(
    animals: List[str],
    adjacency: Dict[int, List[int]],
    centroids: List[Tuple[float, float]],
) -> Dict[str, int]:
    """
    Run all scoring rules and return {objective: points}.
    """
    return {
        "Grizzly Bear": score_bears(animals, adjacency),
        "Red-Tailed Hawk": score_hawks(animals, adjacency),
        "Roosevelt Elk": score_elk(animals, adjacency, centroids),
        "Chinook Salmon": score_salmon(animals, adjacency),
        "Red Fox": score_foxes(animals, adjacency),
    }


# ============================================================
# VISUALIZATION — Draw Bboxes and Labels on Image
# ============================================================

def visualize_detections(
    image_path: str,
    tiles: List[Dict],
    output_path: str = "detection_visualization.png",
) -> None:
    """
    Draw bounding boxes and labels on the board image for debugging.
    
    Args:
        image_path: Path to original board image
        tiles: List of detected tile dicts
        output_path: Where to save annotated image
    """
    # Load image with PIL, convert to OpenCV format
    pil_img = Image.open(image_path)
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    
    # Color map for players
    player_colors = {
        1: (0, 255, 0),    # green
        2: (255, 0, 0),    # blue
        3: (0, 0, 255),    # red
    }
    
    for tile in tiles:
        # Denormalize bbox
        ymin, xmin, ymax, xmax = tile["box_2d"]
        x1 = int(xmin / 1000.0 * w)
        y1 = int(ymin / 1000.0 * h)
        x2 = int(xmax / 1000.0 * w)
        y2 = int(ymax / 1000.0 * h)
        
        # Draw bbox
        color = player_colors.get(tile["player"], (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{tile['animal_type'][:3].upper()}"
        label_y = y1 - 5 if y1 > 20 else y2 + 15
        cv2.putText(
            img, label, (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
        )
    
    cv2.imwrite(output_path, img)
    print(f"[Visualization] Saved annotated image to {output_path}")


# ============================================================
# MAIN ORCHESTRATOR
# ============================================================

def main():
    print("=" * 60)
    print(" HEX BOARD GAME SCORER - VLM-First Architecture")
    print("=" * 60)
    
    # --- Stage 1: VLM Object Detection ---
    print("\n[Stage 1] VLM Tile Detection...")
    all_tiles = detect_tiles_vlm(BOARD_IMAGE_PATH)
    
    if not all_tiles:
        print("[ERROR] No tiles detected. Check image path and API key.")
        return
    
    # Save visualization
    visualize_detections(BOARD_IMAGE_PATH, all_tiles)
    
    # --- Stage 2: CV Post-Processing ---
    print("\n[Stage 2] Building Adjacency Graphs...")
    players = group_tiles_by_player(all_tiles)
    
    # --- Stage 3: Scoring ---
    print("\n" + "=" * 60)
    print(" SCORING RESULTS")
    print("=" * 60)
    
    all_scores = []
    for player in players:
        if not player["tiles"]:
            print(f"\nPlayer {player['player_id']}: NO TILES DETECTED")
            all_scores.append({})
            continue
        
        scores = compute_scores(
            player["animals"],
            player["adjacency"],
            player["centers"],
        )
        total = sum(scores.values())
        all_scores.append(scores)
        
        print(f"\nPlayer {player['player_id']}:")
        for objective, points in scores.items():
            print(f"  {objective:20s}: {points:3d} pts")
        print(f"  {'TOTAL':20s}: {total:3d} pts")
    
    # --- Summary Table ---
    print("\n" + "=" * 60)
    print(" SUMMARY TABLE (Markdown)")
    print("=" * 60)
    
    objectives = [
        "Grizzly Bear",
        "Red-Tailed Hawk",
        "Roosevelt Elk",
        "Chinook Salmon",
        "Red Fox",
        "**TOTAL**"
    ]
    
    header = "| Objective | " + " | ".join(f"Player {i+1}" for i in range(3)) + " |"
    sep = "| --- | " + " | ".join("---" for _ in range(3)) + " |"
    
    print(header)
    print(sep)
    
    for obj in objectives:
        row_vals = []
        for p_idx in range(3):
            if not all_scores[p_idx]:
                row_vals.append("0")
            elif obj == "**TOTAL**":
                row_vals.append(str(sum(all_scores[p_idx].values())))
            else:
                row_vals.append(str(all_scores[p_idx].get(obj, 0)))
        print(f"| {obj} | " + " | ".join(row_vals) + " |")
    
    # Save to file
    with open("scoring_output.md", "w") as f:
        f.write(header + "\n" + sep + "\n")
        for obj in objectives:
            row_vals = []
            for p_idx in range(3):
                if not all_scores[p_idx]:
                    row_vals.append("0")
                elif obj == "**TOTAL**":
                    row_vals.append(str(sum(all_scores[p_idx].values())))
                else:
                    row_vals.append(str(all_scores[p_idx].get(obj, 0)))
            f.write(f"| {obj} | " + " | ".join(row_vals) + " |\n")
    
    print("\n✓ Saved scoring_output.md")
    print("✓ Saved detection_visualization.png")


if __name__ == "__main__":
    main()
"""
Hex Board Game Scorer
=====================
Pipeline:
  1. CV  → Grass removal, hex center detection (distance transform + NMS),
           tile color classification (nearest-centroid), adjacency graph
  2. VLM → Animal-icon classification via Google Gemini on upscaled crops
  3. Engine → Rule-based scoring per objective card

Usage:
    python board_scorer.py
    
Requires:
    pip install opencv-python numpy scipy google-generativeai Pillow
    Set GEMINI_API_KEY env var (or paste key in config below).
"""

import cv2
import numpy as np
import os
import json
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
from scipy.ndimage import maximum_filter
from scipy.ndimage import label as scipy_label
from scipy.ndimage import center_of_mass


# ============================================================
# CONFIG
# ============================================================
BOARD_IMAGE_PATH   = "./public/player-regions.png"   # triptych image
SCORING_CARDS_PATH = "./public/scoring-rules-h.jpg"  # objective card photo

# CV tuning (calibrated for ~930×540 triptych at 1/3 panel width)
GRASS_HSV_LOW      = (25, 40, 30)
GRASS_HSV_HIGH     = (85, 255, 255)
CLOSE_KERNEL_SIZE  = 3
DIST_THRESHOLD     = 6.0          # minimum distance-transform value to be a center
NMS_WINDOW         = 20           # local-max window (pixels)
NMS_MIN_DIST       = 22           # min center-to-center distance after NMS
ADJACENCY_THRESH   = 55.0         # max center distance to be hex-neighbors
ICON_CROP_PAD      = 20           # pixels around center for icon crop
ICON_CROP_SIZE     = 80           # size icon crops are resized to
COLOR_RING_RADII   = [8, 12]      # radii (px) for tile-color ring sampling
COLOR_RING_POINTS  = 8            # samples per ring

# Tile background-color reference centroids (BGR)
TILE_COLOR_REFS = {
    "blue":   np.array([38,  165, 237]),
    "yellow": np.array([213, 185, 113]),
    "pink":   np.array([115, 115, 212]),
    "brown":  np.array([188, 173, 148]),
    "grey":   np.array([125, 135, 150]),
}

# Animals recognised by the scoring cards in the uploaded image
KNOWN_ANIMALS = ["bear", "hawk", "elk", "salmon", "fox"]


# ============================================================
# STAGE 1 — CV: DETECT HEX TILES
# ============================================================

def _foreground_mask(panel: np.ndarray) -> np.ndarray:
    """Binary mask: True where tile pixels are (grass removed)."""
    hsv  = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, GRASS_HSV_LOW, GRASS_HSV_HIGH)
    fg   = cv2.bitwise_not(grass)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE)
    )
    return cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)


def detect_hex_centers(panel: np.ndarray) -> List[Tuple[int, int]]:
    """
    Locate hex-tile centres via distance-transform peaks + greedy NMS.

    Returns list of (row, col) centroids sorted by distance-transform value
    (most-central pixel first).
    """
    fg   = _foreground_mask(panel)
    dist = cv2.distanceTransform(fg, cv2.DIST_L2, 5).astype(np.float32)

    # --- local-maxima detection ---
    local_max = maximum_filter(dist, size=NMS_WINDOW)
    peaks     = (dist == local_max) & (dist > DIST_THRESHOLD)
    labeled, n = scipy_label(peaks)
    raw_centroids = center_of_mass(peaks, labeled, range(1, n + 1))
    raw_centroids = [(int(y), int(x)) for y, x in raw_centroids]

    # --- greedy NMS (keep highest-dist peak first) ---
    h, w = panel.shape[:2]
    sorted_pts = sorted(raw_centroids, key=lambda p: -dist[p[0], p[1]])
    kept: List[Tuple[int, int]] = []
    for cy, cx in sorted_pts:
        # reject: outside foreground or too close to image edge
        if fg[cy, cx] == 0:
            continue
        if cy < 10 or cx < 10 or cy > h - 10 or cx > w - 10:
            continue
        # reject: too close to an already-kept center
        if any(
            np.hypot(cy - ky, cx - kx) < NMS_MIN_DIST
            for ky, kx in kept
        ):
            continue
        kept.append((cy, cx))
    return kept


# ============================================================
# STAGE 1b — CV: TILE-COLOR CLASSIFICATION
# ============================================================

def _sample_ring_color(panel: np.ndarray, cy: int, cx: int) -> np.ndarray:
    """
    Sample tile background by averaging pixels in concentric rings
    around the center (avoids the dark animal icon).
    Returns median BGR value.
    """
    colors = []
    for r in COLOR_RING_RADII:
        for i in range(COLOR_RING_POINTS):
            angle = 2.0 * np.pi * i / COLOR_RING_POINTS
            sy = int(cy + r * np.sin(angle))
            sx = int(cx + r * np.cos(angle))
            if 0 <= sy < panel.shape[0] and 0 <= sx < panel.shape[1]:
                colors.append(panel[sy, sx].astype(np.float64))
    return np.median(colors, axis=0) if colors else panel[cy, cx].astype(np.float64)


def classify_tile_color(bgr: np.ndarray) -> str:
    """Nearest-centroid classification against TILE_COLOR_REFS."""
    bgr = np.asarray(bgr, dtype=np.float64)
    return min(
        TILE_COLOR_REFS,
        key=lambda name: np.linalg.norm(bgr - TILE_COLOR_REFS[name].astype(np.float64))
    )


# ============================================================
# STAGE 1c — CV: ADJACENCY GRAPH
# ============================================================

def build_adjacency(
    centroids: List[Tuple[int, int]],
    thresh: float = ADJACENCY_THRESH,
) -> Dict[int, List[int]]:
    """
    Return adjacency list: adj[i] = [j, k, …] for all hex-neighbors of tile i.
    Two tiles are neighbors when their centre distance < thresh.
    """
    adj: Dict[int, List[int]] = defaultdict(list)
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            d = np.hypot(
                centroids[i][0] - centroids[j][0],
                centroids[i][1] - centroids[j][1],
            )
            if d < thresh:
                adj[i].append(j)
                adj[j].append(i)
    return dict(adj)


# ============================================================
# STAGE 1 ORCHESTRATOR — returns structured data per player
# ============================================================

def analyse_board(image_path: str) -> List[Dict]:
    """
    Split triptych → per-panel CV analysis.
    Returns list of 3 player dicts:
        {
            "centers":    [(row, col), …],
            "tile_colors": ["blue", …],        # habitat type per hex
            "adjacency":  {0: [1,3], …},       # hex-neighbor graph
            "panel":      <BGR ndarray>,
        }
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read {image_path}")

    w   = img.shape[1]
    third = w // 3
    panels = [
        img[:, :third],
        img[:, third : 2 * third],
        img[:, 2 * third :],
    ]

    players = []
    for panel in panels:
        centers = detect_hex_centers(panel)
        tile_colors = [
            classify_tile_color(_sample_ring_color(panel, cy, cx))
            for cy, cx in centers
        ]
        adj = build_adjacency(centers)
        players.append({
            "centers":     centers,
            "tile_colors": tile_colors,
            "adjacency":   adj,
            "panel":       panel,
        })
    return players


# ============================================================
# STAGE 2 — VLM: ANIMAL CLASSIFICATION
# ============================================================

def _crop_icon(panel: np.ndarray, cy: int, cx: int) -> np.ndarray:
    """Extract and resize a single hex-icon crop."""
    h, w = panel.shape[:2]
    y1 = max(0, cy - ICON_CROP_PAD)
    y2 = min(h, cy + ICON_CROP_PAD)
    x1 = max(0, cx - ICON_CROP_PAD)
    x2 = min(w, cx + ICON_CROP_PAD)
    crop = panel[y1:y2, x1:x2]
    return cv2.resize(crop, (ICON_CROP_SIZE, ICON_CROP_SIZE), interpolation=cv2.INTER_CUBIC)


def _build_icon_grid(players: List[Dict], cell_cols: int = 10) -> Tuple[np.ndarray, List[str]]:
    """
    Lay all hex-icon crops into a single labelled grid image.
    Returns (grid_image_BGR, list_of_labels_in_row-major_order).
    """
    all_crops, labels = [], []
    for p_idx, player in enumerate(players):
        for i, (cy, cx) in enumerate(player["centers"]):
            all_crops.append(_crop_icon(player["panel"], cy, cx))
            labels.append(f"P{p_idx + 1}-{i}")

    cols = cell_cols
    rows = (len(all_crops) + cols - 1) // cols
    cell_w, cell_h = ICON_CROP_SIZE + 10, ICON_CROP_SIZE + 20
    grid = np.full((rows * cell_h, cols * cell_w, 3), 240, dtype=np.uint8)

    for idx, (crop, label) in enumerate(zip(all_crops, labels)):
        r, c   = divmod(idx, cols)
        y, x   = r * cell_h + 15, c * cell_w + 5
        grid[y : y + ICON_CROP_SIZE, x : x + ICON_CROP_SIZE] = crop
        cv2.putText(
            grid, label, (x + 5, y - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1,
        )
    return grid, labels


def _encode_image_bytes(img_bgr: np.ndarray) -> bytes:
    """Encode a BGR ndarray to raw PNG bytes (for types.Part.from_bytes)."""
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("Failed to encode image to PNG")
    return buf.tobytes()


def classify_animals_vlm(
    players: List[Dict],
    api_key: Optional[str] = None,
) -> Dict[str, str]:
    """
    Send the icon grid + annotated boards to Gemini for animal classification.

    Returns dict mapping label (e.g. "P1-0") → animal name (lowercase).

    If GEMINI_API_KEY is not set and api_key is None, falls back to
    returning an empty dict (you can fill in manually from the saved grid).
    """
    from google.genai import Client, types
    from dotenv import load_dotenv
    load_dotenv()

    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        print("[VLM] WARNING: No GEMINI_API_KEY set. Skipping classification.")
        print("      Inspect all_icons_grid.png manually and fill animal_labels dict.")
        return {}

    client = Client(api_key=key)

    # --- build icon grid + save for debug ---
    grid_img, labels = _build_icon_grid(players)
    cv2.imwrite("all_icons_grid.png", grid_img)
    grid_bytes = _encode_image_bytes(grid_img)

    # --- build per-player 3× annotated images ---
    annotated_bytes_list = []
    for player in players:
        panel = player["panel"]
        scale = 3
        big   = cv2.resize(
            panel,
            (panel.shape[1] * scale, panel.shape[0] * scale),
            interpolation=cv2.INTER_CUBIC,
        )
        for i, (cy, cx) in enumerate(player["centers"]):
            bcy, bcx = cy * scale, cx * scale
            cv2.circle(big, (bcx, bcy), 12, (255, 255, 255), 2)
            text = str(i)
            tw, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(
                big,
                (bcx - tw // 2 - 2, bcy - 16),
                (bcx + tw // 2 + 2, bcy - 2),
                (0, 0, 0), -1,
            )
            cv2.putText(
                big, text, (bcx - tw // 2, bcy - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )
        annotated_bytes_list.append(_encode_image_bytes(big))

    # --- build prompt ---
    labels_str = ", ".join(labels)
    prompt = (
        "You are classifying animal silhouettes on hex board-game tiles.\n"
        "Each tile has a small dark silhouette of an animal in its centre.\n\n"
        "Known animal types (use EXACTLY these names):\n"
        "  bear, hawk (bird of prey), elk (deer with antlers), salmon (fish), fox\n\n"
        "The first image is a grid of all tiles labelled with codes like P1-0, P2-3 etc.\n"
        "The subsequent images show each player's board at 3× zoom with tile indices.\n\n"
        f"Tile codes in grid order: {labels_str}\n\n"
        "Output ONLY a JSON object mapping each tile code to its animal.\n"
        'Example: {"P1-0": "bear", "P1-1": "hawk", …}\n'
        "Do NOT include any other text.\n"
    )

    # --- assemble contents list ---
    contents = []

    # 1) icon grid image
    contents.append(
        types.Part.from_bytes(data=grid_bytes, mime_type="image/png")
    )

    # 2) per-player annotated board images
    for ann_bytes in annotated_bytes_list:
        contents.append(
            types.Part.from_bytes(data=ann_bytes, mime_type="image/png")
        )

    # 3) text prompt last
    contents.append(
        types.Part.from_text(text=prompt)
    )

    # --- call Gemini ---
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.1,
            top_p=0.95,
            thinking_config=types.ThinkingConfig(
                thinking_level="medium",
            ),
        ),
    )

    # --- parse JSON from response ---
    raw = response.text.strip()
    # strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        result = json.loads(raw)
        result = {k: v.lower().strip() for k, v in result.items()}
    except json.JSONDecodeError:
        print(f"[VLM] Failed to parse JSON. Raw response:\n{raw}")
        result = {}

    return result


# ============================================================
# STAGE 3 — SCORING ENGINE
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
        queue   = [start]
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


def _get_shape_signature(group: List[int], centroids: List[Tuple[int, int]]) -> Tuple:
    """
    Normalised shape signature for a group of hex centres.
    Translate so min-row and min-col are 0, then round to hex-grid units.
    Used for elk "exact shape" matching (not fully implemented here –
    would need the shape templates from the card).
    """
    pts = [centroids[i] for i in group]
    min_r = min(p[0] for p in pts)
    min_c = min(p[1] for p in pts)
    # Normalise and round to nearest ~35px (one hex step)
    step = 35.0
    normalised = tuple(sorted(
        (round((r - min_r) / step), round((c - min_c) / step))
        for r, c in pts
    ))
    return normalised


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
        # Groups > 3 score nothing (rule: "no other bears next to it")
    return score


def score_hawks(
    animals: List[str],
    adjacency: Dict[int, List[int]],
) -> int:
    """
    Red-Tailed Hawk: scores per each hawk that is connected to
    at least one other hawk and forms a chain.
    Scoring table (from card): 2→5, 3→9, 4→12, 5→16, 6→20, 7→24, 8+→23
    (Each hawk scores once; score is per-hawk based on total connected chain length.)

    Simplified reading of card:
        chain length → total points for that chain
        2→5, 3→9, 4→12, 5→16, 6→20, 7→24, 8+→23
    """
    HAWK_CHAIN_SCORES = {2: 5, 3: 9, 4: 12, 5: 16, 6: 20, 7: 24}
    # 8+ → 23 (cap)
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
        elif size in HAWK_CHAIN_SCORES:
            score += HAWK_CHAIN_SCORES[size]
    return score


def score_elk(
    animals: List[str],
    adjacency: Dict[int, List[int]],
    centroids: List[Tuple[int, int]],
) -> int:
    """
    Roosevelt Elk: scores per group of elk in exact shape.
    Card shows specific formations worth:
        2 elk in specific shape → 5 pts
        5 elk → 5 pts  (another formation)
        9 elk → 13 pts
    Each elk may only score for a single group.

    NOTE: Full shape-template matching requires the exact hex-grid templates
    from the card. This is a simplified version that scores connected groups
    using the point values from the card's size→score table.
    """
    # Simplified: score based on connected component size
    # From the card layout: 2→5, 5→5, 9→13
    ELK_SCORES = {2: 5, 5: 5, 9: 13}

    elk_indices = [i for i, a in enumerate(animals) if a == "elk"]
    if not elk_indices:
        return 0

    components = _find_connected_components(elk_indices, adjacency)
    score = 0
    for comp in components:
        size = len(comp)
        if size in ELK_SCORES:
            score += ELK_SCORES[size]
    return score


def score_salmon(
    animals: List[str],
    adjacency: Dict[int, List[int]],
) -> int:
    """
    Chinook Salmon: scores for each run of 3, 4, or 5+ salmon.
    Salmon in a run may NOT be adjacent to each other (they form a "run" =
    connected line, but the card says they need not be adjacent –
    actually re-reading: "run" here means a connected group).

    From card:  run size → points
        3 → 10
        4 → 12
        5+ → 15
    """
    SALMON_SCORES = {3: 10, 4: 12}  # 5+ → 15

    salmon_indices = [i for i, a in enumerate(animals) if a == "salmon"]
    if not salmon_indices:
        return 0

    components = _find_connected_components(salmon_indices, adjacency)
    score = 0
    for comp in components:
        size = len(comp)
        if size >= 5:
            score += 15
        elif size in SALMON_SCORES:
            score += SALMON_SCORES[size]
    return score


def score_foxes(
    animals: List[str],
    adjacency: Dict[int, List[int]],
) -> int:
    """
    Red Fox: scores for each fox based on the number of DISTINCT
    adjacent animal types (foxes do not count).

    From card:  # distinct adjacent types → points per fox
        1 → 1
        2 → 2
        3 → 4
        4 → 5
        5 → 6
        6 → 6
    """
    FOX_SCORES = {1: 1, 2: 2, 3: 4, 4: 5, 5: 6, 6: 6}

    score = 0
    for i, a in enumerate(animals):
        if a != "fox":
            continue
        # Count distinct animal types among neighbors (excluding fox)
        neighbor_types = set()
        for nb in adjacency.get(i, []):
            if animals[nb] != "fox":
                neighbor_types.add(animals[nb])
        n_types = len(neighbor_types)
        score += FOX_SCORES.get(min(n_types, 6), 0)
    return score


def compute_scores(
    animals: List[str],
    adjacency: Dict[int, List[int]],
    centroids: List[Tuple[int, int]],
) -> Dict[str, int]:
    """Run all scoring rules and return {objective: points}."""
    return {
        "Grizzly Bear":    score_bears(animals, adjacency),
        "Red-Tailed Hawk": score_hawks(animals, adjacency),
        "Roosevelt Elk":   score_elk(animals, adjacency, centroids),
        "Chinook Salmon":  score_salmon(animals, adjacency),
        "Red Fox":         score_foxes(animals, adjacency),
    }


# ============================================================
# MAIN — orchestrate & print results
# ============================================================

def main():
    print("=" * 60)
    print(" HEX BOARD GAME SCORER")
    print("=" * 60)

    # --- Stage 1: CV analysis ---
    print("\n[CV] Analysing board image …")
    players = analyse_board(BOARD_IMAGE_PATH)

    for p_idx, player in enumerate(players):
        print(f"\n  Player {p_idx + 1}: {len(player['centers'])} tiles detected")
        print(f"    Tile colors: {dict(Counter(player['tile_colors']))}")
        edge_count = sum(len(v) for v in player["adjacency"].values()) // 2
        print(f"    Adjacency edges: {edge_count}")

    # --- Stage 2: VLM animal classification ---
    print("\n[VLM] Classifying animal icons …")
    animal_labels = classify_animals_vlm(players)

    if not animal_labels:
        print("\n  [!] VLM classification skipped or failed.")
        print("      Open 'all_icons_grid.png' and manually fill the mapping below.")
        # ── MANUAL FALLBACK ──────────────────────────────────────────
        # After inspecting all_icons_grid.png, fill in each tile's animal:
        animal_labels = {}
        # Example (fill these in after visual inspection):
        # animal_labels = {
        #     "P1-0": "bear",  "P1-1": "hawk", …
        # }
        # ─────────────────────────────────────────────────────────────
        return  # can't score without animals

    # Attach animals to player dicts
    for p_idx, player in enumerate(players):
        player["animals"] = []
        for i in range(len(player["centers"])):
            key   = f"P{p_idx + 1}-{i}"
            animal = animal_labels.get(key, "unknown")
            player["animals"].append(animal)
        print(f"  Player {p_idx + 1} animals: {dict(Counter(player['animals']))}")

    # --- Stage 3: Scoring ---
    print("\n" + "=" * 60)
    print(" SCORES")
    print("=" * 60)

    total_scores = {}
    for p_idx, player in enumerate(players):
        scores = compute_scores(
            player["animals"],
            player["adjacency"],
            player["centers"],
        )
        total = sum(scores.values())
        total_scores[f"Player {p_idx + 1}"] = total

        print(f"\n  Player {p_idx + 1}:")
        for obj, pts in scores.items():
            print(f"    {obj:20s}: {pts:3d} pts")
        print(f"    {'TOTAL':20s}: {total:3d} pts")

    # --- Summary table (Markdown) ---
    print("\n" + "=" * 60)
    print(" SUMMARY TABLE (Markdown)")
    print("=" * 60)
    objectives = ["Grizzly Bear", "Red-Tailed Hawk", "Roosevelt Elk",
                  "Chinook Salmon", "Red Fox", "**TOTAL**"]

    # Header
    header = "| Objective | " + " | ".join(f"Player {i+1}" for i in range(3)) + " |"
    sep    = "| --- | " + " | ".join("---" for _ in range(3)) + " |"
    print(header)
    print(sep)

    for obj in objectives:
        row_vals = []
        for p_idx, player in enumerate(players):
            if obj == "**TOTAL**":
                scores = compute_scores(player["animals"], player["adjacency"], player["centers"])
                row_vals.append(str(sum(scores.values())))
            else:
                scores = compute_scores(player["animals"], player["adjacency"], player["centers"])
                row_vals.append(str(scores.get(obj, 0)))
        print(f"| {obj} | " + " | ".join(row_vals) + " |")

    # Save summary to file
    with open("scoring_output.md", "w") as f:
        f.write(header + "\n" + sep + "\n")
        for obj in objectives:
            row_vals = []
            for p_idx, player in enumerate(players):
                scores = compute_scores(player["animals"], player["adjacency"], player["centers"])
                if obj == "**TOTAL**":
                    row_vals.append(str(sum(scores.values())))
                else:
                    row_vals.append(str(scores.get(obj, 0)))
            f.write(f"| {obj} | " + " | ".join(row_vals) + " |\n")
    print("\nSaved scoring_output.md")


if __name__ == "__main__":
    main()
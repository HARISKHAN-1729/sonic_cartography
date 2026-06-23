"""
vjepa/class_map.py
───────────────────
Canonical navigation-relevant object classes for Predictive Sonic Cartography,
plus a mapping from SANPO panoptic label ids to these classes.

The probe predicts one of these classes (or BACKGROUND) per latent patch.
Index 0 is always BACKGROUND. The remaining indices match, in order, the
earcon set in audio/earcon_bank.py so the sonifier needs no changes.
"""

from __future__ import annotations

# Index 0 is BACKGROUND; the rest are the 20 navigation classes used elsewhere.
NAV_CLASSES = [
    "background",      # 0  (not sonified)
    "person",          # 1
    "bicycle",         # 2
    "car",             # 3
    "motorbike",       # 4
    "bus",             # 5
    "truck",           # 6
    "traffic_light",   # 7
    "stop_sign",       # 8
    "bench",           # 9
    "door",            # 10
    "stairs",          # 11
    "curb",            # 12
    "pole",            # 13
    "fire_hydrant",    # 14
    "bollard",         # 15
    "wall",            # 16
    "tree",            # 17
    "trash_can",       # 18
    "table",           # 19
    "chair",           # 20
]

NUM_CLASSES = len(NAV_CLASSES)          # 21 (incl. background)
BACKGROUND_ID = 0

CLASS_TO_ID = {c: i for i, c in enumerate(NAV_CLASSES)}
ID_TO_CLASS = {i: c for i, c in enumerate(NAV_CLASSES)}


# ─────────────────────────────────────────────────────────────────────────────
# SANPO panoptic → NAV_CLASSES mapping.
#
# SANPO defines ~30 semantic classes (surfaces, fences, obstacles, creatures).
# We map the navigation-relevant ones onto NAV_CLASSES and send everything
# else (sky, road surface, vegetation regions we don't sonify, etc.) to
# BACKGROUND. SANPO ships a label-definitions file with its panoptic masks;
# the dict below uses SANPO's published class NAMES so it survives id
# renumbering. At load time we resolve names -> SANPO ids using that file.
#
# >>> ADAPT: if your SANPO release uses slightly different class names, edit
# the right-hand strings here. Unlisted SANPO names default to background.
# ─────────────────────────────────────────────────────────────────────────────
SANPO_NAME_TO_NAV = {
    # creatures
    "person":            "person",
    "pedestrian":        "person",
    "rider":             "person",
    "animal":            "person",        # treat as a dynamic obstacle
    "bicycle":           "bicycle",
    "bicyclist":         "bicycle",
    "motorcycle":        "motorbike",
    "car":               "car",
    "vehicle":           "car",
    "bus":               "bus",
    "truck":             "truck",
    # signage / signals
    "traffic_light":     "traffic_light",
    "traffic_sign":      "stop_sign",
    "sign":              "stop_sign",
    # furniture / fixtures
    "bench":             "bench",
    "door":              "door",
    "stairs":            "stairs",
    "staircase":         "stairs",
    "curb":              "curb",
    "pole":              "pole",
    "utility_pole":      "pole",
    "fire_hydrant":      "fire_hydrant",
    "bollard":           "bollard",
    "wall":              "wall",
    "fence":             "wall",
    "guard_rail":        "wall",
    "gate":              "door",
    "tree":              "tree",
    "trash_can":         "trash_can",
    "garbage_can":       "trash_can",
    "table":             "table",
    "chair":             "chair",
    "bike_rack":         "bollard",
    # everything else (road, sidewalk, sky, terrain, building, water, ...) -> background
}


def sanpo_name_to_nav_id(sanpo_name: str) -> int:
    """Map a SANPO class name to a NAV_CLASSES index (background if unknown)."""
    return CLASS_TO_ID.get(
        SANPO_NAME_TO_NAV.get(sanpo_name.lower().strip(), "background"),
        BACKGROUND_ID,
    )


def build_sanpo_id_to_nav_id(sanpo_label_defs: dict[int, str]) -> dict[int, int]:
    """
    Given SANPO's {panoptic_id: class_name} definition table, build a fast
    {panoptic_id: nav_class_id} lookup used when reading panoptic masks.
    """
    return {
        pid: sanpo_name_to_nav_id(name)
        for pid, name in sanpo_label_defs.items()
    }

"""
AI Style Recommendation Engine.

HONEST FLAG: this is the least-validated module in the project. Face
shape classification here is a rule-based heuristic on landmark ratios
(jaw width / face length / cheekbone width), not a trained classifier.
Rule-based face-shape detection is known to be unreliable at the edges
(oval vs. oblong vs. round get confused easily). Validate the shape
labels against real photos of known face shapes before trusting the
recommendations — do not assume "it runs" means "it's right."

The recommendation logic itself is a simple lookup table (face shape +
density -> suggested style ids), intentionally readable and easy to
correct once you've validated real outputs.
"""

from dataclasses import dataclass

import numpy as np

from src.detection.density_analysis import DensityResult
from src.styles.style_catalogue import BeardStyle, load_style_catalogue

# Landmark indices for rough face-shape ratio estimation (468-point
# MediaPipe topology). Approximate — verify against your own test set.
JAW_LEFT, JAW_RIGHT = 172, 397
CHEEK_LEFT, CHEEK_RIGHT = 234, 454
CHIN_BOTTOM = 152
FOREHEAD_TOP = 10


@dataclass
class FaceShapeResult:
    shape: str  # "oval" | "round" | "square" | "oblong" | "heart"
    jaw_to_cheek_ratio: float
    length_to_width_ratio: float


def estimate_face_shape(landmarks_px: np.ndarray) -> FaceShapeResult:
    jaw_width = np.linalg.norm(landmarks_px[JAW_RIGHT] - landmarks_px[JAW_LEFT])
    cheek_width = np.linalg.norm(landmarks_px[CHEEK_RIGHT] - landmarks_px[CHEEK_LEFT])
    face_length = np.linalg.norm(landmarks_px[CHIN_BOTTOM] - landmarks_px[FOREHEAD_TOP])

    jaw_to_cheek_ratio = jaw_width / cheek_width if cheek_width else 1.0
    length_to_width_ratio = face_length / cheek_width if cheek_width else 1.0

    # Rule-based classification — coarse, deliberately simple to start
    if length_to_width_ratio > 1.5:
        shape = "oblong"
    elif jaw_to_cheek_ratio > 0.95 and length_to_width_ratio < 1.2:
        shape = "square"
    elif jaw_to_cheek_ratio < 0.75:
        shape = "heart"
    elif length_to_width_ratio < 1.15 and jaw_to_cheek_ratio > 0.85:
        shape = "round"
    else:
        shape = "oval"

    return FaceShapeResult(
        shape=shape,
        jaw_to_cheek_ratio=round(jaw_to_cheek_ratio, 3),
        length_to_width_ratio=round(length_to_width_ratio, 3),
    )


# Lookup table: (face_shape) -> ordered list of recommended style ids.
# Density is used to bump sparse-beard users toward lower-coverage
# styles (goatee/stubble) regardless of shape, since a sparse beard
# can't convincingly fill a full-beard style.
_SHAPE_STYLE_MAP: dict[str, list[str]] = {
    "oval": ["full_beard", "short_boxed", "stubble"],
    "round": ["short_boxed", "goatee", "stubble"],
    "square": ["stubble", "full_beard", "goatee"],
    "oblong": ["full_beard", "short_boxed", "stubble"],
    "heart": ["goatee", "stubble", "short_boxed"],
}


def recommend_styles(
    landmarks_px: np.ndarray, density: DensityResult, top_n: int = 3
) -> list[BeardStyle]:
    face_shape = estimate_face_shape(landmarks_px)
    catalogue = {style.id: style for style in load_style_catalogue()}

    ordered_ids = list(_SHAPE_STYLE_MAP.get(face_shape.shape, ["full_beard", "short_boxed", "goatee"]))

    # If beard is sparse, push lower-coverage styles to the front
    if density.density_label == "sparse":
        ordered_ids.sort(key=lambda sid: catalogue[sid].cheek_line_ratio)

    recommended = [catalogue[sid] for sid in ordered_ids if sid in catalogue]
    return recommended[:top_n]
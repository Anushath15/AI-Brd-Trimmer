import numpy as np

from src.detection.density_analysis import DensityResult
from src.styles.recommendation_engine import estimate_face_shape, recommend_styles


def _synthetic_landmarks(jaw_width, cheek_width, face_length) -> np.ndarray:
    """Build a minimal 468-point landmark array with only the indices
    used by estimate_face_shape() set to meaningful values — everything
    else is a harmless placeholder at the origin.
    """
    landmarks = np.zeros((468, 2), dtype=np.float32)
    # JAW_LEFT=172, JAW_RIGHT=397
    landmarks[172] = [-jaw_width / 2, 50]
    landmarks[397] = [jaw_width / 2, 50]
    # CHEEK_LEFT=234, CHEEK_RIGHT=454
    landmarks[234] = [-cheek_width / 2, 0]
    landmarks[454] = [cheek_width / 2, 0]
    # CHIN_BOTTOM=152, FOREHEAD_TOP=10
    landmarks[152] = [0, face_length]
    landmarks[10] = [0, 0]
    return landmarks


def test_estimate_face_shape_oblong_for_long_narrow_face():
    landmarks = _synthetic_landmarks(jaw_width=80, cheek_width=80, face_length=200)
    result = estimate_face_shape(landmarks)
    assert result.shape == "oblong"


def test_estimate_face_shape_square_for_wide_jaw_short_face():
    landmarks = _synthetic_landmarks(jaw_width=95, cheek_width=100, face_length=90)
    result = estimate_face_shape(landmarks)
    assert result.shape == "square"


def test_recommend_styles_returns_requested_count():
    landmarks = _synthetic_landmarks(jaw_width=70, cheek_width=100, face_length=110)
    density = DensityResult(coverage_ratio=0.5, density_label="moderate", patchiness_score=0.1)

    recommended = recommend_styles(landmarks, density, top_n=2)
    assert len(recommended) == 2


def test_recommend_styles_prioritizes_low_coverage_for_sparse_beards():
    landmarks = _synthetic_landmarks(jaw_width=70, cheek_width=100, face_length=110)
    sparse_density = DensityResult(coverage_ratio=0.1, density_label="sparse", patchiness_score=0.3)

    recommended = recommend_styles(landmarks, sparse_density, top_n=1)
    # Sparse beards should be steered away from full-coverage styles first
    assert recommended[0].cheek_line_ratio <= 0.85
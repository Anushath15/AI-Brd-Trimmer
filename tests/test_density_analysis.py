import numpy as np

from src.detection.density_analysis import analyze_beard_density


def test_density_label_sparse_for_low_coverage():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    lower_face_mask = np.zeros((100, 100), dtype=np.uint8)
    lower_face_mask[40:90, 20:80] = 255

    beard_mask = np.zeros((100, 100), dtype=np.uint8)
    beard_mask[80:85, 40:45] = 255  # tiny sliver of "hair"

    result = analyze_beard_density(image, lower_face_mask, beard_mask)
    assert result.density_label == "sparse"
    assert 0.0 <= result.coverage_ratio <= 1.0


def test_density_label_full_for_high_coverage():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    lower_face_mask = np.zeros((100, 100), dtype=np.uint8)
    lower_face_mask[40:90, 20:80] = 255

    beard_mask = lower_face_mask.copy()  # full coverage

    result = analyze_beard_density(image, lower_face_mask, beard_mask)
    assert result.density_label == "full"
    assert result.coverage_ratio > 0.9


def test_zero_face_area_returns_zero_coverage_without_crashing():
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    lower_face_mask = np.zeros((50, 50), dtype=np.uint8)
    beard_mask = np.zeros((50, 50), dtype=np.uint8)

    result = analyze_beard_density(image, lower_face_mask, beard_mask)
    assert result.coverage_ratio == 0.0
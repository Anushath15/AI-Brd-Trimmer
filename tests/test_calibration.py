import numpy as np

from src.capture.calibration import check_distance_ok, check_face_centered, check_lighting_ok


def test_check_face_centered_true_when_landmarks_at_image_center():
    landmarks = np.array([[100, 100], [102, 98], [98, 102]], dtype=np.float32)
    assert check_face_centered(landmarks, image_width=200, image_height=200) is True


def test_check_face_centered_false_when_landmarks_off_to_one_side():
    landmarks = np.array([[10, 10], [12, 8], [8, 12]], dtype=np.float32)
    assert check_face_centered(landmarks, image_width=200, image_height=200) is False


def test_check_lighting_ok_false_for_all_black_image():
    black_image = np.zeros((100, 100, 3), dtype=np.uint8)
    assert check_lighting_ok(black_image) is False


def test_check_lighting_ok_true_for_mid_gray_image():
    gray_image = np.full((100, 100, 3), 128, dtype=np.uint8)
    assert check_lighting_ok(gray_image) is True


def test_check_distance_ok_false_when_face_too_small_in_frame():
    # Landmarks spanning only a tiny fraction of image width
    landmarks = np.array([[95, 100], [105, 100]], dtype=np.float32)
    assert check_distance_ok(landmarks, image_width=1000) is False
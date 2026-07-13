"""
Camera calibration checks — the flowchart's "Face Centered? / Lighting
OK? / Distance OK?" gate before AI detection starts.

These are heuristic, fast checks meant to stop the user before running
the full pipeline on a bad frame (out of frame, too dark, too
close/far) — not full-blown quality scoring.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class CalibrationResult:
    face_centered: bool
    lighting_ok: bool
    distance_ok: bool
    reasons: list[str]

    @property
    def all_ok(self) -> bool:
        return self.face_centered and self.lighting_ok and self.distance_ok


def check_face_centered(
    landmarks_px: np.ndarray, image_width: int, image_height: int, tolerance: float = 0.15
) -> bool:
    """Face is 'centered' if its landmark centroid falls within
    `tolerance` fraction of the image center on both axes.
    """
    centroid = landmarks_px.mean(axis=0)
    center_x, center_y = image_width / 2, image_height / 2

    x_off = abs(centroid[0] - center_x) / image_width
    y_off = abs(centroid[1] - center_y) / image_height

    return x_off <= tolerance and y_off <= tolerance


def check_lighting_ok(image_bgr: np.ndarray, min_brightness: int = 60, max_brightness: int = 200) -> bool:
    """Simple average-brightness check on the grayscale image.
    Not a substitute for real exposure analysis, but catches the
    obvious "too dark" / "blown out" cases cheaply.
    """
    import cv2

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    return min_brightness <= mean_brightness <= max_brightness


def check_distance_ok(
    landmarks_px: np.ndarray, image_width: int, min_face_width_ratio: float = 0.25, max_face_width_ratio: float = 0.75
) -> bool:
    """Estimate face width (in pixels) from landmark spread and compare
    against the image width — too small means user is too far, too
    large means too close.
    """
    face_width_px = landmarks_px[:, 0].max() - landmarks_px[:, 0].min()
    ratio = face_width_px / image_width
    return min_face_width_ratio <= ratio <= max_face_width_ratio


def run_calibration_checks(
    image_bgr: np.ndarray, landmarks_px: np.ndarray, image_width: int, image_height: int
) -> CalibrationResult:
    reasons = []

    centered = check_face_centered(landmarks_px, image_width, image_height)
    if not centered:
        reasons.append("Face not centered — adjust position")

    lighting = check_lighting_ok(image_bgr)
    if not lighting:
        reasons.append("Lighting not adequate — improve lighting")

    distance = check_distance_ok(landmarks_px, image_width)
    if not distance:
        reasons.append("Distance not ideal — move closer or farther")

    return CalibrationResult(
        face_centered=centered, lighting_ok=lighting, distance_ok=distance, reasons=reasons
    )
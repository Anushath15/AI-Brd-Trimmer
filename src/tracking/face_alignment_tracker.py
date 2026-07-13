"""
Face alignment tracker — "AI keeps face map aligned with the live face
view" from the flowchart.

IMPORTANT SCOPE NOTE: this tracks the FACE, frame to frame, and
re-projects the already-generated zone masks onto the new face
position/orientation. It does NOT track the trimmer or blade — that's
V2's job (LED markers), out of scope here. V1's guidance is therefore
"here's your zone map relative to your face right now," not "here's
exactly where your blade is."
"""

from dataclasses import dataclass

import cv2
import numpy as np

from src.detection.face_landmarks import FaceLandmarkResult, detect_face_landmarks


@dataclass
class AlignedZones:
    zones: dict[str, np.ndarray]
    landmarks: FaceLandmarkResult


def _estimate_affine_transform(
    reference_landmarks: np.ndarray, current_landmarks: np.ndarray
) -> np.ndarray:
    """Estimate a similarity transform (rotation/scale/translation) from
    the reference frame's landmarks to the current frame's landmarks,
    using a stable subset of points (eyes + nose + chin) rather than
    all 468, for speed and robustness to occlusion/noise.
    """
    stable_indices = [33, 263, 1, 152]  # left eye, right eye, nose tip, chin
    src = reference_landmarks[stable_indices].astype(np.float32)
    dst = current_landmarks[stable_indices].astype(np.float32)

    transform, _ = cv2.estimateAffinePartial2D(src, dst)
    if transform is None:
        # Fall back to identity if estimation fails (e.g. face lost)
        transform = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    return transform


def realign_zones_to_current_frame(
    reference_landmarks: np.ndarray,
    reference_zones: dict[str, np.ndarray],
    current_frame_bgr: np.ndarray,
) -> AlignedZones | None:
    """Re-detect the face in the current frame and warp the reference
    zone masks to match the current head position/scale/rotation.

    Returns None if no face is found in the current frame (caller
    should hold the last-good overlay or prompt the user to reposition).
    """
    landmarks = detect_face_landmarks(current_frame_bgr)
    if not landmarks.found:
        return None

    transform = _estimate_affine_transform(reference_landmarks, landmarks.landmarks_px)

    height, width = current_frame_bgr.shape[:2]
    warped_zones = {
        name: cv2.warpAffine(mask, transform, (width, height))
        for name, mask in reference_zones.items()
    }

    return AlignedZones(zones=warped_zones, landmarks=landmarks)
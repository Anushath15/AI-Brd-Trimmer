"""
Live overlay: same visual logic as overlay/visualizer.py but applied
per-frame to a live video feed, using the current frame's realigned
zones rather than a single static image.
"""

import cv2
import numpy as np

from config import settings


def draw_live_zone_overlay(
    frame_bgr: np.ndarray, zones: dict[str, np.ndarray], alpha: float = 0.4
) -> np.ndarray:
    """Same blending approach as the static overlay — kept as a
    separate function (rather than importing and reusing directly)
    since live overlays may diverge later (e.g. adding a boundary
    outline stroke for readability at video frame rates, dimmer alpha
    for less flicker). Currently identical logic to the static version.
    """
    overlay = frame_bgr.copy()

    for zone_name, mask in zones.items():
        color = settings.ZONE_COLORS.get(zone_name)
        if color is None:
            continue
        colored_layer = np.zeros_like(frame_bgr)
        colored_layer[:] = color
        overlay = np.where(mask[..., None] > 0, colored_layer, overlay)

    blended = cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)
    return blended


def draw_boundary_outline(
    frame_bgr: np.ndarray, target_mask: np.ndarray, color: tuple[int, int, int] = (0, 165, 255), thickness: int = 2
) -> np.ndarray:
    """Draw the dotted/solid target boundary outline on top of the
    overlay — makes the orange boundary line from the flowchart
    actually visible as a line, not just an implied edge between zones.
    """
    contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = frame_bgr.copy()
    cv2.drawContours(result, contours, -1, color, thickness)
    return result
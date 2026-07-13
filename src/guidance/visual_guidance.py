"""
Visual guidance: the three sub-nodes from the flowchart —
"Guide: trim red areas", "Guide: carefully follow dotted boundaries",
"Warning: do not trim green areas" — rendered as on-screen text/banner.

v1 scope: this reports zone PROPORTIONS in view (how much of the
visible face area is red/yellow/green right now), not "where the
blade is" — V1 has no blade tracking. So the guidance is necessarily
general ("you still have red areas to trim") rather than pinpoint
("trim exactly here").
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class GuidanceMessage:
    text: str
    urgency: str  # "info" | "caution" | "warning"


def determine_guidance(zones: dict[str, np.ndarray]) -> GuidanceMessage:
    remove_px = int(np.count_nonzero(zones.get("remove", np.array([]))))
    blend_px = int(np.count_nonzero(zones.get("blend", np.array([]))))
    keep_px = int(np.count_nonzero(zones.get("keep", np.array([]))))

    total = remove_px + blend_px + keep_px
    if total == 0:
        return GuidanceMessage(text="No beard detected in frame", urgency="info")

    remove_ratio = remove_px / total

    if remove_ratio > 0.15:
        return GuidanceMessage(text="Trim the red areas — outside your target shape", urgency="warning")
    if blend_px / total > 0.1:
        return GuidanceMessage(text="Carefully follow the dotted boundary line", urgency="caution")
    return GuidanceMessage(text="Looking good — avoid trimming the green areas", urgency="info")


def draw_guidance_banner(frame_bgr: np.ndarray, message: GuidanceMessage) -> np.ndarray:
    color_map = {"info": (0, 200, 0), "caution": (0, 200, 200), "warning": (0, 0, 220)}
    color = color_map.get(message.urgency, (255, 255, 255))

    result = frame_bgr.copy()
    height, width = result.shape[:2]
    banner_height = 40

    cv2.rectangle(result, (0, 0), (width, banner_height), (30, 30, 30), -1)
    cv2.putText(
        result, message.text, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
    )
    return result
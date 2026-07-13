"""
Touch-up tracker: "User checks progress in live view" -> "Need
touch-up?" loop from the flowchart.

v1 approach: re-run beard segmentation on the CURRENT frame and compare
against the target mask — if there's still meaningful red-zone area
(hair outside target) above a threshold, flag that touch-up is needed
and the session should loop back to the live overlay rather than
proceeding to the final preview.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class TouchUpDecision:
    needs_touchup: bool
    remaining_remove_ratio: float
    reason: str


def check_needs_touchup(zones: dict[str, np.ndarray], threshold: float = 0.08) -> TouchUpDecision:
    """threshold is the fraction of total zone-covered pixels still in
    the 'remove' zone that we tolerate before calling it done. 0.08 is
    a starting guess — tune against what "looks finished" on real
    trimming sessions, this hasn't been validated.
    """
    remove_px = int(np.count_nonzero(zones.get("remove", np.array([]))))
    keep_px = int(np.count_nonzero(zones.get("keep", np.array([]))))
    blend_px = int(np.count_nonzero(zones.get("blend", np.array([]))))

    total = remove_px + keep_px + blend_px
    if total == 0:
        return TouchUpDecision(
            needs_touchup=False, remaining_remove_ratio=0.0, reason="No beard region detected"
        )

    remaining_ratio = remove_px / total
    needs_touchup = remaining_ratio > threshold

    reason = (
        f"{remaining_ratio:.1%} of the beard region is still outside the target shape"
        if needs_touchup
        else "Trim matches target shape within tolerance"
    )

    return TouchUpDecision(
        needs_touchup=needs_touchup, remaining_remove_ratio=round(remaining_ratio, 3), reason=reason
    )
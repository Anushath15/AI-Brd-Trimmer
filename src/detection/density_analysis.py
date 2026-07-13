"""
Growth & beard density analysis.

HONEST FLAG: like beard_segmentation.py, this is a heuristic, not a
trained model. "Density" here means texture/coverage within the beard
mask — dark-pixel coverage ratio and local contrast — as a rough proxy
for how filled-in the beard is. This has NOT been validated against
real beards of varying density/patchiness. Treat the output as a
rough signal to feed the recommendation engine, not a precise metric.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DensityResult:
    coverage_ratio: float  # fraction of the lower-face region classified as hair
    density_label: str     # "sparse" | "moderate" | "full"
    patchiness_score: float  # 0 = uniform, 1 = highly patchy (rough proxy)


def analyze_beard_density(
    image_bgr: np.ndarray, lower_face_mask: np.ndarray, beard_mask: np.ndarray
) -> DensityResult:
    face_area = int(np.count_nonzero(lower_face_mask))
    beard_area = int(np.count_nonzero(beard_mask))

    coverage_ratio = beard_area / face_area if face_area > 0 else 0.0

    if coverage_ratio < 0.2:
        density_label = "sparse"
    elif coverage_ratio < 0.55:
        density_label = "moderate"
    else:
        density_label = "full"

    patchiness_score = _estimate_patchiness(beard_mask, lower_face_mask)

    return DensityResult(
        coverage_ratio=round(coverage_ratio, 3),
        density_label=density_label,
        patchiness_score=round(patchiness_score, 3),
    )


def _estimate_patchiness(beard_mask: np.ndarray, lower_face_mask: np.ndarray) -> float:
    """Rough proxy: split the lower-face bounding box into a grid and
    measure variance in beard coverage across cells. High variance =
    patchy (some cells full, some empty); low variance = uniform.

    This is NOT a validated patchiness metric — it's a placeholder
    heuristic to unblock the recommendation engine. Revisit if the
    recommendations it drives look wrong on real test photos.
    """
    ys, xs = np.where(lower_face_mask > 0)
    if len(xs) == 0:
        return 0.0

    x_min, x_max, y_min, y_max = xs.min(), xs.max(), ys.min(), ys.max()
    grid_size = 4
    cell_w = max(1, (x_max - x_min) // grid_size)
    cell_h = max(1, (y_max - y_min) // grid_size)

    coverages = []
    for gy in range(grid_size):
        for gx in range(grid_size):
            cx0, cx1 = x_min + gx * cell_w, x_min + (gx + 1) * cell_w
            cy0, cy1 = y_min + gy * cell_h, y_min + (gy + 1) * cell_h

            cell_face = lower_face_mask[cy0:cy1, cx0:cx1]
            cell_beard = beard_mask[cy0:cy1, cx0:cx1]

            face_px = np.count_nonzero(cell_face)
            if face_px == 0:
                continue
            coverages.append(np.count_nonzero(cell_beard) / face_px)

    if not coverages:
        return 0.0

    return float(np.std(coverages))
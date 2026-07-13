"""
Live session orchestrator — this is the module that actually implements
the V1 flowchart end-to-end:

  calibration -> detection -> density -> recommendation -> style choice
  -> zone generation -> live overlay loop -> guidance -> touch-up check
  -> final preview

Kept separate from main.py's CLI argument handling so this can also be
imported and driven from a future GUI/mobile shell without dragging
argparse along with it.
"""

from pathlib import Path

import cv2
import numpy as np

from config import settings
from src.capture.calibration import run_calibration_checks
from src.capture.live_camera import LiveCameraFeed
from src.comparison.before_after import build_side_by_side, save_comparison_image
from src.detection.beard_segmentation import build_lower_face_mask, segment_beard_region
from src.detection.density_analysis import analyze_beard_density
from src.detection.face_landmarks import detect_face_landmarks
from src.guidance.visual_guidance import determine_guidance, draw_guidance_banner
from src.guidance.voice_guidance import VoiceGuidance
from src.overlay.live_overlay import draw_boundary_outline, draw_live_zone_overlay
from src.progress.touchup_tracker import check_needs_touchup
from src.styles.recommendation_engine import recommend_styles
from src.styles.style_catalogue import get_style_by_id
from src.tracking.face_alignment_tracker import realign_zones_to_current_frame
from src.utils.image_utils import ensure_dir_exists
from src.utils.logger import get_logger
from src.zones.zone_generator import generate_target_boundary_mask, generate_zones

logger = get_logger("live_session")


def run_v1_live_session(style_id: str | None = None, camera_index: int = 0) -> Path:
    """Runs the full V1 loop against the webcam. Blocks until the user
    quits ('q') or presses 's' to save the final result and exit.

    Returns the path to the saved before/after comparison image.

    NOTE: this opens display windows (cv2.imshow) — run it where a
    display is available, not headless/SSH-only.
    """
    voice = VoiceGuidance()

    with LiveCameraFeed(camera_index) as feed:
        frame_iter = feed.frames()

        # --- Step 1: grab a calibration/reference frame ---
        reference_frame = next(frame_iter)
        landmarks = detect_face_landmarks(reference_frame)
        if not landmarks.found:
            raise RuntimeError("No face detected in the initial frame — reposition and retry")

        calibration = run_calibration_checks(
            reference_frame, landmarks.landmarks_px, landmarks.image_width, landmarks.image_height
        )
        if not calibration.all_ok:
            logger.warning("Calibration issues detected: %s", "; ".join(calibration.reasons))
            # v1 choice: warn but don't hard-block — a hard gate here
            # would need its own retry UI loop, out of scope for this pass.

        # --- Step 2: detection + density + style selection ---
        lower_face_mask = build_lower_face_mask(
            landmarks.landmarks_px, (landmarks.image_height, landmarks.image_width)
        )
        beard_mask = segment_beard_region(reference_frame, lower_face_mask)
        density = analyze_beard_density(reference_frame, lower_face_mask, beard_mask)

        if style_id is None:
            recommended = recommend_styles(landmarks.landmarks_px, density)
            style = recommended[0]
            logger.info(f"AI-recommended style: {style.name} ({style.id})")
        else:
            style = get_style_by_id(style_id)

        target_mask = generate_target_boundary_mask(beard_mask, style)
        reference_zones = generate_zones(beard_mask, target_mask)
        reference_landmarks_px = landmarks.landmarks_px

        # --- Step 3: live trimming loop ---
        before_snapshot = reference_frame.copy()
        final_frame = reference_frame.copy()

        for frame in frame_iter:
            aligned = realign_zones_to_current_frame(reference_landmarks_px, reference_zones, frame)

            if aligned is None:
                display = frame  # face lost this frame — show raw feed
            else:
                display = draw_live_zone_overlay(frame, aligned.zones)
                display = draw_boundary_outline(display, aligned.zones.get("blend", target_mask))

                guidance = determine_guidance(aligned.zones)
                display = draw_guidance_banner(display, guidance)
                voice.announce(guidance)

                touchup = check_needs_touchup(aligned.zones)
                final_frame = frame.copy()
                if not touchup.needs_touchup:
                    cv2.putText(
                        display, "Trim complete - press 's' to save", (10, display.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2, cv2.LINE_AA,
                    )

            cv2.imshow("AI Brd Trimmer - V1 Live Session (q=quit, s=save)", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                break

        cv2.destroyAllWindows()

    output_dir = ensure_dir_exists(settings.OUTPUT_DIR)
    comparison = build_side_by_side(before_snapshot, final_frame)
    output_path = output_dir / f"v1_session_{style.id}{settings.OUTPUT_IMAGE_EXT}"
    save_comparison_image(comparison, output_path)
    logger.info(f"Saved before/after comparison to {output_path}")

    return output_path
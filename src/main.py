"""
AI Brd Trimmer — v1 CLI entrypoint.

Wires together: image input -> face landmarks -> beard segmentation ->
style selection -> target boundary -> zone generation -> overlay ->
save output.

Usage:
    python -m src.main --input data/input/photo.jpg --style short_boxed
    python -m src.main --webcam --style goatee
"""

import argparse
from pathlib import Path

from config import settings
from src.capture.image_input import capture_single_frame_from_webcam, load_image_from_file
from src.detection.beard_segmentation import build_lower_face_mask, segment_beard_region
from src.detection.face_landmarks import detect_face_landmarks
from src.overlay.visualizer import draw_zone_overlay, save_overlay_image
from src.styles.style_catalogue import get_style_by_id, load_style_catalogue
from src.utils.image_utils import ensure_dir_exists, resize_max_dimension
from src.utils.logger import get_logger
from src.zones.zone_generator import generate_target_boundary_mask, generate_zones

logger = get_logger("ai_brd_trimmer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Brd Trimmer v1 pipeline")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=str, help="Path to an input photo (static pipeline)")
    source.add_argument(
        "--webcam", action="store_true", help="Capture a single frame from the webcam (static pipeline)"
    )
    source.add_argument(
        "--live",
        action="store_true",
        help="Run the full V1 live session (calibration, live overlay, guidance, touch-up loop)",
    )
    parser.add_argument(
        "--style",
        type=str,
        default=None,
        help="Style id from the catalogue. Omit with --live to use the AI recommendation engine; "
        "defaults to 'short_boxed' for --input/--webcam.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the overlay image (defaults to data/output/)",
    )
    parser.add_argument(
        "--list-styles", action="store_true", help="Print available styles and exit"
    )
    return parser.parse_args()


def run_pipeline(image_bgr, style_id: str):
    image_bgr = resize_max_dimension(image_bgr)

    landmarks = detect_face_landmarks(image_bgr)
    if not landmarks.found:
        raise RuntimeError(
            "No face detected. Try a clearer, front-facing, well-lit photo."
        )

    style = get_style_by_id(style_id)

    lower_face_mask = build_lower_face_mask(
        landmarks.landmarks_px, (landmarks.image_height, landmarks.image_width)
    )
    beard_mask = segment_beard_region(image_bgr, lower_face_mask)
    target_mask = generate_target_boundary_mask(beard_mask, style)
    zones = generate_zones(beard_mask, target_mask)

    overlay_image = draw_zone_overlay(image_bgr, zones)
    return overlay_image


def main() -> None:
    args = parse_args()

    if args.list_styles:
        for style in load_style_catalogue():
            print(f"{style.id:15s} - {style.name}: {style.description}")
        return

    if args.live:
        from src.app.live_session import run_v1_live_session

        output_path = run_v1_live_session(style_id=args.style)  # None -> AI recommendation
        logger.info(f"Live session finished. Result saved to {output_path}")
        return

    if args.webcam:
        logger.info("Capturing a single frame from the webcam...")
        image_bgr = capture_single_frame_from_webcam()
    else:
        logger.info(f"Loading image from {args.input}")
        image_bgr = load_image_from_file(args.input)

    style_id = args.style or "short_boxed"
    overlay_image = run_pipeline(image_bgr, style_id)

    output_dir = ensure_dir_exists(settings.OUTPUT_DIR)
    output_path = (
        Path(args.output)
        if args.output
        else output_dir / f"overlay_{style_id}{settings.OUTPUT_IMAGE_EXT}"
    )
    save_overlay_image(overlay_image, output_path)
    logger.info(f"Saved zone overlay to {output_path}")


if __name__ == "__main__":
    main()
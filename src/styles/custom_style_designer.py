"""
Custom Style Designer — lets the user draw/edit their own target beard
boundary instead of picking from the catalogue or AI recommendation.

v1 implementation: a simple OpenCV mouse-drawing tool that lets the
user paint a boundary directly on their photo, producing a binary mask
compatible with the same zone_generator.generate_zones() pipeline as
catalogue styles use. This is a manual/interactive tool, not an AI
component — no model involved here, which also means no accuracy risk.
"""

import cv2
import numpy as np


class CustomStyleDesigner:
    """Interactive mouse-drawing canvas for a custom beard boundary.

    Left-click and drag to paint the "keep beard" region. Press 'c' to
    clear, 'enter' to confirm, 'esc' to cancel.

    NOTE: this opens an OpenCV window and blocks on user input — it's
    meant to be run locally with a display, not headless.
    """

    def __init__(self, image_bgr: np.ndarray, brush_radius: int = 12):
        self.image_bgr = image_bgr
        self.brush_radius = brush_radius
        self.mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        self._drawing = False

    def _on_mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing = True
            cv2.circle(self.mask, (x, y), self.brush_radius, 255, -1)
        elif event == cv2.EVENT_MOUSEMOVE and self._drawing:
            cv2.circle(self.mask, (x, y), self.brush_radius, 255, -1)
        elif event == cv2.EVENT_LBUTTONUP:
            self._drawing = False

    def run(self) -> np.ndarray | None:
        """Open the drawing window. Returns the final binary mask, or
        None if the user cancelled (Esc).
        """
        window_name = "Custom Style Designer - draw target beard area (Enter=confirm, C=clear, Esc=cancel)"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._on_mouse)

        try:
            while True:
                overlay = self.image_bgr.copy()
                overlay[self.mask > 0] = (0, 200, 0)
                display = cv2.addWeighted(overlay, 0.4, self.image_bgr, 0.6, 0)
                cv2.imshow(window_name, display)

                key = cv2.waitKey(20) & 0xFF
                if key == 13:  # Enter
                    return self.mask.copy()
                if key == 27:  # Esc
                    return None
                if key in (ord("c"), ord("C")):
                    self.mask[:] = 0
        finally:
            cv2.destroyWindow(window_name)


def build_custom_style_from_mask(mask: np.ndarray, style_id: str = "custom", name: str = "Custom Style"):
    """Wraps a user-drawn mask so it can be passed directly to
    zone_generator.generate_zones() the same way a catalogue style's
    derived mask would be — skips generate_target_boundary_mask()
    since the user already drew the exact target shape.
    """
    return mask
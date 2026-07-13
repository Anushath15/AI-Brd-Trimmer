"""
Live camera feed: a continuous frame-generator, separate from the
single-shot capture in `image_input.py`. This is the "Open phone live
camera" path staying open for the whole session (calibration → live
overlay → trimming), not just grabbing one frame and closing.
"""

from collections.abc import Iterator

import cv2
import numpy as np


class LiveCameraFeed:
    """Context-manager wrapper around cv2.VideoCapture for a continuous feed.

    Usage:
        with LiveCameraFeed() as feed:
            for frame in feed.frames():
                ...  # process frame
                if user_quits:
                    break
    """

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self._cap: cv2.VideoCapture | None = None

    def __enter__(self) -> "LiveCameraFeed":
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open webcam at index {self.camera_index}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._cap is not None:
            self._cap.release()

    def frames(self) -> Iterator[np.ndarray]:
        """Yield frames until the read fails or the caller breaks out.

        NOTE: this yields raw frames only — it does not do calibration
        or detection. Callers should run calibration/detection on each
        frame themselves (or every Nth frame, for performance).
        """
        if self._cap is None:
            raise RuntimeError("LiveCameraFeed must be used as a context manager")

        while True:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                break
            yield frame
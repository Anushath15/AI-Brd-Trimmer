"""
Voice guidance: speaks the same messages visual_guidance.py generates,
using pyttsx3 (offline TTS — no API key, no network dependency, works
fine for a local demo).

Rate-limited so it doesn't repeat the same message every frame (would
be unusable at 20-30fps) — only speaks on message change, with a
minimum cooldown between announcements.
"""

import time

from src.guidance.visual_guidance import GuidanceMessage
from src.utils.logger import get_logger

logger = get_logger("voice_guidance")

try:
    import pyttsx3
    _TTS_AVAILABLE = True
except ImportError:
    _TTS_AVAILABLE = False


class VoiceGuidance:
    def __init__(self, cooldown_seconds: float = 3.0):
        self.cooldown_seconds = cooldown_seconds
        self._last_text: str | None = None
        self._last_spoken_at: float = 0.0
        self._engine = None

        if _TTS_AVAILABLE:
            self._engine = pyttsx3.init()
        else:
            logger.warning(
                "pyttsx3 not installed — voice guidance will log instead of speak. "
                "Install with: pip install pyttsx3"
            )

    def announce(self, message: GuidanceMessage) -> None:
        now = time.time()
        same_message = message.text == self._last_text
        within_cooldown = (now - self._last_spoken_at) < self.cooldown_seconds

        if same_message and within_cooldown:
            return

        self._last_text = message.text
        self._last_spoken_at = now

        if self._engine is not None:
            self._engine.say(message.text)
            self._engine.runAndWait()
        else:
            logger.info(f"[VOICE] {message.text}")
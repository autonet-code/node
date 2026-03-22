"""
Privacy Controls for Captured Data.

Processes frames before they reach the training pipeline:
  1. Exclude list: skip frames from specified applications
  2. Region blur: blur rectangular screen regions (e.g. taskbar, password fields)
  3. PII scrubber: detect and redact emails, phone numbers, credit card numbers

Story 5.6: Privacy filter for the capture → training pipeline.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .capture import Frame
from .config import AutonetConfig, PrivacyConfig, load_config

logger = logging.getLogger(__name__)


# PII regex patterns
_EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
)
_PHONE_PATTERN = re.compile(
    r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
)
_CREDIT_CARD_PATTERN = re.compile(
    r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
)
_SSN_PATTERN = re.compile(
    r"\b\d{3}-\d{2}-\d{4}\b"
)


@dataclass
class PrivacyStats:
    """Counters for privacy filter activity."""
    frames_processed: int = 0
    frames_excluded: int = 0
    regions_blurred: int = 0
    pii_detections: int = 0


class PrivacyFilter:
    """
    Applies privacy controls to captured frames before training.

    Usage:
        pfilter = PrivacyFilter(config)
        frame = pfilter.process(frame)  # Returns None if excluded
    """

    def __init__(self, config: Optional[AutonetConfig] = None):
        self.config = config or load_config()
        self.privacy_config: PrivacyConfig = self.config.privacy
        self.stats = PrivacyStats()

        # Compile exclude patterns (case-insensitive substring match)
        self._exclude_patterns = [
            app.lower() for app in self.privacy_config.exclude_apps
        ]

        # Parse blur regions
        self._blur_regions: List[Dict] = self.privacy_config.blur_regions

        logger.info(
            f"PrivacyFilter initialized: "
            f"exclude_apps={len(self._exclude_patterns)}, "
            f"blur_regions={len(self._blur_regions)}, "
            f"scrub_pii={self.privacy_config.scrub_pii}"
        )

    def process(self, frame: Frame) -> Optional[Frame]:
        """
        Apply all privacy controls to a frame.

        Returns the (possibly modified) frame, or None if the frame
        should be excluded entirely.
        """
        self.stats.frames_processed += 1

        # 1. Check exclude list
        if self._should_exclude(frame):
            self.stats.frames_excluded += 1
            logger.debug(f"Frame excluded: source={frame.source_id}")
            return None

        # 2. Apply blur regions (only for screen captures with image data)
        if frame.modality == "screen" and self._blur_regions:
            frame = self._apply_blur_regions(frame)

        # 3. PII scrub on metadata
        if self.privacy_config.scrub_pii:
            frame = self._scrub_pii_metadata(frame)

        return frame

    def _should_exclude(self, frame: Frame) -> bool:
        """Check if frame's source matches any exclude pattern."""
        if not self._exclude_patterns:
            return False

        # Check source_id
        source_lower = frame.source_id.lower()
        for pattern in self._exclude_patterns:
            if pattern in source_lower:
                return True

        # Check metadata for window title
        window_title = frame.metadata.get("window_title", "").lower()
        if window_title:
            for pattern in self._exclude_patterns:
                if pattern in window_title:
                    return True

        # Check metadata for URL (browser extension frames)
        url = frame.metadata.get("url", "").lower()
        if url:
            for pattern in self._exclude_patterns:
                if pattern in url:
                    return True

        return False

    def _apply_blur_regions(self, frame: Frame) -> Frame:
        """
        Blur specified rectangular regions of a screen capture.

        Modifies the frame's image_data in place by blurring the
        specified pixel regions. Requires PIL.
        """
        if not self._blur_regions or not frame.image_data:
            return frame

        try:
            import io
            from PIL import Image, ImageFilter

            img = Image.open(io.BytesIO(frame.image_data))

            for region in self._blur_regions:
                x = region.get("x", 0)
                y = region.get("y", 0)
                w = region.get("w", 0)
                h = region.get("h", 0)

                if w <= 0 or h <= 0:
                    continue

                # Crop, blur, paste back
                box = (x, y, x + w, y + h)
                cropped = img.crop(box)
                blurred = cropped.filter(ImageFilter.GaussianBlur(radius=20))
                img.paste(blurred, box)
                self.stats.regions_blurred += 1

            # Re-encode
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            frame.image_data = buf.getvalue()

        except ImportError:
            logger.warning("PIL not available, skipping region blur")
        except Exception as e:
            logger.debug(f"Region blur failed: {e}")

        return frame

    def _scrub_pii_metadata(self, frame: Frame) -> Frame:
        """
        Scrub PII from frame metadata string values.

        Replaces detected emails, phone numbers, credit card numbers,
        and SSNs with redaction markers.
        """
        if not frame.metadata:
            return frame

        scrubbed = {}
        for key, value in frame.metadata.items():
            if isinstance(value, str):
                scrubbed[key] = self._redact_pii(value)
            else:
                scrubbed[key] = value

        frame.metadata = scrubbed
        return frame

    def _redact_pii(self, text: str) -> str:
        """Replace PII patterns with redaction markers."""
        original = text

        text = _EMAIL_PATTERN.sub("[EMAIL_REDACTED]", text)
        text = _CREDIT_CARD_PATTERN.sub("[CC_REDACTED]", text)
        text = _SSN_PATTERN.sub("[SSN_REDACTED]", text)
        text = _PHONE_PATTERN.sub("[PHONE_REDACTED]", text)

        if text != original:
            self.stats.pii_detections += 1

        return text


def scrub_text(text: str) -> str:
    """
    Standalone PII scrubber for arbitrary text.

    Useful for scrubbing text extracted from browser extensions
    or other text sources before they enter the training pipeline.
    """
    text = _EMAIL_PATTERN.sub("[EMAIL_REDACTED]", text)
    text = _CREDIT_CARD_PATTERN.sub("[CC_REDACTED]", text)
    text = _SSN_PATTERN.sub("[SSN_REDACTED]", text)
    text = _PHONE_PATTERN.sub("[PHONE_REDACTED]", text)
    return text

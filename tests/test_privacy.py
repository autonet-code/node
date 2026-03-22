"""Tests for Story 5.6 — Privacy controls."""

import unittest

from nodes.common.config import AutonetConfig, PrivacyConfig
from nodes.common.capture import Frame
from nodes.common.privacy import PrivacyFilter, scrub_text


def _make_frame(source_id="screen-1", metadata=None, modality="screen"):
    return Frame(
        image_data=b"fake_png_data",
        timestamp=1000.0,
        source_id=source_id,
        modality=modality,
        width=1920,
        height=1080,
        metadata=metadata or {},
    )


class TestPrivacyFilter(unittest.TestCase):
    """Tests for PrivacyFilter."""

    def _make_config(self, exclude_apps=None, blur_regions=None, scrub_pii=True):
        cfg = AutonetConfig()
        cfg.privacy = PrivacyConfig(
            exclude_apps=exclude_apps or [],
            blur_regions=blur_regions or [],
            scrub_pii=scrub_pii,
        )
        return cfg

    def test_passthrough_no_filters(self):
        """Frame passes through with no filters configured."""
        cfg = self._make_config()
        pf = PrivacyFilter(config=cfg)
        frame = _make_frame()
        result = pf.process(frame)
        self.assertIsNotNone(result)
        self.assertEqual(pf.stats.frames_processed, 1)
        self.assertEqual(pf.stats.frames_excluded, 0)

    def test_exclude_by_source_id(self):
        """Frames from excluded apps are dropped."""
        cfg = self._make_config(exclude_apps=["1Password", "KeePass"])
        pf = PrivacyFilter(config=cfg)

        frame = _make_frame(source_id="screen-1password-window")
        result = pf.process(frame)
        self.assertIsNone(result)
        self.assertEqual(pf.stats.frames_excluded, 1)

    def test_exclude_by_window_title(self):
        """Frames with excluded window title in metadata are dropped."""
        cfg = self._make_config(exclude_apps=["KeePass"])
        pf = PrivacyFilter(config=cfg)

        frame = _make_frame(metadata={"window_title": "KeePass - Database"})
        result = pf.process(frame)
        self.assertIsNone(result)

    def test_exclude_case_insensitive(self):
        """Exclude matching is case-insensitive."""
        cfg = self._make_config(exclude_apps=["keepass"])
        pf = PrivacyFilter(config=cfg)

        frame = _make_frame(source_id="screen-KEEPASS-main")
        result = pf.process(frame)
        self.assertIsNone(result)

    def test_no_exclude_partial_mismatch(self):
        """Non-matching source passes through."""
        cfg = self._make_config(exclude_apps=["1Password"])
        pf = PrivacyFilter(config=cfg)

        frame = _make_frame(source_id="screen-firefox")
        result = pf.process(frame)
        self.assertIsNotNone(result)

    def test_scrub_email(self):
        """Emails in metadata are redacted."""
        cfg = self._make_config(scrub_pii=True)
        pf = PrivacyFilter(config=cfg)

        frame = _make_frame(metadata={"text": "Contact me at alice@example.com"})
        result = pf.process(frame)
        self.assertIn("[EMAIL_REDACTED]", result.metadata["text"])
        self.assertNotIn("alice@example.com", result.metadata["text"])

    def test_scrub_phone(self):
        """Phone numbers in metadata are redacted."""
        cfg = self._make_config(scrub_pii=True)
        pf = PrivacyFilter(config=cfg)

        frame = _make_frame(metadata={"text": "Call 555-123-4567"})
        result = pf.process(frame)
        self.assertIn("[PHONE_REDACTED]", result.metadata["text"])

    def test_scrub_credit_card(self):
        """Credit card numbers in metadata are redacted."""
        cfg = self._make_config(scrub_pii=True)
        pf = PrivacyFilter(config=cfg)

        frame = _make_frame(metadata={"text": "Card: 4111-1111-1111-1111"})
        result = pf.process(frame)
        self.assertIn("[CC_REDACTED]", result.metadata["text"])

    def test_scrub_ssn(self):
        """SSNs in metadata are redacted."""
        cfg = self._make_config(scrub_pii=True)
        pf = PrivacyFilter(config=cfg)

        frame = _make_frame(metadata={"text": "SSN: 123-45-6789"})
        result = pf.process(frame)
        self.assertIn("[SSN_REDACTED]", result.metadata["text"])

    def test_no_scrub_when_disabled(self):
        """PII not scrubbed when scrub_pii=False."""
        cfg = self._make_config(scrub_pii=False)
        pf = PrivacyFilter(config=cfg)

        frame = _make_frame(metadata={"text": "alice@example.com"})
        result = pf.process(frame)
        self.assertIn("alice@example.com", result.metadata["text"])

    def test_pii_detection_count(self):
        """PII detection counter increments."""
        cfg = self._make_config(scrub_pii=True)
        pf = PrivacyFilter(config=cfg)

        pf.process(_make_frame(metadata={"text": "alice@example.com"}))
        pf.process(_make_frame(metadata={"text": "bob@test.org"}))
        self.assertEqual(pf.stats.pii_detections, 2)

    def test_non_string_metadata_preserved(self):
        """Non-string metadata values pass through unchanged."""
        cfg = self._make_config(scrub_pii=True)
        pf = PrivacyFilter(config=cfg)

        frame = _make_frame(metadata={"count": 42, "flag": True})
        result = pf.process(frame)
        self.assertEqual(result.metadata["count"], 42)
        self.assertEqual(result.metadata["flag"], True)

    def test_audio_frames_not_blurred(self):
        """Audio frames skip blur processing."""
        cfg = self._make_config(
            blur_regions=[{"x": 0, "y": 0, "w": 100, "h": 100}]
        )
        pf = PrivacyFilter(config=cfg)

        frame = _make_frame(modality="audio")
        result = pf.process(frame)
        # Should pass through (no blur on audio)
        self.assertIsNotNone(result)
        self.assertEqual(pf.stats.regions_blurred, 0)


class TestScrubText(unittest.TestCase):
    """Tests for the standalone scrub_text function."""

    def test_scrub_multiple_pii(self):
        text = "Email: a@b.com, Phone: 555-123-4567, Card: 4111 1111 1111 1111"
        result = scrub_text(text)
        self.assertIn("[EMAIL_REDACTED]", result)
        self.assertIn("[PHONE_REDACTED]", result)
        self.assertIn("[CC_REDACTED]", result)

    def test_no_pii(self):
        text = "This is a normal sentence with no PII."
        result = scrub_text(text)
        self.assertEqual(result, text)

    def test_empty_string(self):
        self.assertEqual(scrub_text(""), "")


if __name__ == "__main__":
    unittest.main()

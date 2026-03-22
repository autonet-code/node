"""Tests for Story 5.2 — Browser extension text capture integration."""

import json
import time
import threading
import unittest
from collections import deque
from unittest.mock import MagicMock, patch

from nodes.common.capture import CaptureManager, Frame, CaptureConfig
from nodes.common.browser_capture import BrowserCaptureSource
from nodes.common.config import AutonetConfig
from nodes.common.privacy import PrivacyFilter, scrub_text


class TestBrowserCaptureSource(unittest.TestCase):
    """Tests for BrowserCaptureSource without real WebSocket."""

    def test_source_id(self):
        src = BrowserCaptureSource(instance_id=3)
        self.assertEqual(src.source_id, "browser-3")

    def test_default_source_id(self):
        src = BrowserCaptureSource()
        self.assertEqual(src.source_id, "browser-0")

    def test_modality(self):
        src = BrowserCaptureSource()
        self.assertEqual(src.modality, "text")

    def test_not_running_initially(self):
        src = BrowserCaptureSource()
        self.assertFalse(src.is_running)

    def test_get_frames_empty_buffer(self):
        src = BrowserCaptureSource()
        frames = list(src.get_frames())
        self.assertEqual(frames, [])

    def test_get_frames_yields_buffered(self):
        src = BrowserCaptureSource()
        # Manually push frames into buffer
        frame = Frame(
            image_data=b"Hello world text",
            timestamp=1000.0,
            source_id="browser-0",
            modality="text",
        )
        src._buffer.append(frame)
        src._buffer.append(frame)

        frames = list(src.get_frames())
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0].image_data, b"Hello world text")

    def test_buffer_max_size(self):
        src = BrowserCaptureSource(max_buffer=5)
        for i in range(10):
            src._buffer.append(
                Frame(
                    image_data=f"frame-{i}".encode(),
                    timestamp=float(i),
                    source_id="browser-0",
                    modality="text",
                )
            )
        # deque(maxlen=5) keeps only last 5
        self.assertEqual(len(src._buffer), 5)
        frames = list(src.get_frames())
        self.assertEqual(frames[0].image_data, b"frame-5")

    def test_handle_frame_valid(self):
        src = BrowserCaptureSource(scrub_pii=False)
        data = {
            "type": "training_frame",
            "data": {
                "text": "This is some web page content for training.",
                "metadata": {
                    "url": "https://example.com/article",
                    "title": "Example Article",
                    "lang": "en",
                    "timestamp": 1700000000.0,
                },
                "trigger": "load",
            },
        }
        src._handle_frame(data)
        self.assertEqual(len(src._buffer), 1)

        frame = src._buffer[0]
        self.assertEqual(frame.modality, "text")
        self.assertEqual(frame.source_id, "browser-0")
        self.assertEqual(
            frame.image_data,
            b"This is some web page content for training.",
        )
        self.assertEqual(frame.metadata["url"], "https://example.com/article")
        self.assertEqual(frame.metadata["title"], "Example Article")
        self.assertEqual(frame.metadata["trigger"], "load")
        self.assertAlmostEqual(frame.timestamp, 1700000000.0)

    def test_handle_frame_empty_text_skipped(self):
        src = BrowserCaptureSource()
        data = {
            "type": "training_frame",
            "data": {"text": "", "metadata": {}},
        }
        src._handle_frame(data)
        self.assertEqual(len(src._buffer), 0)

    def test_handle_frame_wrong_type_ignored(self):
        src = BrowserCaptureSource()
        data = {"type": "cdp_event", "method": "something"}
        src._handle_frame(data)
        self.assertEqual(len(src._buffer), 0)

    def test_handle_frame_pii_scrubbing(self):
        src = BrowserCaptureSource(scrub_pii=True)
        data = {
            "type": "training_frame",
            "data": {
                "text": "Contact me at john@example.com or 555-123-4567",
                "metadata": {"url": "https://example.com", "timestamp": 1.0},
                "trigger": "load",
            },
        }
        src._handle_frame(data)
        self.assertEqual(len(src._buffer), 1)

        text = src._buffer[0].image_data.decode("utf-8")
        self.assertNotIn("john@example.com", text)
        self.assertIn("[EMAIL_REDACTED]", text)
        self.assertIn("[PHONE_REDACTED]", text)

    def test_handle_frame_no_pii_scrubbing(self):
        src = BrowserCaptureSource(scrub_pii=False)
        data = {
            "type": "training_frame",
            "data": {
                "text": "Email: john@example.com",
                "metadata": {"url": "https://x.com", "timestamp": 1.0},
                "trigger": "load",
            },
        }
        src._handle_frame(data)
        text = src._buffer[0].image_data.decode("utf-8")
        self.assertIn("john@example.com", text)

    def test_stop_without_start(self):
        src = BrowserCaptureSource()
        src.stop()  # Should not raise
        self.assertFalse(src.is_running)

    def test_configure_without_connection(self):
        src = BrowserCaptureSource()
        # Should not raise, just log warning
        src.configure(enabled=True)

    def test_configure_with_mock_ws(self):
        src = BrowserCaptureSource()
        mock_ws = MagicMock()
        src._ws = mock_ws
        src.configure(enabled=True, exclude_patterns=["bank.com"], capture_interval_ms=3000)
        mock_ws.send.assert_called_once()
        sent = json.loads(mock_ws.send.call_args[0][0])
        self.assertEqual(sent["type"], "training_config")
        self.assertTrue(sent["enabled"])
        self.assertEqual(sent["excludePatterns"], ["bank.com"])
        self.assertEqual(sent["captureIntervalMs"], 3000)


class TestBrowserCaptureWithPrivacy(unittest.TestCase):
    """Test that browser capture frames work with the privacy filter."""

    def _make_browser_frame(self, text="Some content", url="https://example.com"):
        return Frame(
            image_data=text.encode("utf-8"),
            timestamp=time.time(),
            source_id="browser-0",
            modality="text",
            metadata={"url": url, "title": "Test"},
        )

    def test_privacy_filter_passes_browser_frame(self):
        cfg = AutonetConfig()
        pf = PrivacyFilter(config=cfg)
        frame = self._make_browser_frame()
        result = pf.process(frame)
        self.assertIsNotNone(result)

    def test_privacy_filter_excludes_by_url(self):
        cfg = AutonetConfig()
        cfg.privacy.exclude_apps = ["bank.example.com"]
        pf = PrivacyFilter(config=cfg)
        frame = self._make_browser_frame(url="https://bank.example.com/account")
        result = pf.process(frame)
        self.assertIsNone(result)

    def test_privacy_filter_scrubs_metadata_pii(self):
        cfg = AutonetConfig()
        cfg.privacy.scrub_pii = True
        pf = PrivacyFilter(config=cfg)
        frame = self._make_browser_frame()
        frame.metadata["title"] = "Email from alice@secret.org"
        result = pf.process(frame)
        self.assertNotIn("alice@secret.org", result.metadata["title"])
        self.assertIn("[EMAIL_REDACTED]", result.metadata["title"])

    def test_scrub_text_standalone(self):
        text = "Call 555-123-4567 or email test@foo.com, CC: 4111-1111-1111-1111"
        scrubbed = scrub_text(text)
        self.assertNotIn("555-123-4567", scrubbed)
        self.assertNotIn("test@foo.com", scrubbed)
        self.assertNotIn("4111-1111-1111-1111", scrubbed)
        self.assertIn("[PHONE_REDACTED]", scrubbed)
        self.assertIn("[EMAIL_REDACTED]", scrubbed)
        self.assertIn("[CC_REDACTED]", scrubbed)


class TestCaptureManagerBrowserIntegration(unittest.TestCase):
    """Test CaptureManager with browser source type."""

    def test_create_sources_includes_browser(self):
        cfg = AutonetConfig()
        cfg.capture = CaptureConfig(
            enabled_sources=["browser"],
            browser_relay_url="ws://127.0.0.1:9222/training",
        )
        mgr = CaptureManager(config=cfg)
        mgr.create_sources_from_config()
        self.assertEqual(len(mgr._sources), 1)
        self.assertIsInstance(mgr._sources[0], BrowserCaptureSource)
        self.assertEqual(mgr._sources[0].source_id, "browser-0")
        self.assertEqual(mgr._sources[0].modality, "text")

    def test_create_sources_mixed(self):
        cfg = AutonetConfig()
        cfg.capture = CaptureConfig(
            enabled_sources=["screen", "browser"],
            screen_monitor=1,
            resolution=[224, 224],
        )
        mgr = CaptureManager(config=cfg)
        mgr.create_sources_from_config()
        self.assertEqual(len(mgr._sources), 2)

    def test_browser_source_delivers_frames(self):
        """Manually buffer a frame and verify CaptureManager delivers it."""
        cfg = AutonetConfig()
        cfg.capture = CaptureConfig(fps_cap=100, enabled_sources=[])
        mgr = CaptureManager(config=cfg)

        src = BrowserCaptureSource()
        src._running = True
        # Inject a frame
        src._buffer.append(
            Frame(
                image_data=b"web page text",
                timestamp=time.time(),
                source_id="browser-0",
                modality="text",
            )
        )
        mgr.add_source(src)
        mgr._running = True

        frame = mgr.get_frame()
        self.assertIsNotNone(frame)
        self.assertEqual(frame.image_data, b"web page text")
        self.assertEqual(frame.modality, "text")


if __name__ == "__main__":
    unittest.main()

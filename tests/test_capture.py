"""Tests for Epic 5 — Data capture, preprocessing, and privacy."""

import io
import struct
import time
import unittest
from unittest.mock import MagicMock, patch

from nodes.common.config import AutonetConfig, CaptureConfig, ModelConfig, TrainingConfig, PrivacyConfig
from nodes.common.capture import (
    AudioCaptureSource,
    CaptureManager,
    CaptureSource,
    Frame,
    ScreenCaptureSource,
)


class TestFrame(unittest.TestCase):
    """Tests for the Frame dataclass."""

    def test_create_frame(self):
        f = Frame(
            image_data=b"fake_png",
            timestamp=1000.0,
            source_id="screen-1",
            modality="screen",
            width=1920,
            height=1080,
        )
        self.assertEqual(f.source_id, "screen-1")
        self.assertEqual(f.modality, "screen")
        self.assertEqual(f.width, 1920)
        self.assertEqual(f.height, 1080)

    def test_frame_defaults(self):
        f = Frame(image_data=b"", timestamp=0.0, source_id="test", modality="test")
        self.assertEqual(f.width, 0)
        self.assertEqual(f.height, 0)
        self.assertEqual(f.metadata, {})

    def test_frame_metadata(self):
        f = Frame(
            image_data=b"data",
            timestamp=1.0,
            source_id="s",
            modality="screen",
            metadata={"window_title": "Firefox"},
        )
        self.assertEqual(f.metadata["window_title"], "Firefox")


class TestScreenCaptureSource(unittest.TestCase):
    """Tests for ScreenCaptureSource."""

    def test_source_id(self):
        s = ScreenCaptureSource(monitor=2)
        self.assertEqual(s.source_id, "screen-2")

    def test_modality(self):
        s = ScreenCaptureSource()
        self.assertEqual(s.modality, "screen")

    def test_not_running_initially(self):
        s = ScreenCaptureSource()
        self.assertFalse(s.is_running)

    def test_get_frames_when_not_running(self):
        s = ScreenCaptureSource()
        frames = list(s.get_frames())
        self.assertEqual(frames, [])

    def test_stop_when_not_started(self):
        s = ScreenCaptureSource()
        s.stop()  # Should not raise
        self.assertFalse(s.is_running)

    @patch("nodes.common.capture.mss", create=True)
    def test_start_without_mss(self, mock_mss):
        """Start gracefully handles missing mss."""
        s = ScreenCaptureSource()
        # Simulate import error by making mss unavailable
        with patch.dict("sys.modules", {"mss": None}):
            s.start()
        # Should not crash, just won't be running


class TestAudioCaptureSource(unittest.TestCase):
    """Tests for AudioCaptureSource."""

    def test_source_id(self):
        a = AudioCaptureSource()
        self.assertEqual(a.source_id, "audio-system")

    def test_modality(self):
        a = AudioCaptureSource()
        self.assertEqual(a.modality, "audio")

    def test_start_stop(self):
        a = AudioCaptureSource()
        self.assertFalse(a.is_running)
        a.start()
        self.assertTrue(a.is_running)
        a.stop()
        self.assertFalse(a.is_running)

    def test_get_frames_yields_nothing(self):
        a = AudioCaptureSource()
        a.start()
        frames = list(a.get_frames())
        self.assertEqual(frames, [])


class TestCaptureManager(unittest.TestCase):
    """Tests for CaptureManager."""

    def _make_config(self, fps_cap=10):
        cfg = AutonetConfig()
        cfg.capture = CaptureConfig(fps_cap=fps_cap)
        return cfg

    def test_frame_interval(self):
        cfg = self._make_config(fps_cap=10)
        mgr = CaptureManager(config=cfg)
        self.assertAlmostEqual(mgr.frame_interval, 0.1)

    def test_frame_interval_minimum(self):
        cfg = self._make_config(fps_cap=0)
        mgr = CaptureManager(config=cfg)
        # fps_cap=0 -> clamped to 1
        self.assertAlmostEqual(mgr.frame_interval, 1.0)

    def test_add_source(self):
        cfg = self._make_config()
        mgr = CaptureManager(config=cfg)
        src = AudioCaptureSource()
        mgr.add_source(src)
        self.assertEqual(len(mgr._sources), 1)

    def test_start_stop(self):
        cfg = self._make_config()
        mgr = CaptureManager(config=cfg)
        src = AudioCaptureSource()
        mgr.add_source(src)
        mgr.start()
        self.assertTrue(mgr._running)
        self.assertTrue(src.is_running)
        mgr.stop()
        self.assertFalse(mgr._running)
        self.assertFalse(src.is_running)

    def test_get_frame_when_stopped(self):
        cfg = self._make_config()
        mgr = CaptureManager(config=cfg)
        self.assertIsNone(mgr.get_frame())

    def test_get_frame_no_sources(self):
        cfg = self._make_config()
        mgr = CaptureManager(config=cfg)
        mgr._running = True
        self.assertIsNone(mgr.get_frame())

    def test_get_frame_fps_cap(self):
        """FPS cap prevents frames from being delivered too quickly."""
        cfg = self._make_config(fps_cap=2)
        mgr = CaptureManager(config=cfg)

        # Create a mock source that always has frames
        mock_src = MagicMock(spec=CaptureSource)
        mock_src.is_running = True
        mock_src.get_frames.return_value = iter([
            Frame(image_data=b"test", timestamp=time.time(), source_id="mock", modality="screen")
        ])

        mgr.add_source(mock_src)
        mgr._running = True

        # First frame should succeed
        frame = mgr.get_frame()
        self.assertIsNotNone(frame)
        self.assertEqual(mgr.frame_count, 1)

        # Immediate second call should be rate-limited
        frame2 = mgr.get_frame()
        self.assertIsNone(frame2)

    def test_create_sources_from_config(self):
        cfg = AutonetConfig()
        cfg.capture = CaptureConfig(
            enabled_sources=["screen", "audio"],
            screen_monitor=1,
            resolution=[224, 224],
        )
        mgr = CaptureManager(config=cfg)
        mgr.create_sources_from_config()
        self.assertEqual(len(mgr._sources), 2)
        self.assertIsInstance(mgr._sources[0], ScreenCaptureSource)
        self.assertIsInstance(mgr._sources[1], AudioCaptureSource)

    def test_create_sources_unknown_type(self):
        cfg = AutonetConfig()
        cfg.capture = CaptureConfig(enabled_sources=["hologram"])
        mgr = CaptureManager(config=cfg)
        mgr.create_sources_from_config()
        self.assertEqual(len(mgr._sources), 0)


if __name__ == "__main__":
    unittest.main()

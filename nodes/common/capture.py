"""
Data Capture Service for Autonet nodes.

Provides an abstraction over screen capture and audio capture sources.
CaptureManager manages multiple sources and delivers frames at a
configurable FPS cap for use as training data.

Stories 5.1 + 5.4: Screen and audio capture abstraction.
Story 5.5: Source configuration persistence.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

from .config import AutonetConfig, CaptureConfig, load_config

logger = logging.getLogger(__name__)


@dataclass
class Frame:
    """A single captured frame from any modality."""
    image_data: bytes           # Raw image bytes (PNG/JPEG) or audio samples
    timestamp: float            # Unix timestamp of capture
    source_id: str              # Identifier of the source that produced this
    modality: str               # "screen", "audio", "camera"
    width: int = 0              # Image width (0 for non-image)
    height: int = 0             # Image height (0 for non-image)
    metadata: dict = field(default_factory=dict)


class CaptureSource(ABC):
    """Abstract base class for data capture sources."""

    @property
    @abstractmethod
    def source_id(self) -> str:
        """Unique identifier for this source."""
        ...

    @property
    @abstractmethod
    def modality(self) -> str:
        """Type of data: 'screen', 'audio', 'camera'."""
        ...

    @abstractmethod
    def start(self) -> None:
        """Initialize the capture source."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Release resources."""
        ...

    @abstractmethod
    def get_frames(self) -> Iterator[Frame]:
        """Yield captured frames. Non-blocking: yields nothing if no data."""
        ...

    @property
    def is_running(self) -> bool:
        """Whether this source is currently active."""
        return False


class ScreenCaptureSource(CaptureSource):
    """
    Screen capture using mss (cross-platform screenshot library).

    Captures screenshots from a specified monitor and yields them as
    PNG-encoded Frame objects.
    """

    def __init__(self, monitor: int = 1, resolution: Optional[tuple] = None):
        self._monitor = monitor
        self._resolution = resolution  # (width, height) to resize to
        self._sct = None
        self._running = False

    @property
    def source_id(self) -> str:
        return f"screen-{self._monitor}"

    @property
    def modality(self) -> str:
        return "screen"

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        try:
            import mss
            self._sct = mss.mss()
            self._running = True
            logger.info(f"Screen capture started (monitor {self._monitor})")
        except ImportError:
            logger.warning("mss not installed, screen capture unavailable")
            self._running = False
        except Exception as e:
            logger.error(f"Failed to start screen capture: {e}")
            self._running = False

    def stop(self) -> None:
        if self._sct:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None
        self._running = False
        logger.info("Screen capture stopped")

    def get_frames(self) -> Iterator[Frame]:
        if not self._running or not self._sct:
            return

        try:
            import mss.tools

            monitors = self._sct.monitors
            if self._monitor >= len(monitors):
                return

            monitor = monitors[self._monitor]
            screenshot = self._sct.grab(monitor)

            # Convert to PNG bytes
            png_bytes = mss.tools.to_png(screenshot.rgb, screenshot.size)

            yield Frame(
                image_data=png_bytes,
                timestamp=time.time(),
                source_id=self.source_id,
                modality=self.modality,
                width=screenshot.width,
                height=screenshot.height,
                metadata={"monitor": self._monitor},
            )
        except Exception as e:
            logger.debug(f"Screen capture frame error: {e}")


class AudioCaptureSource(CaptureSource):
    """
    Audio capture source (stub).

    Real implementation requires platform-specific audio APIs
    (e.g., sounddevice, pyaudio). This defines the contract.
    """

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self._sample_rate = sample_rate
        self._channels = channels
        self._running = False

    @property
    def source_id(self) -> str:
        return "audio-system"

    @property
    def modality(self) -> str:
        return "audio"

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        # Stub: real implementation would open audio stream
        logger.info("Audio capture started (stub — no real audio capture)")
        self._running = True

    def stop(self) -> None:
        self._running = False
        logger.info("Audio capture stopped")

    def get_frames(self) -> Iterator[Frame]:
        # Stub: yields nothing. Real implementation would yield audio chunks.
        return
        yield  # Make this a generator


class CaptureManager:
    """
    Manages multiple capture sources with round-robin frame delivery.

    Enforces an FPS cap to avoid overwhelming the training pipeline.
    """

    def __init__(self, config: Optional[AutonetConfig] = None):
        self.config = config or load_config()
        self.capture_config: CaptureConfig = self.config.capture
        self._sources: List[CaptureSource] = []
        self._running = False
        self._last_frame_time = 0.0
        self._frame_count = 0

    @property
    def frame_interval(self) -> float:
        """Minimum seconds between frames based on fps_cap."""
        fps = max(self.capture_config.fps_cap, 1)
        return 1.0 / fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def add_source(self, source: CaptureSource) -> None:
        """Register a capture source."""
        self._sources.append(source)
        logger.info(f"Added capture source: {source.source_id} ({source.modality})")

    def start(self) -> None:
        """Start all registered capture sources."""
        for source in self._sources:
            try:
                source.start()
            except Exception as e:
                logger.error(f"Failed to start {source.source_id}: {e}")
        self._running = True
        logger.info(f"CaptureManager started with {len(self._sources)} sources")

    def stop(self) -> None:
        """Stop all capture sources."""
        self._running = False
        for source in self._sources:
            try:
                source.stop()
            except Exception as e:
                logger.error(f"Failed to stop {source.source_id}: {e}")
        logger.info(f"CaptureManager stopped (captured {self._frame_count} frames)")

    def get_frame(self) -> Optional[Frame]:
        """
        Get the next frame from sources, respecting FPS cap.

        Uses round-robin across sources. Returns None if no frame
        is available or FPS cap hasn't elapsed.
        """
        if not self._running or not self._sources:
            return None

        # Enforce FPS cap
        now = time.time()
        if now - self._last_frame_time < self.frame_interval:
            return None

        # Round-robin through sources
        for source in self._sources:
            if not source.is_running:
                continue
            try:
                for frame in source.get_frames():
                    self._last_frame_time = now
                    self._frame_count += 1
                    return frame
            except Exception as e:
                logger.debug(f"Error getting frame from {source.source_id}: {e}")

        return None

    def get_frames_batch(self, count: int) -> List[Frame]:
        """Collect up to count frames (blocking until available or timeout)."""
        frames = []
        deadline = time.time() + 30.0  # 30s timeout

        while len(frames) < count and time.time() < deadline:
            frame = self.get_frame()
            if frame:
                frames.append(frame)
            else:
                time.sleep(self.frame_interval)

        return frames

    def create_sources_from_config(self) -> None:
        """Create capture sources based on configuration."""
        resolution = tuple(self.capture_config.resolution)

        for source_type in self.capture_config.enabled_sources:
            if source_type == "screen":
                self.add_source(ScreenCaptureSource(
                    monitor=self.capture_config.screen_monitor,
                    resolution=resolution,
                ))
            elif source_type == "audio":
                self.add_source(AudioCaptureSource())
            elif source_type == "browser":
                from .browser_capture import BrowserCaptureSource
                self.add_source(BrowserCaptureSource(
                    relay_url=self.capture_config.browser_relay_url,
                    scrub_pii=self.capture_config.browser_scrub_pii,
                ))
            else:
                logger.warning(f"Unknown capture source type: {source_type}")

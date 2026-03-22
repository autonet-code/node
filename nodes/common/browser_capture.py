"""
Browser Extension Capture Source for Autonet training pipeline.

Connects to the ATN CDP Relay's /training WebSocket endpoint
to receive text content extracted from web pages by the browser
extension's training-capture.js content script.

Story 5.2: Browser extension for text extraction.

Architecture:
    Browser Extension (training-capture.js)
        → service-worker.js (forward)
        → CDP Relay ws://127.0.0.1:9222/training
        → BrowserCaptureSource (this module)
        → Frame objects → PrivacyFilter → Preprocessor → JEPA
"""

import json
import logging
import threading
import time
from collections import deque
from typing import Iterator, Optional

from .capture import CaptureSource, Frame
from .privacy import scrub_text

logger = logging.getLogger(__name__)


class BrowserCaptureSource(CaptureSource):
    """
    Capture source that receives text content from the browser extension
    via the CDP relay's /training WebSocket endpoint.

    Text content is stored as UTF-8 bytes in Frame.image_data.
    Frames have modality="text" and source_id="browser-N".

    The source runs a background thread that maintains the WebSocket
    connection and buffers incoming frames.
    """

    def __init__(
        self,
        relay_url: str = "ws://127.0.0.1:9222/training",
        instance_id: int = 0,
        max_buffer: int = 100,
        scrub_pii: bool = True,
    ):
        self._relay_url = relay_url
        self._instance_id = instance_id
        self._max_buffer = max_buffer
        self._scrub_pii = scrub_pii
        self._buffer: deque = deque(maxlen=max_buffer)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._ws = None

    @property
    def source_id(self) -> str:
        return f"browser-{self._instance_id}"

    @property
    def modality(self) -> str:
        return "text"

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        """Start the background WebSocket receiver thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._receiver_loop,
            name=f"browser-capture-{self._instance_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            f"BrowserCaptureSource started (relay={self._relay_url})"
        )

    def stop(self) -> None:
        """Stop the background receiver."""
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("BrowserCaptureSource stopped")

    def get_frames(self) -> Iterator[Frame]:
        """Yield buffered frames (non-blocking)."""
        while self._buffer:
            try:
                yield self._buffer.popleft()
            except IndexError:
                break

    def configure(
        self,
        enabled: bool = True,
        exclude_patterns: Optional[list] = None,
        capture_interval_ms: int = 5000,
    ) -> None:
        """
        Send configuration to the browser extension via the relay.

        This tells the extension's content script whether to capture,
        what URLs to exclude, and how often to capture.
        """
        if not self._ws:
            logger.warning("Cannot configure: not connected to relay")
            return

        config_msg = {
            "type": "training_config",
            "enabled": enabled,
            "excludePatterns": exclude_patterns or [],
            "captureIntervalMs": capture_interval_ms,
        }

        try:
            self._ws.send(json.dumps(config_msg))
            logger.info(f"Sent training config: enabled={enabled}")
        except Exception as e:
            logger.warning(f"Failed to send training config: {e}")

    # -- Internal ---------------------------------------------------------------

    def _receiver_loop(self) -> None:
        """Background thread: connect to relay, receive training frames."""
        while self._running:
            try:
                self._connect_and_receive()
            except Exception as e:
                logger.debug(f"Browser capture connection error: {e}")

            # Reconnect delay
            if self._running:
                time.sleep(3.0)

    def _connect_and_receive(self) -> None:
        """Single connection attempt + receive loop."""
        try:
            import websocket
        except ImportError:
            logger.warning(
                "websocket-client not installed, browser capture unavailable. "
                "Install with: pip install websocket-client"
            )
            self._running = False
            return

        logger.debug(f"Connecting to relay at {self._relay_url}")
        self._ws = websocket.WebSocket()
        self._ws.settimeout(5.0)

        try:
            self._ws.connect(self._relay_url)
        except Exception as e:
            logger.debug(f"Failed to connect to relay: {e}")
            self._ws = None
            return

        logger.info("Connected to CDP relay for training data")

        # Enable capture on the extension side
        self.configure(enabled=True)

        try:
            while self._running:
                try:
                    raw = self._ws.recv()
                    if not raw:
                        break
                    data = json.loads(raw)
                    self._handle_frame(data)
                except websocket.WebSocketTimeoutException:
                    continue
                except websocket.WebSocketConnectionClosedException:
                    break
        finally:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
            logger.debug("Disconnected from relay")

    def _handle_frame(self, data: dict) -> None:
        """Process a training_frame message from the relay."""
        if data.get("type") != "training_frame":
            return

        frame_data = data.get("data", {})
        text = frame_data.get("text", "")
        metadata = frame_data.get("metadata", {})

        if not text:
            return

        # PII scrubbing on the text content
        if self._scrub_pii:
            text = scrub_text(text)

        timestamp = metadata.get("timestamp", time.time())

        frame = Frame(
            image_data=text.encode("utf-8"),
            timestamp=timestamp,
            source_id=self.source_id,
            modality="text",
            width=0,
            height=0,
            metadata={
                "url": metadata.get("url", ""),
                "title": metadata.get("title", ""),
                "lang": metadata.get("lang", ""),
                "trigger": frame_data.get("trigger", "unknown"),
                "text_length": len(text),
            },
        )

        self._buffer.append(frame)
        logger.debug(
            f"Captured text from {metadata.get('url', '?')} "
            f"({len(text)} chars, trigger={frame_data.get('trigger', '?')})"
        )

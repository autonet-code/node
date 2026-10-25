"""The daemon's reader of its robot's scene samples (docs/robots.md,
Phase B). The robot's summary node appends one JSON sample a line to
``<bundle_dir>/scene.jsonl``; this reads what is new since the last call
and keeps the recent ones in memory."""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any


class SceneSource:
    def __init__(self, path: str | Path, keep_s: float = 600.0) -> None:
        self.path = Path(path)
        self.keep_s = keep_s
        self._offset = 0
        self._buf: deque[dict[str, Any]] = deque()

    def _read(self) -> None:
        try:
            size = self.path.stat().st_size
        except OSError:
            return
        if size < self._offset:          # the file was replaced: start over
            self._offset = 0
        if size == self._offset:
            return
        with self.path.open("rb") as f:
            f.seek(self._offset)
            data = f.read()
        end = data.rfind(b"\n") + 1      # leave a half-written last line
        self._offset += end
        for line in data[:end].splitlines():
            try:
                s = json.loads(line)
            except ValueError:
                continue
            if isinstance(s, dict) and "t" in s and "pose" in s:
                self._buf.append(s)
        cutoff = time.time() - self.keep_s
        while self._buf and self._buf[0]["t"] < cutoff:
            self._buf.popleft()

    def between(self, t0: float, t1: float) -> list[dict[str, Any]]:
        """Samples with ``t0 < t <= t1``."""
        self._read()
        return [s for s in self._buf if t0 < s["t"] <= t1]

    def latest(self) -> dict[str, Any] | None:
        self._read()
        return self._buf[-1] if self._buf else None

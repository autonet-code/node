"""Real-time audio mixing and device lookup.

Ported from kevin's voice_service.py (the more advanced of the two lineages —
it keeps channel pause support that autonet dropped), with autonet's lazy
``start()``/``stop()`` stream lifecycle grafted on so the mixer can be created
without immediately opening an output stream.
"""
from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import sounddevice as sd

MIXER_SR = 44100


# ==============================================================
#  Device lookup
# ==============================================================

def find_device(name: str, kind: str | None = None) -> int | None:
    for i, d in enumerate(sd.query_devices()):
        if name.lower() in d["name"].lower():
            if kind == "input" and d["max_input_channels"] == 0:
                continue
            if kind == "output" and d["max_output_channels"] == 0:
                continue
            return i
    return None


def get_device_list() -> tuple[list[dict], list[dict]]:
    all_devs = list(enumerate(sd.query_devices()))
    default_out = sd.default.device[1]
    default_in = sd.default.device[0]

    outputs = []
    for i, d in all_devs:
        if d["max_output_channels"] > 0:
            outputs.append({"id": i, "name": d["name"], "default": i == default_out})

    inputs = []
    for i, d in all_devs:
        if d["max_input_channels"] > 0:
            inputs.append({"id": i, "name": d["name"], "default": i == default_in})

    return outputs, inputs


# ==============================================================
#  Resampling utility
# ==============================================================

def resample(audio: Any, src_rate: int, dst_rate: int) -> Any:
    if src_rate == dst_rate:
        return audio
    n_out = int(len(audio) * dst_rate / src_rate)
    return np.interp(
        np.linspace(0, len(audio) - 1, n_out),
        np.arange(len(audio)),
        audio,
    ).astype(np.float32)


# ==============================================================
#  Audio channel & mixer
# ==============================================================

class AudioChannel:
    """A single audio channel with volume, fade, and pause support."""

    def __init__(self, volume: float = 1.0) -> None:
        self.volume = volume
        self._chunks: list[Any] = []
        self._chunk_offset = 0
        self._total = 0
        self._lock = threading.Lock()
        self._fade_vol = 1.0
        self._fade_rate = 0.0
        self.paused = False

    def write(self, audio_f32: Any) -> None:
        with self._lock:
            self._chunks.append(audio_f32)
            self._total += len(audio_f32)

    def read(self, n: int) -> Any:
        if self.paused:
            return None
        with self._lock:
            if self._total == 0:
                return None
            out_parts = []
            remaining = n
            while remaining > 0 and self._chunks:
                chunk = self._chunks[0]
                avail = len(chunk) - self._chunk_offset
                take = min(avail, remaining)
                out_parts.append(chunk[self._chunk_offset:self._chunk_offset + take])
                self._chunk_offset += take
                remaining -= take
                self._total -= take
                if self._chunk_offset >= len(chunk):
                    self._chunks.pop(0)
                    self._chunk_offset = 0
            if not out_parts:
                return None
            out = np.concatenate(out_parts) if len(out_parts) > 1 else out_parts[0].copy()
            if len(out) < n:
                out = np.pad(out, (0, n - len(out)))

        if self._fade_rate != 0.0:
            start = self._fade_vol
            end = max(0.0, min(1.0, start + self._fade_rate * n))
            envelope = np.linspace(start, end, n, dtype=np.float32)
            out *= envelope
            self._fade_vol = end
            if self._fade_vol <= 0.0:
                self.clear()
                self._fade_rate = 0.0
        elif self._fade_vol < 1.0:
            out *= self._fade_vol

        return out

    def fade_out(self, duration_secs: float, sr: int = MIXER_SR) -> None:
        total = max(1, int(sr * duration_secs))
        self._fade_rate = -self._fade_vol / total

    def fade_reset(self) -> None:
        self._fade_rate = 0.0
        self._fade_vol = 1.0

    def has_data(self) -> bool:
        with self._lock:
            return self._total > 0

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._chunk_offset = 0
            self._total = 0


class AudioMixer:
    """Multi-channel real-time audio mixer with ducking, muting, and per-channel pause.

    The output stream is opened lazily via ``start()`` (autonet's lifecycle), so
    a mixer can be constructed before an audio device is chosen.  ``set_device``
    transparently restarts the stream if it was running.
    """

    def __init__(self, sr: int = MIXER_SR, blocksize: int = 2048,
                 device: int | None = None) -> None:
        self.sr = sr
        self._blocksize = blocksize
        self._device = device
        self.channels: dict[str, AudioChannel] = {}
        self.master_volume = 1.0
        self._duck_level = 0.25
        self._ducking = False
        self._muting = False
        self._stream: Any = None

    def start(self) -> None:
        """Open and start the audio output stream (idempotent)."""
        if self._stream is not None:
            return
        self._stream = sd.OutputStream(
            samplerate=self.sr, channels=1, dtype="float32",
            blocksize=self._blocksize, callback=self._callback, device=self._device,
        )
        self._stream.start()

    def stop(self) -> None:
        """Stop and close the audio output stream (idempotent)."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def set_device(self, device: int | None) -> None:
        was_running = self._stream is not None
        if was_running:
            self.stop()
        self._device = device
        if was_running:
            self.start()

    def _callback(self, outdata: Any, frames: int, _time_info: Any, _status: Any) -> None:
        mixed = np.zeros(frames, dtype=np.float32)
        for ch in self.channels.values():
            chunk = ch.read(frames)
            if chunk is not None:
                mixed += chunk * ch.volume
        if self._muting:
            vol = 0.0
        elif self._ducking:
            vol = self._duck_level
        else:
            vol = self.master_volume
        outdata[:, 0] = mixed * vol

    def add_channel(self, name: str, volume: float = 1.0) -> None:
        self.channels[name] = AudioChannel(volume=volume)

    def play(self, channel: str, audio_f32: Any, sr: int | None = None) -> None:
        if sr and sr != self.sr:
            audio_f32 = resample(audio_f32, sr, self.sr)
        self.channels[channel].write(audio_f32)

    def duck(self) -> None:
        self._ducking = True
        self._muting = False

    def unduck(self) -> None:
        self._ducking = False

    def mute(self) -> None:
        self._muting = True
        self._ducking = False

    def unmute(self) -> None:
        self._muting = False

    def pause(self, channel: str = "voice") -> None:
        self.channels[channel].paused = True

    def unpause(self, channel: str = "voice") -> None:
        self.channels[channel].paused = False

    def is_paused(self, channel: str = "voice") -> bool:
        return self.channels[channel].paused

    def fade_channel(self, name: str, duration_secs: float) -> None:
        self.channels[name].fade_out(duration_secs, sr=self.sr)

    def reset_channel(self, name: str) -> None:
        ch = self.channels[name]
        ch.clear()
        ch.fade_reset()

    def stop_channel(self, name: str) -> None:
        self.channels[name].clear()

    def stop_all(self) -> None:
        for ch in self.channels.values():
            ch.clear()

    def wait_channel(self, name: str) -> None:
        while self.channels[name].has_data():
            time.sleep(0.03)

    def is_playing(self, name: str | None = None) -> bool:
        if name:
            return self.channels[name].has_data()
        return any(ch.has_data() for ch in self.channels.values())

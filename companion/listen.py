from __future__ import annotations

import sounddevice as sd
import numpy as np


class VoiceDetector:
    """Capture microphone audio and decide when the user is talking."""

    def __init__(
        self,
        sample_rate: int = 16000,
        block_ms: int = 30,
        speech_threshold: float = 0.02,
        start_frames: int = 4,
        end_frames: int = 12,
        device: int | str | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.block_size = max(1, int(sample_rate * block_ms / 1000))
        self.speech_threshold = speech_threshold
        self.start_frames = start_frames
        self.end_frames = end_frames
        self.device = device
        self.speaking = False
        self._loud_run = 0
        self._quiet_run = 0
        self._stream: sd.InputStream | None = None

    def energy(self, block: np.ndarray) -> float:
        samples = block.astype(np.float64, copy=False).reshape(-1)
        return float(np.sqrt(np.mean(np.square(samples))))

    def reset(self) -> None:
        self.speaking = False
        self._loud_run = 0
        self._quiet_run = 0

    def update(self, energy: float) -> bool:
        if energy >= self.speech_threshold:
            self._loud_run += 1
            self._quiet_run = 0
            if not self.speaking and self._loud_run >= self.start_frames:
                self.speaking = True
        else:
            self._quiet_run += 1
            self._loud_run = 0
            if self.speaking and self._quiet_run >= self.end_frames:
                self.speaking = False
        return self.speaking

    def open(self) -> VoiceDetector:
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=1,
            dtype="float32",
            device=self.device,
        )
        self._stream.start()
        return self

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def read(self) -> tuple[float, bool]:
        if self._stream is None:
            raise RuntimeError("Microphone stream is not open.")
        block, _overflowed = self._stream.read(self.block_size)
        energy = self.energy(block)
        return energy, self.update(energy)

    def drain(self, blocks: int = 20) -> None:
        if self._stream is None:
            return
        for _ in range(blocks):
            self._stream.read(self.block_size)
        self.reset()


def list_input_devices() -> list[str]:
    lines = []
    devices = sd.query_devices()
    default_input = sd.default.device[0]
    for index, device in enumerate(devices):
        if device["max_input_channels"] <= 0:
            continue
        marker = ">" if index == default_input else " "
        lines.append(
            f"{marker} {index:2d}  {device['name']}  ({device['max_input_channels']} in)"
        )
    return lines

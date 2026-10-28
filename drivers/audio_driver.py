# audio_driver.py
"""[НА БУДУЩЕЕ] Захват живого звука с многоканальной аудиокарты (sounddevice / pyaudio)."""
from typing import Iterator, Optional

import numpy as np

from drivers.base import BaseAudioSource


class LiveAudioDriver(BaseAudioSource):
    def __init__(self, channels: int, fs: int = 48000, device: Optional[str] = None, chunk_samples: int = 512):
        self.channels = channels
        self.fs = fs
        self.device = device
        self.chunk_samples = chunk_samples

    def chunks(self) -> Iterator[np.ndarray]:
        # TODO: sounddevice.InputStream -> очередь -> yield chunk.T (channels, n), float32
        raise NotImplementedError("Живой захват звука ещё не реализован")

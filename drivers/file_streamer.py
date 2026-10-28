# file_streamer.py
import time
from pathlib import Path
from typing import Iterator, Union

import numpy as np
from scipy.io import wavfile

from drivers.base import BaseAudioSource


def to_float32(data: np.ndarray) -> np.ndarray:
    """PCM любого формата -> float32 в диапазоне [-1, 1]."""
    if data.dtype == np.int16:
        return data.astype(np.float32) / 32768.0
    if data.dtype == np.int32:
        return data.astype(np.float32) / 2147483648.0
    if data.dtype == np.uint8:
        return (data.astype(np.float32) - 128.0) / 128.0
    return data.astype(np.float32)


class FileStreamer(BaseAudioSource):
    """
    Стример для HIL-тестов: читает многоканальный .wav и отдаёт его чанками,
    имитируя поток от звуковой карты.

    realtime=True — выдерживать реальную скорость потока (для проверки задержек «как на борту»).
    """

    def __init__(self, path: Union[str, Path], chunk_samples: int = 512, realtime: bool = False):
        self.path = Path(path)
        self.chunk_samples = chunk_samples
        self.realtime = realtime

        fs, data = wavfile.read(self.path)
        data = to_float32(data)
        if data.ndim == 1:
            data = data[:, None]

        self.fs = int(fs)
        self.data = np.ascontiguousarray(data.T)    # (channels, samples)
        self.channels = self.data.shape[0]

    @property
    def duration_s(self) -> float:
        return self.data.shape[1] / self.fs

    def chunks(self) -> Iterator[np.ndarray]:
        chunk_period = self.chunk_samples / self.fs
        next_deadline = time.perf_counter()
        for start in range(0, self.data.shape[1], self.chunk_samples):
            if self.realtime:
                next_deadline += chunk_period
                delay = next_deadline - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            yield self.data[:, start:start + self.chunk_samples]

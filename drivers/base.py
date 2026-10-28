# base.py
from abc import ABC, abstractmethod
from typing import Iterator

import numpy as np


class BaseAudioSource(ABC):
    """Источник многоканального аудио. Отдаёт чанки произвольной длины формы (channels, n), float32."""

    fs: int
    channels: int

    @abstractmethod
    def chunks(self) -> Iterator[np.ndarray]:
        ...

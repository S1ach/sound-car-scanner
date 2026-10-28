# onset_detector.py
from typing import Tuple

import numpy as np

from core.interfaces import BaseEventDetector
from core.registry import register


@register("detector", "onset_detector")
class OnsetDetector(BaseEventDetector):
    """
    Детектор резких звуков/атак. Ответственный: Косолапов.
    При срабатывании конвейер запускает локализаторы с "trigger": "onset_detector".
    """

    def prepare(self) -> None:
        # TODO(Косолапов): пороги, длина истории энергии
        pass

    def detect(self, X_frame: np.ndarray, fs: int) -> Tuple[bool, float]:
        # TODO(Косолапов): скачок энергии / spectral flux относительно истории
        return False, 0.0

# denoiser.py
import numpy as np

from core.interfaces import BasePreProcessor
from core.registry import register


@register("pre_processor", "denoiser")
class SpectralSubtractionDenoiser(BasePreProcessor):
    """
    Спектральное вычитание шума колёс и мотора. Ответственный: Линьков.
    Бюджет: 2–3 мс на фрейм.
    """

    def prepare(self) -> None:
        # TODO(Линьков): оценка профиля шума, параметры alpha/beta, сглаживание
        pass

    def process(self, X_frame: np.ndarray, fs: int) -> np.ndarray:
        # TODO(Линьков): rfft -> вычитание оценки шума -> irfft
        return X_frame

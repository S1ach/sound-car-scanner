# music_localizer.py
from typing import Tuple

import numpy as np

from core.interfaces import BaseDOALocalizer
from core.registry import register


@register("localizer", "music_siren")
class MusicSirenLocalizer(BaseDOALocalizer):
    """Алгоритм MUSIC для узкополосных сигналов сирен. Ответственный: Акулов."""

    def prepare(self) -> None:
        # TODO(Акулов): прерасчёт LUT / steering-векторов (см. scripts/precompute_lut.py)
        pass

    def execute_doa_localization(self, X_frame: np.ndarray, fs: int) -> Tuple[float, float]:
        # TODO(Акулов): реализовать алгоритм
        return 0.0, 0.0

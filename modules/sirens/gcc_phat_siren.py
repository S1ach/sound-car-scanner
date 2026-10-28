# gcc_phat_siren.py
from typing import Tuple

import numpy as np

from core.interfaces import BaseDOALocalizer
from core.registry import register


@register("localizer", "gcc_phat_siren")
class GccPhatSirenLocalizer(BaseDOALocalizer):
    """GCC-PHAT + полосовая фильтрация под сирены. Ответственный: Кузин."""

    def prepare(self) -> None:
        # TODO(Кузин): прерасчёт LUT / steering-векторов (см. scripts/precompute_lut.py)
        pass

    def execute_doa_localization(self, X_frame: np.ndarray, fs: int) -> Tuple[float, float]:
        # TODO(Кузин): реализовать алгоритм
        return 0.0, 0.0

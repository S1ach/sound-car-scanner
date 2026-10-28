# gcc_phat_wide.py
from typing import Tuple

import numpy as np

from core.interfaces import BaseDOALocalizer
from core.registry import register


@register("localizer", "gcc_phat_wide")
class GccPhatWideLocalizer(BaseDOALocalizer):
    """GCC-PHAT для широкой полосы (запускается по триггеру onset_detector). Ответственный: Косолапов."""

    def prepare(self) -> None:
        # TODO(Косолапов): прерасчёт LUT / steering-векторов (см. scripts/precompute_lut.py)
        pass

    def execute_doa_localization(self, X_frame: np.ndarray, fs: int) -> Tuple[float, float]:
        # TODO(Косолапов): реализовать алгоритм
        return 0.0, 0.0

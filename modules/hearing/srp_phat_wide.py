# srp_phat_wide.py
from typing import Tuple

import numpy as np

from core.interfaces import BaseDOALocalizer
from core.registry import register


@register("localizer", "srp_phat_wide")
class SrpPhatWideLocalizer(BaseDOALocalizer):
    """Быстрый SRP-PHAT круговой панорамы (запускается по триггеру onset_detector). Ответственный: Балуев."""

    def prepare(self) -> None:
        # TODO(Балуев): прерасчёт LUT / steering-векторов (см. scripts/precompute_lut.py)
        pass

    def execute_doa_localization(self, X_frame: np.ndarray, fs: int) -> Tuple[float, float]:
        # TODO(Балуев): реализовать алгоритм
        return 0.0, 0.0

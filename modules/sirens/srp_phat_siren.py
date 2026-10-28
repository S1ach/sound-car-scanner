# srp_phat_siren.py
from typing import Tuple

import numpy as np

from core.interfaces import BaseDOALocalizer
from core.registry import register


@register("localizer", "srp_phat_siren")
class SrpPhatSirenLocalizer(BaseDOALocalizer):
    """SRP-PHAT со сканированием по LUT задержек. Ответственный: Сагитов."""

    def prepare(self) -> None:
        # TODO(Сагитов): прерасчёт LUT / steering-векторов (см. scripts/precompute_lut.py)
        pass

    def execute_doa_localization(self, X_frame: np.ndarray, fs: int) -> Tuple[float, float]:
        # TODO(Сагитов): реализовать алгоритм
        return 0.0, 0.0

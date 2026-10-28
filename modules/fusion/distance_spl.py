# distance_spl.py
import math
from typing import Optional, Tuple

import numpy as np

from core.interfaces import BaseDistanceEstimator
from core.registry import register


@register("distance", "distance_spl")
class SplDistanceEstimator(BaseDistanceEstimator):
    """Расстояние по затуханию энергии (SPL). Ответственный: Алехина."""

    def execute_distance_estimation(
        self, X_frame: np.ndarray, fs: int, angle_degrees: Optional[float]
    ) -> Tuple[float, float]:
        # TODO(Алехина): реализовать оценку расстояния
        return math.nan, 0.0

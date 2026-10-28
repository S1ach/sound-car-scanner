# camera_calibration.py
"""[НА БУДУЩЕЕ] Перевод пикселей (x, y) в углы DOA (азимут / угол места) в системе координат массива."""
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: Tuple[float, ...] = ()            # коэффициенты дисторсии (OpenCV)


@dataclass(frozen=True)
class CameraCalibration:
    """
    intrinsics — внутренние параметры камеры.
    rotation_cam_to_array (3, 3) и translation_cam_to_array_m (3,) — положение камеры
    относительно центра микрофонного массива (оси массива — см. core/geometry.py).
    """
    intrinsics: CameraIntrinsics
    rotation_cam_to_array: np.ndarray = field(default_factory=lambda: np.eye(3))
    translation_cam_to_array_m: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def pixel_to_doa(self, x_px: float, y_px: float) -> Tuple[float, float]:
        """
        Returns:
            Tuple[float, float]: azimuth_degrees [0.0, 360.0), elevation_degrees [-90.0, 90.0]
            в соглашениях массива (0° — вдоль +X, против часовой стрелки).
        """
        # TODO: undistort -> луч в системе камеры -> поворот в систему массива -> азимут/угол места
        raise NotImplementedError("Калибровка камеры ещё не реализована")

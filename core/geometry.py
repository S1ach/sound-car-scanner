# geometry.py
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict

DEFAULT_SPEED_OF_SOUND = 343.0


@dataclass(frozen=True)
class ArrayGeometry:
    """
    Геометрия микрофонного массива.

    Система координат: метры, правая тройка XYZ.
    Азимут: 0° — вдоль +X, рост против часовой стрелки (90° — вдоль +Y), диапазон [0, 360).
    Порядок микрофонов в mic_positions == порядок каналов в X_frame.
    """
    name: str
    mic_positions: np.ndarray          # (channels, 3)
    speed_of_sound: float = DEFAULT_SPEED_OF_SOUND

    @property
    def channels(self) -> int:
        return self.mic_positions.shape[0]

    @property
    def center(self) -> np.ndarray:
        return self.mic_positions.mean(axis=0)

    @property
    def max_aperture_m(self) -> float:
        diff = self.mic_positions[:, None, :] - self.mic_positions[None, :, :]
        return float(np.linalg.norm(diff, axis=-1).max())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArrayGeometry":
        positions = np.asarray(data["mic_positions"], dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"mic_positions должен иметь форму (channels, 3), получено {positions.shape}")
        return cls(
            name=data.get("name", "unnamed"),
            mic_positions=positions,
            speed_of_sound=float(data.get("speed_of_sound", DEFAULT_SPEED_OF_SOUND)),
        )


def far_field_delays(geometry: ArrayGeometry, azimuth_degrees: np.ndarray) -> np.ndarray:
    """
    Задержки прихода плоской волны на каждый микрофон относительно центра массива (дальняя зона, плоскость XY).

    Args:
        azimuth_degrees: скаляр или массив азимутов формы (n_angles,).

    Returns:
        np.ndarray формы (n_angles, channels), секунды. Отрицательная задержка — микрофон ближе к источнику
        и слышит сигнал раньше центра. Сигнал на микрофоне m: x_m(t) = s(t - tau_m).
    """
    az = np.deg2rad(np.atleast_1d(np.asarray(azimuth_degrees, dtype=np.float64)))
    directions = np.stack([np.cos(az), np.sin(az), np.zeros_like(az)], axis=-1)   # (n_angles, 3)
    rel_positions = geometry.mic_positions - geometry.center                       # (channels, 3)
    return -(directions @ rel_positions.T) / geometry.speed_of_sound

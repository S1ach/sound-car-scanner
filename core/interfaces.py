# interfaces.py
"""
Архитектурный контракт алгоритмических блоков.

Каждый разработчик наследуется от нужного базового класса, переопределяет единственный абстрактный метод
и регистрирует класс декоратором @register(...) из core.registry — после этого блок подключается
к конвейеру через config/pipeline_config.json без правок pipeline.py.

Общие правила для всех блоков:
  * X_frame — np.ndarray float32 формы (channels, samples), уже умноженный на окно Ханна.
  * X_frame передаётся только для чтения (writeable=False). Нужно изменить — сделайте копию.
  * Тяжёлые вычисления, не зависящие от данных (LUT, steering-векторы, FFT-планы), — в __init__ / prepare().
  * Метод обработки должен укладываться в бюджет реального времени (см. core/profiler.py).
"""
import math
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import numpy as np

from core.geometry import ArrayGeometry


class ContractViolation(RuntimeError):
    """Блок вернул результат, нарушающий архитектурный контракт."""


class BaseBlock(ABC):
    name: str = "base"        # выставляется автоматически декоратором @register

    def __init__(self, geometry: ArrayGeometry, fs: int, **params: Any):
        self.geometry = geometry
        self.fs = fs
        self.params = params
        self.prepare()

    def prepare(self) -> None:
        """Хук для одноразовых прерасчётов (LUT, матрицы, фильтры). Вызывается из __init__."""

    def reset(self) -> None:
        """Сброс внутреннего состояния между прогонами (оценки шума, трекеры и т.п.)."""


class BasePreProcessor(BaseBlock):
    """Блок очистки сигнала (шумоподавление, фильтрация)."""

    @abstractmethod
    def process(self, X_frame: np.ndarray, fs: int) -> np.ndarray:
        """
        Returns:
            np.ndarray той же формы (channels, samples), float32 — очищенный фрейм.
        """


class BaseEventDetector(BaseBlock):
    """Детектор событий (атаки, импульсы). Используется как триггер для других блоков."""

    @abstractmethod
    def detect(self, X_frame: np.ndarray, fs: int) -> Tuple[bool, float]:
        """
        Returns:
            Tuple[bool, float]:
                1. triggered (bool): событие обнаружено в текущем фрейме.
                2. score (float): сила/уверенность детекции в диапазоне [0.0, 1.0].
        """


class BaseDOALocalizer(BaseBlock):
    """Алгоритм определения направления прихода звука (DOA)."""

    @abstractmethod
    def execute_doa_localization(self, X_frame: np.ndarray, fs: int) -> Tuple[float, float]:
        """
        Унифицированный интерфейс вызова алгоритма DOA (Direction of Arrival).

        Args:
            X_frame (np.ndarray): Двумерный массив вещественных чисел формы (channels, samples).
                                  Пример: (8, 1920) для 8 микрофонов и окна 40 мс при fs=48000 Гц.
            fs (int): Текущая частота дискретизации аудиопотока в Гц (например, 44100, 48000).

        Returns:
            Tuple[float, float]: Пары значений:
                1. angle_degrees (float): Расчитанный угол направления (Azimuth) в диапазоне [0.0, 360.0].
                2. confidence_score (float): Метрика уверенности алгоритма в результате в диапазоне [0.0, 1.0].
                                             Где 0.0 - полная неопределенность/тишина, 1.0 - абсолютная точность.
        """
        raise NotImplementedError("Каждый разработчик обязан переопределить эту функцию внутри своего модуля.")


class BaseDistanceEstimator(BaseBlock):
    """Оценка расстояния до источника."""

    @abstractmethod
    def execute_distance_estimation(
        self, X_frame: np.ndarray, fs: int, angle_degrees: Optional[float]
    ) -> Tuple[float, float]:
        """
        Args:
            angle_degrees: лучший азимут текущего фрейма от DOA-блоков (None, если уверенного нет).

        Returns:
            Tuple[float, float]:
                1. distance_m (float): расстояние в метрах, >= 0. math.nan — оценки нет.
                2. confidence_score (float): уверенность в диапазоне [0.0, 1.0].
        """


# ---------------------------------------------------------------------------
# Валидация результатов (используется конвейером и контрактными тестами)
# ---------------------------------------------------------------------------

def _check_unit_interval(value: Any, what: str, block: str) -> float:
    value = float(value)
    if not (0.0 <= value <= 1.0):
        raise ContractViolation(f"[{block}] {what}={value} вне диапазона [0.0, 1.0]")
    return value


def _check_pair(result: Any, block: str) -> Tuple[Any, Any]:
    if not isinstance(result, tuple) or len(result) != 2:
        raise ContractViolation(f"[{block}] ожидался кортеж из 2 значений, получено {result!r}")
    return result


def validate_frame(X_frame: np.ndarray, channels: int, block: str) -> None:
    if not isinstance(X_frame, np.ndarray) or X_frame.ndim != 2 or X_frame.shape[0] != channels:
        shape = getattr(X_frame, "shape", None)
        raise ContractViolation(f"[{block}] ожидался фрейм формы ({channels}, samples), получено {shape}")


def validate_doa_result(result: Any, block: str = "doa") -> Tuple[float, float]:
    angle, confidence = _check_pair(result, block)
    angle = float(angle)
    if not math.isfinite(angle) or not (0.0 <= angle <= 360.0):
        raise ContractViolation(f"[{block}] angle_degrees={angle} вне диапазона [0.0, 360.0]")
    return angle % 360.0, _check_unit_interval(confidence, "confidence_score", block)


def validate_detector_result(result: Any, block: str = "detector") -> Tuple[bool, float]:
    triggered, score = _check_pair(result, block)
    return bool(triggered), _check_unit_interval(score, "score", block)


def validate_distance_result(result: Any, block: str = "distance") -> Tuple[float, float]:
    distance, confidence = _check_pair(result, block)
    distance = float(distance)
    if not math.isnan(distance) and (distance < 0.0 or math.isinf(distance)):
        raise ContractViolation(f"[{block}] distance_m={distance} должно быть >= 0 или nan")
    return distance, _check_unit_interval(confidence, "confidence_score", block)

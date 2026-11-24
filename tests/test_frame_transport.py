"""
Приёмка задачи «каркас передаёт фреймы по контракту (channels, samples) -> (angle, confidence)».

1. Каждый локализатор получает фрейм нужной формы, float32, только для чтения, с правильным fs.
2. Ответ (angle, confidence) доходит до FrameResult и best_doa.
3. Эталонный SRP-PHAT на синтетике находит истинный азимут — каркас передаёт данные без искажений.
"""
from typing import List, Tuple

import numpy as np
import pytest

from core.config import load_array_geometry, load_pipeline_config
from core.interfaces import BaseDOALocalizer
from core.registry import create
from pipeline import Pipeline
from scripts.run_simulation import simulate_synthetic
from tests.test_pipeline_smoke import ArraySource

FS = 48000
GEOMETRY = load_array_geometry()


def circular_error(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


class SpyLocalizer(BaseDOALocalizer):
    """Запоминает всё, что пришло на вход, и возвращает фиксированный ответ."""
    name = "spy"

    def prepare(self) -> None:
        self.calls: List[Tuple[tuple, np.dtype, bool, int]] = []

    def execute_doa_localization(self, X_frame, fs):
        self.calls.append((X_frame.shape, X_frame.dtype, X_frame.flags.writeable, fs))
        return 123.0, 0.9


def synthetic(azimuth: float, duration_s: float = 0.5, snr_db: float = 20.0):
    X, _ = simulate_synthetic(GEOMETRY, FS, duration_s=duration_s, azimuth_deg=azimuth,
                              distance_m=10.0, snr_db=snr_db, seed=0)
    return X


def test_every_frame_reaches_localizer_by_contract():
    pipeline = Pipeline(load_pipeline_config(), GEOMETRY, FS)
    spy = SpyLocalizer(geometry=GEOMETRY, fs=FS)
    pipeline.localizers.append((spy, None))

    results = pipeline.run(ArraySource(synthetic(60.0), FS, chunk=333))   # «неудобный» размер чанка

    assert len(spy.calls) == len(results) > 0
    for shape, dtype, writeable, fs in spy.calls:
        assert shape == (GEOMETRY.channels, 1920)          # 8 каналов × 40 мс при 48 кГц
        assert dtype == np.float32
        assert writeable is False
        assert fs == FS
    for r in results:
        assert r.doa["spy"] == (123.0, 0.9)
        assert r.best_doa is not None


@pytest.mark.parametrize("azimuth", [0.0, 60.0, 135.0, 250.0, 330.0])
def test_reference_localizer_recovers_azimuth_through_pipeline(azimuth):
    pipeline = Pipeline(load_pipeline_config(), GEOMETRY, FS)
    results = pipeline.run(ArraySource(synthetic(azimuth), FS))

    angles = [r.doa["srp_phat_reference"][0] for r in results]
    errors = [circular_error(a, azimuth) for a in angles]
    assert np.median(errors) <= 3.0
    assert all(r.best_doa is not None and r.best_doa[0] == "srp_phat_reference" for r in results)


def test_reference_localizer_low_confidence_on_noise_and_zero_on_silence():
    block = create("localizer", "srp_phat_reference", geometry=GEOMETRY, fs=FS)
    rng = np.random.default_rng(1)
    noise = (rng.normal(0, 0.1, (GEOMETRY.channels, 1920)) * np.hanning(1920)).astype(np.float32)
    silence = np.zeros((GEOMETRY.channels, 1920), dtype=np.float32)

    assert block.execute_doa_localization(noise, FS)[1] < 0.3
    assert block.execute_doa_localization(silence, FS) == (0.0, 0.0)


def test_reference_localizer_works_at_44100():
    X, _ = simulate_synthetic(GEOMETRY, 44100, duration_s=0.3, azimuth_deg=200.0,
                              distance_m=10.0, snr_db=20.0, seed=0)
    pipeline = Pipeline(load_pipeline_config(), GEOMETRY, 44100)
    results = pipeline.run(ArraySource(X, 44100))
    assert np.median([circular_error(r.doa["srp_phat_reference"][0], 200.0) for r in results]) <= 3.0

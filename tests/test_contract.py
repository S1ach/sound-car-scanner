"""
Контрактные тесты: каждый зарегистрированный блок обязан соблюдать единый интерфейс.
Новый модуль проверяется автоматически — достаточно зарегистрировать его через @register.
"""
import numpy as np
import pytest

from core.config import load_array_geometry
from core.interfaces import (
    validate_detector_result, validate_distance_result, validate_doa_result, validate_frame,
)
from core.registry import available, create

FS = 48000
GEOMETRY = load_array_geometry()


def frames():
    rng = np.random.default_rng(0)
    n = int(0.04 * FS)
    window = np.hanning(n).astype(np.float32)
    noise = rng.normal(0, 0.1, (GEOMETRY.channels, n)).astype(np.float32) * window
    silence = np.zeros((GEOMETRY.channels, n), dtype=np.float32)
    for f in (noise, silence):
        f.setflags(write=False)       # блоки не имеют права менять входной фрейм
    return [noise, silence]


def make(kind, name):
    return create(kind, name, geometry=GEOMETRY, fs=FS)


@pytest.mark.parametrize("name", available("localizer"))
def test_localizer_contract(name):
    block = make("localizer", name)
    for frame in frames():
        validate_doa_result(block.execute_doa_localization(frame, FS), name)


@pytest.mark.parametrize("name", available("pre_processor"))
def test_pre_processor_contract(name):
    block = make("pre_processor", name)
    for frame in frames():
        out = block.process(frame, FS)
        validate_frame(out, GEOMETRY.channels, name)
        assert out.shape == frame.shape
        assert np.all(np.isfinite(out))


@pytest.mark.parametrize("name", available("detector"))
def test_detector_contract(name):
    block = make("detector", name)
    for frame in frames():
        validate_detector_result(block.detect(frame, FS), name)


@pytest.mark.parametrize("name", available("distance"))
@pytest.mark.parametrize("angle", [None, 45.0])
def test_distance_contract(name, angle):
    block = make("distance", name)
    for frame in frames():
        validate_distance_result(block.execute_distance_estimation(frame, FS, angle), name)

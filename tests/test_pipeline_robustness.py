"""
Поведение конвейера при сбойных блоках и бюджеты.
Тестовые блоки подставляются напрямую, без @register, чтобы не попасть в контрактные тесты.
"""
import copy
import csv

import numpy as np
import pytest

from core import profiler
from core.config import load_array_geometry, load_pipeline_config
from core.interfaces import (
    BaseDOALocalizer, BasePreProcessor, ContractViolation,
    validate_detector_result, validate_distance_result, validate_doa_result, validate_frame,
)
from pipeline import Pipeline, write_log
from tests.test_pipeline_smoke import ArraySource

FS = 48000
GEOMETRY = load_array_geometry()
N = 1920


def make_pipeline(strict, **overrides):
    config = copy.deepcopy(load_pipeline_config())
    config["strict_contract"] = strict
    config.update(overrides)
    return Pipeline(config, GEOMETRY, FS)


def silence():
    return np.zeros((GEOMETRY.channels, N), dtype=np.float32)


class WrongLength(BasePreProcessor):
    name = "wrong_length"

    def process(self, X_frame, fs):
        return X_frame[:, :100]


class WrongDtype(BasePreProcessor):
    name = "wrong_dtype"

    def process(self, X_frame, fs):
        return X_frame.astype(np.float64)


class ProducesNan(BasePreProcessor):
    name = "produces_nan"

    def process(self, X_frame, fs):
        return X_frame * np.nan


class Raises(BaseDOALocalizer):
    name = "raises"

    def execute_doa_localization(self, X_frame, fs):
        raise ValueError("сбой алгоритма")


class ReturnsNone(BaseDOALocalizer):
    name = "returns_none"

    def execute_doa_localization(self, X_frame, fs):
        return None, 0.5


class Confident(BaseDOALocalizer):
    name = "confident"

    def execute_doa_localization(self, X_frame, fs):
        return 42.0, 0.9


@pytest.mark.parametrize("bad", [WrongLength, WrongDtype, ProducesNan])
def test_bad_pre_processor_strict(bad):
    p = make_pipeline(strict=True)
    p.pre_processors = [bad(GEOMETRY, FS)]
    with pytest.raises(ContractViolation):
        p.process_frame(silence())


@pytest.mark.parametrize("bad", [WrongLength, WrongDtype, ProducesNan])
def test_bad_pre_processor_is_skipped_when_not_strict(bad):
    p = make_pipeline(strict=False)
    p.pre_processors = [bad(GEOMETRY, FS)]
    p.localizers = [(Confident(GEOMETRY, FS), None)]
    with pytest.warns(RuntimeWarning):
        result = p.process_frame(silence())
    assert result.doa["confident"] == (42.0, 0.9)
    assert p.block_failures == {bad.name: 1}


@pytest.mark.parametrize("bad", [Raises, ReturnsNone])
def test_failing_localizer_does_not_stop_others(bad):
    p = make_pipeline(strict=False)
    p.localizers = [(bad(GEOMETRY, FS), None), (Confident(GEOMETRY, FS), None)]
    with pytest.warns(RuntimeWarning):
        result = p.process_frame(silence())
    assert bad.name not in result.doa
    assert result.best_doa == ("confident", 42.0, 0.9)


@pytest.mark.parametrize("bad, error", [(Raises, ValueError), (ReturnsNone, ContractViolation)])
def test_failing_localizer_strict(bad, error):
    p = make_pipeline(strict=True)
    p.localizers = [(bad(GEOMETRY, FS), None)]
    with pytest.raises(error):
        p.process_frame(silence())


def test_validators_reject_non_numbers():
    with pytest.raises(ContractViolation):
        validate_doa_result((None, 0.5))
    with pytest.raises(ContractViolation):
        validate_distance_result(("далеко", 0.5))
    with pytest.raises(ContractViolation):
        validate_detector_result((0.7, 0.5))


def test_validate_frame_samples():
    frame = np.zeros((8, 100), dtype=np.float32)
    validate_frame(frame, 8, "block")
    with pytest.raises(ContractViolation):
        validate_frame(frame, 8, "block", samples=N)


def test_block_budget_from_config():
    profiler.reset_stats()
    p = make_pipeline(strict=True)
    p.process_frame(silence())
    assert profiler.get_stats()["denoiser"].budget_ms == 3.0


def test_frame_budget_is_limited_by_hop():
    assert make_pipeline(strict=True).budget_ms == 20.0
    p = make_pipeline(strict=True, window={"frame_duration_ms": 40.0, "overlap_pct": 75.0})
    assert p.budget_ms == 10.0


def test_run_restarts_stream():
    data = np.zeros((GEOMETRY.channels, FS // 5), dtype=np.float32)
    p = make_pipeline(strict=True)
    first = p.run(ArraySource(data, FS))
    second = p.run(ArraySource(data, FS))
    assert [r.timestamp_s for r in first] == [r.timestamp_s for r in second]


def test_keep_results_false_streams_only_to_callback():
    data = np.zeros((GEOMETRY.channels, FS // 5), dtype=np.float32)
    seen = []
    assert make_pipeline(strict=True).run(ArraySource(data, FS), seen.append, keep_results=False) == []
    assert len(seen) == 9


def test_best_doa_is_logged(tmp_path):
    p = make_pipeline(strict=True)
    p.localizers = [(Confident(GEOMETRY, FS), None)]
    path = tmp_path / "log.csv"
    write_log([p.process_frame(silence())], path)
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["kind"] == "best_doa"]
    assert len(rows) == 1
    assert (rows[0]["block"], float(rows[0]["value"])) == ("confident", 42.0)

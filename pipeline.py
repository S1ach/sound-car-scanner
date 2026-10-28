# pipeline.py
"""
Главный оркестратор: Источник -> Оконный менеджер -> Очистка -> Детекторы -> DOA -> Дистанция -> Лог.

Запуск:
    python pipeline.py --wav data/raw/sim.wav
    python pipeline.py --wav data/raw/sim.wav --realtime --log data/logs/run.csv
"""
import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from core import profiler
from core.config import load_array_geometry, load_pipeline_config
from core.geometry import ArrayGeometry
from core.interfaces import (
    ContractViolation, validate_detector_result, validate_distance_result, validate_doa_result, validate_frame,
)
from core.registry import create
from core.window_manager import RealTimeWindowManager
from drivers.base import BaseAudioSource


@dataclass
class FrameResult:
    frame_index: int
    timestamp_s: float                                   # центр фрейма относительно начала потока
    detections: Dict[str, Tuple[bool, float]] = field(default_factory=dict)
    doa: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    distance: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    best_doa: Optional[Tuple[str, float, float]] = None  # (блок, угол, уверенность) выше порога
    processing_ms: float = 0.0


def _enabled(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [e for e in entries if e.get("enabled", True)]


class Pipeline:
    def __init__(self, config: Dict[str, Any], geometry: ArrayGeometry, fs: int):
        self.config = config
        self.geometry = geometry
        self.fs = fs
        self.budget_ms = float(config.get("realtime_budget_ms", 20.0))
        self.confidence_threshold = float(config.get("confidence_threshold", 0.5))
        self.strict = bool(config.get("strict_contract", True))

        window = config.get("window", {})
        self.window_manager = RealTimeWindowManager(
            channels=geometry.channels,
            fs=fs,
            frame_duration_ms=window.get("frame_duration_ms", 40.0),
            overlap_pct=window.get("overlap_pct", 50.0),
        )

        def build(kind: str, key: str):
            return [
                (create(kind, e["name"], geometry=geometry, fs=fs, **e.get("params", {})), e)
                for e in _enabled(config.get(key, []))
            ]

        self.pre_processors = [b for b, _ in build("pre_processor", "pre_processing")]
        self.detectors = [b for b, _ in build("detector", "detectors")]
        self.localizers = [(b, e.get("trigger")) for b, e in build("localizer", "localizers")]
        self.distance_estimators = [b for b, _ in build("distance", "distance")]

        known_detectors = {d.name for d in self.detectors}
        for loc, trigger in self.localizers:
            if trigger is not None and trigger not in known_detectors:
                raise ValueError(f"Локализатор '{loc.name}' ссылается на неактивный детектор '{trigger}'")

        self.frame_index = 0

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.window_manager.reset()
        self.frame_index = 0
        for block in (*self.pre_processors, *self.detectors, *(l for l, _ in self.localizers),
                      *self.distance_estimators):
            block.reset()

    def _call(self, block, validator: Callable, fn: Callable, *args):
        with profiler.measure(f"{block.name}"):
            result = fn(*args)
        try:
            return validator(result, block.name)
        except ContractViolation:
            if self.strict:
                raise
            return None

    def process_frame(self, frame: np.ndarray) -> FrameResult:
        wm = self.window_manager
        result = FrameResult(
            frame_index=self.frame_index,
            timestamp_s=(self.frame_index * wm.hop_samples + wm.frame_samples / 2) / self.fs,
        )
        self.frame_index += 1

        with profiler.measure("pipeline:frame", self.budget_ms):
            frame.setflags(write=False)
            for pre in self.pre_processors:
                with profiler.measure(pre.name):
                    frame = pre.process(frame, self.fs)
                validate_frame(frame, self.geometry.channels, pre.name)
                if frame.flags.writeable:
                    frame.setflags(write=False)

            for det in self.detectors:
                r = self._call(det, validate_detector_result, det.detect, frame, self.fs)
                if r is not None:
                    result.detections[det.name] = r

            for loc, trigger in self.localizers:
                if trigger is not None and not result.detections.get(trigger, (False, 0.0))[0]:
                    continue
                r = self._call(loc, validate_doa_result, loc.execute_doa_localization, frame, self.fs)
                if r is not None:
                    result.doa[loc.name] = r

            confident = [(n, a, c) for n, (a, c) in result.doa.items() if c >= self.confidence_threshold]
            result.best_doa = max(confident, key=lambda x: x[2]) if confident else None
            best_angle = result.best_doa[1] if result.best_doa else None

            for est in self.distance_estimators:
                r = self._call(est, validate_distance_result, est.execute_distance_estimation,
                               frame, self.fs, best_angle)
                if r is not None:
                    result.distance[est.name] = r

        result.processing_ms = profiler.get_stats()["pipeline:frame"].last_ms
        return result

    def run(self, source: BaseAudioSource,
            on_result: Optional[Callable[[FrameResult], None]] = None) -> List[FrameResult]:
        if source.fs != self.fs or source.channels != self.geometry.channels:
            raise ValueError(
                f"Источник ({source.channels} кан., {source.fs} Гц) не совпадает с конвейером "
                f"({self.geometry.channels} кан., {self.fs} Гц)"
            )
        results: List[FrameResult] = []

        def callback(frame: np.ndarray) -> None:
            r = self.process_frame(frame)
            results.append(r)
            if on_result is not None:
                on_result(r)

        for chunk in source.chunks():
            self.window_manager.process_stream_chunk(chunk, callback)
        return results


# ----------------------------------------------------------------------
LOG_COLUMNS = ["frame_index", "timestamp_s", "kind", "block", "value", "confidence", "processing_ms"]


def write_log(results: List[FrameResult], path: Path) -> None:
    """Длинный формат: одна строка на (фрейм, блок). value — угол (°), расстояние (м) или флаг детектора."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(LOG_COLUMNS)
        for r in results:
            base = [r.frame_index, f"{r.timestamp_s:.4f}"]
            for kind, items in (("detector", r.detections), ("doa", r.doa), ("distance", r.distance)):
                for name, (value, conf) in items.items():
                    value = "" if isinstance(value, float) and math.isnan(value) else float(value)
                    w.writerow(base + [kind, name, value, f"{conf:.4f}", f"{r.processing_ms:.3f}"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Sound Car Scanner — прогон конвейера по .wav файлу")
    parser.add_argument("--wav", required=True, help="многоканальный .wav (каналы в порядке array_config)")
    parser.add_argument("--array-config", default=None)
    parser.add_argument("--pipeline-config", default=None)
    parser.add_argument("--log", default="data/logs/pipeline_log.csv")
    parser.add_argument("--chunk", type=int, default=512, help="размер чанка «звуковой карты», сэмплов")
    parser.add_argument("--realtime", action="store_true", help="выдерживать реальную скорость потока")
    args = parser.parse_args()

    from drivers.file_streamer import FileStreamer

    geometry = load_array_geometry(args.array_config)
    config = load_pipeline_config(args.pipeline_config)
    source = FileStreamer(args.wav, chunk_samples=args.chunk, realtime=args.realtime)

    pipeline = Pipeline(config, geometry, source.fs)
    results = pipeline.run(source)
    write_log(results, Path(args.log))

    print(f"Файл: {args.wav} ({source.duration_s:.2f} с, {source.channels} кан., {source.fs} Гц)")
    print(f"Фреймов: {len(results)}, лог: {args.log}\n")
    print(profiler.report())


if __name__ == "__main__":
    main()

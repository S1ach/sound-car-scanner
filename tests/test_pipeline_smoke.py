import numpy as np

from core.config import load_array_geometry, load_pipeline_config
from drivers.base import BaseAudioSource
from pipeline import Pipeline
from scripts.run_simulation import simulate_synthetic


class ArraySource(BaseAudioSource):
    def __init__(self, data: np.ndarray, fs: int, chunk: int = 512):
        self.data, self.fs, self.channels, self.chunk = data, fs, data.shape[0], chunk

    def chunks(self):
        for start in range(0, self.data.shape[1], self.chunk):
            yield self.data[:, start:start + self.chunk]


def test_pipeline_runs_end_to_end():
    fs = 48000
    geometry = load_array_geometry()
    X, _ = simulate_synthetic(geometry, fs, duration_s=0.5, azimuth_deg=60.0,
                              distance_m=10.0, snr_db=20.0, seed=0)
    pipeline = Pipeline(load_pipeline_config(), geometry, fs)
    results = pipeline.run(ArraySource(X, fs))

    assert len(results) == 1 + (X.shape[1] - 1920) // 960
    assert results[0].doa  # нетриггерные локализаторы отработали
    assert all(np.isclose(r.timestamp_s, (r.frame_index * 960 + 960) / fs) for r in results)

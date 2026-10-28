import numpy as np

from core.window_manager import RealTimeWindowManager


def collect(wm, signal, chunk):
    frames = []
    for start in range(0, signal.shape[1], chunk):
        wm.process_stream_chunk(signal[:, start:start + chunk], lambda f: frames.append(f.copy()))
    return frames


def test_frame_geometry_and_count():
    fs, channels = 48000, 8
    wm = RealTimeWindowManager(channels, fs, 40.0, 50.0)
    assert (wm.frame_samples, wm.hop_samples) == (1920, 960)

    n = fs  # 1 с
    frames = collect(wm, np.ones((channels, n), dtype=np.float32), chunk=333)
    assert len(frames) == 1 + (n - wm.frame_samples) // wm.hop_samples
    assert all(f.shape == (channels, wm.frame_samples) for f in frames)


def test_frames_independent_of_chunk_size():
    fs, channels = 16000, 2
    signal = np.random.default_rng(1).normal(size=(channels, 5000)).astype(np.float32)
    a = collect(RealTimeWindowManager(channels, fs), signal, chunk=1)
    b = collect(RealTimeWindowManager(channels, fs), signal, chunk=4096)
    assert len(a) == len(b)
    for fa, fb in zip(a, b):
        np.testing.assert_allclose(fa, fb)


def test_hop_alignment_and_window():
    fs, channels = 1000, 1
    wm = RealTimeWindowManager(channels, fs, 40.0, 50.0)   # фрейм 40, шаг 20
    signal = np.arange(200, dtype=np.float32)[None, :]
    frames = collect(wm, signal, chunk=7)
    np.testing.assert_allclose(frames[1], signal[:, 20:60] * wm.window)

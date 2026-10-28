# run_simulation.py
"""
Генератор тестовых данных: многоканальный .wav + CSV траектории (Ground Truth).

Режимы:
    synthetic       — только numpy: неподвижный источник в дальней зоне, сирена-«вой» + шум.
                      Нужен для smoke-тестов конвейера и отладки DOA без тяжёлых зависимостей.
    pyroadacoustics — [Сафаров] реалистичный проезд машины (pyroadacoustics / pyroomacoustics).

Примеры:
    python scripts/run_simulation.py --mode synthetic --azimuth 60 --duration 3
    python scripts/run_simulation.py --mode pyroadacoustics
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import DATA_DIR, load_array_geometry          # noqa: E402
from core.geometry import ArrayGeometry, far_field_delays     # noqa: E402

GT_COLUMNS = ["timestamp_s", "azimuth_deg", "distance_m", "x_m", "y_m"]


def siren_signal(fs: int, duration_s: float, f_low: float = 650.0, f_high: float = 1500.0,
                 period_s: float = 1.5) -> np.ndarray:
    """Сирена типа «wail»: синус с плавной частотной модуляцией."""
    t = np.arange(int(fs * duration_s)) / fs
    inst_freq = f_low + (f_high - f_low) * 0.5 * (1 - np.cos(2 * np.pi * t / period_s))
    phase = 2 * np.pi * np.cumsum(inst_freq) / fs
    return np.sin(phase)


def apply_far_field(signal: np.ndarray, fs: int, geometry: ArrayGeometry, azimuth_deg: float) -> np.ndarray:
    """Дробные задержки через FFT: x_m(t) = s(t - tau_m). Возвращает (channels, samples)."""
    n = signal.shape[0]
    n_fft = int(2 ** np.ceil(np.log2(2 * n)))
    spectrum = np.fft.rfft(signal, n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
    tau = far_field_delays(geometry, azimuth_deg)[0]                     # (channels,)
    shifted = spectrum[None, :] * np.exp(-2j * np.pi * freqs[None, :] * tau[:, None])
    return np.fft.irfft(shifted, n_fft)[:, :n]


def simulate_synthetic(geometry: ArrayGeometry, fs: int, duration_s: float, azimuth_deg: float,
                       distance_m: float, snr_db: float, seed: int):
    rng = np.random.default_rng(seed)
    source = siren_signal(fs, duration_s)
    X = apply_far_field(source, fs, geometry, azimuth_deg)
    noise_std = np.sqrt(np.mean(X ** 2) / 10 ** (snr_db / 10))
    X = X + rng.normal(0.0, noise_std, X.shape)
    X = 0.9 * X / np.max(np.abs(X))

    t = np.arange(0.0, duration_s, 0.01)
    az = np.deg2rad(azimuth_deg)
    gt = [(ti, azimuth_deg, distance_m, distance_m * np.cos(az), distance_m * np.sin(az)) for ti in t]
    return X.astype(np.float32), gt


def simulate_pyroadacoustics(geometry: ArrayGeometry, fs: int, duration_s: float, **_):
    # TODO(Сафаров): сцена проезда (pyroadacoustics), траектория источника, отражения от дороги/зданий
    #                (pyroomacoustics). Вернуть (X: (channels, samples) float32, gt: список строк GT_COLUMNS).
    raise NotImplementedError("Симуляция проезда через pyroadacoustics ещё не реализована")


def save(X: np.ndarray, gt, fs: int, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path, csv_path = out_dir / f"{name}.wav", out_dir / f"{name}_gt.csv"
    wavfile.write(wav_path, fs, X.T)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(GT_COLUMNS)
        w.writerows([[f"{v:.4f}" for v in row] for row in gt])
    print(f"Записано: {wav_path} ({X.shape[0]} кан., {X.shape[1] / fs:.2f} с) и {csv_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["synthetic", "pyroadacoustics"], default="synthetic")
    p.add_argument("--array-config", default=None)
    p.add_argument("--fs", type=int, default=48000)
    p.add_argument("--duration", type=float, default=3.0)
    p.add_argument("--azimuth", type=float, default=60.0, help="synthetic: азимут источника, °")
    p.add_argument("--distance", type=float, default=20.0, help="synthetic: расстояние (только для GT), м")
    p.add_argument("--snr", type=float, default=20.0, help="synthetic: SNR, дБ")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=str(DATA_DIR / "raw"))
    p.add_argument("--name", default=None)
    args = p.parse_args()

    geometry = load_array_geometry(args.array_config)
    if args.mode == "synthetic":
        X, gt = simulate_synthetic(geometry, args.fs, args.duration, args.azimuth,
                                   args.distance, args.snr, args.seed)
        name = args.name or f"synthetic_az{args.azimuth:g}"
    else:
        X, gt = simulate_pyroadacoustics(geometry, args.fs, args.duration)
        name = args.name or "road_pass"
    save(X, gt, args.fs, Path(args.out_dir), name)


if __name__ == "__main__":
    main()

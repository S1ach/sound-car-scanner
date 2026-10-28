# precompute_lut.py
"""
Прерасчёт таблицы задержек (LUT) дальней зоны для SRP-PHAT / MUSIC.

    python scripts/precompute_lut.py --resolution 1.0
    -> data/lut/<array_name>_far_field_1deg.npz

Содержимое .npz:
    azimuths_deg : (n_angles,)           сетка азимутов
    delays_s     : (n_angles, channels)  задержки tau_m относительно центра массива, секунды
    mic_positions: (channels, 3)
    speed_of_sound: скаляр

Задержки не зависят от fs; сдвиги в сэмплах / фазовые множители e^{-j 2π f tau} считаются в prepare() блока.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import DATA_DIR, load_array_geometry   # noqa: E402
from core.geometry import far_field_delays             # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Прерасчёт LUT задержек дальней зоны")
    p.add_argument("--array-config", default=None)
    p.add_argument("--resolution", type=float, default=1.0, help="шаг сетки азимутов, °")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    geometry = load_array_geometry(args.array_config)
    azimuths = np.arange(0.0, 360.0, args.resolution)
    delays = far_field_delays(geometry, azimuths)

    out = Path(args.out) if args.out else DATA_DIR / "lut" / f"{geometry.name}_far_field_{args.resolution:g}deg.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, azimuths_deg=azimuths, delays_s=delays,
             mic_positions=geometry.mic_positions, speed_of_sound=geometry.speed_of_sound)
    print(f"LUT {delays.shape} -> {out}")


if __name__ == "__main__":
    main()

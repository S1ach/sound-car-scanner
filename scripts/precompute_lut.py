# precompute_lut.py
"""
Прерасчёт таблицы задержек (LUT) дальней зоны для SRP-PHAT / MUSIC.

    python scripts/precompute_lut.py --resolution 1.0
    -> data/lut/<array_name>_far_field_1deg.npy           delays_s     (n_angles, channels), секунды
    -> data/lut/<array_name>_far_field_1deg_azimuths.npy  azimuths_deg (n_angles,)

delays_s[i, m] — задержка tau_m микрофона m относительно центра массива для азимута azimuths_deg[i].
Геометрия и скорость звука, по которым построена таблица, — в config/array_config.json.

    delays = np.load("data/lut/uca8_r50mm_far_field_1deg.npy")

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

    out = Path(args.out) if args.out else DATA_DIR / "lut" / f"{geometry.name}_far_field_{args.resolution:g}deg.npy"
    out = out.with_suffix(".npy")
    azimuths_out = out.with_name(f"{out.stem}_azimuths.npy")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, delays)
    np.save(azimuths_out, azimuths)
    print(f"LUT {delays.shape} -> {out}\nСетка азимутов -> {azimuths_out}")


if __name__ == "__main__":
    main()

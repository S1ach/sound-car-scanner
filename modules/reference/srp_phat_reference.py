"""
Эталонный DOA-локализатор: SRP-PHAT в частотной области по сетке азимутов.

PHAT нормирует фазу в каждом бине, а бины полосы усредняются с весами |X(f)|^freq_weight_power.
Без весов (power=0) узкополосная сирена тонет в шумовых бинах; с power=2 доминируют тональные бины.

Назначение:
  * доказать, что каркас доносит до алгоритма корректные фреймы (channels, samples) и что
    результат (angle, confidence) проходит по конвейеру до лога (на синтетике виден реальный азимут);
  * служить базой (baseline) для сравнения точности алгоритмов команды.

Это не замена чьему-то модулю: при необходимости отключается в pipeline_config.json ("enabled": false).

Параметры (params в pipeline_config.json):
    resolution_deg  шаг сетки азимутов, ° (по умолчанию 1.0)
    f_min_hz        нижняя граница полосы, Гц (по умолчанию 300)
    f_max_hz        верхняя граница полосы, Гц (по умолчанию 4000)
    freq_weight_power  степень энергетического веса бинов (по умолчанию 2.0; 0 — классический SRP-PHAT)
    min_rms         порог тишины: ниже — (0.0, 0.0) (по умолчанию 1e-6)
    frame_duration_ms  длина фрейма для прерасчёта в prepare() (по умолчанию 40; другие длины считаются лениво)
"""
from typing import Tuple

import numpy as np

from core.geometry import far_field_delays
from core.interfaces import BaseDOALocalizer
from core.registry import register


@register("localizer", "srp_phat_reference")
class SrpPhatReferenceLocalizer(BaseDOALocalizer):

    def prepare(self) -> None:
        self.resolution = float(self.params.get("resolution_deg", 1.0))
        self.f_min = float(self.params.get("f_min_hz", 300.0))
        self.f_max = float(self.params.get("f_max_hz", 4000.0))
        self.min_rms = float(self.params.get("min_rms", 1e-6))
        self.weight_power = float(self.params.get("freq_weight_power", 2.0))
        self.azimuths = np.arange(0.0, 360.0, self.resolution)
        self.tau = far_field_delays(self.geometry, self.azimuths)       # (n_angles, channels)
        self._cache = {}                                                 # (fs, samples) -> (band, steering)
        frame_ms = float(self.params.get("frame_duration_ms", 40.0))
        self._steering(self.fs, int(frame_ms / 1000.0 * self.fs))        # прогрев: не тратить бюджет 1-го фрейма

    def _steering(self, fs: int, samples: int):
        key = (fs, samples)
        if key not in self._cache:
            freqs = np.fft.rfftfreq(samples, 1.0 / fs)
            band = np.flatnonzero((freqs >= self.f_min) & (freqs <= self.f_max))
            # x_m(t) = s(t - tau_m)  =>  X_m(f) = S(f)·e^{-j2πfτ_m}; выравнивание умножением на e^{+j2πfτ_m}
            steering = np.exp(2j * np.pi * freqs[band][None, None, :] * self.tau[:, :, None])
            self._cache[key] = (band, steering.astype(np.complex64))     # (n_angles, channels, n_band)
        return self._cache[key]

    def execute_doa_localization(self, X_frame: np.ndarray, fs: int) -> Tuple[float, float]:
        if float(np.sqrt(np.mean(X_frame.astype(np.float64) ** 2))) < self.min_rms:
            return 0.0, 0.0

        channels, samples = X_frame.shape
        band, steering = self._steering(fs, samples)
        if band.size == 0:
            return 0.0, 0.0

        spectrum = np.fft.rfft(X_frame, axis=1)[:, band]                 # (channels, n_band)
        magnitude = np.abs(spectrum)
        phat = spectrum / np.maximum(magnitude, 1e-12)                   # PHAT: только фаза
        weights = np.mean(magnitude, axis=0) ** self.weight_power        # (n_band,)
        total = weights.sum()
        if not np.isfinite(total) or total <= 0.0:
            return 0.0, 0.0
        weights = weights / total

        beam = np.einsum("acf,cf->af", steering, phat.astype(np.complex64))
        power = (np.abs(beam) ** 2 / channels ** 2) @ weights            # [0, 1]; 1 — идеальная плоская волна

        best = int(np.argmax(power))
        # Некогерентный шум даёт ≈ 1/channels, идеальная плоская волна — 1.0
        floor = 1.0 / channels
        confidence = float(np.clip((power[best] - floor) / (1.0 - floor), 0.0, 1.0))
        return float(self.azimuths[best]), confidence

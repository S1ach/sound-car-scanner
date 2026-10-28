# window_manager.py
import numpy as np
from typing import Callable


class RealTimeWindowManager:
    def __init__(self, channels: int, fs: int, frame_duration_ms: float = 40.0, overlap_pct: float = 50.0):
        if not (0.0 <= overlap_pct < 100.0):
            raise ValueError(f"overlap_pct должен быть в [0, 100), получено {overlap_pct}")

        self.channels = channels
        self.fs = fs

        # Расчет длины фрейма и шага в сэмплах
        self.frame_samples = int((frame_duration_ms / 1000.0) * fs)
        self.hop_samples = int(self.frame_samples * (1.0 - overlap_pct / 100.0))

        # Внутренний FIFO-буфер для входящих аудиоданных
        self.buffer = np.zeros((channels, self.frame_samples), dtype=np.float32)
        self.samples_in_buffer = 0

        # Вычисление окна Ханна
        self.window = np.hanning(self.frame_samples).astype(np.float32)

    @property
    def hop_duration_s(self) -> float:
        return self.hop_samples / self.fs

    def reset(self) -> None:
        self.buffer.fill(0.0)
        self.samples_in_buffer = 0

    def process_stream_chunk(self, chunk: np.ndarray, pipeline_callback: Callable[[np.ndarray], None]) -> None:
        """
        Принимает поступающий от звуковой карты произвольный чанк данных и нарезает его на фреймы.
        """
        if chunk.ndim != 2 or chunk.shape[0] != self.channels:
            raise ValueError(f"Ожидался чанк формы ({self.channels}, n), получено {chunk.shape}")

        chunk_samples = chunk.shape[1]
        ptr = 0

        while ptr < chunk_samples:
            space_left = self.frame_samples - self.samples_in_buffer
            samples_to_copy = min(space_left, chunk_samples - ptr)

            # Векторизованный сдвиг истории влево и запись новой порции данных в хвост буфера
            self.buffer[:, :self.frame_samples - samples_to_copy] = self.buffer[:, samples_to_copy:]
            self.buffer[:, self.frame_samples - samples_to_copy:] = chunk[:, ptr:ptr + samples_to_copy]

            self.samples_in_buffer += samples_to_copy
            ptr += samples_to_copy

            # При полном наполнении буфера — наложение окна и отправка в конвейер
            if self.samples_in_buffer == self.frame_samples:
                windowed_frame = self.buffer * self.window
                pipeline_callback(windowed_frame)

                # Обновление счетчика с учетом шага перекрытия (hop)
                self.samples_in_buffer = self.frame_samples - self.hop_samples

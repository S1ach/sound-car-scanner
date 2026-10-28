# yolo_tracker.py
"""[НА БУДУЩЕЕ] Детекция и трекинг машин (YOLO + BoT-SORT). Требует ultralytics (см. requirements.txt)."""
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Track:
    track_id: int
    bbox_xyxy: Tuple[float, float, float, float]   # пиксели: x1, y1, x2, y2
    class_name: str
    confidence: float                              # [0.0, 1.0]
    timestamp_s: float                             # время кадра в той же шкале, что и аудиопоток

    @property
    def center_px(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


class YoloTracker:
    def __init__(self, model_path: str = "yolov8n.pt", tracker_config: str = "botsort.yaml",
                 classes: Tuple[str, ...] = ("car", "truck", "bus")):
        self.model_path = model_path
        self.tracker_config = tracker_config
        self.classes = classes

    def update(self, image: np.ndarray, timestamp_s: float) -> List[Track]:
        """
        Args:
            image: кадр (H, W, 3), uint8, BGR.
        Returns:
            Активные треки на кадре.
        """
        # TODO: ultralytics.YOLO(model_path).track(image, tracker=tracker_config, persist=True)
        raise NotImplementedError("Трекер YOLO + BoT-SORT ещё не реализован")

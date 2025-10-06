from __future__ import annotations
import numpy as np
from typing import List
from ..core.types import Track


class ConfidencePolicy:
    """Оценивает «здоровье» трекера:
    - средняя длительность треков
    - среднее время без обновления
    - доля треков с малым остатком (Махаланобис)
    """
    def __init__(self, residual_thr=9.0, min_tracks=2, max_pred_frames=8):
        self.residual_thr = residual_thr
        self.min_tracks = min_tracks
        self.max_pred_frames = max_pred_frames
        self.pred_frames_left = 0


    def high_confidence(self, tracks: List[Track]) -> bool:
        active = [t for t in tracks if t.state != "deleted"]
        return len(active) >= self.min_tracks and all(t.time_since_update <= 1 for t in active)


    def step(self, tracks: List[Track]) -> bool:
        """Возвращает: True – использовать только прогноз, False – запустить детектор"""
        if self.pred_frames_left > 0 and self.high_confidence(tracks):
            self.pred_frames_left -= 1
            return True
        # иначе – требуется детектор; если после него всё ок – раздаём кредит на N кадров
        self.pred_frames_left = self.max_pred_frames if self.high_confidence(tracks) else 0
        return False
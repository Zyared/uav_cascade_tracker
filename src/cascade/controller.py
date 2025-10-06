from __future__ import annotations
from typing import List
from ..core.types import Detection, Track
from ..tracking.sort import SORT
from .policy import ConfidencePolicy


class CascadeController:
    def __init__(self, tracker: SORT, policy: ConfidencePolicy):
        self.tracker = tracker
        self.policy = policy


    def step_predict_only(self) -> List[Track]:
        # только прогноз KF
        for t in self.tracker.tracks:
            t.mean, t.cov = self.tracker.kf.predict(t.mean, t.cov)
            t.bbox = self.tracker._to_bbox(t.mean)
            t.age += 1
            t.time_since_update += 1
        return self.tracker.tracks


    def step_with_detections(self, dets: List[Detection]) -> List[Track]:
     return self.tracker.update(dets)


    def forward(self, dets: List[Detection]) -> List[Track]:
        # Решение: если политика говорит «можно предсказывать» и детектор не обязателен – делаем predict-only
        if self.policy.step(self.tracker.tracks):
            return self.step_predict_only()
        else:
            return self.step_with_detections(dets)
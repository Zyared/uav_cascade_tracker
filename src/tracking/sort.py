from __future__ import annotations
import numpy as np
from typing import List
from .kalman_filter import KalmanFilter
from .assignment import linear_assignment
from ..core.types import Detection, Track
from ..core.boxes import iou_matrix


class SORT:
    def __init__(self, iou_match_thr=0.3, max_age=30, min_hits=3, use_mah_dist=True):
        self.iou_thr = iou_match_thr
        self.max_age = max_age
        self.min_hits = min_hits
        self.use_mah = use_mah_dist
        self.kf = KalmanFilter()
        self._next_id = 1
        self.tracks: List[Track] = []


    @staticmethod
    def _xyxy_to_xyah(b):
        x1, y1, x2, y2 = b
        w, h = x2 - x1, y2 - y1
        x, y = x1 + w/2, y1 + h/2
        a = w / max(1e-6, h)
        return np.array([x, y, a, h], dtype=float)


    @staticmethod
    def _xyah_to_xyxy(m):
        x, y, a, h = m
        w = a * h
        return np.array([x - w/2, y - h/2, x + w/2, y + h/2], dtype=float)


    def predict(self):
        for t in self.tracks:
            t.mean, t.cov = self.kf.predict(t.mean, t.cov)
            t.bbox = self._to_bbox(t.mean)
            t.age += 1
            t.time_since_update += 1


    def _to_bbox(self, mean):
        xyxy = self._xyah_to_xyxy(mean[:4])
        from ..core.types import BBox
        return BBox(*xyxy.tolist())


    def update(self, detections: List[Detection]) -> List[Track]:
        # 1) предикт шаг
        for t in self.tracks:
            t.mean, t.cov = self.kf.predict(t.mean, t.cov)
            # 2) построение стоимости сопоставления
            det_xyxy = np.array([[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in detections], dtype=float)
            trk_xyxy = np.array([[tr.bbox.x1, tr.bbox.y1, tr.bbox.x2, tr.bbox.y2] for tr in self.tracks], dtype=float) if self.tracks else np.zeros((0,4))
            cost = 1.0 - iou_matrix(trk_xyxy, det_xyxy) if len(det_xyxy) and len(trk_xyxy) else np.zeros((len(trk_xyxy), len(det_xyxy)))
            matches, um_trk, um_det = linear_assignment(cost, cost_limit=1.0 - self.iou_thr)
        # 3) коррекция по матчам
        for ti, di in matches:
            m = self._xyxy_to_xyah(det_xyxy[di])
            tr = self.tracks[ti]
            tr.mean, tr.cov = self.kf.update(tr.mean, tr.cov, m)
            tr.bbox = self._to_bbox(tr.mean)
            tr.hits += 1
            tr.time_since_update = 0
            tr.state = "confirmed" if tr.hits >= self.min_hits else tr.state
        # 4) новые треки по unmatched detections
        for di in um_det:
            m = self._xyxy_to_xyah(det_xyxy[di])
            mean, cov = self.kf.initiate(m)
            tr = Track(track_id=self._next_id, bbox=self._to_bbox(mean), mean=mean, cov=cov, state="tentative")
            self._next_id += 1
            self.tracks.append(tr)
        # 5) маркация старых треков
        alive = []
        for tr in self.tracks:
            if tr.time_since_update > self.max_age:
                tr.state = "deleted"
            if tr.state != "deleted":
                alive.append(tr)
        self.tracks = alive
        return self.tracks
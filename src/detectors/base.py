from __future__ import annotations
from typing import List
from ..core.types import Detection


class Detector:
    def __init__(self, conf_thr: float = 0.4, iou_thr: float = 0.5):
        self.conf_thr = conf_thr
        self.iou_thr = iou_thr


    def load(self, weights_path: str):
        raise NotImplementedError


    def infer(self, frame) -> List[Detection]:
        raise NotImplementedError
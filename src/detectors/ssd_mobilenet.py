from __future__ import annotations
import cv2
import numpy as np
from typing import List
from .base import Detector
from ..core.types import Detection, BBox


class SSDMobileNet(Detector):
    def __init__(self, conf_thr=0.4, iou_thr=0.5, input_size=(300,300)):
        super().__init__(conf_thr, iou_thr)
        self.model = None
        self.input_size = input_size


    def load(self, weights_path: str):
        # Предпочтительно ONNX: cv2.dnn.readNetFromONNX
        self.model = cv2.dnn.readNetFromONNX(weights_path)
        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


    def infer(self, frame) -> List[Detection]:
        blob = cv2.dnn.blobFromImage(frame, scalefactor=1/127.5, size=self.input_size,
        mean=(127.5,127.5,127.5), swapRB=True, crop=False)
        self.model.setInput(blob)
        out = self.model.forward() # [1,1,N,7]: [img_id, class_id, conf, x1,y1,x2,y2]
        H, W = frame.shape[:2]
        dets: List[Detection] = []
        for det in out[0,0]:
            conf = float(det[2])
            if conf < self.conf_thr:
                continue
            x1, y1, x2, y2 = det[3]*W, det[4]*H, det[5]*W, det[6]*H
            dets.append(Detection(BBox(float(x1), float(y1), float(x2), float(y2)), conf, int(det[1])))
        return dets
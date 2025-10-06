from __future__ import annotations
import cv2
from omegaconf import OmegaConf
from ..detectors.ssd_mobilenet import SSDMobileNet
from ..tracking.sort import SORT
from ..cascade.controller import CascadeController
from ..cascade.policy import ConfidencePolicy
from ..utils.viz import draw_box
from ..core.types import Detection, BBox


class CascadePipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        # детектор
        self.det = SSDMobileNet(cfg.model.conf_thr, cfg.model.iou_thr)
        self.det.load(cfg.model.weights)
        # трекер
        self.trk = SORT(
        iou_match_thr=cfg.tracker.iou_match_thr,
        max_age=cfg.tracker.max_age,
        min_hits=cfg.tracker.min_hits,
        use_mah_dist=cfg.tracker.use_mah_dist,
        )
        # каскад
        self.ctrl = CascadeController(self.trk, ConfidencePolicy(
        residual_thr=cfg.cascade.residual_thr,
        min_tracks=cfg.cascade.high_conf_min_tracks,
        max_pred_frames=cfg.cascade.max_pred_frames,
        ))


    def step(self, frame):
        # базово: каждый кадр прогоняем детектором, а контроллер решает предсказывать или обновлять
        dets = self.det.infer(frame)
        tracks = self.ctrl.forward(dets)
        # визуализация
        for t in tracks:
            if t.state == "deleted":
                continue
            draw_box(frame, t.bbox, text=f"ID {t.track_id}")
        return frame
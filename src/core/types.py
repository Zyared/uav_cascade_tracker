from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class BBox:
    x1:float
    y1:float
    x2:float
    y2:float


def as_xywh(self) -> np.ndarray:
    return np.array([self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1], dtype=float)


def area(self) -> float:
    return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass
class Detection:
    bbox: BBox
    score: float
    cls: int = 0
    embed: Optional[np.ndarray] = None


@dataclass
class Track:
    track_id: int
    bbox: BBox
    cls: int = 0
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    state: str = "tentative" # tentative|confirmed|deleted
    mean: Optional[np.ndarray] = None # KF mean
    cov: Optional[np.ndarray] = None # KF covariance
    feature: Optional[np.ndarray] = None
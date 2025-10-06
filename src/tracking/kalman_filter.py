import numpy as np


class KalmanFilter:
    """CV-модель (x,y,a,h,vx,vy,va,vh) как в SORT/DeepSORT.
    x,y – центр; a – отношение сторон; h – высота бокса.
    """
def __init__(self):
    ndim, dt = 4, 1.0
    self._motion_mat = np.eye(2*ndim)
    for i in range(ndim):
        self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2*ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160


def initiate(self, measurement):
    mean_pos = measurement
    mean_vel = np.zeros_like(mean_pos)
    mean = np.r_[mean_pos, mean_vel]
    std = [2 * self._std_weight_position * measurement[3],
        2 * self._std_weight_position * measurement[3],
        1e-2, 2 * self._std_weight_position * measurement[3],
        10 * self._std_weight_velocity * measurement[3],
        10 * self._std_weight_velocity * measurement[3],
        1e-3, 10 * self._std_weight_velocity * measurement[3]]
    cov = np.diag(np.square(std))
    return mean, cov


def predict(self, mean, cov):

    std_pos = [self._std_weight_position * mean[3],
        self._std_weight_position * mean[3],
        1e-2, self._std_weight_position * mean[3]]

    std_vel = [self._std_weight_velocity * mean[3],
        self._std_weight_velocity * mean[3],
        1e-3, self._std_weight_velocity * mean[3]]

    motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
    mean = self._motion_mat @ mean
    cov = self._motion_mat @ cov @ self._motion_mat.T + motion_cov
    return mean, cov


def project(self, mean, cov):
    std = [self._std_weight_position * mean[3],
    self._std_weight_position * mean[3],
    1e-1, self._std_weight_position * mean[3]]
    innovation_cov = np.diag(np.square(std))
    mean = self._update_mat @ mean
    cov = self._update_mat @ cov @ self._update_mat.T + innovation_cov
    return mean, cov


def update(self, mean, cov, measurement):
    proj_mean, proj_cov = self.project(mean, cov)
    K = cov @ self._update_mat.T @ np.linalg.inv(proj_cov)
    innovation = measurement - proj_mean
    new_mean = mean + K @ innovation
    new_cov = cov - K @ self._update_mat @ cov
    return new_mean, new_cov


def gating_distance(self, mean, cov, measurements, only_position=False):
    mean, cov = self.project(mean, cov)
    d = measurements - mean
    S = cov
    invS = np.linalg.inv(S)
    m = np.einsum("...i,ij,...j->...", d, invS, d)
    return m
import numpy as np
from scipy.optimize import linear_sum_assignment


INF = 1e9


def linear_assignment(cost: np.ndarray, cost_limit: float = np.inf):
    if cost.size == 0:
        return np.empty((0,2), dtype=int), np.arange(cost.shape[0]), np.arange(cost.shape[1])
    row_ind, col_ind = linear_sum_assignment(cost)
    matches, unmatched_rows, unmatched_cols = [], [], []
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] > cost_limit:
            unmatched_rows.append(r)
            unmatched_cols.append(c)
        else:
            matches.append([r, c])
    for r in set(range(cost.shape[0])) - set(row_ind):
        unmatched_rows.append(r)
    for c in set(range(cost.shape[1])) - set(col_ind):
        unmatched_cols.append(c)
    return np.asarray(matches), np.asarray(unmatched_rows), np.asarray(unmatched_cols)
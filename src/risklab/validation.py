import numpy as np
import pandas as pd


def returns_matrix(data, minimum=20):
    matrix = np.asarray(data, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2 or len(matrix) < minimum or matrix.shape[1] < 1 or not np.isfinite(matrix).all():
        raise ValueError("A finite return matrix with sufficient observations is required.")
    if np.any(matrix <= -1):
        raise ValueError("Simple realized returns must exceed -100%.")
    return matrix


def allocation(weights, n):
    w = np.asarray(weights, dtype=float)
    if w.shape != (n,) or not np.isfinite(w).all() or np.any(w < 0) or not np.isclose(w.sum(), 1, atol=1e-8):
        raise ValueError("Weights must be finite, long-only, match assets, and sum to one.")
    return w


def confidence_horizon(confidence, horizon):
    if (
        not 0.5 < confidence < 1
        or isinstance(horizon, bool)
        or not isinstance(horizon, int)
        or not 1 <= horizon <= 21
    ):
        raise ValueError("Confidence must be in (0.5,1); horizon must be an integer from 1 to 21.")


def dated_frame(frame):
    if (
        not isinstance(frame.index, pd.DatetimeIndex)
        or frame.index.has_duplicates
        or not frame.index.is_monotonic_increasing
    ):
        raise ValueError("Dates must be unique and increasing.")
    if frame.columns.has_duplicates:
        raise ValueError("Asset names must be unique.")
    returns_matrix(frame)
    return frame

from typing import List

import numpy as np
from scipy.optimize import least_squares

from .models import Observation


class RankDeficientError(Exception):
    """Raised when the design matrix is not full column rank."""


def _design_matrix(t: np.ndarray, periods: List[float]) -> np.ndarray:
    """Build the harmonic design matrix.

    Columns: [1, cos(w1 t), sin(w1 t), ..., cos(wk t), sin(wk t)]
    where wi = 2*pi / periods[i].
    """
    n = t.shape[0]
    k = len(periods)
    X = np.ones((n, 1 + 2 * k), dtype=float)
    for i, P in enumerate(periods):
        omega = 2.0 * np.pi / float(P)
        X[:, 2 * i + 1] = np.cos(omega * t)
        X[:, 2 * i + 2] = np.sin(omega * t)
    return X


def fit_tide(
    observations: List[Observation],
    periods: List[float],
    delta: float,
    predictions: List[int],
):
    """Fit c + sum[a cos(2pi t/P) + b sin(2pi t/P)] with Huber loss.

    The first observation's time is used as the time origin. All observations
    are used (no outlier removal). Returns baseline, per-period coefficients,
    pointwise residuals and predicted tide levels.
    """
    t = np.array([o.t for o in observations], dtype=float)
    h = np.array([o.h for o in observations], dtype=float)

    # Time origin = first observation.
    t0 = t[0]
    t_rel = t - t0

    X = _design_matrix(t_rel, periods)

    # Full column rank check.
    rank = int(np.linalg.matrix_rank(X))
    if rank < X.shape[1]:
        raise RankDeficientError(
            f"design matrix rank {rank} < {X.shape[1]} (full column rank required)"
        )

    # Initial guess via ordinary least squares.
    beta0, *_ = np.linalg.lstsq(X, h, rcond=None)

    # Minimize Huber loss over all observations. scipy's 'huber' loss uses
    # rho(z) = z^2 if |z| <= 1 else 2|z| - 1 with z = residual / f_scale,
    # so f_scale = delta is the Huber threshold.
    result = least_squares(
        lambda b: h - X @ b,
        beta0,
        loss="huber",
        f_scale=float(delta),
        max_nfev=10000,
    )
    beta = result.x

    baseline = float(beta[0])
    coefficients = []
    for i, P in enumerate(periods):
        a = float(beta[2 * i + 1])
        b = float(beta[2 * i + 2])
        coefficients.append({"period": float(P), "a": a, "b": b})

    residuals = [float(v) for v in (h - X @ beta)]

    pred_results = []
    if predictions:
        t_pred = np.array(predictions, dtype=float) - t0
        Xp = _design_matrix(t_pred, periods)
        hp = Xp @ beta
        pred_results = [
            {"t": int(pt), "h": float(v)} for pt, v in zip(predictions, hp)
        ]

    return {
        "baseline": baseline,
        "coefficients": coefficients,
        "residuals": residuals,
        "predictions": pred_results,
    }

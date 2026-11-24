"""Forecasting beta. Every forecast for day t uses returns up to day t − 1 only.

- rolling_beta: OLS slope over the previous w days.
- ewma_beta: exponentially weighted covariance ÷ variance (no intercept, daily means ≈ 0).
- vasicek: shrink each stock's beta toward the cross-sectional average, by how noisy it is:
  w = σ²_prior / (σ²_prior + se²), β = w·β̂ + (1 − w)·mean, with σ²_prior the spread of the
  estimates minus their average se².
- kalman_beta: β follows a random walk (step variance q); y_t = β_t·x_t + ε_t (variance r).
  Only the ratio q/r matters for the path; fit_kalman_ratio picks it by maximum likelihood on a
  training period.
"""

import numpy as np


def rolling_beta(y, x, w: int) -> np.ndarray:
    y, x = np.asarray(y, float), np.asarray(x, float)
    out = np.full(len(y), np.nan)
    cs = lambda v: np.r_[0.0, np.cumsum(v)]  # noqa: E731
    sx, sy, sxx, sxy = cs(x), cs(y), cs(x * x), cs(x * y)
    t = np.arange(w, len(y))
    mx = (sx[t] - sx[t - w]) / w
    my = (sy[t] - sy[t - w]) / w
    cov = (sxy[t] - sxy[t - w]) / w - mx * my
    var = (sxx[t] - sxx[t - w]) / w - mx * mx
    out[t] = cov / var
    return out


def ewma_beta(y, x, lam: float = 0.99, seed: int = 60) -> np.ndarray:
    y, x = np.asarray(y, float), np.asarray(x, float)
    out = np.full(len(y), np.nan)
    cxy, vx = np.mean(x[:seed] * y[:seed]), np.mean(x[:seed] ** 2)
    for t in range(seed, len(y)):
        if t > seed:
            cxy = lam * cxy + (1 - lam) * x[t - 1] * y[t - 1]
            vx = lam * vx + (1 - lam) * x[t - 1] ** 2
        out[t] = cxy / vx
    return out


def vasicek(betas, ses) -> np.ndarray:
    b, se = np.asarray(betas, float), np.asarray(ses, float)
    # spread of true betas = spread of the estimates minus their average sampling noise
    m, v = b.mean(), max(b.var(ddof=1) - np.mean(se**2), 1e-12)
    w = v / (v + se**2)
    return w * b + (1 - w) * m


def _kalman(y, x, ratio, b0=1.0, p0=1.0, scale=None, seed: int = 250):
    """Forecasts (before seeing day t) and the log-likelihood. `scale` = (residual variance,
    var x) from a training period; by default from an OLS fit on the first `seed` days. Never
    the future. q is tied to r, so the path depends on the ratio only."""
    n = len(y)
    out = np.empty(n)
    b, p = b0, p0
    ll = 0.0
    r, vx = scale if scale is not None else regression_scale(y[:seed], x[:seed])
    q = ratio * r / vx
    for t in range(n):
        p = p + q
        out[t] = b
        f = x[t] ** 2 * p + r
        e = y[t] - b * x[t]
        ll += -0.5 * (np.log(f) + e * e / f)
        k = p * x[t] / f
        b = b + k * e
        p = (1 - k * x[t]) * p
    return out, ll


def regression_scale(y, x) -> tuple[float, float]:
    """(residual variance, variance of x) of an OLS fit: the Kalman filter's noise scales."""
    slope, icpt = np.polyfit(x, y, 1)
    return float(np.var(y - icpt - slope * x)), float(np.var(x))


def kalman_beta(y, x, ratio: float, scale=None, seed: int = 250) -> np.ndarray:
    """Without a training `scale`, the scale comes from the first `seed` days, so forecasts for
    those days would see the future: they are left as NaN."""
    out = _kalman(np.asarray(y, float), np.asarray(x, float), ratio, scale=scale, seed=seed)[0]
    if scale is None:
        out[:seed] = np.nan
    return out


def fit_kalman_ratio(y, x, grid=None) -> float:
    grid = np.logspace(-7, -1, 25) if grid is None else grid
    lls = [_kalman(np.asarray(y, float), np.asarray(x, float), g)[1] for g in grid]
    return float(grid[int(np.argmax(lls))])

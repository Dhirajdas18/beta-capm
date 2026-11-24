"""How beta is measured: return frequency, late-reacting prices, and extreme days.

- resampled_returns: weekly or monthly compounded returns from daily ones (unfinished last period
  left out).
- dimson_beta (Dimson 1979): regress on the market's previous, same and next day and add the three
  slopes, so a stock whose price catches up a day late still gets its full beta.
- huber_beta: robust regression (Huber loss, iteratively reweighted least squares): days with
  large residuals get less weight, so a handful of wild days can't swing the slope.
"""

import numpy as np
import pandas as pd


def resampled_returns(R: pd.DataFrame, freq: str) -> pd.DataFrame:
    px = (1 + R.fillna(0)).cumprod().where(R.notna())
    out = px.resample(freq).last().pct_change(fill_method=None).iloc[1:]
    if len(out) and out.index[-1] > R.index[-1]:
        out = out.iloc[:-1]
    return out


def dimson_beta(y, x, lags: int = 1) -> dict:
    """Sum of slopes on x[t−lags] … x[t+lags]; `parts` lists them from the earliest."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    n = len(x)
    cols = [
        np.r_[np.full(k, np.nan), x[: n - k]]
        if k > 0
        else np.r_[x[-k:], np.full(-k, np.nan)]
        if k < 0
        else x
        for k in range(lags, -lags - 1, -1)
    ]
    A = np.column_stack(cols)
    ok = np.isfinite(A).all(axis=1) & np.isfinite(y)
    coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(ok.sum()), A[ok]]), y[ok], rcond=None)
    return {"beta": float(coef[1:].sum()), "parts": coef[1:].tolist()}


def huber_beta(y, x, c: float = 1.345, iters: int = 50) -> float:
    y, x = np.asarray(y, float), np.asarray(x, float)
    X = np.column_stack([np.ones(len(x)), x])
    w = np.ones(len(x))
    coef = np.zeros(2)
    for _ in range(iters):
        sw = np.sqrt(w)
        coef_new, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
        u = y - X @ coef_new
        scale = np.median(np.abs(u - np.median(u))) / 0.6745  # robust spread (MAD)
        if scale == 0:
            return float(coef_new[1])
        a = np.abs(u / scale)
        w = np.where(a <= c, 1.0, c / np.maximum(a, c))  # no division by a zero residual
        if np.allclose(coef_new, coef, rtol=0, atol=1e-12):
            break
        coef = coef_new
    return float(coef_new[1])

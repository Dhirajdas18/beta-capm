"""Asymmetric betas: does a stock fall harder with the market than it rises with it?

- updown_beta: one regression y = a + β_up·max(x, 0) + β_down·min(x, 0), so the two slopes share
  an intercept, with a Newey-West t-statistic for β_down − β_up.
- tail_beta: the slope on the market's worst q share of days only (crash beta).
"""

import numpy as np

from bc.capm import _nw_cov


def updown_beta(y, x, lags: int = 5) -> dict:
    y, x = np.asarray(y, float), np.asarray(x, float)
    X = np.column_stack([np.ones(len(x)), np.maximum(x, 0), np.minimum(x, 0)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    cov = _nw_cov(X, y - X @ coef, lags)
    diff = coef[2] - coef[1]
    se = np.sqrt(cov[1, 1] + cov[2, 2] - 2 * cov[1, 2])
    return {"up": float(coef[1]), "down": float(coef[2]), "diff": float(diff),
            "t_diff": float(diff / se), "n": len(y)}


def tail_beta(y, x, q: float = 0.05) -> dict:
    y, x = np.asarray(y, float), np.asarray(x, float)
    cut = np.quantile(x, q)
    m = x <= cut
    slope = np.polyfit(x[m], y[m], 1)[0]
    return {"beta": float(slope), "n": int(m.sum()), "cut": float(cut)}

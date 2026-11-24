"""CAPM regression: r_i − r_f = α + β(r_m − r_f) + ε.

β measures how much a stock moves with the market; α is the average return the market doesn't
explain. Standard errors come in two versions: plain OLS (assumes each day's error is independent)
and Newey-West (allows errors to be correlated over nearby days and to change in size). Annual
figures use the calendar: trading days per year = (days − 1) / years elapsed.
"""

import numpy as np
import pandas as pd


def excess(r: pd.Series, rf_annual: float) -> pd.Series:
    """Return minus the risk-free rate earned over the calendar days since the previous row."""
    days = r.index.to_series().diff().dt.days.fillna(1).to_numpy()
    return r - ((1 + rf_annual) ** (days / 365) - 1)


def _nw_cov(X: np.ndarray, u: np.ndarray, lags: int) -> np.ndarray:
    """Newey-West (Bartlett) covariance of OLS coefficients."""
    n = len(u)
    xu = X * u[:, None]
    s = xu.T @ xu
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        g = xu[lag:].T @ xu[:-lag]
        s += w * (g + g.T)
    inv = np.linalg.inv(X.T @ X)
    return inv @ s @ inv * n / (n - X.shape[1])


def ols(y, x, lags: int = 5) -> dict:
    y, x = np.asarray(y, float), np.asarray(x, float)
    m = np.isfinite(y) & np.isfinite(x)
    y, x = y[m], x[m]
    X = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ coef
    n = len(y)
    s2 = u @ u / (n - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    nw = _nw_cov(X, u, lags)
    return {"alpha": float(coef[0]), "beta": float(coef[1]),
            "se_alpha": float(np.sqrt(cov[0, 0])), "se_beta": float(np.sqrt(cov[1, 1])),
            "se_alpha_nw": float(np.sqrt(nw[0, 0])), "se_beta_nw": float(np.sqrt(nw[1, 1])),
            "r2": float(1 - u @ u / ((y - y.mean()) @ (y - y.mean()))),
            "resid_sd": float(np.sqrt(s2)), "n": n}


def bh_reject(p, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg: which hypotheses to reject keeping the false discovery rate at q."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    passed = np.flatnonzero(ranked <= q)
    out = np.zeros(len(p), dtype=bool)
    if len(passed):
        out[order[: passed[-1] + 1]] = True
    return out


def capm_table(R: pd.DataFrame, symbols: list[str], market: str, rf: float,
               lags: int = 5) -> pd.DataFrame:
    """CAPM regression of each symbol's excess return on the market's, over the rows where both
    have returns; α annualised by calendar time."""
    from scipy.stats import norm

    rows = {}
    for s in symbols:
        d = R[[s, market]].dropna()
        if len(d) < 60:
            continue
        y, x = excess(d[s], rf).iloc[1:], excess(d[market], rf).iloc[1:]
        res = ols(y.to_numpy(), x.to_numpy(), lags)
        years = (d.index[-1] - d.index[0]).days / 365.25
        per_year = (len(d) - 1) / years
        res["alpha_annual"] = res["alpha"] * per_year
        res["t_alpha_nw"] = res["alpha"] / res["se_alpha_nw"]
        res["p_alpha_nw"] = float(2 * norm.sf(abs(res["t_alpha_nw"])))
        res["years"] = years
        rows[s] = res
    return pd.DataFrame(rows).T.astype(float)

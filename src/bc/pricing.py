"""Does beta pay? Cross-sectional tests of CAPM's prediction that higher beta earns more.

- fama_macbeth (Fama and MacBeth 1973): each month, regress the stocks' next-month excess returns on
  their betas known at the start of the month; the average slope is the reward per unit of beta,
  with a Newey-West t from the month-to-month spread of the slopes.
- bab_weights, bab_return (Frazzini and Pedersen 2014, betting against beta): rank stocks by beta;
  the low leg weights below-median ranks, the high leg above-median ranks, each summing to 1. Each
  leg is scaled to beta 1: long (low leg ÷ its beta), short (high leg ÷ its beta).
"""

import numpy as np
import pandas as pd


def _nw_t_mean(d: np.ndarray, lags: int) -> float:
    d = d[np.isfinite(d)]
    n, m = len(d), d.mean()
    u = d - m
    lr = u @ u / n + 2 * sum(
        (1 - j / (lags + 1)) * (u[j:] @ u[:-j]) / n for j in range(1, lags + 1)
    )
    return float(m / np.sqrt(lr / (n - 1))) if lr > 0 else float("nan")  # no spread: t undefined


def fama_macbeth(B: pd.DataFrame, Rnext: pd.DataFrame, lags: int = 3, min_stocks: int = 10) -> dict:
    """B: betas known at each month's start (rows months, columns stocks); Rnext: that month's
    excess returns, same shape."""
    icpt, slope = {}, {}
    for t in B.index:
        b, r = B.loc[t], Rnext.loc[t]
        ok = b.notna() & r.notna()
        if ok.sum() < min_stocks:
            continue
        s, a = np.polyfit(b[ok].to_numpy(float), r[ok].to_numpy(float), 1)
        icpt[t], slope[t] = a, s
    slopes, icpts = pd.Series(slope), pd.Series(icpt)
    return {
        "slope": float(slopes.mean()),
        "t_slope": _nw_t_mean(slopes.to_numpy(), lags),
        "intercept": float(icpts.mean()),
        "t_intercept": _nw_t_mean(icpts.to_numpy(), lags),
        "slopes": slopes,
        "intercepts": icpts,
    }


def bab_weights(beta: pd.Series) -> tuple[pd.Series, pd.Series]:
    z = beta.rank()
    d = z - z.mean()
    wl, wh = (-d).clip(lower=0), d.clip(lower=0)
    return wl / wl.sum(), wh / wh.sum()


def bab_return(wl: pd.Series, wh: pd.Series, beta: pd.Series, r: pd.Series) -> float:
    """One period's betting-against-beta excess return from the legs' excess returns."""
    return float((wl @ r) / (wl @ beta) - (wh @ r) / (wh @ beta))


def beta_sorts(B: pd.DataFrame, Rnext: pd.DataFrame) -> dict:
    """Each month: equal-weight quintiles by beta (0 = lowest), the betting-against-beta return, and
    turnover (sum of absolute weight changes; 0 in the first month) of the BAB position and of the
    lowest-beta quintile. Weights reset each month; drift within the month is ignored."""
    q, bab, low, bab_turn, low_turn = [], [], [], [], []
    w_prev = lw_prev = None
    for t in B.index:
        b = B.loc[t].dropna()
        b = b[Rnext.loc[t, b.index].notna()]
        r = Rnext.loc[t, b.index]
        grp = pd.qcut(b.rank(method="first"), 5, labels=False)
        q.append([float(r[grp == k].mean()) for k in range(5)])
        wl, wh = bab_weights(b)
        bab.append(bab_return(wl, wh, b, r))
        w = (wl / (wl @ b)).sub(wh / (wh @ b), fill_value=0)
        bab_turn.append(0.0 if w_prev is None else float(w.sub(w_prev, fill_value=0).abs().sum()))
        lw = (grp == 0).astype(float) / (grp == 0).sum()
        dl = 0.0 if lw_prev is None else float(lw.sub(lw_prev, fill_value=0).abs().sum())
        low_turn.append(dl)
        low.append(float(lw @ r))
        w_prev, lw_prev = w, lw
    idx = B.index
    return {"quintiles": pd.DataFrame(q, index=idx), "bab": pd.Series(bab, index=idx),
            "low": pd.Series(low, index=idx), "bab_turnover": pd.Series(bab_turn, index=idx),
            "low_turnover": pd.Series(low_turn, index=idx)}

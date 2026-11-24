"""Price-based factors: momentum, low volatility and sectors, tested beyond beta.

- momentum: return over months t−11 … t−1 (the classic 12-1: the latest month is skipped, since
  one-month returns tend to reverse).
- zscore, sector_demean: make characteristics comparable across months; demeaning within sector
  asks whether a factor works *inside* sectors rather than by picking sectors.
- fama_macbeth_multi: each month regress next-month returns on several characteristics together;
  average each slope; Newey-West t.
- long_short: equal-weight top third minus bottom third by a characteristic.
"""

import numpy as np
import pandas as pd

from bc.pricing import _nw_t_mean


def momentum(M: pd.DataFrame) -> pd.DataFrame:
    """M: monthly returns (rows months). Row t: compounded return of rows t−11 … t−1."""
    log = np.log1p(M).shift(1)
    return np.expm1(log.rolling(11, min_periods=11).sum())


def zscore(X: pd.DataFrame) -> pd.DataFrame:
    return X.sub(X.mean(axis=1), axis=0).div(X.std(axis=1, ddof=1), axis=0)


def sector_demean(X: pd.DataFrame, sectors: dict) -> pd.DataFrame:
    groups = pd.Series(sectors).reindex(X.columns)
    return X - X.T.groupby(groups).transform("mean").T


def fama_macbeth_multi(
    chars: dict, Rnext: pd.DataFrame, lags: int = 3, min_stocks: int = 10
) -> dict:
    names = list(chars)
    slopes = {k: {} for k in names}
    for t in Rnext.index:
        cols = [chars[k].loc[t] for k in names]
        ok = Rnext.loc[t].notna()
        for c in cols:
            ok &= c.notna()
        if ok.sum() < min_stocks:
            continue
        X = np.column_stack([np.ones(ok.sum())] + [c[ok].to_numpy(float) for c in cols])
        coef, *_ = np.linalg.lstsq(X, Rnext.loc[t, ok].to_numpy(float), rcond=None)
        for k, b in zip(names, coef[1:], strict=True):
            slopes[k][t] = b
    out = {}
    for k in names:
        s = pd.Series(slopes[k])
        out[k] = {"slope": float(s.mean()), "t": _nw_t_mean(s.to_numpy(), lags), "slopes": s}
    return out


def long_short(x: pd.Series, r: pd.Series) -> float:
    ok = x.notna() & r.notna()
    x, r = x[ok], r[ok]
    third = pd.qcut(x.rank(method="first"), 3, labels=False)
    return float(r[third == 2].mean() - r[third == 0].mean())

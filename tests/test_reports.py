import numpy as np
import pandas as pd
import pytest

from bc.data import ETFS, SYMBOLS
from bc.reports import RF, _monthly_panel


def _returns(seed=0):
    idx = pd.bdate_range("2011-01-03", "2014-06-30")
    cols = SYMBOLS + ETFS + ["NIFTY"]
    return pd.DataFrame(np.random.default_rng(seed).normal(0, 0.01, (len(idx), len(cols))),
                        index=idx, columns=cols)


def test_monthly_panel_pairs_month_t_betas_with_month_t_plus_1_returns():
    R = _returns()
    B, Rn, mkt = _monthly_panel(R)
    t = pd.Period("2013-03", "M")
    # month t+1's excess return, compounded from its days
    apr = R.loc["2013-04", SYMBOLS[0]]
    days = (R.loc["2013-04"].index[-1] - R.loc["2013-03"].index[-1]).days
    want = (1 + apr).prod() - 1 - ((1 + RF) ** (days / 365) - 1)
    assert Rn.loc[t, SYMBOLS[0]] == pytest.approx(want, abs=2e-3)  # calendar-month rf ≈ trading
    # betas at t ignore everything after month t
    R2 = R.copy()
    R2.loc["2013-04-01":] = R2.loc["2013-04-01":] * 3
    B2, _, _ = _monthly_panel(R2)
    assert B2.loc[t].equals(B.loc[t])

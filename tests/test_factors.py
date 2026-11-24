import numpy as np
import pandas as pd
import pytest

from bc.factors import fama_macbeth_multi, long_short, momentum, sector_demean, zscore


def test_momentum_compounds_months_t_minus_11_to_t_minus_1_and_skips_month_t():
    idx = pd.period_range("2020-01", periods=13, freq="M")
    M = pd.DataFrame({"A": [0.01] * 12 + [0.5]}, index=idx)
    M.iloc[11, 0] = 0.9  # month t-1 counts; month t (0.5) must not
    mom = momentum(M)
    assert np.isnan(mom.iloc[10, 0])  # needs 11 past months
    assert mom.iloc[12, 0] == pytest.approx(1.01**10 * 1.9 - 1)


def test_zscore_and_sector_demean_work_across_each_row():
    X = pd.DataFrame([[1.0, 2.0, 3.0, np.nan]], columns=list("abcd"))
    z = zscore(X)
    assert z.iloc[0, :3].tolist() == pytest.approx([-1, 0, 1]) and np.isnan(z.iloc[0, 3])
    sectors = {"a": "x", "b": "x", "c": "y", "d": "y"}
    d = sector_demean(X, sectors)
    assert d.iloc[0, :3].tolist() == pytest.approx([-0.5, 0.5, 0.0])


def test_fama_macbeth_multi_separates_two_characteristics():
    rng = np.random.default_rng(0)
    months, n = 300, 50
    f1 = pd.DataFrame(rng.normal(size=(months, n)))
    f2 = pd.DataFrame(0.5 * f1.to_numpy() + rng.normal(size=(months, n)))  # correlated
    r = 0.01 * f1 + rng.normal(0, 0.05, (months, n))  # only f1 pays
    fm = fama_macbeth_multi({"f1": f1, "f2": f2}, r, lags=0)
    assert fm["f1"]["slope"] == pytest.approx(0.01, abs=0.002)
    assert abs(fm["f2"]["t"]) < 3 and fm["f1"]["t"] > 5
    assert len(fm["f1"]["slopes"]) == months


def test_long_short_is_top_third_minus_bottom_third():
    x = pd.Series(range(9), dtype=float)
    r = pd.Series(np.arange(9) / 100)
    assert long_short(x, r) == pytest.approx(0.07 - 0.01)

import numpy as np
import pandas as pd
import pytest

from bc.capm import bh_reject, capm_table, excess, ols


def test_excess_returns_use_calendar_days():
    idx = pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"])  # Fri, Mon, Tue
    r = pd.Series([0.0, 0.01, 0.01], index=idx)
    ex = excess(r, 0.0657)
    daily = (1.0657) ** (1 / 365) - 1
    assert ex.iloc[1] == pytest.approx(0.01 - ((1 + daily) ** 3 - 1))  # weekend: 3 days
    assert ex.iloc[2] == pytest.approx(0.01 - daily)


def test_ols_recovers_alpha_and_beta_and_newey_west_widens_with_autocorrelation():
    rng = np.random.default_rng(0)
    n = 20_000
    x = rng.normal(0, 0.01, n)
    e = rng.normal(0, 0.01, n)
    y = 0.0002 + 1.3 * x + e
    res = ols(y, x, lags=5)
    assert res["beta"] == pytest.approx(1.3, abs=0.03)
    assert res["alpha"] == pytest.approx(0.0002, abs=0.0002)
    assert res["se_beta_nw"] == pytest.approx(res["se_beta"], rel=0.1)  # iid: about equal
    assert res["r2"] == pytest.approx(np.corrcoef(x, y)[0, 1] ** 2)
    # autocorrelated errors and x: Newey-West's standard error is wider than plain OLS
    xa = np.convolve(rng.normal(0, 0.01, n + 9), np.ones(10), "valid")
    ea = np.convolve(rng.normal(0, 0.01, n + 9), np.ones(10), "valid")
    res2 = ols(0.5 * xa + ea, xa, lags=20)
    assert res2["se_beta_nw"] > 1.5 * res2["se_beta"]


def test_benjamini_hochberg():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    assert bh_reject(p, 0.05).tolist() == [True, True] + [False] * 8


def test_capm_table_annualises_by_calendar_time():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2020-01-01", "2024-12-31")
    m = rng.normal(0.0005, 0.01, len(idx))
    R = pd.DataFrame({"MKT": m, "A": 1.2 * m + rng.normal(0, 0.01, len(idx))}, index=idx)
    t = capm_table(R, ["A"], "MKT", rf=0.0)
    per_year = (len(idx) - 1) / ((idx[-1] - idx[0]).days / 365.25)
    res = ols(R["A"].to_numpy()[1:], R["MKT"].to_numpy()[1:], lags=5)
    assert t.loc["A", "alpha_annual"] == pytest.approx(res["alpha"] * per_year, rel=1e-6)
    assert t.loc["A", "beta"] == pytest.approx(res["beta"], rel=1e-9)

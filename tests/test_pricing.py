import numpy as np
import pandas as pd
import pytest

from bc.pricing import bab_return, bab_weights, fama_macbeth


def test_fama_macbeth_recovers_a_known_price_of_beta():
    rng = np.random.default_rng(0)
    months, n = 240, 40
    B = pd.DataFrame(rng.uniform(0.5, 1.5, (months, n)))
    gamma = rng.normal(0.004, 0.02, months)  # monthly reward per unit of beta
    Rn = B.mul(gamma, axis=0) + rng.normal(0, 0.03, (months, n))
    fm = fama_macbeth(B, Rn, lags=0)
    assert len(fm["slopes"]) == months
    assert fm["slopes"].iloc[5] == pytest.approx(np.polyfit(B.iloc[5], Rn.iloc[5], 1)[0])
    assert fm["slope"] == pytest.approx(gamma.mean(), abs=0.002)
    assert fm["t_slope"] == pytest.approx(
        fm["slope"] / (fm["slopes"].std(ddof=1) / np.sqrt(months)), rel=0.02
    )


def test_fama_macbeth_skips_stocks_missing_a_beta_or_return():
    B = pd.DataFrame([[0.5, 1.0, 1.5, np.nan]] * 3)
    Rn = pd.DataFrame([[0.01, 0.02, 0.03, 0.9]] * 3)
    assert fama_macbeth(B, Rn, lags=0, min_stocks=3)["slope"] == pytest.approx(0.02)


def test_bab_legs_are_rank_weighted_and_each_sum_to_one():
    beta = pd.Series([0.5, 0.8, 1.0, 1.2, 1.6], index=list("abcde"))
    wl, wh = bab_weights(beta)
    assert wl.sum() == pytest.approx(1) and wh.sum() == pytest.approx(1)
    assert wl["a"] > wl["b"] > 0 and wl["d"] == 0
    assert wh["e"] > wh["d"] > 0 and wh["b"] == 0
    assert wl["c"] == wh["c"] == 0  # median rank sits in neither leg


def test_bab_return_scales_each_leg_to_beta_one():
    beta = pd.Series([0.5, 1.5], index=["lo", "hi"])
    wl, wh = pd.Series([1.0, 0.0], index=beta.index), pd.Series([0.0, 1.0], index=beta.index)
    r = pd.Series([0.01, 0.03], index=beta.index)  # excess returns
    # 0.01 / 0.5 − 0.03 / 1.5 = 0.02 − 0.02
    assert bab_return(wl, wh, beta, r) == pytest.approx(0.0)


def test_beta_sorts_quintiles_bab_and_turnover():
    from bc.pricing import beta_sorts

    cols = [f"s{i}" for i in range(10)]
    B = pd.DataFrame([np.linspace(0.5, 1.4, 10), np.linspace(1.4, 0.5, 10)], columns=cols)
    Rn = pd.DataFrame([np.linspace(0.0, 0.09, 10), np.full(10, 0.01)], columns=cols)
    out = beta_sorts(B, Rn)
    assert out["quintiles"].iloc[0].tolist() == pytest.approx([0.005, 0.025, 0.045, 0.065, 0.085])
    wl, wh = bab_weights(B.iloc[0])
    assert out["bab"].iloc[0] == pytest.approx(bab_return(wl, wh, B.iloc[0], Rn.iloc[0]))
    # the low-beta fifth flips from s0, s1 to s8, s9: everything is sold and bought
    assert out["low"].iloc[1] == pytest.approx(0.01)
    assert out["low_turnover"].tolist() == pytest.approx([0.0, 2.0])
    assert len(out["bab_turnover"]) == 2 and out["bab_turnover"].iloc[0] == 0

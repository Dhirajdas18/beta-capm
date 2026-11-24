import numpy as np
import pandas as pd
import pytest

from bc.hedge import FUT, futures_cost, hedge_lots, run_hedge


def test_futures_cost_on_one_roll():
    n = 1_700_000.0  # one lot's notional, ₹
    c = futures_cost(buy=n, sell=n, orders=2)
    brokerage = 2 * min(20.0, 0.0003 * n)
    fees = 2 * n * (FUT["exchange"] + FUT["sebi"])
    expected = (
        (brokerage + fees) * 1.18
        + n * FUT["stt_sell"]
        + n * FUT["stamp_buy"]
        + 2 * n * FUT["slippage"]
    )
    assert c == pytest.approx(expected)
    assert FUT["stt_sell"] == 0.0005 and FUT["lot"] == 65


def test_hedge_lots_round_to_whole_contracts():
    # ₹50 lakh at beta 0.8 against a ₹17 lakh lot: 2.35 lots -> 2
    assert hedge_lots(5_000_000, 0.8, 1_700_000) == 2
    assert hedge_lots(800_000, 1.0, 1_700_000) == 0  # under half a lot: no hedge possible
    assert hedge_lots(5_000_000, 0.8, 1_700_000, whole=False) == pytest.approx(
        5_000_000 * 0.8 / 1_700_000
    )


def test_run_hedge_removes_the_market_and_charges_each_roll():
    idx = pd.period_range("2020-01", periods=4, freq="M")
    fut = pd.Series([0.02, -0.03, 0.01, 0.04], idx)
    port = 1.0 * fut + pd.Series([0.001, -0.002, 0.0, 0.001], idx)
    beta = pd.Series(1.0, idx)
    lot_value = pd.Series(1_000_000.0, idx)
    out = run_hedge(port, fut, beta, lot_value, capital=5_000_000, whole=True)
    assert (out["lots"] == 5).all()
    gross = port - 5 * 1_000_000 / 5_000_000 * fut
    # first month: only open (sell); later months: close (buy) the old and open the new
    c0 = futures_cost(0.0, 5e6, 1) / 5e6
    c1 = futures_cost(5e6, 5e6, 2) / 5e6
    assert out["hedged"].tolist() == pytest.approx((gross - [c0, c1, c1, c1]).tolist())
    assert np.std(out["hedged"]) < np.std(port) / 5

import numpy as np
import pandas as pd
import pytest

from bc.measure import dimson_beta, huber_beta, resampled_returns


def test_dimson_adds_the_lagged_response():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.01, 20_000)
    # half the response arrives a day late (a price that updates slowly)
    y = 0.6 * x + 0.4 * np.r_[0.0, x[:-1]] + rng.normal(0, 0.005, len(x))
    d = dimson_beta(y, x)
    assert d["parts"][0] == pytest.approx(0.4, abs=0.02)  # yesterday's market
    assert d["parts"][1] == pytest.approx(0.6, abs=0.02)
    assert d["beta"] == pytest.approx(1.0, abs=0.03)


def test_huber_resists_a_few_wild_days():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.01, 2000)
    y = 1.0 * x + rng.normal(0, 0.005, 2000)
    y[:10] += np.sign(x[:10]) * 0.3  # ten huge moves in the market's direction
    plain = np.polyfit(x, y, 1)[0]
    robust = huber_beta(y, x)
    assert abs(robust - 1.0) < abs(plain - 1.0) / 3
    assert huber_beta(1.2 * x, x) == pytest.approx(1.2)


def test_resampled_returns_compound_and_drop_unfinished_periods():
    idx = pd.bdate_range("2024-01-01", "2024-03-20")
    R = pd.DataFrame({"A": 0.01}, index=idx)
    m = resampled_returns(R, "ME")
    feb = idx[idx.month == 2]
    assert m.loc["2024-02-29", "A"] == pytest.approx(1.01 ** len(feb) - 1)
    assert m.index[-1] == pd.Timestamp("2024-02-29")  # March isn't finished

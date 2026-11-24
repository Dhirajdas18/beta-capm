import numpy as np
import pytest

from bc.asym import tail_beta, updown_beta


def test_updown_beta_recovers_separate_slopes():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.01, 20_000)
    y = np.where(x > 0, 0.8, 1.3) * x + rng.normal(0, 0.004, len(x))
    r = updown_beta(y, x)
    assert r["up"] == pytest.approx(0.8, abs=0.02)
    assert r["down"] == pytest.approx(1.3, abs=0.02)
    assert r["t_diff"] > 10


def test_updown_beta_t_is_small_when_slopes_match():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.01, 5000)
    y = x + rng.normal(0, 0.01, 5000)
    r = updown_beta(y, x)
    assert abs(r["t_diff"]) < 3
    assert r["down"] - r["up"] == pytest.approx(r["diff"])


def test_tail_beta_uses_only_the_worst_market_days():
    x = np.r_[np.linspace(-0.05, -0.03, 50), np.linspace(-0.01, 0.01, 950)]
    y = np.where(x < -0.025, 2.0 * x, 0.5 * x)
    r = tail_beta(y, x, q=0.05)
    assert r["n"] == 50
    assert r["beta"] == pytest.approx(2.0)

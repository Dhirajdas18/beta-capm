import numpy as np
import pytest

from bc.forecast import ewma_beta, fit_kalman_ratio, kalman_beta, rolling_beta, vasicek


def test_rolling_beta_matches_ols_on_each_window_and_uses_past_days_only():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.01, 600)
    y = 1.3 * x + rng.normal(0, 0.01, 600)
    b = rolling_beta(y, x, 250)
    assert np.isnan(b[249])  # needs 250 days before t
    t = 400
    assert b[t] == pytest.approx(np.polyfit(x[t - 250:t], y[t - 250:t], 1)[0], rel=1e-9)


def test_ewma_beta_recursion_uses_days_before_t():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.01, 300)
    y = 0.8 * x + rng.normal(0, 0.005, 300)
    b = ewma_beta(y, x, 0.97, seed=60)
    cxy, vx = np.mean(x[:60] * y[:60]), np.mean(x[:60] ** 2)
    for t in range(60, 100):
        cxy, vx = (0.97 * cxy + 0.03 * x[t - 1] * y[t - 1] if t > 60 else cxy,
                   0.97 * vx + 0.03 * x[t - 1] ** 2 if t > 60 else vx)
    assert b[99] == pytest.approx(cxy / vx, rel=1e-9)


def test_vasicek_shrinks_noisy_estimates_more():
    b = np.array([0.5, 1.5, 1.0])
    se = np.array([0.05, 0.4, 0.1])
    out = vasicek(b, se)
    prior_m, prior_v = b.mean(), b.var(ddof=1)
    w = prior_v / (prior_v + se**2)
    assert out == pytest.approx(w * b + (1 - w) * prior_m)
    assert abs(out[1] - prior_m) / abs(b[1] - prior_m) < abs(out[0] - prior_m) / abs(b[0] - prior_m)


def test_kalman_follows_a_change_in_beta():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 0.01, 2000)
    beta = np.r_[np.full(1000, 0.6), np.full(1000, 1.4)]
    y = beta * x + rng.normal(0, 0.005, 2000)
    k = kalman_beta(y, x, ratio=1e-4)
    assert k[900] == pytest.approx(0.6, abs=0.15)
    assert k[1900] == pytest.approx(1.4, abs=0.15)
    ratio = fit_kalman_ratio(y[:1000], x[:1000])
    assert ratio < 1e-3  # a steady beta wants little drift


def test_kalman_forecast_ignores_the_day_itself():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.01, 500)
    y = x + rng.normal(0, 0.005, 500)
    k1 = kalman_beta(y, x, 1e-4)
    y2 = y.copy()
    y2[300:] += 0.05
    assert kalman_beta(y2, x, 1e-4)[300] == pytest.approx(k1[300])

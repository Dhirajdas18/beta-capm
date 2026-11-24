"""README findings, one function per topic: `bc report <topic>` writes charts to out/."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bc import plots

RF = 0.065  # assumed risk-free rate (India 91-day T-bills ran roughly 3-7% over 2009-2026)
MARKET = "NIFTYBEES"  # NIFTY 50 with dividends reinvested, as traded


def returns() -> pd.DataFrame:
    from bc.data import DATA_PATH, daily_returns

    return daily_returns(pd.read_parquet(DATA_PATH)["adj_close"])


def last_years(R: pd.DataFrame, years: int) -> pd.DataFrame:
    return R.loc[R.index[-1] - pd.DateOffset(years=years) :]


def report_capm(out_dir: Path) -> dict:
    """CAPM regressions of today's NIFTY 50 stocks on the NIFTY ETF over five years: betas with
    honest uncertainty, how much the market explains, and whether any alpha survives testing."""
    from scipy.stats import norm

    from bc.capm import bh_reject, capm_table, excess
    from bc.data import SYMBOLS

    W = last_years(returns(), 5)
    syms = [s for s in SYMBOLS if W[s].notna().mean() > 0.95]
    t = capm_table(W, syms, MARKET, RF)
    t = t.sort_values("beta")

    lo, hi = t.index[0], t.index[-1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True, sharey=True)
    for ax, s, c in ((axes[0], lo, plots.SERIES[2]), (axes[1], hi, plots.SERIES[1])):
        d = W[[s, MARKET]].dropna()
        x, y = excess(d[MARKET], RF).iloc[1:] * 100, excess(d[s], RF).iloc[1:] * 100
        ax.plot(x, y, "o", ms=2, color=plots.MUTED, alpha=0.5)
        xs = np.linspace(-8, 8, 10)
        r = t.loc[s]
        ax.plot(
            xs,
            r["alpha"] * 100 + r["beta"] * xs,
            color=c,
            lw=2,
            label=f"beta {r['beta']:.2f}, R² {r['r2']:.2f}",
        )
        ax.plot(xs, xs, color=plots.TEXT_2, lw=0.8, ls="--", label="beta 1")
        ax.set_title(s, fontsize=10, fontweight="normal")
        ax.set_xlim(-8, 8)
        ax.set_ylim(-15, 15)
        ax.set_xlabel("NIFTY ETF excess return, % a day")
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("stock excess return, % a day")
    fig.suptitle(
        f"The lowest and highest betas in today's NIFTY 50, last 5 years: "
        f"{t.loc[lo, 'beta']:.2f} and {t.loc[hi, 'beta']:.2f}",
        fontweight="bold",
    )
    plots.save(fig, out_dir, "capm_scatter")

    fig, ax = plt.subplots(figsize=(9, 9))
    y = np.arange(len(t))
    ax.errorbar(
        t["beta"],
        y,
        xerr=1.96 * t["se_beta_nw"],
        fmt="o",
        ms=4,
        color=plots.SERIES[0],
        ecolor=plots.SERIES[0],
        elinewidth=1.2,
        capsize=0,
        label="Newey-West 95% range",
    )
    ax.errorbar(
        t["beta"],
        y + 0.3,
        xerr=1.96 * t["se_beta"],
        fmt="none",
        ecolor=plots.MUTED,
        elinewidth=1,
        label="plain OLS 95% range",
    )
    ax.axvline(1, color=plots.TEXT_2, lw=0.8, ls="--")
    ax.set_yticks(y, t.index, fontsize=7)
    ax.set_xlabel("beta to the NIFTY ETF (daily excess returns, last 5 years)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8)
    ratio = float((t["se_beta_nw"] / t["se_beta"]).median())
    ax.set_title(
        f"Betas ran from {t['beta'].min():.2f} to {t['beta'].max():.2f}; honest "
        f"(Newey-West) ranges were {ratio - 1:.0%} wider than plain OLS"
    )
    plots.save(fig, out_dir, "capm_betas")

    sig = t["p_alpha_nw"] < 0.05
    bh = bh_reject(t["p_alpha_nw"].to_numpy(), 0.05)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bins = np.arange(-4, 4.01, 0.5)
    ax.hist(
        t["t_alpha_nw"],
        bins=bins,
        color=plots.SERIES[0],
        alpha=0.85,
        label=f"{len(t)} stocks' alpha t-statistics",
    )
    xs = np.linspace(-4, 4, 200)
    ax.plot(
        xs,
        norm.pdf(xs) * len(t) * 0.5,
        color=plots.SERIES[1],
        lw=2,
        label="what pure luck gives (no true alpha)",
    )
    for v in (-1.96, 1.96):
        ax.axvline(v, color=plots.TEXT_2, lw=0.8, ls="--")
    ax.set_xlabel("alpha t-statistic (Newey-West)")
    ax.set_ylabel("number of stocks")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(
        f"{int(sig.sum())} of {len(t)} alphas looked significant, {0.05 * len(t):.1f} "
        f"expected by luck; {int(bh.sum())} survived a multiple-testing correction"
    )
    plots.save(fig, out_dir, "capm_alphas")

    return {
        "window": f"{W.index[0]:%d %b %Y} to {W.index[-1]:%d %b %Y}",
        "stocks": len(t),
        "beta: min (who), median, max (who)": (
            round(t["beta"].min(), 2),
            lo,
            round(t["beta"].median(), 2),
            round(t["beta"].max(), 2),
            hi,
        ),
        "Newey-West / OLS standard error, beta: median, max": (
            round(ratio, 2),
            round(float((t["se_beta_nw"] / t["se_beta"]).max()), 2),
        ),
        "Newey-West / OLS standard error, alpha: median": round(
            float((t["se_alpha_nw"] / t["se_alpha"]).median()), 2
        ),
        "R²: median, min, max": (
            round(t["r2"].median(), 2),
            round(t["r2"].min(), 2),
            round(t["r2"].max(), 2),
        ),
        "alpha a year: mean, share positive": (
            round(t["alpha_annual"].mean(), 4),
            round(float((t["alpha_annual"] > 0).mean()), 2),
        ),
        "significant at 5% (all positive?), expected, BH, Bonferroni": (
            int(sig.sum()),
            bool((t.loc[sig, "alpha_annual"] > 0).all()),
            round(0.05 * len(t), 1),
            int(bh.sum()),
            int((t["p_alpha_nw"] < 0.05 / len(t)).sum()),
        ),
        "significant ones (alpha, t)": {
            s: (round(r["alpha_annual"], 3), round(r["t_alpha_nw"], 2))
            for s, r in t[sig].iterrows()
        },
        "lowest betas": t["beta"].head(3).round(2).to_dict(),
        "highest betas": t["beta"].tail(3).round(2).to_dict(),
    }


MEASURE_START = "2014-01-01"


def report_measure(out_dir: Path) -> dict:
    """Beta by return frequency, by market proxy (ETF or index), with Dimson lags and robust
    regression, for the stocks with prices every day since 2014."""
    from bc.capm import excess, ols
    from bc.data import SYMBOLS
    from bc.measure import dimson_beta, huber_beta, resampled_returns

    R = returns().loc[MEASURE_START:]
    syms = [s for s in SYMBOLS if R[s].notna().mean() > 0.99]
    X = R[syms + [MARKET, "NIFTY"]].dropna()
    freqs = {
        "daily": (X, 5),
        "weekly": (resampled_returns(X, "W-FRI"), 2),
        "monthly": (resampled_returns(X, "ME"), 1),
    }
    beta = {}
    for lab, (D, lags) in freqs.items():
        for mkt in (MARKET, "NIFTY"):
            ex_m = excess(D[mkt], RF).iloc[1:]
            beta[(lab, mkt)] = pd.Series(
                {
                    s: ols(excess(D[s], RF).iloc[1:].to_numpy(), ex_m.to_numpy(), lags)["beta"]
                    for s in syms
                }
            )
    avg = {k: float(v.mean()) for k, v in beta.items()}

    fig, ax = plt.subplots(figsize=(9, 4.6))
    xs = np.arange(3)
    for j, (mkt, c, lab) in enumerate(
        (
            (MARKET, plots.SERIES[0], "against the NIFTY ETF"),
            ("NIFTY", plots.SERIES[1], "against the NIFTY index"),
        )
    ):
        vals = [avg[(f, mkt)] for f in freqs]
        ax.bar(xs - 0.2 + 0.4 * j, vals, 0.38, color=c, label=lab)
        for i, v in enumerate(vals):
            ax.text(i - 0.2 + 0.4 * j, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(xs, list(freqs))
    ax.set_ylim(0.9, 1.1)
    ax.set_ylabel(f"average beta of {len(syms)} stocks, 2014–26")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=8)
    gd = avg[("daily", MARKET)] / avg[("daily", "NIFTY")] - 1
    gm = avg[("monthly", MARKET)] / avg[("monthly", "NIFTY")] - 1
    ax.set_title(
        f"Against the ETF, daily betas averaged {gd:.0%} higher than against the index; "
        f"monthly, {gm:.0%}"
    )
    plots.save(fig, out_dir, "meas_market")
    etf_d, idx_d = X[MARKET], X["NIFTY"]
    etf_vs_idx = {
        "daily vol ratio (ETF / index)": float(etf_d.std() / idx_d.std()),
        "daily correlation": float(etf_d.corr(idx_d)),
    }
    for lab, (D, _) in list(freqs.items())[1:]:
        etf_vs_idx[f"{lab} vol ratio"] = float(D[MARKET].std() / D["NIFTY"].std())
        etf_vs_idx[f"{lab} correlation"] = float(D[MARKET].corr(D["NIFTY"]))

    fig, ax = plt.subplots(figsize=(9, 4.8))
    d = beta[("daily", MARKET)]
    for lab, c in (("weekly", plots.SERIES[2]), ("monthly", plots.SERIES[3])):
        ax.plot(d, beta[(lab, MARKET)], "o", ms=6, color=c, label=f"{lab} beta")
    ax.plot([0.3, 2], [0.3, 2], color=plots.TEXT_2, lw=0.8, ls="--", label="same as daily")
    ax.set_xlabel("daily beta to the NIFTY ETF")
    ax.set_ylabel("weekly or monthly beta")
    ax.legend(loc="upper left", fontsize=8)
    cw = float(np.corrcoef(d, beta[("weekly", MARKET)])[0, 1])
    cm = float(np.corrcoef(d, beta[("monthly", MARKET)])[0, 1])
    ax.set_title(
        f"Daily, weekly and monthly betas agreed: correlation {cw:.2f} and {cm:.2f} across stocks"
    )
    plots.save(fig, out_dir, "meas_freq")

    ex_m = excess(X[MARKET], RF).iloc[1:].to_numpy()
    dim, hub, plain = {}, {}, {}
    big = np.argsort(-np.abs(ex_m))[:10]
    keep = np.setdiff1d(np.arange(len(ex_m)), big)
    noext = {}
    for s in syms:
        y = excess(X[s], RF).iloc[1:].to_numpy()
        plain[s] = ols(y, ex_m, 5)["beta"]
        dim[s] = dimson_beta(y, ex_m)
        hub[s] = huber_beta(y, ex_m)
        noext[s] = ols(y[keep], ex_m[keep], 5)["beta"]
    parts = np.array([dim[s]["parts"] for s in syms]).mean(axis=0)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    labels = {
        "plain OLS": plain,
        "Dimson (yesterday, today, tomorrow)": {s: v["beta"] for s, v in dim.items()},
        "Huber (robust)": hub,
        "without the 10 biggest market days": noext,
    }
    vals = [float(np.mean(list(v.values()))) for v in labels.values()]
    ax.bar(
        range(len(vals)),
        vals,
        0.55,
        color=[plots.TEXT, plots.SERIES[2], plots.SERIES[0], plots.MUTED],
    )
    for i, v in enumerate(vals):
        ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(vals)), [k.replace(" (", "\n(") for k in labels], fontsize=8)
    ax.set_ylim(0.9, 1.1)
    ax.set_ylabel("average daily beta to the NIFTY ETF")
    ax.grid(axis="x", visible=False)
    ax.set_title(
        f"Lags and robust weights moved the average beta by at most "
        f"{max(abs(v - vals[0]) for v in vals[1:]):.2f}"
    )
    plots.save(fig, out_dir, "meas_robust")

    diffs = {k: pd.Series(v) - pd.Series(plain) for k, v in labels.items() if k != "plain OLS"}
    return {
        "stocks, days": (len(syms), len(X), f"{X.index[0]:%d %b %Y} to {X.index[-1]:%d %b %Y}"),
        "average beta (freq, market)": {f"{a} vs {b}": round(v, 3) for (a, b), v in avg.items()},
        "ETF / index gap: daily, weekly, monthly": tuple(
            round(avg[(f, MARKET)] / avg[(f, "NIFTY")] - 1, 3) for f in freqs
        ),
        "ETF vs index": {k: round(v, 3) for k, v in etf_vs_idx.items()},
        "weekly - daily (ETF): mean, sd, min, max": tuple(
            round(x, 3)
            for x in (
                (beta[("weekly", MARKET)] - d).mean(),
                (beta[("weekly", MARKET)] - d).std(),
                (beta[("weekly", MARKET)] - d).min(),
                (beta[("weekly", MARKET)] - d).max(),
            )
        ),
        "monthly - daily (ETF): mean, sd": (
            round((beta[("monthly", MARKET)] - d).mean(), 3),
            round((beta[("monthly", MARKET)] - d).std(), 3),
        ),
        "correlation daily-weekly, daily-monthly": (round(cw, 3), round(cm, 3)),
        "Dimson average parts (yesterday, today, tomorrow)": parts.round(3).tolist(),
        "averages: plain, Dimson, Huber, no extremes": [round(v, 3) for v in vals],
        "largest per-stock change vs plain": {
            k: (round(float(v.abs().max()), 3), v.abs().idxmax()) for k, v in diffs.items()
        },
    }


HEDGE_MARKET = "NIFTY"  # futures track the index (QL-097)


def _nw_t(d: np.ndarray, lags: int) -> float:
    """t-statistic of the mean of d with a Newey-West (Bartlett) standard error."""
    d = d[np.isfinite(d)]
    n, m = len(d), d.mean()
    g = [np.mean((d[j:] - m) * (d[: n - j] - m)) for j in range(lags + 1)]
    lr = g[0] + 2 * sum((1 - j / (lags + 1)) * g[j] for j in range(1, lags + 1))
    return float(m / np.sqrt(lr / n))


def report_forecast(out_dir: Path) -> dict:
    """Which beta forecast best predicts next year's beta and best hedges next month, monthly from
    2014; settings fixed in advance or fitted on 2011-13 only."""
    from bc.data import SYMBOLS
    from bc.forecast import (
        ewma_beta,
        fit_kalman_ratio,
        kalman_beta,
        regression_scale,
        rolling_beta,
        vasicek,
    )

    R = returns().loc["2011-01-01":]
    syms = [s for s in SYMBOLS if R.loc[MEASURE_START:, s].notna().mean() > 0.99]
    X = R[syms + [HEDGE_MARKET]].dropna()
    x = X[HEDGE_MARKET].to_numpy()
    pre = X.index < MEASURE_START
    month = X.index.to_period("M")
    ends = np.flatnonzero(np.append(month[1:] != month[:-1], True))
    ends = ends[
        (X.index[ends] >= pd.Timestamp(MEASURE_START) - pd.Timedelta(days=1)) & (ends < len(X) - 1)
    ]

    fc = {
        k: {}
        for k in (
            "rolling 1 year",
            "rolling 3 years",
            "EWMA (λ 0.99)",
            "Blume",
            "Vasicek",
            "Kalman (own drift)",
            "Kalman (shared drift)",
        )
    }
    se1 = {}
    ratios = {}
    for s in syms:
        y = X[s].to_numpy()
        r1, r3 = rolling_beta(y, x, 250), rolling_beta(y, x, 750)
        fc["rolling 1 year"][s], fc["rolling 3 years"][s] = r1, r3
        fc["EWMA (λ 0.99)"][s] = ewma_beta(y, x, 0.99)
        fc["Blume"][s] = 0.67 * r1 + 0.33
        ratios[s] = fit_kalman_ratio(y[pre], x[pre])
        fc["Kalman (own drift)"][s] = kalman_beta(
            y, x, ratios[s], scale=regression_scale(y[pre], x[pre])
        )
        # standard error of the 1-year slope, for Vasicek
        resid_var = np.array(
            [
                np.var(y[t - 250 : t] - r1[t] * x[t - 250 : t]) if np.isfinite(r1[t]) else np.nan
                for t in range(len(y))
            ]
        )
        var_x = np.array([np.var(x[t - 250 : t]) if t >= 250 else np.nan for t in range(len(y))])
        se1[s] = np.sqrt(resid_var / (250 * var_x))
    shared = float(np.median(list(ratios.values())))  # one drift for all, still from 2011-13
    for s in syms:
        y = X[s].to_numpy()
        fc["Kalman (shared drift)"][s] = kalman_beta(
            y, x, shared, scale=regression_scale(y[pre], x[pre])
        )
    vas = {s: np.full(len(X), np.nan) for s in syms}
    for t in ends:
        b = np.array([fc["rolling 1 year"][s][t + 1] for s in syms])
        se = np.array([se1[s][t + 1] for s in syms])
        out = vasicek(b, se)
        for s, v in zip(syms, out, strict=True):
            vas[s][t + 1] = v
    fc["Vasicek"] = vas

    # score only months where every method has a forecast (same months for all)
    ends = np.array(
        [t for t in ends if all(np.isfinite(fc[k][s][t + 1]) for k in fc for s in syms)]
    )
    err_year = {k: [] for k in fc}
    hedge = {k: [] for k in fc}
    for i, t in enumerate(ends):
        nxt = ends[i + 1] if i + 1 < len(ends) else len(X) - 1
        year = slice(t + 1, t + 251)
        month_days = slice(t + 1, nxt + 1)
        for k in fc:
            ey, hv = [], []
            for s in syms:
                y = X[s].to_numpy()
                b = fc[k][s][t + 1]
                if t + 251 <= len(X):
                    real = np.polyfit(x[year], y[year], 1)[0]
                    ey.append((b - real) ** 2)
                hv.append(np.var(y[month_days] - b * x[month_days]))
            err_year[k].append(np.mean(ey) if ey else np.nan)
            hedge[k].append(np.mean(hv))
    base = "rolling 1 year"
    ey = {k: np.array(v) for k, v in err_year.items()}
    hv = {k: np.array(v) for k, v in hedge.items()}
    rel_y = {k: float(np.nanmean(ey[k]) / np.nanmean(ey[base])) for k in fc}
    rel_h = {k: float(hv[k].mean() / hv[base].mean()) for k in fc}
    t_y = {k: _nw_t(ey[k] - ey[base], 12) for k in fc if k != base}
    t_h = {k: _nw_t(hv[k] - hv[base], 2) for k in fc if k != base}

    s_ex = "HDFCBANK" if "HDFCBANK" in syms else syms[0]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    idx = X.index
    test = idx >= pd.Timestamp(MEASURE_START)
    for k, c, lw in (
        ("rolling 1 year", plots.MUTED, 1.0),
        ("rolling 3 years", plots.TEXT, 1.4),
        ("EWMA (λ 0.99)", plots.SERIES[1], 1.2),
        ("Kalman (shared drift)", plots.SERIES[0], 1.8),
    ):
        ax.plot(idx[test], fc[k][s_ex][test], color=c, lw=lw, label=k)
    ax.set_ylabel(f"{s_ex}'s beta to NIFTY (forecast for the next day)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    v = pd.Series(fc["rolling 1 year"][s_ex][test], index=idx[test])
    ax.set_title(
        f"{s_ex}'s one-year beta ranged from {v.min():.2f} to {v.max():.2f} between 2014 and 2026"
    )
    plots.save(fig, out_dir, "fc_paths")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    names = [k for k in fc if k != base]
    for ax, rel, tt, title in (
        (axes[0], rel_y, t_y, "error predicting next year's beta"),
        (axes[1], rel_h, t_h, "risk left by next month's hedge"),
    ):
        vals = [(rel[k] - 1) * 100 for k in names]
        ax.barh(
            range(len(names)),
            vals,
            color=[plots.SERIES[0] if v < 0 else plots.SERIES[1] for v in vals],
        )
        ax.axvline(0, color=plots.TEXT_2, lw=0.8)
        ax.set_yticks(
            range(len(names)),
            [f"{k} (t {tt[k]:.1f})".replace("t -", "t −") for k in names],
            fontsize=8,
        )
        ax.set_xlabel("% change against the 1-year rolling beta (below 0 = better)")
        ax.set_title(title, fontsize=10, fontweight="normal")
        ax.grid(axis="y", visible=False)
    best = min(names, key=lambda k: rel_y[k])
    spread = max(abs(rel_h[k] - 1) for k in names)
    fig.suptitle(
        f"{best} cut the error on next year's beta by {1 - rel_y[best]:.0%}; hedging "
        f"results differed by under {spread:.1%}",
        fontweight="bold",
    )
    fig.tight_layout()
    plots.save(fig, out_dir, "fc_score")

    return {
        "stocks, months": (
            len(syms),
            len(ends),
            f"{X.index[ends[0]]:%b %Y} to {X.index[ends[-1]]:%b %Y}",
        ),
        "months with a full year ahead": int(np.isfinite(ey[base]).sum()),
        "stocks with own drift ratio > 1e-3": int(sum(v > 1e-3 for v in ratios.values())),
        "Kalman q/r ratio (2011-13): median, min, max": (
            float(np.median(list(ratios.values()))),
            float(min(ratios.values())),
            float(max(ratios.values())),
        ),
        "next-year beta MSE (absolute)": {k: round(float(np.nanmean(v)), 4) for k, v in ey.items()},
        "next-year beta MSE vs rolling 1y (ratio, NW t)": {
            k: (round(rel_y[k], 3), round(t_y.get(k, 0.0), 1)) for k in fc
        },
        "next-month hedged variance vs rolling 1y (ratio, NW t)": {
            k: (round(rel_h[k], 4), round(t_h.get(k, 0.0), 1)) for k in fc
        },
        "example stock": s_ex,
    }


def report_asym(out_dir: Path) -> dict:
    """Up and down betas, crash betas, and whether a stock's asymmetry repeats, 2014-26, daily
    excess returns against the NIFTY index."""
    from scipy.stats import norm

    from bc.asym import tail_beta, updown_beta
    from bc.capm import bh_reject, excess
    from bc.data import SYMBOLS

    R = returns().loc[MEASURE_START:]
    syms = [s for s in SYMBOLS if R[s].notna().mean() > 0.99]
    X = R[syms + [HEDGE_MARKET]].dropna()
    E = X.apply(lambda c: excess(c, RF)).iloc[1:]
    E["equal weight"] = excess(X[syms].mean(axis=1), RF).iloc[1:]
    x = E[HEDGE_MARKET].to_numpy()
    halves = {"2014–19": E.index < "2020-01-01", "2020–26": E.index >= "2020-01-01"}
    rows = {}
    for s in syms + ["equal weight"]:
        y = E[s].to_numpy()
        u = updown_beta(y, x)
        rows[s] = {
            "beta": float(np.polyfit(x, y, 1)[0]),
            "up": u["up"],
            "down": u["down"],
            "t": u["t_diff"],
            "tail": tail_beta(y, x, 0.05)["beta"],
        }
        for h, m in halves.items():
            uh = updown_beta(y[m], x[m])
            rows[s].update(
                {
                    f"up {h}": uh["up"],
                    f"down {h}": uh["down"],
                    f"diff {h}": uh["diff"],
                    f"tail {h}": tail_beta(y[m], x[m], 0.05)["beta"],
                }
            )
    D = pd.DataFrame(rows).T
    St = D.loc[syms]
    p = 2 * norm.sf(np.abs(St["t"].to_numpy(float)))
    bh = bh_reject(p)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0.4, 1.7], [0.4, 1.7], color=plots.TEXT_2, lw=0.8, ls="--", label="same both ways")
    ax.plot(
        St["up"][~bh],
        St["down"][~bh],
        "o",
        ms=6,
        color=plots.MUTED,
        label="difference not significant",
    )
    ax.plot(
        St["up"][bh],
        St["down"][bh],
        "o",
        ms=8,
        color=plots.SERIES[1],
        label="down > up after Benjamini-Hochberg",
    )
    for i, s in enumerate(St.loc[bh].sort_values("down", ascending=False).index):
        ax.annotate(
            s,
            (St.loc[s, "up"], St.loc[s, "down"]),
            xytext=(8, (6, -8, -3)[i % 3]),
            textcoords="offset points",
            fontsize=8,
            color=plots.TEXT,
        )
    ax.set_xlabel("up beta (days NIFTY rose)")
    ax.set_ylabel("down beta (days NIFTY fell)")
    ax.legend(loc="upper left", fontsize=8)
    above = int((St["down"] > St["up"]).sum())
    ax.set_title(f"{above} of {len(syms)} stocks fell harder with NIFTY than they rose with it")
    plots.save(fig, out_dir, "asym_updown")

    h1, h2 = (f"diff {h}" for h in halves)
    pc = float(St[h1].astype(float).corr(St[h2].astype(float)))
    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.axhline(0, color=plots.TEXT_2, lw=0.8)
    ax.axvline(0, color=plots.TEXT_2, lw=0.8)
    ax.plot(St[h1], St[h2], "o", ms=6, color=plots.SERIES[0])
    ax.set_xlabel("down beta − up beta, 2014–19")
    ax.set_ylabel("down beta − up beta, 2020–26")
    ax.set_title(f"A stock's own asymmetry barely repeated: correlation {pc:.2f} across halves")
    plots.save(fig, out_dir, "asym_persist")

    ew = D.loc["equal weight"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    xs = np.arange(2)
    for j, (k, c, lab) in enumerate(
        (
            ("up", plots.SERIES[0], "up beta"),
            ("down", plots.SERIES[1], "down beta"),
            ("tail", plots.TEXT, "crash beta (worst 5% of days)"),
        )
    ):
        vals = [float(ew[f"{k} {h}"]) for h in halves]
        ax.bar(xs - 0.27 + 0.27 * j, vals, 0.25, color=c, label=lab)
        for i, v in enumerate(vals):
            ax.text(i - 0.27 + 0.27 * j, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(xs, list(halves))
    ax.set_ylim(0.6, 1.25)
    ax.set_ylabel(f"beta of an equal-weight portfolio of the {len(syms)} stocks")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    ax.set_title("The equal-weight portfolio's down beta exceeded its up beta in both halves")
    plots.save(fig, out_dir, "asym_halves")

    return {
        "stocks, days": (len(syms), len(E), f"{E.index[0]:%d %b %Y} to {E.index[-1]:%d %b %Y}"),
        "stock averages: beta, up, down, tail": tuple(
            round(float(St[k].mean()), 3) for k in ("beta", "up", "down", "tail")
        ),
        "down > up (count)": above,
        "p < 0.05: down > up, up > down, expected by luck": (
            int(((p < 0.05) & (St["t"] > 0)).sum()),
            int(((p < 0.05) & (St["t"] < 0)).sum()),
            round(0.05 * len(syms), 1),
        ),
        "Benjamini-Hochberg survivors": list(St.index[bh]),
        "largest t (down − up)": {
            s: round(float(St.loc[s, "t"]), 2) for s in St["t"].astype(float).nlargest(3).index
        },
        "smallest t": {
            s: round(float(St.loc[s, "t"]), 2) for s in St["t"].astype(float).nsmallest(3).index
        },
        "tail - beta: mean, corr": (
            round(float((St["tail"] - St["beta"]).mean()), 3),
            round(float(St["tail"].astype(float).corr(St["beta"].astype(float))), 3),
        ),
        "diff by half: means": {h: round(float(St[f"diff {h}"].mean()), 3) for h in halves},
        "diff persistence corr": round(pc, 3),
        "equal weight (beta, up, down, t, tail)": tuple(
            round(float(ew[k]), 3) for k in ("beta", "up", "down", "t", "tail")
        ),
        "equal weight by half (up, down, tail)": {
            h: tuple(round(float(ew[f"{k} {h}"]), 3) for k in ("up", "down", "tail"))
            for h in halves
        },
    }


# Zerodha equity delivery charges (as used in correlation-heatmap), per rupee traded
DELIVERY = {
    "stt": 0.001,
    "stamp_buy": 0.00015,
    "exchange": 0.0000322,
    "sebi": 0.000001,
    "gst": 0.18,
    "slippage": 0.0005,
}


def _delivery_cost(turnover: float) -> float:
    """Cost of buying turnover/2 and selling turnover/2 of the portfolio (DP charges left out)."""
    d = DELIVERY
    side = (d["exchange"] + d["sebi"]) * (1 + d["gst"]) + d["slippage"] + d["stt"]
    return turnover * side + turnover / 2 * d["stamp_buy"]


def _monthly_panel(R: pd.DataFrame | None = None):
    """Blume-shrunk 1-year betas at each month end (daily excess returns to the NIFTY index, at
    least 200 of the last 250 days) and the next month's excess returns."""
    from bc.capm import excess
    from bc.data import SYMBOLS
    from bc.measure import resampled_returns

    R = (returns() if R is None else R).loc["2011-01-01":]
    E = R[SYMBOLS + [HEDGE_MARKET]].apply(lambda c: excess(c, RF))
    month = E.index.to_period("M")
    ends = np.flatnonzero(np.append(month[1:] != month[:-1], True))
    ends = ends[ends >= 249]
    B = {}
    x_all = E[HEDGE_MARKET].to_numpy()
    for t in ends:
        win = slice(t - 249, t + 1)
        x = x_all[win]
        row = {}
        for s in SYMBOLS:
            y = E[s].to_numpy()[win]
            ok = np.isfinite(y) & np.isfinite(x)
            if ok.sum() >= 200:
                row[s] = 0.67 * np.polyfit(x[ok], y[ok], 1)[0] + 0.33
        B[E.index[t].to_period("M")] = row
    B = pd.DataFrame(B).T
    M = resampled_returns(R[SYMBOLS + [MARKET, HEDGE_MARKET]], "ME")
    days = M.index.to_series().diff().dt.days
    M = M.sub((1 + RF) ** (days / 365) - 1, axis=0).iloc[1:]
    M.index = M.index.to_period("M")
    nxt = M.shift(-1)  # row t holds month t+1's excess return
    common = B.index.intersection(nxt.dropna(subset=[MARKET]).index)
    return B.loc[common], nxt.loc[common, SYMBOLS], nxt.loc[common, MARKET]


def report_pricing(out_dir: Path) -> dict:
    """Does beta pay? Fama-MacBeth, beta-sorted portfolios against the CAPM line, betting against
    beta, and a long-only low-beta portfolio after costs. Monthly; betas known at month start."""
    from bc.pricing import _nw_t_mean, beta_sorts, fama_macbeth

    B, Rn, mkt = _monthly_panel()
    fm = fama_macbeth(B, Rn, lags=3)
    prem = float(mkt.mean())

    srt = beta_sorts(B, Rn)
    Q, bab = srt["quintiles"], srt["bab"]
    turn, lo_turn, lo_ret = srt["bab_turnover"].iloc[1:], srt["low_turnover"], srt["low"]
    m = mkt.to_numpy()
    real_beta = {k: float(np.polyfit(m, Q[k], 1)[0]) for k in range(5)}
    ann = {k: float(Q[k].mean() * 12) for k in range(5)}

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    xs = np.array([0, 1.8])
    ax.plot(
        xs,
        xs * prem * 12 * 100,
        color=plots.TEXT_2,
        lw=1,
        ls="--",
        label="CAPM line: cash at beta 0, NIFTY ETF at beta 1",
    )
    ax.plot(
        [real_beta[k] for k in range(5)],
        [ann[k] * 100 for k in range(5)],
        "o-",
        ms=8,
        color=plots.SERIES[0],
        label="beta quintiles (1 = lowest beta)",
    )
    for k in range(5):
        ax.annotate(
            str(k + 1),
            (real_beta[k], ann[k] * 100),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=plots.TEXT,
        )
    ax.plot([1], [prem * 12 * 100], "s", ms=8, color=plots.TEXT, label="NIFTY ETF")
    ax.set_xlabel("realised beta of the quintile portfolio (monthly, against the NIFTY ETF)")
    ax.set_ylabel("average return above cash, % a year")
    ax.set_xlim(0, 1.8)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(
        f"Lowest-beta fifth earned {ann[0] * 100:.1f}% a year above cash; the other "
        f"four {min(ann[k] for k in range(1, 5)) * 100:.1f}–"
        f"{max(ann[k] for k in range(1, 5)) * 100:.1f}%"
    )
    plots.save(fig, out_dir, "price_sml")

    from bc.capm import ols

    o = ols(bab.to_numpy(), m, 3)
    bab_alpha = o["alpha"]
    resid = bab.to_numpy() - o["alpha"] - o["beta"] * m
    t_alpha = o["alpha"] / o["se_alpha_nw"]
    mean_turn = float(np.mean(turn))
    breakeven = bab_alpha / mean_turn  # per-side cost (per rupee traded) that zeroes the alpha
    lo = lo_ret
    lo_cost = lo_turn.map(_delivery_cost)
    lo_net = lo - lo_cost
    ew = pd.Series(
        [float(Rn.loc[t, B.loc[t].dropna().index].mean()) for t in B.index], index=B.index
    )  # all stocks, equal weight: the same survivors
    wealth = {
        "betting against beta (gross)": (1 + bab).cumprod(),
        "lowest-beta quintile, after costs": (1 + lo_net).cumprod(),
        "all stocks, equal weight (same survivors, no costs)": (1 + ew).cumprod(),
        "NIFTY ETF": (1 + mkt).cumprod(),
    }

    fig, ax = plt.subplots(figsize=(9, 4.8))
    idx = B.index.to_timestamp(how="end") + pd.offsets.MonthEnd(1)
    for (k, w), c in zip(
        wealth.items(), (plots.SERIES[1], plots.SERIES[0], plots.MUTED, plots.TEXT), strict=True
    ):
        ax.plot(idx, w.to_numpy(), color=c, lw=1.8, label=k)
    ax.set_yscale("log")
    from matplotlib.ticker import FuncFormatter, LogLocator

    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 3, 4, 6)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"₹{v:g}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_ylabel("₹1 grown above cash (log scale)")
    ax.legend(loc="upper left", fontsize=8)
    sh = lambda v: float(v.mean() / v.std() * np.sqrt(12))  # noqa: E731
    ax.set_title(
        f"Sharpe: betting against beta {sh(bab):.2f} gross, low-beta fifth {sh(lo_net):.2f} net, "
        f"all stocks {sh(ew):.2f}, NIFTY ETF {sh(mkt):.2f}"
    )
    plots.save(fig, out_dir, "price_bab")

    lo_o = ols(lo_net.to_numpy(), m, 3)
    ew_o = ols(ew.to_numpy(), m, 3)
    diff = lo_net - ew
    return {
        "months, stocks per month (min, max)": (
            len(B),
            int(B.notna().sum(axis=1).min()),
            int(B.notna().sum(axis=1).max()),
            f"{B.index[0] + 1} to {B.index[-1] + 1}",
        ),
        "market premium (ETF) % a year": round(prem * 12 * 100, 2),
        "Fama-MacBeth slope % a year, NW t": (
            round(fm["slope"] * 1200, 2),
            round(fm["t_slope"], 2),
        ),
        "Fama-MacBeth intercept % a year, NW t": (
            round(fm["intercept"] * 1200, 2),
            round(fm["t_intercept"], 2),
        ),
        "slope minus premium, t": (
            round((fm["slope"] - prem) * 1200, 2),
            round(_nw_t_mean((fm["slopes"] - mkt.loc[fm["slopes"].index]).to_numpy(), 3), 2),
        ),
        "quintiles: realised beta": [round(real_beta[k], 3) for k in range(5)],
        "quintiles: % a year above cash": [round(ann[k] * 100, 2) for k in range(5)],
        "BAB: mean % a year, vol % a year, Sharpe": (
            round(bab.mean() * 1200, 2),
            round(bab.std() * np.sqrt(12) * 100, 2),
            round(sh(bab), 2),
        ),
        "BAB: alpha % a year, NW t, market beta": (
            round(o["alpha"] * 1200, 2),
            round(t_alpha, 2),
            round(o["beta"], 3),
        ),
        "BAB: resid vol % a year": round(float(resid.std() * np.sqrt(12) * 100), 2),
        "BAB: mean monthly turnover (gross, both legs)": round(mean_turn, 3),
        "BAB: breakeven per-side cost (%)": round(breakeven * 100, 3),
        "low-beta quintile: net % a year, vol, Sharpe": (
            round(lo_net.mean() * 1200, 2),
            round(lo_net.std() * np.sqrt(12) * 100, 2),
            round(sh(lo_net), 2),
        ),
        "low-beta quintile: cost % a year, mean turnover": (
            round(lo_cost.mean() * 1200, 2),
            round(float(lo_turn.iloc[1:].mean()), 3),
        ),
        "low-beta quintile: alpha % a year, NW t, beta": (
            round(lo_o["alpha"] * 1200, 2),
            round(lo_o["alpha"] / lo_o["se_alpha_nw"], 2),
            round(lo_o["beta"], 3),
        ),
        "equal-weight all stocks: alpha % a year, NW t, beta": (
            round(ew_o["alpha"] * 1200, 2),
            round(ew_o["alpha"] / ew_o["se_alpha_nw"], 2),
            round(ew_o["beta"], 3),
        ),
        "low-beta net minus equal weight: % a year, NW t": (
            round(diff.mean() * 1200, 2),
            round(_nw_t_mean(diff.to_numpy(), 3), 2),
        ),
        "NIFTY ETF: vol % a year, Sharpe": (
            round(mkt.std() * np.sqrt(12) * 100, 2),
            round(sh(mkt), 2),
        ),
    }


REPORTS = {
    "capm": report_capm,
    "measure": report_measure,
    "forecast": report_forecast,
    "asym": report_asym,
    "pricing": report_pricing,
}

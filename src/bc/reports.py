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
    return R.loc[R.index[-1] - pd.DateOffset(years=years):]


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
        ax.plot(xs, r["alpha"] * 100 + r["beta"] * xs, color=c, lw=2,
                label=f"beta {r['beta']:.2f}, R² {r['r2']:.2f}")
        ax.plot(xs, xs, color=plots.TEXT_2, lw=0.8, ls="--", label="beta 1")
        ax.set_title(s, fontsize=10, fontweight="normal")
        ax.set_xlim(-8, 8)
        ax.set_ylim(-15, 15)
        ax.set_xlabel("NIFTY ETF excess return, % a day")
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("stock excess return, % a day")
    fig.suptitle(f"The lowest and highest betas in today's NIFTY 50, last 5 years: "
                 f"{t.loc[lo, 'beta']:.2f} and {t.loc[hi, 'beta']:.2f}", fontweight="bold")
    plots.save(fig, out_dir, "capm_scatter")

    fig, ax = plt.subplots(figsize=(9, 9))
    y = np.arange(len(t))
    ax.errorbar(t["beta"], y, xerr=1.96 * t["se_beta_nw"], fmt="o", ms=4, color=plots.SERIES[0],
                ecolor=plots.SERIES[0], elinewidth=1.2, capsize=0, label="Newey-West 95% range")
    ax.errorbar(t["beta"], y + 0.3, xerr=1.96 * t["se_beta"], fmt="none", ecolor=plots.MUTED,
                elinewidth=1, label="plain OLS 95% range")
    ax.axvline(1, color=plots.TEXT_2, lw=0.8, ls="--")
    ax.set_yticks(y, t.index, fontsize=7)
    ax.set_xlabel("beta to the NIFTY ETF (daily excess returns, last 5 years)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8)
    ratio = float((t["se_beta_nw"] / t["se_beta"]).median())
    ax.set_title(f"Betas ran from {t['beta'].min():.2f} to {t['beta'].max():.2f}; honest "
                 f"(Newey-West) ranges were {ratio - 1:.0%} wider than plain OLS")
    plots.save(fig, out_dir, "capm_betas")

    sig = t["p_alpha_nw"] < 0.05
    bh = bh_reject(t["p_alpha_nw"].to_numpy(), 0.05)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bins = np.arange(-4, 4.01, 0.5)
    ax.hist(t["t_alpha_nw"], bins=bins, color=plots.SERIES[0], alpha=0.85,
            label=f"{len(t)} stocks' alpha t-statistics")
    xs = np.linspace(-4, 4, 200)
    ax.plot(xs, norm.pdf(xs) * len(t) * 0.5, color=plots.SERIES[1], lw=2,
            label="what pure luck gives (no true alpha)")
    for v in (-1.96, 1.96):
        ax.axvline(v, color=plots.TEXT_2, lw=0.8, ls="--")
    ax.set_xlabel("alpha t-statistic (Newey-West)")
    ax.set_ylabel("number of stocks")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"{int(sig.sum())} of {len(t)} alphas looked significant, {0.05 * len(t):.1f} "
                 f"expected by luck; {int(bh.sum())} survived a multiple-testing correction")
    plots.save(fig, out_dir, "capm_alphas")

    return {
        "window": f"{W.index[0]:%d %b %Y} to {W.index[-1]:%d %b %Y}",
        "stocks": len(t),
        "beta: min (who), median, max (who)": (round(t["beta"].min(), 2), lo,
                                               round(t["beta"].median(), 2),
                                               round(t["beta"].max(), 2), hi),
        "Newey-West / OLS standard error, beta: median, max": (
            round(ratio, 2), round(float((t["se_beta_nw"] / t["se_beta"]).max()), 2)),
        "Newey-West / OLS standard error, alpha: median": round(
            float((t["se_alpha_nw"] / t["se_alpha"]).median()), 2),
        "R²: median, min, max": (round(t["r2"].median(), 2), round(t["r2"].min(), 2),
                                 round(t["r2"].max(), 2)),
        "alpha a year: mean, share positive": (round(t["alpha_annual"].mean(), 4),
                                               round(float((t["alpha_annual"] > 0).mean()), 2)),
        "significant at 5% (all positive?), expected, BH, Bonferroni": (
            int(sig.sum()), bool((t.loc[sig, "alpha_annual"] > 0).all()),
            round(0.05 * len(t), 1), int(bh.sum()),
            int((t["p_alpha_nw"] < 0.05 / len(t)).sum())),
        "significant ones (alpha, t)": {s: (round(r["alpha_annual"], 3), round(r["t_alpha_nw"], 2))
                                        for s, r in t[sig].iterrows()},
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
    freqs = {"daily": (X, 5), "weekly": (resampled_returns(X, "W-FRI"), 2),
             "monthly": (resampled_returns(X, "ME"), 1)}
    beta = {}
    for lab, (D, lags) in freqs.items():
        for mkt in (MARKET, "NIFTY"):
            ex_m = excess(D[mkt], RF).iloc[1:]
            beta[(lab, mkt)] = pd.Series({s: ols(excess(D[s], RF).iloc[1:].to_numpy(),
                                                 ex_m.to_numpy(), lags)["beta"] for s in syms})
    avg = {k: float(v.mean()) for k, v in beta.items()}

    fig, ax = plt.subplots(figsize=(9, 4.6))
    xs = np.arange(3)
    for j, (mkt, c, lab) in enumerate(((MARKET, plots.SERIES[0], "against the NIFTY ETF"),
                                       ("NIFTY", plots.SERIES[1], "against the NIFTY index"))):
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
    ax.set_title(f"Against the ETF, daily betas averaged {gd:.0%} higher than against the index; "
                 f"monthly, {gm:.0%}")
    plots.save(fig, out_dir, "meas_market")
    etf_d, idx_d = X[MARKET], X["NIFTY"]
    etf_vs_idx = {"daily vol ratio (ETF / index)": float(etf_d.std() / idx_d.std()),
                  "daily correlation": float(etf_d.corr(idx_d))}
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
    ax.set_title(f"Daily, weekly and monthly betas agreed: correlation {cw:.2f} and {cm:.2f} "
                 "across stocks")
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
    labels = {"plain OLS": plain, "Dimson (yesterday, today, tomorrow)": {s: v["beta"] for s, v in
                                                                         dim.items()},
              "Huber (robust)": hub, "without the 10 biggest market days": noext}
    vals = [float(np.mean(list(v.values()))) for v in labels.values()]
    ax.bar(range(len(vals)), vals, 0.55, color=[plots.TEXT, plots.SERIES[2], plots.SERIES[0],
                                                plots.MUTED])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(vals)), [k.replace(" (", "\n(") for k in labels], fontsize=8)
    ax.set_ylim(0.9, 1.1)
    ax.set_ylabel("average daily beta to the NIFTY ETF")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Lags and robust weights moved the average beta by at most "
                 f"{max(abs(v - vals[0]) for v in vals[1:]):.2f}")
    plots.save(fig, out_dir, "meas_robust")

    diffs = {k: pd.Series(v) - pd.Series(plain) for k, v in labels.items() if k != "plain OLS"}
    return {
        "stocks, days": (len(syms), len(X), f"{X.index[0]:%d %b %Y} to {X.index[-1]:%d %b %Y}"),
        "average beta (freq, market)": {f"{a} vs {b}": round(v, 3) for (a, b), v in avg.items()},
        "ETF / index gap: daily, weekly, monthly": tuple(
            round(avg[(f, MARKET)] / avg[(f, "NIFTY")] - 1, 3) for f in freqs),
        "ETF vs index": {k: round(v, 3) for k, v in etf_vs_idx.items()},
        "weekly - daily (ETF): mean, sd, min, max": tuple(round(x, 3) for x in (
            (beta[("weekly", MARKET)] - d).mean(), (beta[("weekly", MARKET)] - d).std(),
            (beta[("weekly", MARKET)] - d).min(), (beta[("weekly", MARKET)] - d).max())),
        "monthly - daily (ETF): mean, sd": (round((beta[("monthly", MARKET)] - d).mean(), 3),
                                            round((beta[("monthly", MARKET)] - d).std(), 3)),
        "correlation daily-weekly, daily-monthly": (round(cw, 3), round(cm, 3)),
        "Dimson average parts (yesterday, today, tomorrow)": parts.round(3).tolist(),
        "averages: plain, Dimson, Huber, no extremes": [round(v, 3) for v in vals],
        "largest per-stock change vs plain": {k: (round(float(v.abs().max()), 3), v.abs().idxmax())
                                              for k, v in diffs.items()},
    }


REPORTS = {"capm": report_capm, "measure": report_measure}

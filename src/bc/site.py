"""Interactive dashboard: a live beta calculator for the NIFTY 50, as one static page."""

import json
import re
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from bc import plots

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
ICON = (
    "data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 16 16%22%3E"
    "%3Cpath d=%22M1 14 15 2%22 stroke=%22%232a78d6%22 stroke-width=%222%22/%3E"
    "%3Ccircle cx=%225%22 cy=%229%22 r=%221.3%22 fill=%22%23a3a29c%22/%3E"
    "%3Ccircle cx=%2210%22 cy=%227%22 r=%221.3%22 fill=%22%23a3a29c%22/%3E%3C/svg%3E"
)
HERE = Path(__file__).parent
JS = (HERE / "beta.js").read_text(encoding="utf-8")
CALCULATOR = (HERE / "calculator.html").read_text(encoding="utf-8")
PAGE = Template((HERE / "page.html").read_text(encoding="utf-8"))
REPO = "https://github.com/Dhirajdas18/beta-capm"
DAYS = 1250  # five years of trading days embedded in the page
SCALE = 1e-5  # returns stored as integers in units of 0.001%


def latest_level() -> float:
    """Latest NIFTY index level, for sizing futures lots."""
    from bc.data import DATA_PATH

    return float(pd.read_parquet(DATA_PATH)["adj_close"]["NIFTY"].dropna().iloc[-1])


def calculator_data(r: pd.DataFrame) -> dict:
    from bc.data import SECTORS, SYMBOLS
    from bc.hedge import FUT
    from bc.reports import MARKET, RF

    symbols = sorted(SYMBOLS)
    recent = r[symbols + [MARKET, "NIFTY"]].iloc[-DAYS:]
    data = {
        s: [None if pd.isna(x) else int(round(x / SCALE)) for x in recent[s]]
        for s in symbols + [MARKET, "NIFTY"]
    }
    return {
        "symbols": symbols,
        "sectors": {s: SECTORS[s] for s in symbols},
        "market": MARKET,
        "rf": RF,
        "dates": list(recent.index.strftime("%Y-%m-%d")),
        "last": f"{recent.index[-1]:%d %b %Y}",
        "r": data,
        "fut": FUT,
        "nifty_level": round(latest_level(), 2),
    }


def calculator_section(data: pd.DataFrame) -> str:
    return CALCULATOR.replace("$js", JS).replace(
        "$data", json.dumps(calculator_data(data), separators=(",", ":"))
    )


def _pricing(data: pd.DataFrame) -> dict:
    from bc.pricing import beta_sorts
    from bc.reports import _delivery_cost, _monthly_panel

    B, Rn, mkt = _monthly_panel(data)
    srt = beta_sorts(B, Rn)
    ew = pd.Series([float(Rn.loc[t, B.loc[t].dropna().index].mean()) for t in B.index], B.index)
    lo_net = srt["low"] - srt["low_turnover"].map(_delivery_cost)
    return {
        "Q": srt["quintiles"],
        "bab": srt["bab"],
        "low": lo_net,
        "ew": ew,
        "mkt": mkt,
        "dates": list((B.index + 1).strftime("%Y-%m-%d")),
    }


LOG_TICKS = [0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 7, 10, 15, 20]


def _layout(fig: go.Figure, ytitle: str, **y) -> go.Figure:
    if y.get("type") == "log":  # label every tick in rupees; Plotly's minor log labels read "2"
        y = {**y, "tickvals": LOG_TICKS, "ticktext": [f"₹{v:g}" for v in LOG_TICKS]}
        y.pop("tickprefix", None)
    fig.update_layout(
        paper_bgcolor=plots.SURFACE,
        plot_bgcolor=plots.SURFACE,
        font=dict(family=FONT, color=plots.TEXT_2, size=13),
        margin=dict(l=56, r=16, t=56, b=40),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0),
        xaxis=dict(gridcolor=plots.GRID),
        yaxis=dict(title=ytitle, gridcolor=plots.GRID, **y),
    )
    return fig


def sml_figure(data: pd.DataFrame) -> go.Figure:
    """Average next-month return above cash of each beta fifth, against the NIFTY ETF's."""
    p = _pricing(data)
    m = p["mkt"].to_numpy()
    labels = ["1 lowest", "2", "3", "4", "5 highest"]
    real = [round(float(np.polyfit(m, p["Q"][k], 1)[0]), 2) for k in range(5)]
    fig = go.Figure()
    fig.add_bar(
        x=labels,
        y=[round(float(p["Q"][k].mean() * 1200), 2) for k in range(5)],
        marker_color=plots.SERIES[0],
        customdata=real,
        name="beta fifth",
        hovertemplate="fifth %{x}: %{y:.1f}% a year, realised beta %{customdata}<extra></extra>",
    )
    fig.add_scatter(
        x=labels,
        y=[round(float(p["mkt"].mean() * 1200), 2)] * 5,
        mode="lines",
        line=dict(color=plots.TEXT, width=1.5, dash="dash"),
        name="NIFTY ETF",
        hovertemplate="NIFTY ETF: %{y:.1f}% a year<extra></extra>",
    )
    return _layout(fig, "average return above cash, % a year", rangemode="tozero")


def bab_figure(data: pd.DataFrame) -> go.Figure:
    """₹1 grown above cash: betting against beta, the low-beta fifth, all stocks, NIFTY ETF."""
    p = _pricing(data)
    fig = go.Figure()
    for name, v, c in (
        ("betting against beta (gross)", p["bab"], plots.SERIES[1]),
        ("low-beta fifth, after costs", p["low"], plots.SERIES[0]),
        ("all stocks, equal weight", p["ew"], plots.MUTED),
        ("NIFTY ETF", p["mkt"], plots.TEXT),
    ):
        fig.add_scatter(
            x=p["dates"],
            y=(1 + v).cumprod().round(3).tolist(),
            mode="lines",
            name=name,
            line=dict(color=c, width=2),
            hovertemplate=f"{name}: ₹%{{y:.2f}}<extra></extra>",
        )
    fig.update_layout(hovermode="x unified")
    return _layout(fig, "₹1 grown above cash (log scale)", type="log", tickprefix="₹")


def factors_figure(data: pd.DataFrame) -> go.Figure:
    """₹1 in three long-short portfolios, before costs: momentum, low volatility, low beta."""
    from bc.factors import long_short
    from bc.reports import _factor_panel

    B, Rn, _, chars, _, _ = _factor_panel(data)
    dates = list((B.index + 1).strftime("%Y-%m-%d"))
    fig = go.Figure()
    for k, sign, name, c in (
        ("momentum", 1, "momentum: winners − losers", plots.SERIES[1]),
        ("volatility", -1, "low − high volatility", plots.SERIES[0]),
        ("beta", -1, "low − high beta", plots.TEXT),
    ):
        v = pd.Series([sign * long_short(chars[k].loc[t], Rn.loc[t]) for t in B.index], B.index)
        fig.add_scatter(
            x=dates,
            y=(1 + v).cumprod().round(3).tolist(),
            mode="lines",
            name=name,
            line=dict(color=c, width=2),
            hovertemplate=f"{name}: ₹%{{y:.2f}}<extra></extra>",
        )
    fig.update_layout(hovermode="x unified")
    return _layout(fig, "₹1 in each long-short (log scale)", type="log", tickprefix="₹")


SECTIONS = [
    (
        "Beta calculator",
        "Pick one of today’s NIFTY 50 stocks, a window and daily or weekly returns. The page "
        "regresses the stock’s return above cash (6.5% a year) on the NIFTY ETF’s: beta is how "
        "much the stock moves per 1% of NIFTY, alpha the average return NIFTY doesn’t explain, and "
        "R² the share of the stock’s ups and downs the market accounts for. Newey-West ranges "
        "allow for errors that cluster in time.",
        calculator_section,
    ),
    (
        "Does beta pay",
        "Each month, today’s NIFTY 50 stocks are sorted by their beta at the start of the month "
        "(Blume-shrunk, past year against the NIFTY index) and held for the month. CAPM says "
        "higher beta should earn more. Betting against beta (below) buys the low-beta half and "
        "sells the "
        "high-beta half, each scaled to beta 1. These are today’s members looking back, so every "
        "bar is flattered by survivorship: compare the fifths with each other, not with the ETF.",
        sml_figure,
    ),
    (
        "Betting against beta",
        "₹1 in betting against beta (before costs; shorting would need stock futures), in the "
        "lowest-beta fifth after delivery costs, in all the stocks equally weighted (the same "
        "survivors) and in the NIFTY ETF, all measured above cash.",
        bab_figure,
    ),
    (
        "Price factors",
        "Each month, today’s NIFTY 50 stocks are split into thirds by a measure known at the "
        "month start; each line is one third’s next-month return minus the other’s, before costs. "
        "Momentum: the past 12 months’ return, skipping the latest. Volatility: the past year’s "
        "daily swings. Beta: Blume-shrunk, past year. Survivorship can flatter volatile stocks "
        "most: one that crashed would have left the index, so it isn’t in this data.",
        factors_figure,
    ),
]


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def build_html(data: pd.DataFrame) -> str:
    parts, nav = [], []
    for title, blurb, make in SECTIONS:
        out = make(data)
        body = (
            out
            if isinstance(out, str)
            else out.to_html(
                full_html=False,
                include_plotlyjs=False,
                div_id=f"fig-{_slug(title)}",
                config={"displaylogo": False, "responsive": True},
            )
        )
        if not isinstance(out, str):  # give each chart an accessible name
            body = f'<div role="figure" aria-label="{title}: chart">\n{body}\n</div>'
        sid = _slug(title)
        nav.append(f'<li><a href="#{sid}">{title}</a></li>')
        parts.append(f'<section id="{sid}">\n<h2>{title}</h2>\n<p>{blurb}</p>\n{body}\n</section>')
    d = plots.DARK
    return PAGE.substitute(
        icon=ICON,
        font=FONT,
        repo=REPO,
        updated=f"{data.index[-1]:%d %b %Y}",
        surface=plots.SURFACE,
        text=plots.TEXT,
        text2=plots.TEXT_2,
        grid=plots.GRID,
        accent=plots.SERIES[0],
        muted=plots.MUTED,
        d_surface=d[plots.SURFACE],
        d_text=d[plots.TEXT],
        d_text2=d[plots.TEXT_2],
        d_grid=d[plots.GRID],
        d_accent=d[plots.SERIES[0]],
        d_muted="#6b6a64",
        dark=json.dumps({k.lower(): v for k, v in d.items()}),
        dark_lines=json.dumps({k.lower(): v for k, v in plots.DARK_LINES.items()}),
        nav="\n".join(nav),
        sections="\n".join(parts),
    )

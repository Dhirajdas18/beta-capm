"""Interactive dashboard: a live beta calculator for the NIFTY 50, as one static page."""

import json
import re
from pathlib import Path
from string import Template

import pandas as pd

from bc import plots

FONT = "system-ui, -apple-system, \"Segoe UI\", Roboto, sans-serif"
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


def calculator_data(r: pd.DataFrame) -> dict:
    from bc.data import SECTORS, SYMBOLS
    from bc.reports import MARKET, RF

    symbols = sorted(SYMBOLS)
    recent = r[symbols + [MARKET, "NIFTY"]].iloc[-DAYS:]
    data = {s: [None if pd.isna(x) else int(round(x / SCALE)) for x in recent[s]]
            for s in symbols + [MARKET, "NIFTY"]}
    return {"symbols": symbols, "sectors": {s: SECTORS[s] for s in symbols}, "market": MARKET,
            "rf": RF, "dates": list(recent.index.strftime("%Y-%m-%d")),
            "last": f"{recent.index[-1]:%d %b %Y}", "r": data}


def calculator_section(data: pd.DataFrame) -> str:
    return (CALCULATOR.replace("$js", JS)
            .replace("$data", json.dumps(calculator_data(data), separators=(",", ":"))))


SECTIONS = [
    ("Beta calculator",
     "Pick one of today’s NIFTY 50 stocks, a window and daily or weekly returns. The page "
     "regresses the stock’s return above cash (6.5% a year) on the NIFTY ETF’s: beta is how "
     "much the stock moves per 1% of NIFTY, alpha the average return NIFTY doesn’t explain, and "
     "R² the share of the stock’s ups and downs the market accounts for. Newey-West ranges allow "
     "for errors that cluster in time.",
     calculator_section),
]


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def build_html(data: pd.DataFrame) -> str:
    parts, nav = [], []
    for title, blurb, make in SECTIONS:
        out = make(data)
        body = out if isinstance(out, str) else out.to_html(
            full_html=False, include_plotlyjs=False, div_id=f"fig-{_slug(title)}",
            config={"displaylogo": False, "responsive": True})
        sid = _slug(title)
        nav.append(f'<li><a href="#{sid}">{title}</a></li>')
        parts.append(f'<section id="{sid}">\n<h2>{title}</h2>\n<p>{blurb}</p>\n{body}\n</section>')
    d = plots.DARK
    return PAGE.substitute(
        icon=ICON, font=FONT, repo=REPO, updated=f"{data.index[-1]:%d %b %Y}",
        surface=plots.SURFACE, text=plots.TEXT, text2=plots.TEXT_2, grid=plots.GRID,
        accent=plots.SERIES[0], muted=plots.MUTED, d_surface=d[plots.SURFACE], d_text=d[plots.TEXT],
        d_text2=d[plots.TEXT_2], d_grid=d[plots.GRID], d_accent=d[plots.SERIES[0]],
        d_muted="#6b6a64",
        dark=json.dumps({k.lower(): v for k, v in d.items()}),
        dark_lines=json.dumps({k.lower(): v for k, v in plots.DARK_LINES.items()}),
        nav="\n".join(nav), sections="\n".join(parts),
    )

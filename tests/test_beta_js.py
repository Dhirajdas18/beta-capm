import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bc.capm import excess, ols

JS = Path(__file__).resolve().parents[1] / "src" / "bc" / "beta.js"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def _node(expr: str):
    script = f"const m = require({json.dumps(str(JS))}); console.log(JSON.stringify({expr}));"
    out = subprocess.run(["node", "-"], input=script, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_browser_ols_matches_python():
    rng = np.random.default_rng(0)
    x = np.convolve(rng.normal(0, 0.01, 1009), np.ones(10) / 3, "valid")
    y = 0.0003 + 1.2 * x + np.convolve(rng.normal(0, 0.01, 1009), np.ones(10) / 3, "valid")
    js = _node(f"m.ols({json.dumps(y.tolist())}, {json.dumps(x.tolist())}, 5)")
    py = ols(y, x, lags=5)
    for k in ("alpha", "beta", "se_alpha", "se_beta", "se_alpha_nw", "se_beta_nw", "r2"):
        assert js[k] == pytest.approx(py[k], rel=1e-9), k


def test_browser_excess_uses_calendar_days():
    idx = pd.bdate_range("2024-01-01", periods=30)
    r = pd.Series(np.random.default_rng(1).normal(0, 0.01, 30), index=idx)
    js = _node(f"m.excess({json.dumps(r.tolist())}, {json.dumps(list(idx.strftime('%Y-%m-%d')))},"
               " 0.065)")
    assert np.array(js[1:]) == pytest.approx(excess(r, 0.065).to_numpy()[1:], rel=1e-9)


def test_browser_weekly_returns_match_pandas():
    idx = pd.bdate_range("2024-01-03", "2024-04-10")  # starts and ends mid-week
    r = pd.Series(np.random.default_rng(2).normal(0, 0.01, len(idx)), index=idx)
    px = (1 + r).cumprod()
    py = px.resample("W-FRI").last().pct_change().iloc[1:]
    py = py[py.index <= idx[-1]]  # unfinished last week left out
    js = _node(f"m.toWeekly({{a: {json.dumps(r.tolist())}}}, "
               f"{json.dumps(list(idx.strftime('%Y-%m-%d')))})")
    assert np.array(js["cols"]["a"]) == pytest.approx(py.to_numpy(), rel=1e-9)
    assert js["dates"] == list(py.index.strftime("%Y-%m-%d"))

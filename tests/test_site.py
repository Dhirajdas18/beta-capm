import json

import numpy as np
import pandas as pd

from bc.data import ETFS, SYMBOLS
from bc.site import build_html, calculator_data


def _returns():
    idx = pd.bdate_range("2020-01-01", periods=1300)
    r = pd.DataFrame(np.random.default_rng(0).normal(0, 0.01, (1300, len(SYMBOLS) + 3)),
                     index=idx, columns=SYMBOLS + ETFS + ["NIFTY"])
    r.iloc[:400, 0] = np.nan
    return r


def test_calculator_data_is_compact_and_keeps_gaps():
    d = calculator_data(_returns())
    assert len(d["dates"]) == 1250 and d["market"] == "NIFTYBEES" and d["rf"] == 0.065
    assert d["r"][SYMBOLS[0]][0] is None and isinstance(d["r"][SYMBOLS[0]][-1], int)
    assert "NIFTYBEES" in d["r"] and "NIFTY" in d["r"] and "NIFTYBEES" not in d["symbols"]
    assert json.loads(json.dumps(d)) == d


def test_page_builds():
    html = build_html(_returns())
    assert "function ols" in html and 'id="beta-calculator"' in html
    assert "$" + "js" not in html and "$" + "data" not in html


def test_page_has_the_beta_pays_section_from_the_same_returns():
    html = build_html(_returns())
    assert "fig-does-beta-pay" in html and "fig-betting-against-beta" in html

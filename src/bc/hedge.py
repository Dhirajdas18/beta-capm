"""Hedging a stock portfolio with NIFTY futures, sold at each month start and rolled monthly.

The futures' return is approximated by the NIFTY ETF's return above cash: a fairly priced future
moves with the index plus the carry (cash rate − dividend yield) built into its price, which is
what the ETF's dividend-inclusive return above cash gives. Basis risk (the future drifting from
its fair price) is left out.

FUT holds Zerodha's equity-futures charges as listed on zerodha.com/charges in November 2026,
applied to every year (futures STT was lower before 2024, so earlier years are charged more than
they paid). Slippage is an assumption: half a typical bid-ask spread per side.
"""

import numpy as np
import pandas as pd

FUT = {
    "brokerage_rate": 0.0003,
    "brokerage_cap": 20.0,
    "stt_sell": 0.0005,
    "exchange": 0.0000183,
    "sebi": 0.000001,
    "stamp_buy": 0.00002,
    "gst": 0.18,
    "slippage": 0.00005,
    "lot": 65,
}


def futures_cost(buy: float, sell: float, orders: int) -> float:
    """₹ cost of buying `buy` and selling `sell` of futures notional in `orders` orders."""
    if orders == 0:
        return 0.0
    per_order = (buy + sell) / orders
    brokerage = orders * min(FUT["brokerage_cap"], FUT["brokerage_rate"] * per_order)
    fees = (buy + sell) * (FUT["exchange"] + FUT["sebi"])
    return (
        (brokerage + fees) * (1 + FUT["gst"])
        + sell * FUT["stt_sell"]
        + buy * FUT["stamp_buy"]
        + (buy + sell) * FUT["slippage"]
    )


def hedge_lots(value: float, beta: float, lot_value: float, whole: bool = True) -> float:
    n = value * beta / lot_value
    return float(round(n)) if whole else n


def run_hedge(
    port: pd.Series,
    fut: pd.Series,
    beta: pd.Series,
    lot_value: pd.Series,
    capital: float,
    whole: bool = True,
) -> pd.DataFrame:
    """Each month: hold `capital` of the portfolio, sell beta × capital of futures (in whole lots
    unless whole=False), roll at month end. Returns are on `capital`; costs include closing last
    month's position and opening this month's."""
    rows, prev = [], 0.0
    for t in port.index:
        n = hedge_lots(capital, beta.loc[t], lot_value.loc[t], whole)
        notional = n * lot_value.loc[t]
        prev_notional = prev * lot_value.loc[t]  # rolled at this month's start price
        cost = futures_cost(buy=prev_notional, sell=notional, orders=int(prev > 0) + int(n > 0))
        rows.append(
            {
                "lots": n,
                "hedge_ratio": notional / capital,
                "hedged": port.loc[t] - notional / capital * fut.loc[t] - cost / capital,
                "cost": cost / capital,
            }
        )
        prev = n
    out = pd.DataFrame(rows, index=port.index)
    out["unhedged"] = port
    return out


def variance_removed(out: pd.DataFrame) -> float:
    return float(1 - np.var(out["hedged"]) / np.var(out["unhedged"]))

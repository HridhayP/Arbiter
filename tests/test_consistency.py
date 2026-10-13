import json

import numpy as np
import pandas as pd
import pytest

from arbiter.research import consistency as R

T = pd.Timestamp("2026-10-13T20:00:00Z")


def kalshi_row(mid, ev, bid, ask, strike=None, strike_type=None, legs=None, size=100.0):
    return {"ts": T, "market_id": mid, "event_id": ev, "yes_bid": bid, "yes_ask": ask, "bid_size": size,
            "ask_size": size, "last": (bid + ask) / 2, "strike": strike, "strike_type": strike_type,
            "legs": json.dumps(legs) if legs else None}


def test_ladder_flags_only_inversions_that_survive_fees():
    k = pd.DataFrame([
        kalshi_row("F-T4.00", "F", 0.80, 0.82, 4.00, "greater"),
        kalshi_row("F-T4.25", "F", 0.60, 0.62, 4.25, "greater"),
        kalshi_row("F-T4.50", "F", 0.90, 0.92, 4.50, "greater"),  # bid 0.90 > ask 0.62 below it: inverted
        kalshi_row("G-T1", "G", 0.50, 0.52, 1.0, "greater"),
        kalshi_row("G-T2", "G", 0.525, 0.53, 2.0, "greater"),  # 0.5 cent inversion, eaten by fees
    ])
    v = R.ladder_violations(k)
    f = v[v.event_id == "F"].iloc[0]
    assert (f.buy, f.sell) == ("F-T4.25", "F-T4.50")
    assert f.gross == pytest.approx(0.28) and f.net > 0.25
    g = v[v.event_id == "G"].iloc[0]
    assert g.gross > 0 and g.net < 0  # flagged, but not profitable after fees


def test_exclusive_sets_sell_all_and_buy_all():
    rows = [kalshi_row(f"E-{i}", "E", b, a) for i, (b, a) in enumerate([(0.40, 0.42), (0.35, 0.37), (0.30, 0.32)])]
    rows += [kalshi_row(f"H-{i}", "H", b, a) for i, (b, a) in enumerate([(0.30, 0.31), (0.30, 0.31), (0.30, 0.31)])]
    k = pd.DataFrame(rows).assign(grp=lambda d: d.event_id)
    s = R.exclusive_sets(k, "grp", lambda r, p: 0.0).set_index("group")
    assert s.loc["E", "sum_bids"] == pytest.approx(1.05) and s.loc["E", "sell_all_net"] == pytest.approx(0.05)
    assert s.loc["H", "sum_asks"] == pytest.approx(0.93) and s.loc["H", "buy_all_net_if_exhaustive"] == pytest.approx(0.07)
    assert s.loc["H", "sell_all_net"] < 0


def test_combo_premium_uses_leg_sides():
    legs = {"markets": "A,B", "sides": "yes,no"}
    k = pd.DataFrame([
        kalshi_row("A", "EA", 0.59, 0.61),           # P(A yes) ~ 0.60
        kalshi_row("B", "EB", 0.29, 0.31),           # P(B no)  ~ 0.70
        kalshi_row("KXMVE-1", "MVE", 0.38, 0.45, legs=legs),
    ])
    c = R.combo_premium(k).iloc[0]
    assert c.independent_price == pytest.approx(0.42)
    assert c.ask_minus_indep == pytest.approx(0.03)


def test_ladders_never_mix_different_quantities():
    # Same event and strike, different players: not a ladder.
    k = pd.DataFrame([kalshi_row("NHL-G1-PLAYERA-1", "NHL-G1", 0.40, 0.41, 0.5, "greater"),
                      kalshi_row("NHL-G1-PLAYERB-2", "NHL-G1", 0.60, 0.61, 1.5, "greater"),
                      kalshi_row("RAIN-D-TTN", "RAIN-D", 0.90, 0.91, 0.0, "greater"),
                      kalshi_row("RAIN-D-TPA", "RAIN-D", 0.10, 0.11, 0.0, "greater")])
    assert R.ladder_violations(k).empty
    assert R.ladder_key("KXFED-26OCT-T4.00") == "KXFED-26OCT-T"
    assert R.ladder_key("RAIN-D-TTN") is None


def test_unquoted_rows_are_not_used_as_prices():
    k = pd.DataFrame([kalshi_row("X-T1", "X", 0.0, 1.0, 1.0, "greater"),
                      kalshi_row("X-T2", "X", 0.5, 0.52, 2.0, "greater")])
    assert R.ladder_violations(k).empty

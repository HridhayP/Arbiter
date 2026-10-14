import pandas as pd
import pytest

from arbiter.research import crossvenue as X

T = pd.Timestamp("2026-10-13T20:00:00Z")


def frames(k_quotes, p_quotes, poly_fees=False, skew=pd.Timedelta(minutes=5)):
    pairs = pd.DataFrame({"kalshi_id": ["K1"], "poly_id": ["P1"], "p_match": [0.9], "kalshi_text": ["x"], "poly_text": ["x"]})
    k = pd.DataFrame([{"ts": T + pd.Timedelta(minutes=20 * i), "market_id": "K1", "yes_bid": b, "yes_ask": a,
                       "bid_size": 100.0, "ask_size": 100.0, "legs": None} for i, (b, a) in enumerate(k_quotes)])
    p = pd.DataFrame([{"ts": T + pd.Timedelta(minutes=20 * i) + skew, "market_id": "P1", "yes_bid": b, "yes_ask": a,
                       "fees_enabled": poly_fees} for i, (b, a) in enumerate(p_quotes)])
    return pairs, k, p


def test_executable_gap_needs_bid_above_other_ask_after_fees():
    # snapshot 0: no overlap; 1: Polymarket bid 0.62 > Kalshi ask 0.55 (7c gross); 2: 1c overlap
    pairs, k, p = frames([(0.50, 0.52), (0.53, 0.55), (0.53, 0.55)], [(0.51, 0.53), (0.62, 0.64), (0.56, 0.58)])
    g = X.gaps(pairs, k, p).sort_values("ts_k")
    assert g.gross_best.round(4).tolist() == [-0.01, 0.07, 0.01]
    assert g.net_best.iloc[1] == pytest.approx(0.07 - 0.07 * 0.55 * 0.45 - 0.0)  # Kalshi taker fee only
    assert g.net_best.iloc[2] < 0  # 1 cent is eaten by the Kalshi fee
    pers = X.persistence(g)
    assert pers.snapshots_positive.tolist() == [1]


def test_polymarket_fees_are_charged_when_enabled():
    pairs, k, p = frames([(0.53, 0.55)], [(0.62, 0.64)], poly_fees=True)
    g = X.gaps(pairs, k, p)
    expected = 0.07 - 0.07 * 0.55 * 0.45 - X.POLY_WORST_RATE * 0.38 * 0.62  # NO on Polymarket at 1 - 0.62
    assert g.net_best.iloc[0] == pytest.approx(expected)


def test_snapshots_too_far_apart_are_not_joined():
    pairs, k, p = frames([(0.53, 0.55)], [(0.62, 0.64)], skew=pd.Timedelta(hours=2))
    assert X.gaps(pairs, k, p).empty

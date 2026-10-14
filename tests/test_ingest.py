import json

import pandas as pd
import pytest

from arbiter import store
from arbiter.venues import kalshi as K
from arbiter.venues import polymarket as P

LIVE = {"ticker": "KXFED-26SEP-T5.25", "event_ticker": "KXFED-26SEP", "result": "no",
        "open_time": "2025-08-06T14:30:00Z", "close_time": "2026-09-16T17:55:00Z",
        "settlement_ts": "2026-09-16T18:10:00Z", "floor_strike": 5.25, "volume_fp": "608.51",
        "last_price_dollars": "0.0100", "rules_primary": "If the rate is above 5.25%..."}
HIST = {"ticker": "KXFED-26JUN-T5.25", "event_ticker": "KXFED-26JUN", "result": "no",
        "close_time": "2026-06-17T18:15:00Z", "volume": "1200", "last_price": "0.0200"}


def test_kalshi_market_row_live_and_historical_fields():
    meta = {"category": "Economics", "fee_type": "quadratic_with_maker_fees", "fee_multiplier": 1}
    a = K.market_row(LIVE, meta)
    b = K.market_row(HIST, meta)
    assert a["series_id"] == "KXFED" and a["volume"] == 608.51 and a["last_price"] == 0.01
    assert a["close_time"] == pd.Timestamp("2026-09-16T17:55:00Z")
    assert b["volume"] == 1200.0 and b["last_price"] == 0.02  # "/historical" field names
    assert a["fee_type"] == "quadratic_with_maker_fees" and a["category"] == "Economics"


def test_kalshi_candles_both_formats():
    live = {"end_period_ts": 1791924360, "yes_bid": {"close_dollars": "0.6800"},
            "yes_ask": {"close_dollars": "0.6900"}, "price": {"close_dollars": "0.6900"}, "volume_fp": "683.39"}
    hist = {"end_period_ts": 1781139600, "yes_bid": {"close": "0.0000"}, "yes_ask": {"close": "0.0100"},
            "price": {"close": None}, "volume": "0.00", "open_interest": "766.00"}
    r1, r2 = K.candle_rows("X", [live, hist])
    assert (r1["yes_bid"], r1["yes_ask"], r1["last"], r1["volume"]) == (0.68, 0.69, 0.69, 683.39)
    assert (r2["yes_bid"], r2["yes_ask"], r2["last"], r2["open_interest"]) == (0.0, 0.01, None, 766.0)


def test_combo_detection():
    assert K.is_combo("KXMVECROSSCATEGORY-S2026...") and not K.is_combo("KXFED-26SEP-T5.25")


def _pm(prices, closed=True, outcomes=("Yes", "No")):
    return {"conditionId": "0xabc", "question": "Will X happen?", "outcomes": json.dumps(list(outcomes)),
            "outcomePrices": json.dumps(prices), "clobTokenIds": json.dumps(["111", "222"]), "closed": closed,
            "endDate": "2025-01-01T00:00:00Z", "volumeNum": 5000, "feesEnabled": True,
            "feeSchedule": {"rate": 0.05, "exponent": 1}}


@pytest.mark.parametrize("prices,result", [(["1", "0"], "yes"), (["0", "1"], "no"), (["0.5", "0.5"], "void"),
                                           (["0.9995", "0.0005"], "yes")])
def test_polymarket_resolution(prices, result):
    r = P.market_row(_pm(prices))
    assert r["result"] == result and r["token_yes"] == "111" and r["fee_multiplier"] == 0.05


def test_polymarket_skips_non_binary():
    m = _pm(["1", "0", "0"], outcomes=("A", "B", "C"))
    m["clobTokenIds"] = json.dumps(["1", "2", "3"])
    assert P.market_row(m) is None


class FakeGamma:
    """Serves `n` markets spread over a window and rejects deep offsets, like the real API."""

    def __init__(self, ends):
        self.ends = sorted(ends)
        self.calls = 0

    def get_json(self, url, params):
        self.calls += 1
        if params["offset"] > 2000:
            raise AssertionError("pager asked for an offset the API rejects")
        lo, hi = pd.Timestamp(params["end_date_min"]), pd.Timestamp(params["end_date_max"])
        hits = [e for e in self.ends if lo <= e < hi]
        page = hits[params["offset"]: params["offset"] + 100]
        return [{"id": str(e.value), "endDate": e.isoformat()} for e in page]


def test_polymarket_window_pager_splits_dense_windows():
    # 5,000 markets inside one week, far past the offset cap
    base = pd.Timestamp("2025-01-01", tz="UTC")
    ends = [base + pd.Timedelta(seconds=120 * i) for i in range(5000)]
    api = P.Polymarket(FakeGamma(ends))
    got = [m for page in api.closed_market_pages(0, start="2025-01-01", end="2025-01-08") for m in page]
    assert len(got) == 5000 and len({m["id"] for m in got}) == 5000


def test_state_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "ROOT", str(tmp_path))
    s = store.State("job", every=2)
    s.add("a")
    s.add("b")  # flushed
    s.add("c")  # pending
    s.flush()
    assert set(store.State("job").done) == {"a", "b", "c"}


def test_polymarket_window_pager_never_repeats_edge_markets():
    # markets exactly on window boundaries are returned by both windows
    base = pd.Timestamp("2025-01-01", tz="UTC")
    ends = [base + pd.Timedelta(days=d) for d in range(0, 15)]
    api = P.Polymarket(FakeGamma(ends))
    # FakeGamma uses half-open windows; make it inclusive like the real API
    api.c.get_json = lambda url, params, f=api.c.get_json: [
        {"id": str(e.value), "endDate": e.isoformat()} for e in ends
        if pd.Timestamp(params["end_date_min"]) <= e <= pd.Timestamp(params["end_date_max"])][params["offset"]:params["offset"] + 100]
    got = [m["id"] for page in api.closed_market_pages(0, start="2025-01-01", end="2025-01-15") for m in page]
    assert len(got) == len(set(got)) == 15

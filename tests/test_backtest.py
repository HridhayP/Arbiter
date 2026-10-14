import numpy as np
import pandas as pd
import pytest

pytest.importorskip("arbiter._sim")

from arbiter import backtest as B  # noqa: E402
from arbiter.fees import kalshi_fees  # noqa: E402

T0 = pd.Timestamp("2026-10-01", tz="UTC")


def candles(mids, spread=0.02):
    ts = [T0 + pd.Timedelta(minutes=i) for i in range(len(mids))]
    return pd.DataFrame({"ts": ts, "yes_bid": np.round(np.array(mids) - spread / 2, 2),
                         "yes_ask": np.round(np.array(mids) + spread / 2, 2)})


def test_ns_is_resolution_independent():
    assert B._ns(pd.Timestamp("1970-01-01 00:00:01", tz="UTC")) == 1_000_000_000
    assert B._ns(pd.Timestamp("1970-01-01 00:00:01").as_unit("us")) == 1_000_000_000


def test_candle_events_skip_one_sided_and_order_clears_first():
    c = candles([0.5, 0.6])
    c.loc[1, "yes_ask"] = np.nan
    ev = B.events_from_candles(c)
    assert len(ev) == 3 and ev.type.tolist() == [B.CLEAR, B.SET, B.SET]


def test_kalshi_recording_maps_no_bids_to_yes_asks():
    book = pd.DataFrame({"ts": [T0] * 2, "seq": [1, 1], "side": ["yes", "no"], "price": [0.40, 0.55],
                         "size": [10.0, 20.0], "level": [0, 0]})
    trades = pd.DataFrame({"ts": [T0 + pd.Timedelta(seconds=1)], "price": [0.45], "size": [3.0], "taker_side": ["yes"]})
    ev = B.events_from_kalshi_recording(book, trades)
    sets = [(int(s), int(t)) for ty, s, t in zip(ev.type, ev.side, ev.tick) if ty == B.SET]
    assert sets == [(B.BID, 400), (B.ASK, 450)]  # NO bid 0.55 = YES ask 0.45
    assert ev.type[-1] == B.TRADE and ev.side[-1] == B.ASK  # YES buyer lifted the offer


def test_touch_quotes_never_fill_on_snapshots_alone_but_do_with_trades():
    """Top-of-book snapshots without trade prints can't reveal fills at the
    touch (the conservative cross rule needs the far side to move through
    us); trade prints at the touch, beyond the queue ahead, do fill."""
    rng = np.random.default_rng(0)
    mids = np.clip(0.5 + 0.03 * np.sin(np.arange(600) / 15) + rng.normal(0, 0.005, 600), 0.1, 0.9).round(2)
    c = candles(mids, spread=0.04)
    params = {"gamma": 0.1, "k": 20.0, "quote_size": 5, "tick_size": 10}
    close = T0 + pd.Timedelta(days=5)
    snap_only = B.run_market_maker(B.events_from_candles(c, nominal_size=50), kalshi_fees("quadratic", 1), True, close,
                                   params, fill_on_cross=True)
    assert snap_only.fills == 0
    rows = []
    for r in c.itertuples():
        t = B._ns(r.ts)
        rows += [(t, B.CLEAR, 0, 0, 0.0), (t, B.SET, B.BID, B._tick(r.yes_bid), 50.0), (t, B.SET, B.ASK, B._tick(r.yes_ask), 50.0)]
        # 30 seconds later, 80 contracts trade at each touch: the 50 ahead of us, then 30 more
        rows += [(t + 30 * B.NS, B.TRADE, B.BID, B._tick(r.yes_bid), 80.0), (t + 30 * B.NS, B.TRADE, B.ASK, B._tick(r.yes_ask), 80.0)]
    r = B.run_market_maker(B.EventStream.build(rows), kalshi_fees("quadratic", 1), True, close, params,
                           fill_on_cross=False, max_position=50)
    assert r.fills > 100 and r.maker_share == 1.0
    assert r.max_abs_inventory <= 50


def test_walk_forward_only_uses_past_folds():
    rng = np.random.default_rng(1)
    ms = []
    for i in range(10):
        mids = np.clip(0.5 + np.cumsum(rng.normal(0, 0.01, 600)), 0.05, 0.95).round(2)
        ms.append({"market_id": f"m{i}", "events": B.events_from_candles(candles(mids)), "fees": kalshi_fees("quadratic", 1),
                   "resolved_yes": bool(i % 2), "close_ts": T0 + pd.Timedelta(days=i + 1)})
    grid = [{"gamma": 0.05, "k": 2.0}, {"gamma": 0.5, "k": 2.0}]
    wf = B.walk_forward(ms, grid, n_folds=5)
    assert set(wf.fold) == {1, 2, 3, 4}  # fold 0 is training only
    assert wf.market_id.tolist() == [f"m{i}" for i in range(2, 10)]

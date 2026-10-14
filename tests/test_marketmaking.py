import pandas as pd
import pytest

from arbiter.backtest import ASK, BID, SET, TRADE, EventStream
from arbiter.fees import FeeModel
from arbiter.research import marketmaking as R

FREE = FeeModel("polymarket", 0.0, 0.0, 1.0)


def test_tick_is_inferred_from_displayed_prices():
    assert R.infer_tick(pd.DataFrame({"price": [0.45, 0.46, 0.5]})) == 10
    assert R.infer_tick(pd.DataFrame({"price": [0.45, 0.001, 0.5]})) == 1


def test_walk_forward_only_uses_earlier_blocks():
    # grid 0 wins block 0, grid 1 wins blocks 1 and 2: block 1 must still trade grid 0
    r = pd.DataFrame([
        {"token": "t", "block": b, "grid": g, "pnl": p, "fills": 1, "volume": 10.0, "fees": 0.0,
         "maker_share": 1.0, "markout_1m": 0.0, "markout_5m": 0.0}
        for b, g, p in [(0, 0, 5.0), (0, 1, -5.0), (1, 0, -1.0), (1, 1, 3.0), (2, 0, -1.0), (2, 1, 3.0)]])
    out = R.walk_forward(r)
    assert out["grid_chosen_by_block"] == {1: 0, 2: 0}  # through block 1, grid 0 still leads (4 vs -2)
    assert out["total_pnl"] == pytest.approx(-2.0)


def test_block_closes_out_at_the_touch_not_the_mid():
    # 0.40 / 0.60 book; a taker sells into our bid, then nothing else happens
    rows = [(1, SET, BID, 400, 100.0), (1, SET, ASK, 600, 100.0)]
    rows += [(2 + i, SET, BID, 400, 100.0) for i in range(3)]  # book events let the quoter place orders
    rows += [(10, TRADE, BID, 410, 1000.0), (11, SET, BID, 400, 100.0)]
    ev = EventStream.build(rows)
    params = R.GRID[0] | {"k": 1e6, "gamma": 0.01, "tick_size": 10, "min_spread_ticks": 1}
    r = R.run_block(ev, params, FREE, queue_ahead=1.0)
    # our bid sits inside the 0.40/0.60 spread, so the sell sweeping to 0.41 fills it first
    assert r["fills"] >= 1 and r["max_inv"] > 0
    # bought near 0.49, closed at the 0.40 bid: a loss, where marking at the 0.50 mid would show a gain
    assert r["pnl"] < 0


def test_paper_trader_turns_live_messages_into_events():
    from arbiter import papertrade as P

    book = P.to_rows({"event_type": "book", "asset_id": "t", "bids": [{"price": "0.45", "size": "100"}],
                      "asks": [{"price": "0.47", "size": "50"}]})
    assert [r[1:] for r in book] == [(P.CLEAR, 0, 0, 0.0), (P.SET, P.BID, 450, 100.0), (P.SET, P.ASK, 470, 50.0)]
    ch = P.to_rows({"event_type": "price_change", "price_changes": [{"asset_id": "t", "side": "SELL", "price": "0.47", "size": "0"}]})
    assert [r[1:] for r in ch] == [(P.SET, P.ASK, 470, 0.0)]
    tr = P.to_rows({"event_type": "last_trade_price", "asset_id": "t", "side": "BUY", "price": "0.47", "size": "5"})
    assert [r[1:] for r in tr] == [(P.TRADE, P.ASK, 470, 5.0)]
    # a book fed message by message keeps state across calls
    b = P.Book({"token": "t", "question": "q", "fees_enabled": False, "tick": 10}, R.GRID[0], 1.0)
    b.feed(book)
    assert b.sim.book.best_bid == 450 and b.sim.book.best_ask == 470
    b.feed(ch)
    assert b.sim.book.best_ask > 470


def test_pnl_decomposition_adds_up():
    rows = [(1, SET, BID, 400, 100.0), (1, SET, ASK, 600, 100.0)]
    rows += [(2 + i, SET, BID, 400, 100.0) for i in range(3)]
    rows += [(10, TRADE, BID, 410, 1000.0), (11, SET, BID, 380, 100.0), (12, SET, ASK, 560, 100.0)]
    ev = EventStream.build(rows)
    params = R.GRID[0] | {"k": 1e6, "gamma": 0.01, "tick_size": 10, "min_spread_ticks": 1}
    r = R.run_block(ev, params, FeeModel("polymarket", 0.07, 0.0, 1.0), queue_ahead=1.0)
    assert r["fills"] >= 1
    assert r["pnl"] == pytest.approx(r["edge"] + r["drift"] + r["closeout"] - r["fees"])

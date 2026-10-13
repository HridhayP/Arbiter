import numpy as np
import pytest

sim = pytest.importorskip("arbiter._sim")


def events(rows):
    ts, ty, sd, tk, sz = zip(*rows)
    return (np.array(ts, np.int64), np.array(ty, np.uint8), np.array(sd, np.uint8), np.array(tk, np.int16),
            np.array(sz, np.float64))


class JoinBid(sim.Strategy):
    """Places one 10-lot bid at the best bid on the first two-sided book."""

    def __init__(self):
        super().__init__()
        self.placed = False
        self.fills = []

    def on_event(self, s, ev):
        if not self.placed and s.book.two_sided:
            s.place(sim.Side.bid, s.book.best_bid, 10)
            self.placed = True

    def on_fill(self, s, f):
        self.fills.append((f.qty, f.price, f.maker))


def test_python_strategy_gets_queue_aware_fills():
    fees = sim.FeeModel(sim.Venue.kalshi, 0.07, 0.0175)
    s = sim.Simulator(sim.SimConfig(fees))
    strat = JoinBid()
    BID, ASK, SET, TRADE = 0, 1, 0, 2
    s.run(*events([(1, SET, BID, 500, 100), (1, SET, ASK, 520, 100),
                   (2, TRADE, BID, 500, 105)]), strategy=strat)  # 100 ahead, 5 to us
    assert strat.fills == [(5.0, 0.5, True)]
    assert s.position == 5 and s.events_processed == 3
    assert s.settle(True) == pytest.approx(5 * 0.5 - 0.03)  # maker fee 0.0175*5*0.25 = 0.0219, rounded up to 0.03


def test_builtin_market_maker_runs_from_numpy():
    rng = np.random.default_rng(0)
    n = 20000
    mid = 500 + np.cumsum(rng.choice([-10, 0, 0, 0, 10], n)).clip(-400, 400)
    side = rng.integers(0, 2, n).astype(np.uint8)
    tick = np.where(side == 0, mid - 10, mid + 10).astype(np.int16)
    ty = np.where(rng.random(n) < 0.1, 2, 0).astype(np.uint8)
    ts = (np.arange(n, dtype=np.int64) + 1) * 1_000_000_000
    p = sim.MMParams()
    p.close_ts = int(ts[-1] + 10 * 3600 * 1_000_000_000)
    mm = sim.LogitMarketMaker(p)
    s = sim.Simulator(sim.SimConfig(sim.FeeModel(sim.Venue.kalshi, 0.07, 0.0), max_position=200))
    s.run(ts, ty, side, tick, rng.uniform(10, 200, n), strategy=mm)
    assert s.events_processed == n and mm.quotes_placed > 0 and len(s.fills) > 0
    assert abs(s.position) <= 200

"""R4: market-making backtests on recorded and historical data.

Turns market data into simulator events, runs the C++ LogitMarketMaker, and
measures what a market maker cares about:

- P&L after fees and settlement, per market and in aggregate
- volume traded, maker/taker share, fees
- inventory (max and time-weighted absolute)
- markouts: for each fill, the mid move against us 1 and 5 minutes later,
  which separates spread capture from adverse selection

Data sources
- Kalshi 1-minute candles (historical, top of book only, no trades): each
  minute becomes a book snapshot at the candle's closing bid/ask with a
  nominal size, and fills use the simulator's conservative fill-on-cross rule
- Recorded books (data/recorder): Kalshi REST snapshots (YES bids and NO bids,
  so YES asks are 1 - NO bid) plus trade prints fetched for the same market;
  Polymarket WebSocket snapshots, level changes and trades. Fills use the
  queue model.

Parameters are chosen walk-forward: markets are ordered by close time and
each fold's parameters are the best on earlier folds only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from arbiter import _sim as S
from arbiter.fees import FeeModel

NS = 1_000_000_000
SET, CLEAR, TRADE, TIMER = 0, 1, 2, 3
BID, ASK = 0, 1


@dataclass
class EventStream:
    ts: np.ndarray
    type: np.ndarray
    side: np.ndarray
    tick: np.ndarray
    size: np.ndarray

    def __len__(self):
        return len(self.ts)

    @staticmethod
    def build(rows: list[tuple]) -> "EventStream":
        if not rows:
            z = np.zeros(0)
            return EventStream(z.astype(np.int64), z.astype(np.uint8), z.astype(np.uint8), z.astype(np.int16), z)
        a = np.array(rows, dtype=[("ts", "i8"), ("type", "u1"), ("side", "u1"), ("tick", "i2"), ("size", "f8")])
        # Stable sort on time only: rows were built clear -> levels -> trades,
        # and that order must survive within a timestamp (sorting on the type
        # code would put CLEAR=1 after SET=0 and wipe every snapshot). Note
        # ndarray.sort(order="ts") still breaks ties on the other fields, so
        # argsort the time column alone.
        a = a[np.argsort(a["ts"], kind="stable")]
        return EventStream(a["ts"], a["type"], a["side"], a["tick"], a["size"])


def _tick(p) -> int:
    return int(round(float(p) * 1000))


_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _ns(ts) -> int:
    """Nanoseconds since the epoch, independent of pandas' timestamp
    resolution (pandas 3 defaults to microseconds); naive times are UTC."""
    t = pd.Timestamp(ts)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return (t - _EPOCH) // pd.Timedelta(1, "ns")


def events_from_candles(c: pd.DataFrame, nominal_size: float = 1000.0) -> EventStream:
    """One snapshot per candle with both a bid and an ask."""
    rows = []
    for r in c.itertuples():
        if pd.isna(r.yes_bid) or pd.isna(r.yes_ask) or not (0 < r.yes_bid < r.yes_ask < 1):
            continue
        t = _ns(r.ts)
        rows += [(t, CLEAR, 0, 0, 0.0), (t, SET, BID, _tick(r.yes_bid), nominal_size),
                 (t, SET, ASK, _tick(r.yes_ask), nominal_size)]
    return EventStream.build(rows)


def events_from_kalshi_recording(book: pd.DataFrame, trades: pd.DataFrame | None) -> EventStream:
    """Kalshi REST snapshots: YES bids are bids; NO bids at q are YES asks at 1 - q."""
    rows = []
    for (ts, seq), g in book.groupby(["ts", "seq"], sort=True):
        t = _ns(ts)
        rows.append((t, CLEAR, 0, 0, 0.0))
        for r in g[g.side.isin(["yes", "no"])].itertuples():
            if r.side == "yes":
                rows.append((t, SET, BID, _tick(r.price), float(r.size)))
            else:
                rows.append((t, SET, ASK, _tick(1 - r.price), float(r.size)))
    if trades is not None and len(trades):
        for r in trades.itertuples():
            # taker bought YES -> asks were lifted; taker sold YES (bought NO) -> bids were hit
            hit = ASK if r.taker_side == "yes" else BID
            rows.append((_ns(r.ts), TRADE, hit, _tick(r.price), float(r.size)))
    return EventStream.build(rows)


def events_from_polymarket_recording(snaps: pd.DataFrame, deltas: pd.DataFrame, trades: pd.DataFrame) -> EventStream:
    rows = []
    for ts, g in snaps.groupby("ts", sort=True):
        t = _ns(ts)
        rows.append((t, CLEAR, 0, 0, 0.0))
        rows += [(t, SET, BID if r.side == "bid" else ASK, _tick(r.price), float(r.size)) for r in g.itertuples()]
    rows += [(_ns(r.ts), SET, BID if r.side == "bid" else ASK, _tick(r.price), float(r.size_after))
             for r in deltas.itertuples()]
    for r in trades.itertuples():
        hit = ASK if r.taker_side == "buy" else BID
        rows.append((_ns(r.ts), TRADE, hit, _tick(r.price), float(r.size)))
    return EventStream.build(rows)


@dataclass
class MMResult:
    market_id: str
    pnl: float
    fees: float
    fills: int
    volume: float
    maker_share: float
    max_abs_inventory: float
    markout_1m: float  # mean mid move in our favour per contract 1 minute after fills (negative = adverse)
    markout_5m: float
    events: int


def _cpp_fees(f: FeeModel):
    venue = S.Venue.kalshi if f.venue == "kalshi" else S.Venue.polymarket
    return S.FeeModel(venue, f.taker_rate, f.maker_rate, f.exponent)


def run_market_maker(ev: EventStream, fees: FeeModel, resolved_yes: bool, close_ts, params: dict,
                     fill_on_cross: bool, market_id: str = "", max_position: float = 500.0) -> MMResult:
    p = S.MMParams()
    for k, v in params.items():
        setattr(p, k, v)
    p.close_ts = _ns(close_ts)
    mm = S.LogitMarketMaker(p)
    sim = S.Simulator(S.SimConfig(_cpp_fees(fees), fill_on_cross=fill_on_cross, max_position=max_position))
    sim.run(ev.ts, ev.type, ev.side, ev.tick, ev.size, strategy=mm)
    fills = sim.fills
    pnl = sim.settle(resolved_yes)
    if not fills:
        return MMResult(market_id, pnl, 0.0, 0, 0.0, 0.0, 0.0, np.nan, np.nan, len(ev))
    f = pd.DataFrame({"ts": [x.ts for x in fills], "side": [1 if x.side == S.Side.bid else -1 for x in fills],
                      "price": [x.price for x in fills], "qty": [x.qty for x in fills],
                      "maker": [x.maker for x in fills], "fee": [x.fee for x in fills]})
    pos = (f.side * f.qty).cumsum()
    mids = _mid_series(ev)
    marks = {}
    for lag in (60, 300):
        later = np.interp(f.ts + lag * NS, mids[0], mids[1])
        marks[lag] = float(np.average(f.side * (later - f.price), weights=f.qty))
    return MMResult(market_id, pnl, float(f.fee.sum()), len(f), float(f.qty.sum()),
                    float(f.qty[f.maker].sum() / f.qty.sum()), float(pos.abs().max()), marks[60], marks[300], len(ev))


def _mid_series(ev: EventStream) -> tuple[np.ndarray, np.ndarray]:
    """Mid after each event, reconstructed with the C++ book (for markouts)."""
    class Tap(S.Strategy):
        def __init__(self):
            super().__init__()
            self.t, self.m = [], []

        def on_event(self, s, e):
            if e.type != S.EventType.trade and s.book.two_sided:
                self.t.append(e.ts)
                self.m.append(s.book.mid)

    tap = Tap()
    S.Simulator(S.SimConfig(S.FeeModel(S.Venue.kalshi, 0, 0))).run(ev.ts, ev.type, ev.side, ev.tick, ev.size, strategy=tap)
    if not tap.t:
        return np.array([0, 1]), np.array([0.5, 0.5])
    return np.array(tap.t, dtype=np.int64), np.array(tap.m)


def walk_forward(markets: list[dict], grid: list[dict], n_folds: int = 5, fill_on_cross: bool = True,
                 min_train: int = 1) -> pd.DataFrame:
    """markets: dicts with market_id, events, fees, resolved_yes, close_ts.
    Markets are sorted by close and split into folds; fold i is traded with
    the grid point that had the highest total P&L over folds < i."""
    ms = sorted(markets, key=lambda m: pd.Timestamp(m["close_ts"]))
    folds = np.array_split(np.arange(len(ms)), n_folds)
    results = {}  # (grid index, market index) -> MMResult
    for gi, params in enumerate(grid):
        for mi, m in enumerate(ms):
            results[(gi, mi)] = run_market_maker(m["events"], m["fees"], m["resolved_yes"], m["close_ts"], params,
                                                 fill_on_cross, m["market_id"])
    rows = []
    for fi, idx in enumerate(folds):
        if fi < min_train:
            continue
        past = np.concatenate(folds[:fi])
        best = max(range(len(grid)), key=lambda gi: sum(results[(gi, mi)].pnl for mi in past))
        for mi in idx:
            r = results[(best, mi)]
            rows.append({**r.__dict__, "fold": fi, "params": grid[best], "close_ts": ms[mi]["close_ts"],
                         "event_id": ms[mi].get("event_id")})
    return pd.DataFrame(rows)

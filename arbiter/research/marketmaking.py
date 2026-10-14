"""R4: can a passive market maker earn the spread on prediction markets?

Data: the recorder's Polymarket capture (full-depth book snapshots and
deltas, plus every trade print, from the market WebSocket), so resting
orders fill only when trades actually reach them through the queue.

Evaluation
- The recording is cut into equal time blocks. Each block is a separate
  session: quoting starts flat and, at the block's end, any inventory is
  closed at the touch (long sells at the bid, short buys at the ask), so the
  spread is paid on the way out rather than marked at the mid.
- Walk-forward: the parameters used in block i are those with the best total
  P&L over blocks < i. Nothing is tuned on the block it is scored on.
- Sensitivity: queue_ahead (the share of displayed size assumed ahead of a
  new order: 1 = back of the queue) and a no-quote baseline.
- Diagnostics: fills, maker share, inventory, 1- and 5-minute markouts
  (adverse selection: how the mid moves against us after a fill).

Before any of that, trade prints are checked against the reconstructed book:
a taker buy should print at or above the prevailing ask.
"""

from __future__ import annotations

import glob
import itertools
import json
import os

import numpy as np
import pandas as pd

from arbiter import _sim as S
from arbiter import store
from arbiter.backtest import (ASK, BID, CLEAR, NS, SET, TRADE, EventStream, _cpp_fees, _mid_series, _ns, _tick,
                              events_from_kalshi_recording)
from arbiter.fees import FeeModel

# Quotes on the market's own tick (1 cent, or 0.1 cent for markets quoted below a
# cent; tick_size is in 0.001 units and set per market). In logit
# units the Avellaneda-Stoikov spread is gamma*sigma^2*tau + (2/gamma)ln(1 + gamma/k):
# k from 20 to 2000 spans about 1.5 cents per side down to quoting at the touch.
GRID = [dict(gamma=g, k=k, min_spread_ticks=m, quote_size=10.0, max_inventory=100.0, sigma=0.5,
             horizon_ns=int(3600 * 1e9), requote_ticks=1, tick_size=10)
        for g, k, m in itertools.product([0.05, 0.5], [20.0, 200.0, 2000.0], [1, 2])]


def load(kind: str, venue: str = "polymarket") -> pd.DataFrame:
    fs = sorted(glob.glob(os.path.join(store.ROOT, "recorder", venue, "date=*", f"{kind}-*.parquet")))
    return pd.concat((pd.read_parquet(f) for f in fs), ignore_index=True) if fs else pd.DataFrame()


def token_events(snaps: pd.DataFrame, deltas: pd.DataFrame, trades: pd.DataFrame) -> EventStream:
    rows = []
    for ts, g in snaps.groupby("ts", sort=True):
        t = _ns(ts)
        rows.append((t, CLEAR, 0, 0, 0.0))
        rows += [(t, SET, BID if s == "bid" else ASK, _tick(p), float(z)) for s, p, z in zip(g.side, g.price, g["size"])]
    rows += [(_ns(t), SET, BID if s == "bid" else ASK, _tick(p), float(z))
             for t, s, p, z in zip(deltas.ts, deltas.side, deltas.price, deltas.size_after)]
    rows += [(_ns(t), TRADE, ASK if s == "buy" else BID, _tick(p), float(z))
             for t, s, p, z in zip(trades.ts, trades.taker_side, trades.price, trades["size"])]
    return EventStream.build(rows)


def trade_side_check(ev: EventStream) -> dict:
    """Where each trade printed relative to the book just before it."""
    # replay the book events only, recording the touch after each one
    book_ev = ev.type != TRADE
    t_book = ev.ts[book_ev]
    tap_bid, tap_ask = [], []

    class Quotes(S.Strategy):
        def __init__(self):
            super().__init__()

        def on_event(self, s, e):
            tap_bid.append(s.book.best_bid)  # -1 when empty
            tap_ask.append(s.book.best_ask)  # 1001 when empty

    q = Quotes()
    S.Simulator(S.SimConfig(S.FeeModel(S.Venue.polymarket, 0, 0))).run(
        ev.ts[book_ev], ev.type[book_ev], ev.side[book_ev], ev.tick[book_ev], ev.size[book_ev], strategy=q)
    bids, asks = np.array(tap_bid), np.array(tap_ask)
    tr = ~book_ev
    i = np.searchsorted(t_book, ev.ts[tr], side="right") - 1  # last book event strictly before or at the trade
    ok = i >= 0
    side, tick = ev.side[tr][ok], ev.tick[tr][ok]
    b, a = bids[i[ok]], asks[i[ok]]
    hit_ask = side == ASK
    return {"trades": int(tr.sum()),
            "taker_buy_at_or_above_ask": float(np.mean(tick[hit_ask] >= a[hit_ask])) if hit_ask.any() else None,
            "taker_buy_at_or_below_bid": float(np.mean(tick[hit_ask] <= b[hit_ask])) if hit_ask.any() else None,
            "taker_sell_at_or_below_bid": float(np.mean(tick[~hit_ask] <= b[~hit_ask])) if (~hit_ask).any() else None,
            "taker_sell_at_or_above_ask": float(np.mean(tick[~hit_ask] >= a[~hit_ask])) if (~hit_ask).any() else None}


def infer_tick(snaps: pd.DataFrame) -> int:
    """Venue tick in 0.001 units: 1 if any displayed price is off the cent grid."""
    cents = snaps.price.values * 100
    return 1 if np.any(np.abs(cents - np.round(cents)) > 1e-6) else 10


def run_block(ev: EventStream, params: dict | None, fees: FeeModel, queue_ahead: float = 1.0,
              mids: tuple | None = None) -> dict:
    """One flat-to-flat session. params=None runs no strategy (baseline)."""
    sim = S.Simulator(S.SimConfig(_cpp_fees(fees), fill_on_cross=False, max_position=params["max_inventory"] if params else 1e18,
                                  queue_ahead=queue_ahead))
    if params is None:
        return {"pnl": 0.0, "fills": 0, "volume": 0.0, "fees": 0.0, "maker_share": np.nan, "max_inv": 0.0,
                "markout_1m": np.nan, "markout_5m": np.nan}
    p = S.MMParams()
    for k, v in params.items():
        setattr(p, k, v)
    p.close_ts = int(ev.ts[-1]) + int(3600 * 1e9) * 24 * 30  # resolution is far away: no close-out widening
    mm = S.LogitMarketMaker(p)
    sim.run(ev.ts, ev.type, ev.side, ev.tick, ev.size, strategy=mm)
    pos = sim.position
    # close out at the touch: long sells at the bid, short buys at the ask
    # (an empty side means the position is worth nothing to the exit: 0 or 1)
    exit_px = max(sim.book.best_bid, 0) / 1000.0 if pos > 0 else min(sim.book.best_ask, 1000) / 1000.0
    pnl = sim.cash + pos * exit_px - (fees.fee(exit_px, abs(pos), maker=False) if pos else 0.0)
    fills = sim.fills
    if not fills:
        return {"pnl": pnl, "fills": 0, "volume": 0.0, "fees": 0.0, "maker_share": np.nan, "max_inv": 0.0,
                "markout_1m": np.nan, "markout_5m": np.nan}
    f = pd.DataFrame({"ts": [x.ts for x in fills], "side": [1 if x.side == S.Side.bid else -1 for x in fills],
                      "price": [x.price for x in fills], "qty": [x.qty for x in fills],
                      "maker": [x.maker for x in fills], "fee": [x.fee for x in fills]})
    mids = mids if mids is not None else _mid_series(ev)
    marks = {lag: float(np.average(f.side * (np.interp(f.ts + lag * NS, mids[0], mids[1]) - f.price), weights=f.qty))
             for lag in (60, 300)}
    # Exact P&L decomposition (dollars):
    #   edge     = sum side*qty*(mid just before the fill - fill price)   spread captured
    #   drift    = sum side*qty*(mid at block end - mid before the fill)   inventory P&L
    #   closeout = pos*(exit price - mid at block end) - exit taker fee    paying the spread to go flat
    #   pnl      = edge + drift + closeout - fill fees
    j = np.clip(np.searchsorted(mids[0], f.ts.values, side="left") - 1, 0, len(mids[1]) - 1)
    mid_fill, mid_end = mids[1][j], mids[1][-1]
    q = f.side.values * f.qty.values
    edge = float(np.sum(q * (mid_fill - f.price.values)))
    drift = float(np.sum(q * (mid_end - mid_fill)))
    closeout = float(pos * (exit_px - mid_end) - (fees.fee(exit_px, abs(pos), maker=False) if pos else 0.0))
    return {"pnl": pnl, "fills": len(f), "volume": float(f.qty.sum()), "fees": float(f.fee.sum()),
            "maker_share": float(f.qty[f.maker].sum() / f.qty.sum()), "max_inv": float((f.side * f.qty).cumsum().abs().max()),
            "markout_1m": marks[60], "markout_5m": marks[300], "edge": edge, "drift": drift, "closeout": closeout}


def slice_events(ev: EventStream, t0: int, t1: int, snaps_at: np.ndarray) -> EventStream | None:
    """Events in [t0, t1), starting from the last full snapshot at or before t0
    so the block opens with a real book."""
    j = np.searchsorted(snaps_at, t0, side="right") - 1
    if j < 0:
        return None
    start = np.searchsorted(ev.ts, snaps_at[j], side="left")
    end = np.searchsorted(ev.ts, t1, side="left")
    if end - start < 10:
        return None
    sub = EventStream.__new__(EventStream)
    for name in ("ts", "type", "side", "tick", "size"):
        setattr(sub, name, getattr(ev, name)[start:end])
    return sub


def evaluate(streams: list[dict], t_start: int, t_end: int, n_blocks: int) -> tuple[pd.DataFrame, dict]:
    """streams: dicts with id, ev (EventStream), snaps_at (full-book times),
    fees (FeeModel) and tick. Runs every grid point and queue assumption on
    every (stream, block) and scores the walk-forward choice."""
    edges = np.linspace(t_start, t_end, n_blocks + 1).astype(np.int64)
    rows = []
    for st in streams:
        for b in range(n_blocks):
            sub = slice_events(st["ev"], edges[b], edges[b + 1], st["snaps_at"])
            if sub is None:
                continue
            mids = _mid_series(sub)
            for qa in (1.0, 0.5, 0.0):
                for gi, params in enumerate(GRID):
                    rows.append({"token": st["id"], "block": b, "grid": gi, "queue_ahead": qa, "tick": st["tick"],
                                 **run_block(sub, params | {"tick_size": st["tick"]}, st["fees"], qa, mids)})
    r = pd.DataFrame(rows)
    out = {f"queue_ahead={qa}": walk_forward(g) for qa, g in r.groupby("queue_ahead")} if len(r) else {}
    return r, out


def run_polymarket(n_blocks: int = 6, min_trades: int = 50) -> dict:
    snaps, deltas, trades = load("book_snapshots"), load("book_deltas"), load("trades")
    listing = load("listing")
    yes_tokens = set(listing.token_yes.dropna())
    fees_on = dict(zip(listing.token_yes, listing.fees_enabled))
    counts = trades[trades.token.isin(yes_tokens)].groupby("token").size()
    tokens = counts[counts >= min_trades].index.tolist()
    streams, checks = [], []
    for tok in tokens:
        s, d, t = snaps[snaps.token == tok], deltas[deltas.token == tok], trades[trades.token == tok]
        ev = token_events(s, d, t)
        checks.append({"token": tok, **trade_side_check(ev)})
        streams.append({"id": tok, "ev": ev, "snaps_at": np.array(sorted(_ns(x) for x in s.ts.unique()), dtype=np.int64),
                        "fees": FeeModel("polymarket", 0.07 if fees_on.get(tok) else 0.0, 0.0, 1.0),  # worst-case taker rate
                        "tick": infer_tick(s)})
    t_start, t_end = _ns(trades.ts.min()), _ns(trades.ts.max())
    r, wf = evaluate(streams, t_start, t_end, n_blocks)
    store.write_table("r4_blocks", r)
    return {"tokens": len(tokens), "blocks": n_blocks, "hours": (t_end - t_start) / 3.6e12,
            "trades": int(trades.token.isin(tokens).sum()),
            "trade_side_check": pd.DataFrame(checks).mean(numeric_only=True).to_dict(), **wf}


def kalshi_trades(markets: pd.DataFrame, rps: float = 2.0) -> pd.DataFrame:
    """Trade prints for each recorded Kalshi market over its recorded window,
    cached in the raw store (network on first use)."""
    from arbiter.http import RateLimitedClient
    from arbiter.venues import kalshi as K

    out = []
    with RateLimitedClient(rates={"api.elections.kalshi.com": rps}) as c:
        api = K.Kalshi(c)
        for r in markets.itertuples():
            key = f"{r.market_id}_{int(r.t0.timestamp())}_{int(r.t1.timestamp())}"
            raw = store.load_raw("kalshi", "trades", key)
            if raw is None:
                raw = api.trades(r.market_id, min_ts=int(r.t0.timestamp()), max_ts=int(r.t1.timestamp()) + 1)
                store.save_raw("kalshi", "trades", key, raw)
            out += K.trade_rows(raw)
    return pd.DataFrame(out)


def run_kalshi(n_blocks: int = 6, min_trades: int = 20) -> dict:
    """Kalshi: the recorder polls each watched market's full book over REST
    (about every 30 s) and trade prints come from the public trades endpoint.
    The book between polls is unknown, so queue positions are coarser than on
    Polymarket's streamed deltas. Fees are each series' own (some Kalshi
    series charge makers too)."""
    from arbiter.fees import kalshi_fees

    book = load("book_snapshots", "kalshi")
    win = book.groupby("market_id").ts.agg(t0="min", t1="max").reset_index()
    trades = kalshi_trades(win)
    counts = trades.groupby("market_id").size()
    ids = counts[counts >= min_trades].index.tolist()
    meta = {x["ticker"]: x for x in (store.load_raw("kalshi", "series", "all") or [])}
    streams, checks = [], []
    for mid in ids:
        b, t = book[book.market_id == mid], trades[trades.market_id == mid]
        ev = events_from_kalshi_recording(b, t)
        checks.append({"token": mid, **trade_side_check(ev)})
        sm = meta.get(mid.split("-")[0], {})
        prices = pd.DataFrame({"price": np.concatenate([b.price.values, 1 - b.price.values])})
        streams.append({"id": mid, "ev": ev, "snaps_at": np.array(sorted(_ns(x) for x in b.ts.unique()), dtype=np.int64),
                        "fees": kalshi_fees(sm.get("fee_type"), sm.get("fee_multiplier")), "tick": infer_tick(prices),
                        "maker_fees": "maker" in str(sm.get("fee_type"))})
    t_start, t_end = _ns(book.ts.min()), _ns(book.ts.max())
    r, wf = evaluate(streams, t_start, t_end, n_blocks)
    store.write_table("r4_kalshi_blocks", r)
    return {"markets": len(ids), "markets_recorded": int(win.shape[0]), "markets_with_maker_fees": int(sum(s["maker_fees"] for s in streams)),
            "blocks": n_blocks, "hours": (t_end - t_start) / 3.6e12, "trades": int(trades.market_id.isin(ids).sum()),
            "trade_side_check": pd.DataFrame(checks).mean(numeric_only=True).to_dict(), **wf}


def run() -> dict:
    return {"polymarket": run_polymarket(), "kalshi": run_kalshi()}


def walk_forward(r: pd.DataFrame) -> dict:
    """Block b is traded with the grid point that made the most over blocks < b
    (across all tokens)."""
    chosen, picked = {}, []
    for b in sorted(r.block.unique()):
        past = r[r.block < b]
        if past.empty:
            continue
        best = int(past.groupby("grid").pnl.sum().idxmax())
        chosen[int(b)] = best
        picked.append(r[(r.block == b) & (r.grid == best)])
    if not picked:
        return {}
    t = pd.concat(picked)
    per_block = t.groupby("block").pnl.sum()
    per_token = t.groupby("token").pnl.sum()
    return {"token_blocks": int(len(t)), "total_pnl": float(t.pnl.sum()), "pnl_per_block_mean": float(per_block.mean()),
            "pnl_per_block_t": float(per_block.mean() / (per_block.std(ddof=1) / np.sqrt(len(per_block)))) if len(per_block) > 1 and per_block.std() > 0 else None,
            "tokens_positive_share": float((per_token > 0).mean()), "fills": int(t.fills.sum()), "volume": float(t.volume.sum()),
            "fees": float(t.fees.sum()), "maker_share": float(np.nanmean(t.maker_share)),
            "markout_1m_per_contract": float(np.average(t.markout_1m.fillna(0), weights=t.volume)) if t.volume.sum() else None,
            "markout_5m_per_contract": float(np.average(t.markout_5m.fillna(0), weights=t.volume)) if t.volume.sum() else None,
            "pnl_per_contract": float(t.pnl.sum() / t.volume.sum()) if t.volume.sum() else None,
            **({f"{k}_per_contract": float(t[k].sum() / t.volume.sum()) for k in ("edge", "drift", "closeout")}
               if "edge" in t and t.volume.sum() else {}),
            "fees_per_contract": float(t.fees.sum() / t.volume.sum()) if t.volume.sum() else None,
            "grid_chosen_by_block": chosen}


if __name__ == "__main__":
    res = run()
    with open(store.path("reports", "marketmaking.json"), "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(json.dumps(res, indent=2, default=str))

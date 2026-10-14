"""Paper trading: the logit market maker quoting live Polymarket books.

Nothing is sent to any venue. Each live WebSocket message becomes simulator
events, fed to a C++ Simulator + LogitMarketMaker per market as it arrives,
so the quoter sees exactly what it would have seen live, and fills happen
only when real trades reach its queue position (the same fill model as the
backtest). Every minute it logs position, cash and P&L marked at the touch.
At the end, inventory is closed at the touch and a summary is written.

    python -m arbiter.papertrade --markets 10 --minutes 60 [--queue-ahead 1.0]

Output: data/papertrade/<run>/fills.parquet, pnl.parquet, summary.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time

import numpy as np
import pandas as pd
import websockets

from arbiter import _sim as S
from arbiter import store
from arbiter.backtest import ASK, BID, CLEAR, SET, TRADE, _cpp_fees, _tick
from arbiter.fees import FeeModel
from arbiter.research.marketmaking import GRID

log = logging.getLogger("arbiter.papertrade")
PM_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
GAMMA = "https://gamma-api.polymarket.com"


def pick_markets(n: int) -> list[dict]:
    """The n most-traded open binary markets with a two-sided book in 10-90 cents."""
    from arbiter.http import RateLimitedClient

    with RateLimitedClient() as c:
        ms = c.get_json(f"{GAMMA}/markets", {"closed": "false", "active": "true", "order": "volume24hr",
                                             "ascending": "false", "limit": 100}) or []
    out = []
    for m in ms:
        toks = json.loads(m.get("clobTokenIds") or "[]")
        bid, ask = m.get("bestBid"), m.get("bestAsk")
        if len(toks) != 2 or bid is None or ask is None or not (0.1 <= float(bid) and float(ask) <= 0.9):
            continue
        tick = float(m.get("orderPriceMinTickSize") or 0.01)
        out.append({"token": toks[0], "question": m.get("question"), "fees_enabled": bool(m.get("feesEnabled")),
                    "tick": max(1, int(round(tick * 1000)))})
        if len(out) == n:
            break
    return out


class Book:
    """One market: its simulator, quoter and fill log."""

    def __init__(self, m: dict, params: dict, queue_ahead: float):
        self.m = m
        self.fees = FeeModel("polymarket", 0.07 if m["fees_enabled"] else 0.0, 0.0, 1.0)  # worst-case taker rate
        p = S.MMParams()
        for k, v in (params | {"tick_size": m["tick"]}).items():
            setattr(p, k, v)
        p.close_ts = int(time.time() * 1e9) + int(30 * 86400 * 1e9)  # far away: no close-out widening
        self.mm = S.LogitMarketMaker(p)
        self.sim = S.Simulator(S.SimConfig(_cpp_fees(self.fees), fill_on_cross=False,
                                           max_position=params["max_inventory"], queue_ahead=queue_ahead))
        self.seen_fills = 0
        self.events = 0

    def feed(self, rows: list[tuple]) -> list:
        if not rows:
            return []
        self.events += len(rows)
        a = np.array(rows, dtype=object)
        self.sim.run(a[:, 0].astype(np.int64), a[:, 1].astype(np.uint8), a[:, 2].astype(np.uint8),
                     a[:, 3].astype(np.int16), a[:, 4].astype(np.float64), strategy=self.mm)
        new = self.sim.fills[self.seen_fills:]
        self.seen_fills += len(new)
        return new

    def close_out_value(self) -> float:
        """Cash plus inventory sold at the bid (long) or bought at the ask (short), after taker fees."""
        pos, b = self.sim.position, self.sim.book
        px = max(b.best_bid, 0) / 1000.0 if pos > 0 else min(b.best_ask, 1000) / 1000.0
        return self.sim.cash + pos * px - (self.fees.fee(px, abs(pos), maker=False) if pos else 0.0)


def to_rows(msg: dict) -> list[tuple]:
    """A Polymarket market-channel message as simulator event rows."""
    t = time.time_ns()
    et = msg.get("event_type")
    if et == "book":
        rows = [(t, CLEAR, 0, 0, 0.0)]
        rows += [(t, SET, BID, _tick(float(lv["price"])), float(lv["size"])) for lv in msg.get("bids") or []]
        rows += [(t, SET, ASK, _tick(float(lv["price"])), float(lv["size"])) for lv in msg.get("asks") or []]
        return rows
    if et == "price_change":
        return [(t, SET, BID if ch.get("side") == "BUY" else ASK, _tick(float(ch["price"])), float(ch["size"]))
                for ch in msg.get("price_changes") or msg.get("changes") or []]
    if et == "last_trade_price":
        hit = ASK if (msg.get("side") or "").lower() == "buy" else BID
        return [(t, TRADE, hit, _tick(float(msg["price"])), float(msg.get("size") or 0))]
    return []


async def run(markets: list[dict], minutes: float, params: dict, queue_ahead: float, out_dir: str) -> dict:
    books = {m["token"]: Book(m, params, queue_ahead) for m in markets}
    fills, pnl_rows = [], []
    t_end = time.monotonic() + minutes * 60
    last_log = time.monotonic()

    def log_pnl():
        for tok, b in books.items():
            pnl_rows.append({"ts": pd.Timestamp.now(tz="UTC"), "token": tok, "position": b.sim.position,
                             "cash": b.sim.cash, "pnl_at_touch": b.close_out_value(), "fills": b.seen_fills})
        tot = sum(b.close_out_value() for b in books.values())
        quoting = sum(1 for b in books.values() if b.sim.orders)
        log.info("paper P&L (closed at the touch) %+.2f across %d markets, %d fills; %d events, %d markets with live quotes",
                 tot, len(books), len(fills), sum(b.events for b in books.values()), quoting)

    while time.monotonic() < t_end:
        try:
            async with websockets.connect(PM_WS, ping_interval=20, ping_timeout=20, max_size=2**24) as ws:
                await ws.send(json.dumps({"assets_ids": list(books), "type": "market"}))
                log.info("paper trading %d markets for %.0f minutes (no orders are sent anywhere)", len(books), minutes)
                while time.monotonic() < t_end:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=5)
                    except asyncio.TimeoutError:
                        await ws.send("PING")
                        continue
                    if raw == "PONG":
                        continue
                    data = json.loads(raw)
                    for msg in data if isinstance(data, list) else [data]:
                        groups: dict[str, list] = {}
                        if msg.get("event_type") == "price_change":
                            for ch in msg.get("price_changes") or []:  # one message can touch several tokens
                                groups.setdefault(ch.get("asset_id"), []).extend(to_rows({"event_type": "price_change", "price_changes": [ch]}))
                        else:
                            groups[msg.get("asset_id")] = to_rows(msg)
                        for tok, rows in groups.items():
                            if tok in books:
                                for f in books[tok].feed(rows):
                                    fills.append({"ts": pd.Timestamp(f.ts, unit="ns", tz="UTC"), "token": tok,
                                                  "side": "buy" if f.side == S.Side.bid else "sell", "price": f.price,
                                                  "qty": f.qty, "maker": f.maker, "fee": f.fee})
                    if time.monotonic() - last_log > 60:
                        log_pnl()
                        last_log = time.monotonic()
        except Exception as e:  # reconnect: books resync from the next full snapshot
            log.warning("websocket error: %s; reconnecting", e)
            await asyncio.sleep(2)
    log_pnl()
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(fills).to_parquet(os.path.join(out_dir, "fills.parquet"), index=False)
    pd.DataFrame(pnl_rows).to_parquet(os.path.join(out_dir, "pnl.parquet"), index=False)
    vol = sum(f["qty"] for f in fills)
    total = sum(b.close_out_value() for b in books.values())
    summary = {"markets": [m["question"] for m in markets], "minutes": minutes, "queue_ahead": queue_ahead,
               "params": params, "fills": len(fills), "contracts": vol, "pnl_closed_at_touch": total,
               "pnl_per_contract": total / vol if vol else None,
               "events": sum(b.events for b in books.values()),
               "per_market": {b.m["question"]: {"events": b.events, "fills": b.seen_fills, "position": b.sim.position,
                                                "resting_orders": len(b.sim.orders), "quotes_placed": b.mm.quotes_placed,
                                                "pnl_closed_at_touch": b.close_out_value()} for b in books.values()}}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", type=int, default=10)
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--grid", type=int, default=7, help="index into marketmaking.GRID (7: the R4 walk-forward's most frequent pick)")
    ap.add_argument("--queue-ahead", type=float, default=1.0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    markets = pick_markets(a.markets)
    out_dir = store.path("papertrade", pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S"), "x")[:-2]
    s = asyncio.run(run(markets, a.minutes, GRID[a.grid], a.queue_ahead, out_dir))
    print(json.dumps({k: v for k, v in s.items() if k != "per_market"}, indent=2, default=str))


if __name__ == "__main__":
    main()

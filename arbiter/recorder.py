"""Live order-book recorder for both venues.

    python -m arbiter.recorder [--kalshi 60] [--polymarket 150] [--kalshi-rps 4]

Neither venue publishes historical depth, so this builds it going forward.

- Polymarket: the public market WebSocket streams a full book snapshot per
  token on subscribe, then price-level changes and trade prices. Every
  message is stored as received (parsed to rows), which is enough to
  replay the book exactly.
- Kalshi: its WebSocket requires an API key, so the recorder polls the public
  REST order book round-robin over the watched markets instead, within a
  request budget, and stores full snapshots.

The watchlist (most active open markets) is refreshed every 30 minutes.
Rows are buffered and flushed to data/recorder/<venue>/date=YYYY-MM-DD/
every minute; a heartbeat line per minute in the log shows coverage, so gaps
can be excluded from backtests.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import time
from datetime import datetime, timezone

import httpx
import pandas as pd
import websockets

from arbiter import store
from arbiter.venues.kalshi import BASE as KALSHI
from arbiter.venues.polymarket import GAMMA

log = logging.getLogger("arbiter.recorder")
PM_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


def now() -> datetime:
    return datetime.now(timezone.utc)


class Sink:
    """Buffers rows per (venue, table) and writes one parquet file per flush."""

    def __init__(self):
        self.buf: dict[tuple[str, str], list[dict]] = {}
        self.counts: dict[tuple[str, str], int] = {}

    def add(self, venue: str, table: str, row: dict) -> None:
        self.buf.setdefault((venue, table), []).append(row)
        self.counts[(venue, table)] = self.counts.get((venue, table), 0) + 1

    def flush(self) -> None:
        t = now()
        for (venue, table), rows in self.buf.items():
            if not rows:
                continue
            p = store.path("recorder", venue, f"date={t:%Y-%m-%d}", f"{table}-{t:%H%M%S}.parquet")
            pd.DataFrame(rows).to_parquet(p, index=False)
        self.buf = {}


class Recorder:
    def __init__(self, n_kalshi: int, n_poly: int, kalshi_rps: float):
        self.n_kalshi, self.n_poly, self.kalshi_rps = n_kalshi, n_poly, kalshi_rps
        self.sink = Sink()
        self.kalshi_markets: list[str] = []
        self.poly_tokens: dict[str, str] = {}  # token -> market (condition) id
        self.poly_changed = asyncio.Event()
        self.stop = asyncio.Event()

    # ---- watchlist ----

    async def refresh_watchlist(self, client: httpx.AsyncClient) -> None:
        try:
            r = await client.get(f"{KALSHI}/markets", params={"status": "open", "limit": 1000,
                                                              "mve_filter": "exclude"})
            ms = [m for m in r.json().get("markets", []) if not m["ticker"].startswith("KXMVE")]
            cursor = r.json().get("cursor")
            for _ in range(10):  # a few more pages so the volume ranking is meaningful
                if not cursor:
                    break
                r = await client.get(f"{KALSHI}/markets", params={"status": "open", "limit": 1000, "cursor": cursor,
                                                                  "mve_filter": "exclude"})
                ms += [m for m in r.json().get("markets", []) if not m["ticker"].startswith("KXMVE")]
                cursor = r.json().get("cursor")
                await asyncio.sleep(0.3)
            ms.sort(key=lambda m: float(m.get("volume_24h_fp") or 0), reverse=True)
            self.kalshi_markets = [m["ticker"] for m in ms[: self.n_kalshi]]
        except Exception as e:  # keep the old list on failure
            log.warning("kalshi watchlist refresh failed: %s", e)
        try:
            r = await client.get(f"{GAMMA}/markets", params={"active": "true", "closed": "false", "limit": 500,
                                                             "order": "volume24hr", "ascending": "false"})
            tokens = {}
            for m in r.json()[: self.n_poly]:
                ids = json.loads(m.get("clobTokenIds") or "[]")
                if len(ids) == 2:
                    tokens[ids[0]] = m.get("conditionId")
            if tokens and set(tokens) != set(self.poly_tokens):
                self.poly_tokens = tokens
                self.poly_changed.set()
        except Exception as e:
            log.warning("polymarket watchlist refresh failed: %s", e)
        log.info("watchlist: %d kalshi markets, %d polymarket tokens", len(self.kalshi_markets), len(self.poly_tokens))

    async def watchlist_loop(self, client):
        while not self.stop.is_set():
            await self.refresh_watchlist(client)
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=1800)
            except asyncio.TimeoutError:
                pass

    # ---- Kalshi polling ----

    async def kalshi_loop(self, client: httpx.AsyncClient):
        interval = 1.0 / self.kalshi_rps
        i = 0
        while not self.stop.is_set():
            if not self.kalshi_markets:
                await asyncio.sleep(1)
                continue
            ticker = self.kalshi_markets[i % len(self.kalshi_markets)]
            i += 1
            t0 = time.monotonic()
            try:
                r = await client.get(f"{KALSHI}/markets/{ticker}/orderbook")
                if r.status_code == 200:
                    ob = r.json().get("orderbook_fp") or {}
                    ts, seq = now(), i
                    for side, key in (("yes", "yes_dollars"), ("no", "no_dollars")):
                        for level, (price, size) in enumerate(reversed(ob.get(key) or [])):  # best bid first
                            self.sink.add("kalshi", "book_snapshots", {
                                "ts": ts, "market_id": ticker, "seq": seq, "side": side, "level": level,
                                "price": float(price), "size": float(size)})
                    if not (ob.get("yes_dollars") or ob.get("no_dollars")):
                        self.sink.add("kalshi", "book_snapshots", {"ts": ts, "market_id": ticker, "seq": seq,
                                                                   "side": "empty", "level": -1, "price": None, "size": None})
                elif r.status_code == 429:
                    await asyncio.sleep(5)
            except Exception as e:
                log.debug("kalshi %s: %s", ticker, e)
            await asyncio.sleep(max(0.0, interval - (time.monotonic() - t0)))

    # ---- Polymarket WebSocket ----

    def _poly_message(self, msg: dict) -> None:
        ts = now()
        et = msg.get("event_type")
        asset = msg.get("asset_id")
        market = self.poly_tokens.get(asset, msg.get("market"))
        if et == "book":
            for side, levels in (("bid", msg.get("bids") or []), ("ask", msg.get("asks") or [])):
                for lv in levels:
                    self.sink.add("polymarket", "book_snapshots", {
                        "ts": ts, "exch_ts": msg.get("timestamp"), "market_id": market, "token": asset,
                        "side": side, "price": float(lv["price"]), "size": float(lv["size"]),
                        "hash": msg.get("hash")})
        elif et == "price_change":
            for ch in msg.get("price_changes") or msg.get("changes") or []:
                self.sink.add("polymarket", "book_deltas", {
                    "ts": ts, "exch_ts": msg.get("timestamp"), "market_id": market,
                    "token": ch.get("asset_id", asset), "side": "bid" if ch.get("side") == "BUY" else "ask",
                    "price": float(ch["price"]), "size_after": float(ch["size"])})
        elif et == "last_trade_price":
            self.sink.add("polymarket", "trades", {
                "ts": ts, "exch_ts": msg.get("timestamp"), "market_id": market, "token": asset,
                "price": float(msg["price"]), "size": float(msg.get("size") or 0),
                "taker_side": (msg.get("side") or "").lower()})

    async def poly_loop(self):
        backoff = 1
        while not self.stop.is_set():
            if not self.poly_tokens:
                await asyncio.sleep(1)
                continue
            try:
                async with websockets.connect(PM_WS, ping_interval=None, max_size=2**24) as ws:
                    self.poly_changed.clear()
                    await ws.send(json.dumps({"assets_ids": list(self.poly_tokens), "type": "market"}))
                    log.info("polymarket ws subscribed to %d tokens", len(self.poly_tokens))
                    backoff = 1
                    last_ping = time.monotonic()
                    while not self.stop.is_set() and not self.poly_changed.is_set():
                        if time.monotonic() - last_ping > 10:
                            await ws.send("PING")
                            last_ping = time.monotonic()
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=5)
                        except asyncio.TimeoutError:
                            continue
                        if raw == "PONG":
                            continue
                        data = json.loads(raw)
                        for msg in data if isinstance(data, list) else [data]:
                            self._poly_message(msg)
            except Exception as e:
                log.warning("polymarket ws error: %s (reconnect in %ds)", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    # ---- flushing / heartbeat ----

    async def flush_loop(self):
        while not self.stop.is_set():
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=60)
            except asyncio.TimeoutError:
                pass
            counts = {f"{v}.{t}": n for (v, t), n in self.sink.counts.items()}
            self.sink.flush()
            log.info("heartbeat %s", json.dumps(counts))
            self.sink.counts = {}

    async def run(self):
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.stop.set)
            except NotImplementedError:
                pass
        async with httpx.AsyncClient(timeout=20, headers={"User-Agent": "arbiter-research/0.1"}) as client:
            await asyncio.gather(self.watchlist_loop(client), self.kalshi_loop(client), self.poly_loop(),
                                 self.flush_loop())
        self.sink.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kalshi", type=int, default=60, help="Kalshi markets to poll")
    ap.add_argument("--polymarket", type=int, default=150, help="Polymarket markets to stream")
    ap.add_argument("--kalshi-rps", type=float, default=4.0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(store.path("recorder", "recorder.log"))])
    asyncio.run(Recorder(a.kalshi, a.polymarket, a.kalshi_rps).run())


if __name__ == "__main__":
    main()

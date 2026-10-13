"""Kalshi public market data.

Markets settled before the exchange's historical cutoff live under
/historical/... (same shapes, older field names such as "close" instead of
"close_dollars"); newer ones under the live endpoints. Both are crawled per
series: paging the global settled list is dominated by multivariate combo
markets (thousands per hour), which are sampled separately for R3.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from arbiter.http import RateLimitedClient

BASE = "https://api.elections.kalshi.com/trade-api/v2"


def ts(s: str | None) -> pd.Timestamp | None:
    return pd.Timestamp(s) if s else None


def _num(x):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _dollars(d: dict, key: str):
    """Field that is "<key>_dollars" in the live API and "<key>" in /historical."""
    v = d.get(f"{key}_dollars", d.get(key))
    return _num(v)


def is_combo(ticker: str) -> bool:
    return ticker.startswith("KXMVE")


class Kalshi:
    def __init__(self, client: RateLimitedClient):
        self.c = client

    def _pages(self, path: str, params: dict, key: str, limit: int | None = None, page_size: int = 1000):
        """Cursor pagination. Page caps differ by endpoint (markets 1000,
        events 200; larger values are rejected with HTTP 400)."""
        cursor, n = None, 0
        while True:
            p = dict(params, limit=min(page_size, limit - n) if limit else page_size)
            if cursor:
                p["cursor"] = cursor
            r = self.c.get_json(f"{BASE}/{path}", p)
            if not r:
                return
            items = r.get(key) or []
            yield from items
            n += len(items)
            cursor = r.get("cursor")
            if not cursor or not items or (limit and n >= limit):
                return

    def series(self) -> list[dict]:
        r = self.c.get_json(f"{BASE}/series")
        return r.get("series", []) if r else []

    def settled_markets(self, series: str, limit: int) -> list[dict]:
        """Newest `limit` settled markets of a series, live then historical."""
        out = list(self._pages("markets", {"series_ticker": series, "status": "settled"}, "markets", limit))
        if len(out) < limit:
            out += list(self._pages("historical/markets", {"series_ticker": series}, "markets", limit - len(out)))
        return out

    def events(self, series: str, status: str) -> list[dict]:
        return list(self._pages("events", {"series_ticker": series, "status": status,
                                           "with_nested_markets": "true"}, "events", page_size=200))

    def candles(self, series: str, ticker: str, start: int, end: int, period: int, historical: bool) -> list[dict]:
        path = (f"historical/markets/{ticker}/candlesticks" if historical
                else f"series/{series}/markets/{ticker}/candlesticks")
        r = self.c.get_json(f"{BASE}/{path}", {"start_ts": start, "end_ts": end, "period_interval": period})
        return (r or {}).get("candlesticks", [])

    def orderbook(self, ticker: str) -> dict | None:
        r = self.c.get_json(f"{BASE}/markets/{ticker}/orderbook")
        return (r or {}).get("orderbook_fp") or (r or {}).get("orderbook")

    def open_markets(self, limit: int = 20000) -> list[dict]:
        return list(self._pages("markets", {"status": "open"}, "markets", limit))


def market_row(m: dict, series_meta: dict | None = None) -> dict:
    s = series_meta or {}
    return {
        "venue": "kalshi",
        "market_id": m["ticker"],
        "event_id": m.get("event_ticker"),
        "series_id": m["ticker"].split("-")[0] if not m.get("event_ticker") else m["event_ticker"].split("-")[0],
        "title": m.get("title"),
        "subtitle": m.get("yes_sub_title") or m.get("subtitle"),
        "category": s.get("category"),
        "frequency": s.get("frequency"),
        "market_type": m.get("market_type"),
        "open_time": ts(m.get("open_time")),
        "close_time": ts(m.get("close_time")),
        "settle_time": ts(m.get("settlement_ts") or m.get("expiration_time")),
        "result": m.get("result") or None,
        "strike": _num(m.get("floor_strike")),
        "strike_type": m.get("strike_type"),
        "volume": _num(m.get("volume_fp", m.get("volume"))),
        "open_interest": _num(m.get("open_interest_fp", m.get("open_interest"))),
        "last_price": _dollars(m, "last_price"),
        "rules_text": m.get("rules_primary"),
        "fee_type": s.get("fee_type"),
        "fee_multiplier": _num(s.get("fee_multiplier")),
    }


def candle_rows(market_id: str, candles: list[dict]) -> list[dict]:
    rows = []
    for c in candles:
        bid, ask, price = c.get("yes_bid") or {}, c.get("yes_ask") or {}, c.get("price") or {}
        rows.append({
            "venue": "kalshi", "market_id": market_id,
            "ts": datetime.fromtimestamp(c["end_period_ts"], tz=timezone.utc),
            "yes_bid": _dollars(bid, "close"), "yes_ask": _dollars(ask, "close"),
            "last": _dollars(price, "close"), "volume": _num(c.get("volume_fp", c.get("volume"))),
            "open_interest": _num(c.get("open_interest_fp", c.get("open_interest"))),
        })
    return rows

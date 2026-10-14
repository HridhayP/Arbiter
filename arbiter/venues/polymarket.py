"""Polymarket public market data (gamma API for metadata, CLOB for prices).

A binary market has two outcomes and two CLOB tokens; this project calls the
first outcome "yes" (for "Will X happen?" markets it is literally "Yes").
A closed market resolved cleanly when its final outcome prices are 1 and 0.
Trade prints from data-api carry wallet addresses, which are never stored.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from arbiter.http import RateLimitedClient

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"


def _list(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            return []
    return x or []


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


class Polymarket:
    def __init__(self, client: RateLimitedClient):
        self.c = client

    MAX_OFFSET = 1900  # the API rejects offsets past ~2000 (HTTP 422)

    def closed_market_pages(self, min_volume: float, start: str = "2020-06-01", end: str | None = None,
                            window_days: float = 7.0):
        yield from self.market_pages(True, min_volume, start, end, window_days)

    def market_pages(self, closed: bool, min_volume: float, start: str = "2020-06-01", end: str | None = None,
                     window_days: float = 7.0):
        """All closed markets with volume >= min_volume, by end-date windows.

        Offset paging is capped, so walk the timeline in windows and page
        within each; a window that would need offsets past the cap is split
        in half. The page size is capped by the API (100), so advance by what
        came back and stop on an empty page."""
        t0 = pd.Timestamp(start, tz="UTC")
        t_end = pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)
        stack = []
        t = t0
        while t < t_end:
            stack.append((t, min(t + pd.Timedelta(days=window_days), t_end)))
            t = stack[-1][1]
        stack.reverse()
        seen: set[str] = set()  # window bounds are inclusive, so edge markets repeat
        while stack:
            lo, hi = stack.pop()
            pages, offset, overflow = [], 0, False
            while True:
                if offset > self.MAX_OFFSET:
                    overflow = True
                    break
                page = self.c.get_json(f"{GAMMA}/markets", {
                    "closed": "true" if closed else "false", **({} if closed else {"active": "true"}),
                    "limit": 100, "offset": offset, "volume_num_min": min_volume,
                    "end_date_min": lo.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "end_date_max": hi.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "order": "id", "ascending": "true"})
                if not page:
                    break
                pages.append(page)
                offset += len(page)
            if overflow and hi - lo > pd.Timedelta(hours=1):
                mid = lo + (hi - lo) / 2
                stack += [(mid, hi), (lo, mid)]
                continue
            for page in pages:
                fresh = [m for m in page if str(m.get("id")) not in seen]
                seen.update(str(m.get("id")) for m in fresh)
                if fresh:
                    yield fresh

    def active_markets(self, min_volume: float = 0, horizon_days: int = 730) -> list[dict]:
        """Open markets ending between a month ago and `horizon_days` ahead
        (some open markets have stale end dates)."""
        now = pd.Timestamp.now(tz="UTC")
        start = (now - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        end = (now + pd.Timedelta(days=horizon_days)).strftime("%Y-%m-%d")
        return [m for page in self.market_pages(False, min_volume, start, end, window_days=3) for m in page]

    def price_history(self, token: str, start: int, end: int, fidelity: int = 60) -> list[dict]:
        r = self.c.get_json(f"{CLOB}/prices-history", {"market": token, "startTs": start, "endTs": end,
                                                       "fidelity": fidelity})
        return (r or {}).get("history", [])

    def book(self, token: str) -> dict | None:
        return self.c.get_json(f"{CLOB}/book", {"token_id": token})


def market_row(m: dict) -> dict | None:
    outcomes, prices, tokens = _list(m.get("outcomes")), _list(m.get("outcomePrices")), _list(m.get("clobTokenIds"))
    if len(outcomes) != 2 or len(tokens) != 2:
        return None
    p = [_num(x) for x in prices] if len(prices) == 2 else [None, None]
    result = None
    if m.get("closed") and p[0] is not None and p[1] is not None:
        if p[0] >= 0.99 and p[1] <= 0.01:
            result = "yes"
        elif p[1] >= 0.99 and p[0] <= 0.01:
            result = "no"
        else:
            result = "void"  # 50/50 or unresolved
    ev = (m.get("events") or [{}])[0]
    sched = m.get("feeSchedule") or {}
    return {
        "venue": "polymarket",
        "market_id": str(m.get("conditionId") or m.get("id")),
        "event_id": str(ev.get("id")) if ev.get("id") else None,
        "series_id": ev.get("seriesSlug") or ev.get("slug"),
        "title": m.get("question"),
        "subtitle": f"{outcomes[0]} vs {outcomes[1]}" if outcomes[0].lower() != "yes" else None,
        "category": m.get("category") or ev.get("category"),
        "frequency": None,
        "market_type": "binary",
        "open_time": pd.Timestamp(m["startDate"]) if m.get("startDate") else None,
        "close_time": pd.Timestamp(m.get("closedTime") or m.get("endDate")) if (m.get("closedTime") or m.get("endDate")) else None,
        "settle_time": pd.Timestamp(m["closedTime"]) if m.get("closedTime") else None,
        # closedTime moves with the outcome ("by date X" markets close early on
        # YES); endDate is the deadline set at listing
        "scheduled_close": pd.Timestamp(m["endDate"]) if m.get("endDate") else None,
        "result": result,
        "strike": None,
        "strike_type": None,
        "volume": _num(m.get("volumeNum") or m.get("volume")),
        "open_interest": None,
        "last_price": _num(m.get("lastTradePrice")),
        "rules_text": m.get("description"),
        "fee_type": "polymarket" if m.get("feesEnabled") else "free",
        "fee_multiplier": _num(sched.get("rate")) if m.get("feesEnabled") else 0.0,
        "token_yes": tokens[0],
        "token_no": tokens[1],
        "neg_risk": bool(m.get("negRisk")),
    }


def price_rows(market_id: str, history: list[dict]) -> list[dict]:
    return [{"venue": "polymarket", "market_id": market_id,
             "ts": datetime.fromtimestamp(h["t"], tz=timezone.utc), "yes_bid": None, "yes_ask": None,
             "last": _num(h.get("p")), "volume": None, "open_interest": None} for h in history]

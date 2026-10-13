"""Data collection CLI (resumable; raw responses are kept under data/raw).

    python -m arbiter.collect kalshi-series
    python -m arbiter.collect kalshi-markets --per-series 300
    python -m arbiter.collect kalshi-candles --horizon-days 7 --min-volume 500
    python -m arbiter.collect polymarket-markets --min-volume 1000
    python -m arbiter.collect polymarket-prices --horizon-days 7
    python -m arbiter.collect tables          # rebuild parquet tables from raw
"""

from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from arbiter import store
from arbiter.http import RateLimitedClient
from arbiter.venues import kalshi as K
from arbiter.venues import polymarket as P

log = logging.getLogger("arbiter.collect")


def kalshi_series(client) -> None:
    s = K.Kalshi(client).series()
    store.save_raw("kalshi", "series", "all", s)
    log.info("kalshi series: %d", len(s))


def _series_meta() -> dict[str, dict]:
    return {s["ticker"]: s for s in (store.load_raw("kalshi", "series", "all") or [])}


def kalshi_markets(client, per_series: int, workers: int) -> None:
    api = K.Kalshi(client)
    meta = _series_meta()
    todo = [t for t in meta if not K.is_combo(t)]
    state = store.State("kalshi-markets")
    todo = [t for t in todo if t not in state]
    log.info("kalshi markets: %d series to crawl (%d done)", len(todo), len(state.done))
    t0, n = time.time(), 0

    def one(series):
        ms = api.settled_markets(series, per_series)
        store.save_raw("kalshi", "markets", series, ms)
        return series, len(ms)

    with ThreadPoolExecutor(workers) as ex:
        for series, count in ex.map(one, todo):
            state.add(series)
            n += count
            if len(state.done) % 500 == 0:
                log.info("  %d series done, %d markets, %.0f req/s", len(state.done), n,
                         client.requests / max(1e-9, time.time() - t0))
    state.flush()


def kalshi_candles(client, horizon_days: int, min_volume: float, per_series: int, workers: int) -> None:
    """Hourly candles over the last `horizon_days` before close, for settled
    markets with enough volume (at most `per_series` per series)."""
    api = K.Kalshi(client)
    mk = store.read_table("markets")
    mk = mk[(mk.venue == "kalshi") & mk.result.isin(["yes", "no"]) & (mk.volume >= min_volume)]
    mk = mk.sort_values("close_time", ascending=False).groupby("series_id").head(per_series)
    cutoff = pd.Timestamp("2026-08-14", tz="UTC")  # /historical/cutoff market_settled_ts
    state = store.State("kalshi-candles", every=200)
    rows = [r for r in mk.itertuples() if r.market_id not in state]
    log.info("kalshi candles: %d markets to fetch", len(rows))

    def one(r):
        end = int(r.close_time.timestamp())
        start = max(end - horizon_days * 86400, int(r.open_time.timestamp()) if pd.notna(r.open_time) else 0)
        hist = r.settle_time is not None and pd.notna(r.settle_time) and r.settle_time < cutoff
        c = api.candles(r.series_id, r.market_id, start, end, 60, historical=hist)
        if not c and not hist:  # settled near the cutoff: try the other endpoint
            c = api.candles(r.series_id, r.market_id, start, end, 60, historical=True)
        store.save_raw("kalshi", "candles", r.market_id, c)
        return r.market_id

    with ThreadPoolExecutor(workers) as ex:
        for i, mid in enumerate(ex.map(one, rows)):
            state.add(mid)
            if i % 2000 == 0:
                log.info("  %d / %d", i, len(rows))
    state.flush()


def polymarket_markets(client, min_volume: float) -> None:
    api = P.Polymarket(client)
    n = 0
    for i, page in enumerate(api.closed_market_pages(min_volume)):
        store.save_raw("polymarket", "markets", f"page{i:06d}", page)
        n += len(page)
        if i % 20 == 0:
            log.info("  polymarket: %d markets", n)
    log.info("polymarket closed markets: %d", n)


def polymarket_prices(client, horizon_days: int, workers: int, min_volume: float, per_event: int,
                      max_markets: int | None = None, seed: int = 0) -> None:
    """Hourly price history before close for resolved markets. There are
    ~500k closed markets over $1k volume (mostly sports micro-markets), so
    sample: volume >= min_volume and at most `per_event` per event, so no
    single event or category dominates the calibration sample."""
    api = P.Polymarket(client)
    mk = store.read_table("markets")
    mk = mk[(mk.venue == "polymarket") & mk.result.isin(["yes", "no"]) & mk.token_yes.notna()
            & (mk.volume >= min_volume)]
    mk = (mk.sort_values("volume", ascending=False)
          .groupby(mk.event_id.fillna(mk.market_id), group_keys=False).head(per_event))
    if max_markets and len(mk) > max_markets:
        mk = mk.sample(max_markets, random_state=seed)  # uniform over the eligible set
    state = store.State("polymarket-prices", every=200)
    rows = [r for r in mk.itertuples() if r.market_id not in state]
    log.info("polymarket prices: %d markets", len(rows))

    def one(r):
        end = int(r.close_time.timestamp())
        h = api.price_history(r.token_yes, end - horizon_days * 86400, end, fidelity=60)
        store.save_raw("polymarket", "prices", r.market_id, h)
        return r.market_id

    with ThreadPoolExecutor(workers) as ex:
        for i, mid in enumerate(ex.map(one, rows)):
            state.add(mid)
            if i % 2000 == 0:
                log.info("  %d / %d", i, len(rows))
    state.flush()


def build_tables() -> None:
    meta = _series_meta()
    rows = []
    for series, ms in store.iter_raw("kalshi", "markets"):
        rows += [K.market_row(m, meta.get(series)) for m in ms]
    for _, page in store.iter_raw("polymarket", "markets"):
        rows += [r for r in (P.market_row(m) for m in page) if r]
    markets = pd.DataFrame(rows).drop_duplicates(["venue", "market_id"])
    store.write_table("markets", markets)
    log.info("markets table: %d rows (%s)", len(markets), markets.venue.value_counts().to_dict())

    candles = []
    for mid, c in store.iter_raw("kalshi", "candles"):
        candles += K.candle_rows(mid, c)
    for mid, h in store.iter_raw("polymarket", "prices"):
        candles += P.price_rows(mid, h)
    if candles:
        cdf = pd.DataFrame(candles)
        store.write_table("candles", cdf)
        log.info("candles table: %d rows", len(cdf))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=["kalshi-series", "kalshi-markets", "kalshi-candles", "polymarket-markets",
                                    "polymarket-prices", "tables"])
    ap.add_argument("--per-series", type=int, default=300)
    ap.add_argument("--horizon-days", type=int, default=7)
    ap.add_argument("--min-volume", type=float, default=500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--per-event", type=int, default=5, help="polymarket-prices: max markets per event")
    ap.add_argument("--max-markets", type=int, default=None, help="polymarket-prices: random cap on the sample")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    with RateLimitedClient() as c:
        if a.job == "kalshi-series":
            kalshi_series(c)
        elif a.job == "kalshi-markets":
            kalshi_markets(c, a.per_series, a.workers)
        elif a.job == "kalshi-candles":
            kalshi_candles(c, a.horizon_days, a.min_volume, a.per_series, a.workers)
        elif a.job == "polymarket-markets":
            polymarket_markets(c, a.min_volume)
        elif a.job == "polymarket-prices":
            polymarket_prices(c, a.horizon_days, a.workers, a.min_volume, a.per_event, a.max_markets)
        else:
            build_tables()
        log.info("requests made: %d", c.requests)


if __name__ == "__main__":
    main()

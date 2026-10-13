"""A polite HTTP client for public market-data APIs.

Neither venue returns rate-limit headers on public endpoints, so the client
throttles itself (a minimum interval per host), retries transient failures
with exponential backoff, and honours Retry-After on 429s.
"""

from __future__ import annotations

import random
import threading
import time
from urllib.parse import urlparse

import httpx

DEFAULT_RATES = {  # requests per second, deliberately below documented limits
    "api.elections.kalshi.com": 4.0,  # 8/s drew 429s on /historical
    "gamma-api.polymarket.com": 4.0,
    "clob.polymarket.com": 8.0,
    "data-api.polymarket.com": 4.0,
}


class RateLimitedClient:
    def __init__(self, rates: dict[str, float] | None = None, timeout: float = 30.0, max_retries: int = 6):
        self.rates = {**DEFAULT_RATES, **(rates or {})}
        self.max_retries = max_retries
        self._client = httpx.Client(timeout=timeout, headers={"User-Agent": "arbiter-research/0.1"},
                                    follow_redirects=True)
        self._next_ok: dict[str, float] = {}
        self._lock = threading.Lock()
        self.requests = 0

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _throttle(self, host: str) -> None:
        interval = 1.0 / self.rates.get(host, 2.0)
        with self._lock:
            now = time.monotonic()
            wait = self._next_ok.get(host, now) - now
            self._next_ok[host] = max(now, self._next_ok.get(host, now)) + interval
        if wait > 0:
            time.sleep(wait)

    def get_json(self, url: str, params: dict | None = None):
        host = urlparse(url).hostname or ""
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            self._throttle(host)
            try:
                r = self._client.get(url, params=params)
                self.requests += 1
            except (httpx.TransportError, httpx.TimeoutException):
                if attempt == self.max_retries:
                    raise
            else:
                if r.status_code == 200:
                    return r.json()
                if r.status_code == 404:
                    return None
                if r.status_code not in (429, 500, 502, 503, 504) or attempt == self.max_retries:
                    r.raise_for_status()
                if r.status_code == 429 and r.headers.get("Retry-After", "").isdigit():
                    delay = max(delay, float(r.headers["Retry-After"]))
            time.sleep(delay * (1 + 0.25 * random.random()))
            delay = min(delay * 2, 60.0)
        raise RuntimeError("unreachable")

"""On-disk layout.

    data/raw/<venue>/<kind>/<key>.json.gz   every API response, verbatim, so
                                            tables can be rebuilt without re-crawling
    data/tables/<name>.parquet              normalized tables (see PLAN.md)
    data/state/<job>.json                   crawl progress, for resuming
    data/recorder/<venue>/date=YYYY-MM-DD/  live book and trade capture
"""

from __future__ import annotations

import gzip
import json
import os
import re

import duckdb
import pandas as pd

ROOT = os.environ.get("ARBITER_DATA", os.path.join(os.path.dirname(__file__), "..", "data"))


def path(*parts: str) -> str:
    p = os.path.join(ROOT, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def _safe(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", key)[:200]


def save_raw(venue: str, kind: str, key: str, obj) -> None:
    with gzip.open(path("raw", venue, kind, _safe(key) + ".json.gz"), "wt", encoding="utf-8") as f:
        json.dump(obj, f)


def load_raw(venue: str, kind: str, key: str):
    p = path("raw", venue, kind, _safe(key) + ".json.gz")
    if not os.path.exists(p):
        return None
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def iter_raw(venue: str, kind: str):
    d = os.path.join(ROOT, "raw", venue, kind)
    if not os.path.isdir(d):
        return
    for name in sorted(os.listdir(d)):
        if name.endswith(".json.gz"):
            with gzip.open(os.path.join(d, name), "rt", encoding="utf-8") as f:
                yield name[: -len(".json.gz")], json.load(f)


class State:
    """A JSON set of finished keys, flushed every `every` additions."""

    def __init__(self, job: str, every: int = 50):
        self.file = path("state", job + ".json")
        self.done: set[str] = set(json.load(open(self.file))) if os.path.exists(self.file) else set()
        self.every, self._pending = every, 0

    def __contains__(self, key: str) -> bool:
        return key in self.done

    def add(self, key: str) -> None:
        self.done.add(key)
        self._pending += 1
        if self._pending >= self.every:
            self.flush()

    def flush(self) -> None:
        tmp = self.file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(sorted(self.done), f)
        os.replace(tmp, self.file)
        self._pending = 0


def write_table(name: str, df: pd.DataFrame) -> str:
    p = path("tables", name + ".parquet")
    df.to_parquet(p, index=False)
    return p


def read_table(name: str) -> pd.DataFrame:
    return pd.read_parquet(path("tables", name + ".parquet"))


def db() -> duckdb.DuckDBPyConnection:
    """DuckDB with a view per table in data/tables."""
    con = duckdb.connect()
    tdir = os.path.join(ROOT, "tables")
    if os.path.isdir(tdir):
        for f in os.listdir(tdir):
            if f.endswith(".parquet"):
                con.execute(f"CREATE VIEW {f[:-8]} AS SELECT * FROM read_parquet('{os.path.join(tdir, f)}')")
    return con

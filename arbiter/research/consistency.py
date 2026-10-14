"""R3: are related prediction-market prices consistent with each other?

Every check uses executable prices (bid to sell, ask to buy) from the same
listing snapshot, and charges taker fees on every leg. Each violation is
classified by whether profiting from it needs any assumption:

1. Strike ladders (Kalshi "above X" markets in one event): P(above X) can
   not rise with X. If bid(above X_hi) > ask(above X_lo), buy the low strike
   and sell the high one: the pair pays >= 0 in every outcome. Model-free.
2. Mutually exclusive outcome sets (Kalshi events flagged
   mutually_exclusive, Polymarket neg-risk groups):
   - sum of YES bids > 1: sell every outcome (buy every NO); at most one
     outcome resolves YES, so the payout is >= n - 1 for a cost of
     n - sum(bids). Model-free.
   - sum of YES asks < 1: buy every outcome; only riskless if the set is
     also exhaustive (exactly one resolves YES), which the venues don't
     flag. Most such sets are not: the listing is missing outcomes (a
     third-party candidate, "draw"), so verify_polymarket_buy_all() fetches
     each candidate's event and keeps only sets whose every market was
     quoted; the survivors' rules are then read by hand (labels/r3_review.csv).
3. Combos (Kalshi multivariate markets): a combo pays only if every leg
   resolves as specified. Its price is compared with the product of its
   legs' prices (independence) to measure the combo premium; correlated
   legs make the independence price only a reference, not an arbitrage.
"""

from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import pandas as pd

from arbiter import store
from arbiter.fees import FeeModel, kalshi_fees

KALSHI_TAKER = kalshi_fees("quadratic", 1.0)
POLY_WORST_RATE = 0.07  # listing has no category: charge the highest published taker rate


def load_listing(venue: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(store.ROOT, "recorder", venue, "date=*", "listing-*.parquet")))
    if not files:
        return pd.DataFrame()
    d = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    if venue == "kalshi":
        d = d[d.ts.isin(complete_kalshi_snapshots(d))]
    # Early Polymarket snapshots repeated markets on window edges (pager bug, fixed)
    return d.drop_duplicates(["ts", "market_id"])


def complete_kalshi_snapshots(listing: pd.DataFrame) -> set:
    """Snapshot times whose Kalshi paging finished. Snapshots recorded before
    completeness was logged count as complete only if they hold at least 90%
    of the largest snapshot's market count."""
    files = sorted(glob.glob(os.path.join(store.ROOT, "recorder", "kalshi", "date=*", "snapshots-*.parquet")))
    flagged = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True) if files else pd.DataFrame()
    counts = listing[listing.legs.isna()].groupby("ts").size()
    ok = set(counts[counts >= 0.9 * counts.max()].index)
    if len(flagged):
        ok -= set(flagged.loc[~flagged.complete, "ts"])
        ok |= set(flagged.loc[flagged.complete, "ts"])
    return ok


def _valid(df: pd.DataFrame) -> pd.Series:
    """Rows with a real two-sided quote."""
    return df.yes_bid.gt(0) & df.yes_ask.lt(1) & df.yes_ask.gt(df.yes_bid)


_LADDER = re.compile(r"^(.*-[A-Z]*)(\d+(?:\.\d+)?)$")


def ladder_key(ticker: str) -> str | None:
    """The quantity a strike market is about: its ticker without the trailing
    strike number. Kalshi events group *different* quantities that share a
    strike (rain in different cities, goals by different players), so the
    event is not the ladder; KXFED-26OCT-T4.00 -> KXFED-26OCT-T,
    ...-GIRGENSONS28-1 -> ...-GIRGENSONS28-, ...-TTN (a city) -> None."""
    m = _LADDER.match(ticker)
    return m.group(1) if m else None


def ladder_violations(k: pd.DataFrame, fee: FeeModel = KALSHI_TAKER) -> pd.DataFrame:
    """Within each ladder (same quantity, close time and strike type, one
    snapshot): any pair with a higher strike's bid above a lower strike's
    ask, net of two taker fees."""
    d = k[k.strike.notna() & k.strike_type.isin(["greater", "greater_or_equal"]) & _valid(k)].copy()
    d["ladder"] = d.market_id.map(ladder_key)
    d = d[d.ladder.notna()]
    if "close_time" not in d:
        d["close_time"] = None
    out = []
    # dropna=False: a missing close time must not silently drop a ladder
    for keys, g in d.groupby(["ts", "event_id", "ladder", "close_time", "strike_type"], dropna=False):
        ts, ev = keys[0], keys[1]
        if len(g) < 2:
            continue
        g = g.sort_values("strike")
        strikes, bids, asks = g.strike.values, g.yes_bid.values, g.yes_ask.values
        ids, bsz, asz = g.market_id.values, g.bid_size.values, g.ask_size.values
        # best (cheapest) ask at or below each strike vs bids above it
        for j in range(1, len(g)):
            i = int(np.argmin(asks[:j]))
            edge = bids[j] - asks[i]
            if edge <= 0:
                continue
            net = edge - fee.per_contract(asks[i], maker=False) - fee.per_contract(bids[j], maker=False)
            out.append({"ts": ts, "event_id": ev, "buy": ids[i], "buy_strike": strikes[i], "ask": asks[i],
                        "sell": ids[j], "sell_strike": strikes[j], "bid": bids[j], "gross": edge, "net": net,
                        "size": float(np.nanmin([bsz[j], asz[i]]))})
    return pd.DataFrame(out)


def exclusive_sets(listing: pd.DataFrame, group_col: str, fee_fn) -> pd.DataFrame:
    """Per (snapshot, group): sums of bids and asks over the group's outcomes."""
    d = listing[listing[group_col].notna()]
    rows = []
    for (ts, gid), g in d.groupby(["ts", group_col]):
        if len(g) < 2:
            continue
        quoted = g[_valid(g)]
        rows.append({
            "ts": ts, "group": gid, "n": len(g), "n_quoted": len(quoted),
            "sum_bids": float(g.yes_bid.fillna(0).sum()), "sum_asks": float(g.yes_ask.fillna(1).sum()),
            "fees_sell_all": float(sum(fee_fn(r, b) for r, b in zip(g.itertuples(), g.yes_bid.fillna(0)))),
            "fees_buy_all": float(sum(fee_fn(r, a) for r, a in zip(g.itertuples(), g.yes_ask.fillna(1)))),
            "min_bid_size": float(g.bid_size.min()) if "bid_size" in g else np.nan,
        })
    r = pd.DataFrame(rows)
    if r.empty:
        return r
    # selling every YES at its bid: payout >= n - 1, cost = n - sum_bids
    r["sell_all_net"] = r.sum_bids - 1 - r.fees_sell_all
    # buying every YES at its ask: payout 1 only if exactly one resolves YES
    r["buy_all_net_if_exhaustive"] = 1 - r.sum_asks - r.fees_buy_all
    return r


def combo_premium(k: pd.DataFrame) -> pd.DataFrame:
    """Kalshi combo price vs the product of its legs' prices (same snapshot)."""
    combos = k[k.legs.notna() & _valid(k)]  # most combos are priced on request and rest unquoted
    if combos.empty:
        return pd.DataFrame()
    mid = k.assign(mid=np.where(_valid(k), (k.yes_bid + k.yes_ask) / 2, np.nan)).set_index(["ts", "market_id"]).mid
    rows = []
    for r in combos.itertuples():
        legs = json.loads(r.legs)
        markets = [m.strip() for m in (legs.get("markets") or "").split(",") if m.strip()]
        sides = [s.strip() for s in (legs.get("sides") or "").split(",") if s.strip()]
        if not markets or len(markets) != len(sides):
            continue
        ps = []
        for m, s in zip(markets, sides):
            p = mid.get((r.ts, m), np.nan)
            ps.append(p if s == "yes" else 1 - p)
        if any(np.isnan(ps)):
            continue
        indep = float(np.prod(ps))
        rows.append({"ts": r.ts, "market_id": r.market_id, "legs": len(ps), "independent_price": indep,
                     "yes_bid": r.yes_bid, "yes_ask": r.yes_ask, "last": r.last})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["ask_over_indep"] = out.yes_ask / out.independent_price
    out["ask_minus_indep"] = out.yes_ask - out.independent_price
    return out


def run(verify: bool = False) -> dict:
    k = load_listing("kalshi")
    p = load_listing("polymarket")
    res: dict = {"kalshi_snapshots": int(k.ts.nunique()) if len(k) else 0,
                 "polymarket_snapshots": int(p.ts.nunique()) if len(p) else 0}
    if len(k):
        lad = ladder_violations(k)
        res["ladder"] = {"violations": int(len(lad)), "net_positive": int((lad.net > 0).sum()) if len(lad) else 0,
                         "events": int(lad.event_id.nunique()) if len(lad) else 0,
                         "median_gross_cents": float(lad.gross.median() * 100) if len(lad) else None}
        store.write_table("r3_ladder_violations", lad) if len(lad) else None
        ev = store.load_raw("kalshi", "open", "events") or []
        me = {e["event_ticker"] for e in ev if e.get("mutually_exclusive")}
        kk = k.assign(me_event=np.where(k.event_id.isin(me), k.event_id, None))
        sets = exclusive_sets(kk, "me_event", lambda r, price: KALSHI_TAKER.per_contract(price, maker=False))
        if len(sets):
            store.write_table("r3_kalshi_exclusive", sets)
            res["kalshi_exclusive"] = _summ(sets)
        combo = combo_premium(k)
        if len(combo):
            store.write_table("r3_combos", combo)
            res["combos"] = {"priced": int(len(combo)), "median_ask_over_indep": float(combo.ask_over_indep.median()),
                             "median_ask_minus_indep_cents": float(combo.ask_minus_indep.median() * 100),
                             "share_ask_below_indep": float((combo.ask_minus_indep < 0).mean()),
                             "by_legs": combo.groupby("legs").ask_over_indep.median().to_dict()}
    if len(p):
        pfee = lambda r, price: (POLY_WORST_RATE * price * (1 - price)) if getattr(r, "fees_enabled", False) else 0.0  # noqa: E731
        sets = exclusive_sets(p, "neg_risk_id", pfee)
        if len(sets):
            store.write_table("r3_polymarket_negrisk", sets)
            res["polymarket_negrisk"] = _summ(sets)
            if verify:
                v = verify_polymarket_buy_all(sets, p)
                if len(v):
                    store.write_table("r3_polymarket_buy_all_verified", v)
                    res["polymarket_negrisk"]["buy_all_candidates_complete_sets"] = int(v.complete.sum())
                    res["polymarket_negrisk"]["buy_all_candidates_missing_outcomes"] = int((~v.complete).sum())
    return res


def verify_polymarket_buy_all(sets: pd.DataFrame, listing: pd.DataFrame) -> pd.DataFrame:
    """For neg-risk groups whose asks sum below 1 (after fees): does the
    listing hold every market of the group's event? Network: one gamma call
    per group."""
    from arbiter.http import RateLimitedClient
    from arbiter.venues.polymarket import GAMMA

    pos = sets[sets.buy_all_net_if_exhaustive > 0]
    event_of = listing.drop_duplicates("neg_risk_id").set_index("neg_risk_id").event_id
    rows = []
    with RateLimitedClient() as c:
        for gid, d in pos.groupby("group"):
            e = c.get_json(f"{GAMMA}/events/{event_of.get(gid)}") or {}
            ms = e.get("markets") or []
            rows.append({"group": gid, "event_id": event_of.get(gid), "title": e.get("title"),
                         "n_listing": int(d.n.max()), "n_event": len(ms),
                         "snapshots_positive": int(len(d)), "best_net": float(d.buy_all_net_if_exhaustive.max())})
    r = pd.DataFrame(rows)
    if len(r):
        r["complete"] = r.n_listing >= r.n_event
    return r


def _summ(s: pd.DataFrame) -> dict:
    return {"group_snapshots": int(len(s)), "groups": int(s.group.nunique()),
            "sell_all_net_positive": int((s.sell_all_net > 0).sum()),
            "buy_all_net_positive_if_exhaustive": int((s.buy_all_net_if_exhaustive > 0).sum()),
            "median_sum_bids": float(s.sum_bids.median()), "median_sum_asks": float(s.sum_asks.median())}


if __name__ == "__main__":
    import sys

    r = run(verify="--verify" in sys.argv)
    with open(store.path("reports", "consistency.json"), "w") as f:
        json.dump(r, f, indent=2, default=str)
    print(json.dumps(r, indent=2, default=str))

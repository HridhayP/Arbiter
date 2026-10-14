"""R2: do Kalshi and Polymarket price the same proposition differently, and
can the difference be traded after fees?

Pairs come from the matcher (research/matching.py, p_match >= 0.7, 95%
precision in cross-validation on hand labels). For every complete listing
snapshot that quotes both sides of a pair:

- mid gap: Polymarket mid - Kalshi mid (who prices it higher, by how much)
- executable gap: buying YES on one venue at its ask and NO on the other at
  1 - its bid costs ask_A + 1 - bid_B, and pays exactly 1 whichever way the
  proposition resolves, if both venues resolve it the same way. It is an
  opportunity only if bid_B - ask_A exceeds both venues' taker fees.
- persistence: how many consecutive snapshots an opportunity survives.

Polymarket's listing has no category, so fee-enabled Polymarket markets are
charged the highest published taker rate (0.07): conservative. Every pair
with a positive net gap is printed for manual review of both venues'
resolution rules before it is reported as real.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from arbiter import store
from arbiter.fees import kalshi_fees
from arbiter.research import consistency as R
from arbiter.research import matching as M

KALSHI = kalshi_fees("quadratic", 1.0)
POLY_WORST_RATE = 0.07


def poly_fee(price: float, enabled: bool) -> float:
    return POLY_WORST_RATE * price * (1 - price) if enabled else 0.0


def latest_rows(listing: pd.DataFrame) -> pd.DataFrame:
    """Most recent row per market (for matching on current titles)."""
    return listing.sort_values("ts").groupby("market_id").tail(1)


def match_listing(k: pd.DataFrame, p: pd.DataFrame, labels_path: str = "labels/match_labels.csv",
                  threshold: float = 0.7) -> pd.DataFrame:
    """Runs the matcher on the markets in the listing snapshots."""
    kk = latest_rows(k[k.legs.isna()]).assign(close_time=lambda d: pd.to_datetime(d.close_time, utc=True, format="ISO8601"))
    pp = latest_rows(p).assign(close_time=lambda d: pd.to_datetime(d.close_time, utc=True, format="ISO8601"))
    kk = kk[kk.close_time.notna()]
    pp = pp[pp.close_time.notna()]
    cand = M.add_features(M.candidates(kk, pp))
    lab = M.add_features(pd.read_csv(labels_path))
    model = M.train(lab)["model"]
    cand["p_match"] = M.score(model, cand)
    return cand[cand.p_match >= threshold].sort_values("p_match", ascending=False)


def gaps(pairs: pd.DataFrame, k: pd.DataFrame, p: pd.DataFrame, max_skew: pd.Timedelta = pd.Timedelta(minutes=30)) -> pd.DataFrame:
    """Pair x snapshot table of mid and executable gaps.

    The venues' listing passes start at the same moment but take minutes, so
    each Kalshi snapshot is joined to the Polymarket snapshot nearest in
    time (within max_skew)."""
    kq = k[R._valid(k) & k.market_id.isin(pairs.kalshi_id)][["ts", "market_id", "yes_bid", "yes_ask", "bid_size", "ask_size"]]
    pq = p[R._valid(p) & p.market_id.isin(pairs.poly_id)][["ts", "market_id", "yes_bid", "yes_ask", "fees_enabled"]]
    kq = kq.rename(columns={"market_id": "kalshi_id", "ts": "ts_k", "yes_bid": "k_bid", "yes_ask": "k_ask"})
    pq = pq.rename(columns={"market_id": "poly_id", "ts": "ts_p", "yes_bid": "p_bid", "yes_ask": "p_ask"})
    snap_k = pd.DatetimeIndex(sorted(kq.ts_k.unique()))
    snap_p = pd.DatetimeIndex(sorted(pq.ts_p.unique()))
    if not len(snap_k) or not len(snap_p):
        return pd.DataFrame()
    j = snap_p.get_indexer(snap_k, method="nearest")
    nearest = {tk: snap_p[i] for tk, i in zip(snap_k, j) if abs(snap_p[i] - tk) <= max_skew}
    kq = kq[kq.ts_k.isin(nearest)].assign(ts_p=lambda d: d.ts_k.map(nearest))
    df = (pairs[["kalshi_id", "poly_id", "p_match", "kalshi_text", "poly_text"]]
          .merge(kq, on="kalshi_id").merge(pq, on=["poly_id", "ts_p"]))
    if df.empty:
        return df
    df["mid_gap"] = (df.p_bid + df.p_ask) / 2 - (df.k_bid + df.k_ask) / 2
    # Direction 1: buy YES on Kalshi at its ask, buy NO on Polymarket at 1 - p_bid.
    f1 = [KALSHI.per_contract(a, maker=False) + poly_fee(1 - b, e) for a, b, e in zip(df.k_ask, df.p_bid, df.fees_enabled)]
    # Direction 2: buy YES on Polymarket at its ask, buy NO on Kalshi at 1 - k_bid.
    f2 = [poly_fee(a, e) + KALSHI.per_contract(1 - b, maker=False) for a, b, e in zip(df.p_ask, df.k_bid, df.fees_enabled)]
    df["gross_1"] = df.p_bid - df.k_ask
    df["gross_2"] = df.k_bid - df.p_ask
    df["net_1"] = df.gross_1 - np.array(f1)
    df["net_2"] = df.gross_2 - np.array(f2)
    df["net_best"] = df[["net_1", "net_2"]].max(axis=1)
    df["gross_best"] = df[["gross_1", "gross_2"]].max(axis=1)
    return df


def persistence(g: pd.DataFrame) -> pd.DataFrame:
    """Consecutive-snapshot runs of net_best > 0 per pair."""
    rows = []
    snaps = sorted(g.ts_k.unique())
    order = {t: i for i, t in enumerate(snaps)}
    for (k, p), d in g[g.net_best > 0].groupby(["kalshi_id", "poly_id"]):
        idx = sorted(order[t] for t in d.ts_k)
        run, best = 1, 1
        for a, b in zip(idx, idx[1:]):
            run = run + 1 if b == a + 1 else 1
            best = max(best, run)
        rows.append({"kalshi_id": k, "poly_id": p, "snapshots_positive": len(idx), "longest_run": best,
                     "max_net": float(d.net_best.max())})
    return pd.DataFrame(rows)


NOT_EQUIVALENT = {"false_match", "rules_differ"}


def apply_review(g: pd.DataFrame, review: pd.DataFrame) -> pd.DataFrame:
    """Drops pairs a manual read of both venues' rules found not to be the
    same proposition. Matched on the Kalshi id and Polymarket question."""
    bad = review[review.verdict.isin(NOT_EQUIVALENT)][["kalshi_id", "poly_text"]].assign(_bad=True)
    g = g.merge(bad, on=["kalshi_id", "poly_text"], how="left")
    return g[g._bad.isna()].drop(columns="_bad")


def summarize(g: pd.DataFrame) -> dict:
    per_pair = g.groupby(["kalshi_id", "poly_id"])
    res = {
        "pair_snapshots": int(len(g)), "pairs_quoted": int(per_pair.ngroups),
        "abs_mid_gap_median_cents": float(g.mid_gap.abs().median() * 100),
        "abs_mid_gap_p90_cents": float(g.mid_gap.abs().quantile(0.9) * 100),
        "share_gross_positive": float((g.gross_best > 0).mean()),
        "share_net_positive": float((g.net_best > 0).mean()),
        "pairs_ever_net_positive": int(g[g.net_best > 0][["kalshi_id", "poly_id"]].drop_duplicates().shape[0]),
        "max_net_cents": float(g.net_best.max() * 100),
    }
    pers = persistence(g)
    if len(pers):
        res["persistence_median_snapshots"] = float(pers.snapshots_positive.median())
        res["longest_run_max"] = int(pers.longest_run.max())
    return res


def run(review_path: str = "labels/r2_review.csv") -> dict:
    k = R.load_listing("kalshi")
    p = R.load_listing("polymarket")
    pairs = match_listing(k, p)
    store.write_table("r2_pairs", pairs)
    g = gaps(pairs, k, p)
    store.write_table("r2_gaps", g) if len(g) else None
    res = {"pairs": int(len(pairs)), "kalshi_snapshots": int(k.ts.nunique()), "polymarket_snapshots": int(p.ts.nunique())}
    if len(g):
        res["all_matched"] = summarize(g)
        review = pd.read_csv(review_path)
        clean = apply_review(g, review)
        res["after_review"] = summarize(clean) if len(clean) else {}
        res["review"] = review.verdict.value_counts().to_dict()
        # pairs with a positive net gap that nobody has read the rules for yet
        pos = g[g.net_best > 0][["kalshi_id", "poly_text"]].drop_duplicates()
        res["unreviewed_positive"] = int(len(pos.merge(review[["kalshi_id", "poly_text"]], how="left", indicator=True)
                                             .query("_merge == 'left_only'")))
        pers = persistence(g)
        if len(pers):
            store.write_table("r2_persistence", pers)
        # one row per pair, with the review verdict, for the report and its chart
        per = (g.groupby(["kalshi_id", "poly_id", "kalshi_text", "poly_text"])
               .agg(snapshots=("ts_k", "size"), max_net=("net_best", "max"), mean_abs_mid_gap=("mid_gap", lambda x: x.abs().mean()))
               .reset_index().merge(review[["kalshi_id", "poly_text", "verdict"]], on=["kalshi_id", "poly_text"], how="left"))
        default = pd.Series(np.where(per.max_net > 0, "unreviewed", "not_reviewed_never_positive"), index=per.index)
        per["verdict"] = per.verdict.fillna(default)
        per.to_csv(store.path("reports", "r2_pairs.csv"), index=False)
    return res


if __name__ == "__main__":
    import json

    r = run()
    with open(store.path("reports", "crossvenue.json"), "w") as f:
        json.dump(r, f, indent=2, default=str)
    print(json.dumps(r, indent=2, default=str))

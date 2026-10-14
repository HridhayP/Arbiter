"""R5b: can a model beat the market's own price at forecasting resolution?

For each resolved market, features are computed from information available
at the sampling time (1 day before close by default): the price itself (as
log-odds), recent price moves, the bid/ask spread, volume, time since open,
category and venue. Models are fit on markets that closed earlier and scored
on markets that closed later (time-ordered folds), and compared with the
market price as a forecast by Brier score, log loss and calibration.

The market price is a strong baseline: it aggregates traders' information,
so "the model adds nothing" is a plausible, reportable answer. A positive
Brier skill relative to the price, out of sample, is the only result that
counts as beating the market.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from arbiter.research.calibration import logit, scheduled_close
from arbiter.timeutil import utc_us

EPS = 1e-4


def features(markets: pd.DataFrame, candles: pd.DataFrame, horizon=pd.Timedelta(days=1)) -> pd.DataFrame:
    """One row per resolved market, sampled `horizon` before its scheduled
    deadline and only if it was still open then (see calibration.snapshots:
    sampling before the actual close leaks the outcome)."""
    m = markets[markets.result.isin(["yes", "no"]) & markets.close_time.notna()].copy()
    m["close_time"] = utc_us(m.close_time)
    m["open_time"] = utc_us(m.open_time)
    m["t"] = utc_us(scheduled_close(m) - horizon)
    m = m[m.t < m.close_time]
    c = candles.copy()
    c["ts"] = utc_us(c.ts)
    c["mid"] = np.where(c.yes_bid.notna() & c.yes_ask.notna(), (c.yes_bid + c.yes_ask) / 2, c["last"])
    c["spread"] = c.yes_ask - c.yes_bid
    c = c[c.mid.notna()].sort_values("ts")
    out = []
    lags = {"now": pd.Timedelta(0), "lag6h": pd.Timedelta(hours=6), "lag1d": pd.Timedelta(days=1),
            "lag3d": pd.Timedelta(days=3)}
    base = m[["venue", "market_id", "t"]].sort_values("t")
    for name, lag in lags.items():
        q = base.assign(tq=utc_us(base.t - lag)).sort_values("tq")  # timedelta math upcasts to ns
        j = pd.merge_asof(q, c[["venue", "market_id", "ts", "mid", "spread", "volume"]], left_on="tq", right_on="ts",
                          by=["venue", "market_id"], direction="backward", tolerance=pd.Timedelta(hours=12))
        j = j[["venue", "market_id", "mid", "spread", "volume"]].rename(
            columns={"mid": f"p_{name}", "spread": f"spread_{name}", "volume": f"vol_{name}"})
        out.append(j)
    f = m[["venue", "market_id", "event_id", "category", "close_time", "open_time", "result", "t"]]
    for j in out:
        f = f.merge(j, on=["venue", "market_id"], how="left")
    f = f[f.p_now.notna()].copy()
    f["y"] = (f.result == "yes").astype(int)
    f["x_now"] = logit(f.p_now)
    for lag in ("lag6h", "lag1d", "lag3d"):
        f[f"dx_{lag}"] = (f.x_now - logit(f[f"p_{lag}"])).fillna(0.0)
    f["spread"] = f.spread_now.fillna(f.spread_now.median() if f.spread_now.notna().any() else 0.0)
    f["log_age_days"] = np.log1p(((f.t - f.open_time) / pd.Timedelta(days=1)).clip(lower=0).fillna(0))
    f["extreme"] = (np.abs(f.p_now - 0.5) * 2) ** 2
    return f


NUMERIC = ["x_now", "dx_lag6h", "dx_lag1d", "dx_lag3d", "spread", "log_age_days", "extreme"]


def _design(df: pd.DataFrame, cats: list[str]) -> np.ndarray:
    x = df[NUMERIC].astype(float).values
    onehot = np.stack([(df.category.fillna("other") == c).astype(float).values for c in cats], axis=1) if cats else \
        np.zeros((len(df), 0))
    venue = (df.venue == "kalshi").astype(float).values[:, None]
    # interactions: lets the price's weight differ by category and venue
    inter = onehot * df.x_now.values[:, None]
    return np.hstack([x, onehot, venue, inter, venue * df.x_now.values[:, None]])


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def logloss(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def evaluate(f: pd.DataFrame, n_folds: int = 5, min_cat: int = 200, seed: int = 0) -> dict:
    """Time-ordered folds: fit on all earlier folds, score the next.
    Returns Brier and log loss for the market price, a logistic model and a
    gradient-boosted model, with a paired bootstrap CI (by event) on the
    Brier difference model - market."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    f = f.sort_values("close_time").reset_index(drop=True)
    folds = np.array_split(np.arange(len(f)), n_folds)
    preds = {"market": [], "logistic": [], "gbm": []}
    rows = []
    for i in range(1, n_folds):
        tr = f.iloc[np.concatenate(folds[:i])]
        te = f.iloc[folds[i]]
        cats = [c for c, n in tr.category.fillna("other").value_counts().items() if n >= min_cat]
        xtr, xte = _design(tr, cats), _design(te, cats)
        sc = StandardScaler().fit(xtr)
        lr = LogisticRegression(C=0.5, max_iter=2000).fit(sc.transform(xtr), tr.y)
        gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                            l2_regularization=1.0, random_state=seed).fit(xtr, tr.y)
        p_lr = lr.predict_proba(sc.transform(xte))[:, 1]
        p_gb = gb.predict_proba(xte)[:, 1]
        preds["market"].append(te.p_now.values)
        preds["logistic"].append(p_lr)
        preds["gbm"].append(p_gb)
        rows.append(te[["venue", "market_id", "event_id", "category", "y", "p_now"]].assign(fold=i, p_lr=p_lr, p_gb=p_gb))
    scored = pd.concat(rows, ignore_index=True)
    y = scored.y.values
    res = {"n_scored": int(len(scored)), "events": int(scored.event_id.nunique()), "folds": n_folds - 1}
    for name, col in (("market", "p_now"), ("logistic", "p_lr"), ("gbm", "p_gb")):
        res[name] = {"brier": brier(scored[col], y), "log_loss": logloss(scored[col], y)}
    rng = np.random.default_rng(seed)
    groups = scored.event_id.fillna(scored.market_id).values
    uniq, inv = np.unique(groups, return_inverse=True)
    idx = pd.Series(np.arange(len(scored))).groupby(inv).apply(np.array).values
    for name, col in (("logistic", "p_lr"), ("gbm", "p_gb")):
        d = (scored[col].values - y) ** 2 - (scored.p_now.values - y) ** 2  # < 0 means the model beats the market
        boots = []
        for _ in range(1000):
            pick = np.concatenate([idx[j] for j in rng.integers(0, len(uniq), len(uniq))])
            boots.append(d[pick].mean())
        res[name]["brier_minus_market"] = float(d.mean())
        res[name]["ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        res[name]["brier_skill_vs_market"] = float(1 - res[name]["brier"] / res["market"]["brier"])
    return {"summary": res, "scored": scored}


if __name__ == "__main__":
    import json

    from arbiter import store

    f = features(store.read_table("markets"), store.read_table("candles"))
    out = evaluate(f)
    store.write_table("r5_forecast_scored", out["scored"])
    with open(store.path("reports", "forecast.json"), "w") as fh:
        json.dump(out["summary"], fh, indent=2)
    print(json.dumps(out["summary"], indent=2))

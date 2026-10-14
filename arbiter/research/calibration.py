"""R1: are prediction-market prices calibrated? Is there a favorite-longshot bias?

For every resolved binary market, the price is read at fixed horizons
before close (1 hour, 1 day, 1 week): Kalshi uses the bid/ask midpoint of
the last hourly candle at or before that time; Polymarket uses its
price-history series. Settlement prices are never used.

Outputs
- reliability table (price bins vs realized frequency, event-clustered
  bootstrap CIs)
- calibration regression  logit P(yes) = a + b * logit(price), GLM with
  event-clustered SEs: b > 1 means realized odds are more extreme than prices
  (longshots overpriced, favorites underpriced: the favorite-longshot bias)
- Brier score / log loss and Brier skill vs the base rate
- a trading test: buy the favorite side when the price is beyond a threshold,
  paying the real ask plus fees (Kalshi), with thresholds chosen
  walk-forward on earlier months only
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from arbiter.fees import kalshi_fees
from arbiter.timeutil import utc_us

HORIZONS = {"1h": pd.Timedelta(hours=1), "1d": pd.Timedelta(days=1), "1w": pd.Timedelta(days=7)}
EPS = 1e-4


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def scheduled_close(m: pd.DataFrame) -> pd.Series:
    """The deadline a market was listed with, which does not depend on the
    outcome. Polymarket keeps it (endDate). Kalshi overwrites close_time on an
    early close, so the latest close among the event's markets stands in."""
    sched = utc_us(m.scheduled_close) if "scheduled_close" in m else m.close_time.copy()
    is_k = m.venue == "kalshi"
    ev_max = m[is_k].groupby(m[is_k].event_id.fillna(m[is_k].market_id)).close_time.transform("max")
    sched[is_k] = ev_max
    return sched.fillna(m.close_time)


def snapshots(markets: pd.DataFrame, candles: pd.DataFrame, max_staleness: pd.Timedelta = pd.Timedelta(hours=6),
              max_spread: float = 0.10, anchor: str = "scheduled") -> pd.DataFrame:
    """One row per (market, horizon) with the price at anchor - horizon.

    anchor="scheduled" (default) samples before the listed deadline and keeps
    a market only if it was still open then. That condition is known at the
    time, so it adds no look-ahead. anchor="close" samples before the
    *actual* close. That is biased: markets that close early because the
    event happened ("by date X" resolving YES) are sampled before the surprise
    instead of at the dead end, which makes longshots look underpriced.

    Kalshi rows require both a bid and an ask with spread <= max_spread (so
    the midpoint is a real price, not an empty book); the candle used must be
    at most max_staleness old. Polymarket rows use the history price."""
    m = markets[markets.result.isin(["yes", "no"]) & markets.close_time.notna()].copy()
    m["close_time"] = utc_us(m.close_time)
    m["open_time"] = utc_us(m.open_time)
    m["anchor"] = scheduled_close(m) if anchor == "scheduled" else m.close_time
    m["y"] = (m.result == "yes").astype(int)
    c = candles.copy()
    c["ts"] = utc_us(c.ts)
    c["mid"] = np.where(c.yes_bid.notna() & c.yes_ask.notna(), (c.yes_bid + c.yes_ask) / 2, np.nan)
    c["spread"] = c.yes_ask - c.yes_bid
    is_k = c.venue == "kalshi"
    c["price"] = np.where(is_k, c.mid, c["last"])
    c = c[c.price.notna() & (~is_k | (c.spread <= max_spread)) & (~is_k | (c.yes_bid > 0) | (c.yes_ask < 1))]
    c = c.sort_values("ts")
    out = []
    for name, h in HORIZONS.items():
        t = m[["venue", "market_id", "close_time", "open_time", "anchor"]].copy()
        t["t"] = utc_us(t.anchor - h)  # timedelta math upcasts to ns
        t = t[(t.open_time.isna() | (t.open_time <= t.t)) & (t.t < t.close_time)].sort_values("t")
        j = pd.merge_asof(t, c[["venue", "market_id", "ts", "price", "yes_bid", "yes_ask", "spread"]],
                          left_on="t", right_on="ts", by=["venue", "market_id"], direction="backward")
        j = j[j.ts.notna() & (j.t - j.ts <= max_staleness)]
        j["horizon"] = name
        out.append(j)
    s = pd.concat(out, ignore_index=True)
    keep = ["venue", "market_id", "event_id", "series_id", "category", "volume", "y", "fee_type", "fee_multiplier"]
    return s.merge(m[keep], on=["venue", "market_id"])


def reliability(s: pd.DataFrame, bins=(0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, 1.0001),
                n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """Realized frequency per price bin with 95% CIs from an event-clustered bootstrap."""
    s = s.assign(bin=pd.cut(s.price, bins, right=False))
    g = s.groupby("bin", observed=True)
    table = g.agg(n=("y", "size"), mean_price=("price", "mean"), freq=("y", "mean"),
                  events=("event_id", "nunique")).reset_index()
    rng = np.random.default_rng(seed)
    events = s.event_id.fillna(s.market_id).values
    uniq, inv = np.unique(events, return_inverse=True)
    idx_by_event = pd.Series(np.arange(len(s))).groupby(inv).apply(np.array).values
    bin_codes = s["bin"].cat.codes.values
    y = s.y.values
    boots = np.zeros((n_boot, len(table)))
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        rows = np.concatenate([idx_by_event[i] for i in pick])
        bc, yy = bin_codes[rows], y[rows]
        for k in range(len(table)):
            sel = bc == k
            boots[b, k] = yy[sel].mean() if sel.any() else np.nan
    table["ci_lo"] = np.nanpercentile(boots, 2.5, axis=0)
    table["ci_hi"] = np.nanpercentile(boots, 97.5, axis=0)
    table["bin"] = table["bin"].astype(str)
    return table


@dataclass
class CalibrationFit:
    n: int
    events: int
    intercept: float
    slope: float
    slope_se: float
    slope_ci: tuple[float, float]
    brier: float
    brier_base: float
    brier_skill: float
    log_loss: float

    def as_dict(self):
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


def fit(s: pd.DataFrame) -> CalibrationFit:
    x = sm.add_constant(logit(s.price.values))
    groups = pd.factorize(s.event_id.fillna(s.market_id))[0]
    res = sm.GLM(s.y.values, x, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": groups})
    p = np.clip(s.price.values, EPS, 1 - EPS)
    y = s.y.values
    brier = float(np.mean((p - y) ** 2))
    base = float(np.mean((y.mean() - y) ** 2))
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    b, se = float(res.params[1]), float(res.bse[1])
    return CalibrationFit(len(s), int(len(np.unique(groups))), float(res.params[0]), b, se,
                          (b - 1.96 * se, b + 1.96 * se), brier, base, 1 - brier / base, ll)


def favorite_strategy(s: pd.DataFrame, horizon: str = "1d", thresholds=(0.7, 0.75, 0.8, 0.85, 0.9, 0.95),
                      min_train_months: int = 3) -> dict:
    """Walk-forward test on Kalshi: when one side trades above a threshold,
    buy it at the ask and hold to resolution, paying taker fees.

    Each month's threshold is the one with the best mean return per contract
    over all earlier months. Returns per-trade P&L with event-clustered stats."""
    k = s[(s.venue == "kalshi") & (s.horizon == horizon) & s.yes_bid.notna() & s.yes_ask.notna()].copy()
    if k.empty:
        return {}
    k["month"] = k.close_time.dt.tz_convert(None).dt.to_period("M")
    # favorite side and its executable price: YES at yes_ask, or NO at 1 - yes_bid
    fav_yes = k.price >= 0.5
    k["fav_price"] = np.where(fav_yes, k.price, 1 - k.price)
    k["cost"] = np.where(fav_yes, k.yes_ask, 1 - k.yes_bid)
    k["win"] = np.where(fav_yes, k.y == 1, k.y == 0).astype(float)
    fees = [kalshi_fees(ft, fm).per_contract(c, maker=False) for ft, fm, c in zip(k.fee_type, k.fee_multiplier, k.cost)]
    k["pnl"] = k.win - k.cost - np.array(fees)

    months = sorted(k.month.unique())
    trades = []
    chosen = {}
    for i, mo in enumerate(months):
        past = k[k.month < mo]
        if past.month.nunique() < min_train_months:
            continue
        best = max(thresholds, key=lambda t: past[past.fav_price >= t].pnl.mean() if (past.fav_price >= t).sum() >= 30 else -1)
        chosen[str(mo)] = best
        trades.append(k[(k.month == mo) & (k.fav_price >= best)])
    if not trades:
        return {}
    t = pd.concat(trades)
    ev = t.groupby(t.event_id.fillna(t.market_id)).pnl.sum()
    mean_ev = ev.mean()
    se_ev = ev.std(ddof=1) / np.sqrt(len(ev))
    return {"horizon": horizon, "trades": int(len(t)), "events": int(len(ev)),
            "mean_pnl_per_contract": float(t.pnl.mean()), "total_pnl_per_1_contract_each": float(t.pnl.sum()),
            "event_clustered_t": float(mean_ev / se_ev) if se_ev > 0 else None,
            "win_rate": float(t.win.mean()), "mean_cost": float(t.cost.mean()),
            "thresholds_by_month": chosen}


def run(markets: pd.DataFrame, candles: pd.DataFrame) -> dict:
    s = snapshots(markets, candles)
    out = {"snapshots": int(len(s)), "by": {}}
    for (venue, horizon), g in s.groupby(["venue", "horizon"]):
        if len(g) < 200:
            continue
        key = f"{venue}/{horizon}"
        out["by"][key] = {"fit": fit(g).as_dict(), "reliability": reliability(g).to_dict("records")}
        mid = g[(g.price > 0.02) & (g.price < 0.98)]  # exclude effectively-decided markets
        if len(mid) >= 200:
            out["by"][key]["fit_excluding_decided"] = fit(mid).as_dict()
    for (venue, cat), g in s[s.horizon == "1d"].groupby(["venue", "category"]):
        if len(g) >= 300:
            out.setdefault("by_category_1d", {})[f"{venue}/{cat}"] = fit(g).as_dict()
    # Markets were sampled by lifetime volume, and upsets attract volume:
    # if the bias grows with volume, it is (partly) selection.
    d1 = s[s.horizon == "1d"].copy()
    d1["tier"] = d1.groupby("venue").volume.transform(lambda v: pd.qcut(v.rank(method="first"), 3, labels=["low", "mid", "high"]))
    for (venue, tier), g in d1.groupby(["venue", "tier"], observed=True):
        if len(g) >= 300:
            out.setdefault("by_volume_tier_1d", {})[f"{venue}/{tier}"] = fit(g).as_dict()
    # The biased alternative, for comparison: horizons before the actual close.
    naive = snapshots(markets, candles, anchor="close")
    for (venue, horizon), g in naive.groupby(["venue", "horizon"]):
        if len(g) >= 200:
            out.setdefault("anchored_on_actual_close", {})[f"{venue}/{horizon}"] = fit(g).as_dict()
    for h in HORIZONS:
        r = favorite_strategy(s, h)
        if r:
            out.setdefault("favorite_strategy", {})[h] = r
    return out


if __name__ == "__main__":
    from arbiter import store

    res = run(store.read_table("markets"), store.read_table("candles"))
    p = store.path("reports", "calibration.json")
    with open(p, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(json.dumps({k: v for k, v in res.items() if k != "by"}, indent=2, default=str)[:3000])

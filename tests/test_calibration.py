import numpy as np
import pandas as pd
import pytest

from arbiter.research import calibration as C


def synthetic(n_events=1500, per_event=3, bias=1.0, seed=0, venue="kalshi"):
    """Markets whose quoted price compresses the true log-odds by `bias`
    (bias > 1: longshots overpriced, the favorite-longshot pattern), with
    hourly candles carrying that price as a 2-cent-wide quote."""
    rng = np.random.default_rng(seed)
    rows, candles = [], []
    t0 = pd.Timestamp("2025-01-01", tz="UTC")
    for e in range(n_events):
        shared = rng.normal(0, 1.5)  # markets in an event share information
        close = t0 + pd.Timedelta(hours=int(rng.integers(0, 24 * 365)))
        for j in range(per_event):
            z = shared + rng.normal(0, 0.7)
            p_true = 1 / (1 + np.exp(-z))
            price = float(np.clip(1 / (1 + np.exp(-z / bias)), 0.02, 0.98))
            y = int(rng.random() < p_true)
            mid = f"E{e}-M{j}"
            rows.append({"venue": venue, "market_id": mid, "event_id": f"E{e}", "series_id": "S", "category": "Test",
                         "volume": 1000.0, "result": "yes" if y else "no", "close_time": close,
                         "open_time": close - pd.Timedelta(days=30), "fee_type": "quadratic", "fee_multiplier": 1.0})
            for h in (1, 24, 24 * 7):
                bid, ask = max(0.0, price - 0.01), min(1.0, price + 0.01)
                candles.append({"venue": venue, "market_id": mid, "ts": close - pd.Timedelta(hours=h),
                                "yes_bid": bid, "yes_ask": ask, "last": price})
    return pd.DataFrame(rows), pd.DataFrame(candles)


@pytest.mark.parametrize("bias,lo,hi", [(1.0, 0.85, 1.15), (1.3, 1.15, 1.45)])
def test_recovers_planted_bias(bias, lo, hi):
    m, c = synthetic(bias=bias)
    s = C.snapshots(m, c)
    assert set(s.horizon) == {"1h", "1d", "1w"}
    f = C.fit(s[s.horizon == "1d"])
    assert lo < f.slope < hi, f
    assert f.events == 1500  # clustered by event, not by market


def test_reliability_bins_track_truth():
    m, c = synthetic(bias=1.0, seed=1)
    s = C.snapshots(m, c)
    t = C.reliability(s[s.horizon == "1d"], n_boot=100)
    big = t[t.n > 300]
    assert (abs(big.freq - big.mean_price) < 0.06).all()
    assert (big.ci_lo <= big.freq).all() and (big.freq <= big.ci_hi).all()


def test_snapshots_never_use_prices_after_the_horizon():
    m, c = synthetic(n_events=50, seed=2)
    # Poison every candle at/after close with the realized outcome.
    leaks = m.assign(ts=m.close_time, yes_bid=m.result.eq("yes").astype(float) - 0.01,
                     yes_ask=m.result.eq("yes").astype(float) + 0.01, last=np.nan)[
        ["venue", "market_id", "ts", "yes_bid", "yes_ask", "last"]]
    s = C.snapshots(m, pd.concat([c, leaks]))
    assert (s.ts <= s.t).all()
    assert not ((s.price < 0.0) | (s.price > 1.0)).any()


def test_favorite_strategy_finds_planted_edge_and_not_a_fake_one():
    m, c = synthetic(n_events=3000, bias=1.4, seed=3)
    r = C.favorite_strategy(C.snapshots(m, c), "1d")
    assert r["trades"] > 100 and r["mean_pnl_per_contract"] > 0 and r["event_clustered_t"] > 2
    m, c = synthetic(n_events=3000, bias=1.0, seed=4)
    r = C.favorite_strategy(C.snapshots(m, c), "1d")
    assert r["event_clustered_t"] < 2  # calibrated prices: spread + fees leave no edge


def by_date_markets(n=1500, days=30, seed=3):
    """'Will it happen by the deadline?' markets with perfectly calibrated
    prices: the event arrives at rate lam per day, the price at time t is
    P(arrival before the deadline | none yet) = 1 - exp(-lam * time left),
    and a market closes the moment the event happens (YES) or at the
    deadline (NO), like Polymarket's closedTime and Kalshi's early close."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2025-01-01", tz="UTC")
    rows, candles = [], []
    for i in range(n):
        lam = rng.uniform(0.002, 0.15)
        open_ = t0 + pd.Timedelta(days=int(rng.integers(0, 300)))
        deadline = open_ + pd.Timedelta(days=days)
        tau = open_ + pd.Timedelta(days=float(rng.exponential(1 / lam)))
        yes = tau < deadline
        close = tau.floor("h") if yes else deadline
        rows.append({"venue": "polymarket", "market_id": f"M{i}", "event_id": f"E{i}", "series_id": "S", "category": "T",
                     "volume": 1.0, "result": "yes" if yes else "no", "close_time": close, "scheduled_close": deadline,
                     "open_time": open_, "fee_type": "free", "fee_multiplier": 0.0})
        for h in range(1, 24 * 8 + 1):
            ts = close - pd.Timedelta(hours=h)
            if ts < open_:
                break
            left = (deadline - ts).total_seconds() / 86400
            p = 1 - np.exp(-lam * left)
            candles.append({"venue": "polymarket", "market_id": f"M{i}", "ts": ts, "yes_bid": np.nan, "yes_ask": np.nan, "last": p})
    return pd.DataFrame(rows), pd.DataFrame(candles)


def test_anchoring_on_the_actual_close_fakes_a_longshot_bias():
    m, c = by_date_markets(n=4000)
    honest = C.snapshots(m, c, anchor="scheduled").query("horizon == '1d'")
    naive = C.snapshots(m, c, anchor="close").query("horizon == '1d'")
    # calibrated prices: the win rate matches the average price (4 SE band)
    se = np.sqrt(honest.price.mean() / len(honest))
    assert abs(honest.y.mean() - honest.price.mean()) < 4 * se, (honest.y.mean(), honest.price.mean())
    # YES markets get sampled a day before the surprise: cheap, and they win
    assert naive.y.mean() - naive.price.mean() > 0.1, (naive.y.mean(), naive.price.mean())

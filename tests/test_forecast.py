import numpy as np
import pandas as pd

from arbiter.research import forecast as F
from tests.test_calibration import synthetic


def test_features_use_only_past_prices():
    m, c = synthetic(n_events=300, bias=1.0, seed=5)
    # Poison every candle after close - 1 day with the realized outcome.
    leak = m.assign(ts=m.close_time - pd.Timedelta(hours=1), yes_bid=m.result.eq("yes").astype(float),
                    yes_ask=m.result.eq("yes").astype(float), last=np.nan)[["venue", "market_id", "ts", "yes_bid", "yes_ask", "last"]]
    f = F.features(m, pd.concat([c, leak]).assign(volume=1.0))
    assert len(f) > 0
    assert f.p_now.between(0.0, 1.0).all() and not f.p_now.isin([0.0, 1.0]).any()  # never the leaked outcome


def test_model_beats_a_biased_market_and_not_a_calibrated_one():
    # Biased prices (longshots overpriced): a model on the price can learn the bias.
    m, c = synthetic(n_events=4000, bias=1.6, seed=6)
    r = F.evaluate(F.features(m, c.assign(volume=1.0)), n_folds=4, min_cat=10)["summary"]
    assert r["logistic"]["brier_minus_market"] < 0 and r["logistic"]["ci95"][1] < 0
    # Calibrated prices: nothing to learn; the CI should include zero or favour the market.
    m, c = synthetic(n_events=4000, bias=1.0, seed=7)
    r = F.evaluate(F.features(m, c.assign(volume=1.0)), n_folds=4, min_cat=10)["summary"]
    assert r["logistic"]["ci95"][0] < 0.002 and r["logistic"]["ci95"][1] > -0.002

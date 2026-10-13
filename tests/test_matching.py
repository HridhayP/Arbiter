import numpy as np
import pandas as pd

from arbiter.research import matching as M


def test_rank_extraction_ignores_round_names():
    assert M.ranks("Will X finish 3rd in the first round of the election?") == {3}
    assert M.ranks("Will X win the second-most votes in the first round?") == {2}
    assert M.ranks("Will X win the most seats?") == {1}
    assert M.ranks("Will X advance to the second round?") == set()


def test_dates_numbers_and_stages():
    assert M.dates("Bitcoin price on Oct 16, 2026?") == {(10, 16)}
    assert M.dates("above $90,000 on October 18?") == {(10, 18)}
    assert M.numbers("above $92,000 and 4.5%") == {"92000", "4.5"}
    assert M.numbers("2026 election") == set()  # years are handled separately
    assert M.numbers("Above 8M daily views") == {"8e+06"}
    assert M.stages("qualify for the Worlds final") == {"qualify"}
    assert M.stages("Will T1 win Worlds 2026?") == {"win"}


def test_features_flag_the_near_misses_seen_in_labeling():
    c = pd.DataFrame({
        "kalshi_text": ["Will Ruffin finish 3rd in the first round?", "Bitcoin price on Oct 16, 2026? $90,000 or above",
                        "Will KOI qualify for the Worlds final?", "Will Yuka Asaka win the 2026 Yokohama mayoral election?"],
        "poly_text": ["Will Ruffin win the second-most votes in the first round?", "Will Bitcoin be above $90,000 on October 18?",
                      "Will KOI win Worlds 2026?", "Will Yuka Asaka win the 2026 Yokohama mayoral election?"],
        "emb_cos": [0.9, 0.9, 0.9, 1.0], "gap_days": [0.4, 0.0, 0.0, 0.4]})
    f = M.add_features(c)
    assert f.rank_conflict.tolist() == [True, False, False, False]
    assert f.date_conflict.tolist() == [False, True, False, False]
    assert f.stage_conflict.tolist() == [False, False, True, False]
    h = M.heuristic(f)
    assert h[3] > max(h[:3])  # the genuine match outranks every near miss


def test_candidates_respect_the_close_time_window(monkeypatch):
    # Deterministic "embeddings": identical texts get identical vectors.
    def fake_embed(texts, batch=256):
        out = np.array([[hash(t) % 7, hash(t) % 11, 1.0] for t in texts], dtype=np.float32)
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    monkeypatch.setattr(M, "embed", fake_embed)
    t = pd.Timestamp("2026-10-01", tz="UTC")
    kal = pd.DataFrame({"market_id": ["k1", "k2"], "title": ["Will A win?", "Will A win?"], "subtitle": ["", ""],
                        "close_time": [t, t + pd.Timedelta(days=10)]})
    poly = pd.DataFrame({"market_id": ["p1"], "title": ["Will A win?"], "close_time": [t + pd.Timedelta(days=1)]})
    c = M.candidates(kal, poly, k=5, max_gap_days=3, min_cos=0.5)
    assert c.kalshi_id.tolist() == ["k1"]  # k2 closes 9 days apart: excluded

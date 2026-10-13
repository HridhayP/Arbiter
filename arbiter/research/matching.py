"""R5a: find the same proposition listed on Kalshi and Polymarket.

The venues word things differently ("Atletico Madrid wins" vs "Will Club
Atletico de Madrid win on 2026-10-13?") and slice them differently (price
ranges vs up/down, vote-count thresholds vs winners), so text similarity
alone produces many near misses. The matcher therefore:

1. generates candidates: for each Polymarket market, the most similar Kalshi
   markets by sentence embedding among those closing within `max_gap` days;
2. computes features: embedding and TF-IDF similarity, close-time gap,
   agreement of the numbers in each text (strikes, scores, dates), named
   entity overlap, and conflicting years;
3. scores pairs, first with a transparent heuristic, then with a logistic
   regression trained on hand-labeled pairs (labels/match_labels.csv),
   reporting precision and recall by cross-validation.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_NUM = re.compile(r"(?<![a-z])[-+]?\$?\d[\d,]*(?:\.\d+)?\s*(?:%|k|m|b|bps)?", re.I)
_YEAR = re.compile(r"\b(20[2-3]\d)\b")
_STOP = {"Will", "The", "A", "An", "On", "In", "Of", "Who", "What", "Which", "Before", "After", "By", "Yes", "No",
         "Above", "Below", "Over", "Under", "Win", "Wins", "And", "Or", "Than", "More", "Less", "At", "To", "For"}


def fold(s: str) -> str:
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()


def kalshi_text(row) -> str:
    sub = row.get("subtitle") or ""
    return fold(f"{row['title']} {sub}".strip())


def poly_text(row) -> str:
    return fold(row["title"] or "")


def numbers(s: str) -> set[str]:
    out = set()
    for m in _NUM.findall(s):
        t = m.lower().replace("$", "").replace(",", "").replace(" ", "")
        mult = 1.0
        for suf, f in (("bps", 0.01), ("k", 1e3), ("m", 1e6), ("b", 1e9), ("%", 1.0)):
            if t.endswith(suf):
                t, mult = t[: -len(suf)], f
                break
        try:
            v = float(t) * mult
        except ValueError:
            continue
        if 1900 <= v <= 2100 and float(v).is_integer():
            continue  # years are compared separately
        out.add(f"{v:g}")
    return out


def entities(s: str) -> set[str]:
    return {w for w in re.findall(r"\b[A-Z][a-zA-Z'&.-]+", s) if w not in _STOP and len(w) > 1}


_ORD = {"1st": 1, "first": 1, "2nd": 2, "second": 2, "3rd": 3, "third": 3, "4th": 4, "fourth": 4,
        "5th": 5, "fifth": 5}
_RANK = re.compile(r"\b(1st|2nd|3rd|4th|5th|first|second|third|fourth|fifth)(?:[- ]most)?\b(?![- ]round)", re.I)
_MOST = re.compile(r"\b(?:win|finish with|receive) the most\b|\bfinish 1st\b", re.I)
_MONTHS = "jan feb mar apr may jun jul aug sep oct nov dec".split()
_DATE = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.? (\d{1,2})\b", re.I)
_TOUCH = re.compile(r"\b(hit|hits|reach|reaches|dip|dips|touch|touches)\b", re.I)
# What has to happen: qualifying for a stage is not winning it.
_STAGE = re.compile(r"\b(qualify|qualifies|advance|advances|make the|reach the|win|wins|champion|relegated)\b", re.I)


def ranks(s: str) -> set[int]:
    """Finishing positions mentioned ("finish 2nd", "second-most votes");
    "second round" is a stage, not a rank, and is ignored."""
    out = {_ORD[m.lower()] for m in _RANK.findall(s)}
    if _MOST.search(s):
        out.add(1)
    return out


def dates(s: str) -> set[tuple[int, int]]:
    return {(_MONTHS.index(m[:3].lower()) + 1, int(d)) for m, d in _DATE.findall(s)}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else np.nan


def embed(texts: list[str], batch: int = 256) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(MODEL, device="cpu")
    return m.encode(texts, batch_size=batch, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)


def candidates(kal: pd.DataFrame, poly: pd.DataFrame, k: int = 5, max_gap_days: float = 3.0,
               min_cos: float = 0.55) -> pd.DataFrame:
    """Top-k Kalshi neighbours for each Polymarket market, within the close-time window."""
    kal = kal.reset_index(drop=True)
    poly = poly.reset_index(drop=True)
    kt = [kalshi_text(r) for r in kal.to_dict("records")]
    pt = [poly_text(r) for r in poly.to_dict("records")]
    ke, pe = embed(kt), embed(pt)
    kday = (kal.close_time.astype("int64") // 10**9 / 86400).values
    pday = (poly.close_time.astype("int64") // 10**9 / 86400).values
    # Sort both sides by close day so every block of Polymarket markets is
    # compared with one contiguous slice of Kalshi embeddings (a matmul on a
    # view, no per-market gathers), then mask pairs outside the window.
    korder, porder = np.argsort(kday), np.argsort(pday)
    ks, ke_s = kday[korder], ke[korder]
    rows = []
    block = 512
    for b0 in range(0, len(porder), block):
        pi = porder[b0:b0 + block]
        lo = np.searchsorted(ks, pday[pi].min() - max_gap_days)
        hi = np.searchsorted(ks, pday[pi].max() + max_gap_days, side="right")
        if hi <= lo:
            continue
        sims = pe[pi] @ ke_s[lo:hi].T  # [block, window]
        far = np.abs(pday[pi][:, None] - ks[lo:hi][None, :]) > max_gap_days
        sims[far] = -1.0
        kk = min(k, hi - lo)
        top = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
        for r, i in enumerate(pi):
            for j in top[r]:
                if sims[r, j] >= min_cos:
                    rows.append((int(i), int(korder[lo + j]), float(sims[r, j])))
    c = pd.DataFrame(rows, columns=["pi", "ki", "emb_cos"])
    c["kalshi_id"] = kal.market_id.values[c.ki]
    c["poly_id"] = poly.market_id.values[c.pi]
    c["kalshi_text"] = [kt[j] for j in c.ki]
    c["poly_text"] = [pt[i] for i in c.pi]
    c["gap_days"] = np.abs(kday[c.ki] - pday[c.pi])
    return c.drop(columns=["pi", "ki"])


def add_features(c: pd.DataFrame) -> pd.DataFrame:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1).fit(
        pd.concat([c.kalshi_text, c.poly_text]))
    a, b = vec.transform(c.kalshi_text), vec.transform(c.poly_text)
    c = c.copy()
    c["tfidf_cos"] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    kn = c.kalshi_text.map(numbers)
    pn = c.poly_text.map(numbers)
    c["num_jaccard"] = [jaccard(x, y) for x, y in zip(kn, pn)]
    c["num_conflict"] = [bool(x) and bool(y) and not (x & y) for x, y in zip(kn, pn)]
    c["ent_jaccard"] = [jaccard(entities(x), entities(y)) for x, y in zip(c.kalshi_text, c.poly_text)]
    ky = c.kalshi_text.map(lambda s: set(_YEAR.findall(s)))
    py = c.poly_text.map(lambda s: set(_YEAR.findall(s)))
    c["year_conflict"] = [bool(x) and bool(y) and not (x & y) for x, y in zip(ky, py)]
    kr, pr = c.kalshi_text.map(ranks), c.poly_text.map(ranks)
    c["rank_conflict"] = [bool(x) and bool(y) and not (x & y) for x, y in zip(kr, pr)]
    kd, pd_ = c.kalshi_text.map(dates), c.poly_text.map(dates)
    c["date_conflict"] = [bool(x) and bool(y) and not (x & y) for x, y in zip(kd, pd_)]
    c["touch_conflict"] = [bool(_TOUCH.search(x)) != bool(_TOUCH.search(y)) for x, y in zip(c.kalshi_text, c.poly_text)]
    ks, ps = c.kalshi_text.map(stages), c.poly_text.map(stages)
    c["stage_conflict"] = [bool(x) and bool(y) and not (x & y) for x, y in zip(ks, ps)]
    return c


def stages(s: str) -> set[str]:
    norm = {"qualifies": "qualify", "advances": "advance", "advance": "qualify", "make the": "qualify",
            "reach the": "qualify", "wins": "win", "champion": "win"}
    return {norm.get(m.lower(), m.lower()) for m in _STAGE.findall(s)}


FEATURES = ["emb_cos", "tfidf_cos", "gap_days", "num_jaccard_f", "num_conflict", "ent_jaccard_f", "year_conflict",
            "has_nums", "rank_conflict", "date_conflict", "touch_conflict", "stage_conflict"]


def design(c: pd.DataFrame) -> np.ndarray:
    x = c.copy()
    x["has_nums"] = x.num_jaccard.notna().astype(float)
    x["num_jaccard_f"] = x.num_jaccard.fillna(0)
    x["ent_jaccard_f"] = x.ent_jaccard.fillna(0)
    return x[FEATURES].astype(float).values


def heuristic(c: pd.DataFrame) -> np.ndarray:
    """Transparent pre-label score used to choose pairs to label."""
    s = c.emb_cos + 0.3 * c.tfidf_cos + 0.2 * c.ent_jaccard.fillna(0) - 0.05 * c.gap_days
    s = s - 0.5 * c.num_conflict.astype(float) - 0.5 * c.year_conflict.astype(float)
    for col in ("rank_conflict", "date_conflict", "touch_conflict", "stage_conflict"):
        if col in c:
            s = s - 0.5 * c[col].astype(float)
    return s.values


def train(labeled: pd.DataFrame, seed: int = 0) -> dict:
    """Logistic regression on labeled pairs; precision/recall by 5-fold CV."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x, y = design(labeled), labeled.label.astype(int).values
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    proba = cross_val_predict(model, x, y, cv=cv, method="predict_proba")[:, 1]
    curve = []
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        pred = proba >= t
        tp = int((pred & (y == 1)).sum())
        curve.append({"threshold": t, "precision": tp / max(1, pred.sum()), "recall": tp / max(1, y.sum()),
                      "predicted": int(pred.sum())})
    base = heuristic(labeled)
    model.fit(x, y)
    return {"model": model, "cv_curve": curve, "n_labeled": int(len(y)), "positives": int(y.sum()),
            "cv_proba": proba, "heuristic": base}


def score(model, c: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(design(c))[:, 1]

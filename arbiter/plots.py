"""Figures for the README, drawn from the measured reports in results/.

    python -m arbiter.plots [--results results] [--out docs/img]
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREY, RED, GREEN = "#2563eb", "#ea580c", "#9ca3af", "#dc2626", "#16a34a"


def _reliability(ax, rows, color, label):
    t = pd.DataFrame(rows)
    t = t[t.n >= 50]
    ax.errorbar(t.mean_price, t.freq, yerr=[t.freq - t.ci_lo, t.ci_hi - t.freq], fmt="o-", ms=3.5, lw=1.2,
                capsize=2, color=color, label=label)


def calibration(results: str, out: str) -> str:
    r = json.load(open(os.path.join(results, "calibration.json")))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, venue, title in zip(axes, ["kalshi", "polymarket"], ["Kalshi (bid/ask midpoint)", "Polymarket (hourly price)"]):
        ax.plot([0, 1], [0, 1], color=GREY, lw=1, ls="--", label="perfectly calibrated")
        naive = r.get("reliability_anchored_on_actual_close", {}).get(venue)
        if naive:
            _reliability(ax, naive, RED, "sampled before the actual close (biased)")
        fit = r["by"][f"{venue}/1d"]["fit"]
        _reliability(ax, r["by"][f"{venue}/1d"]["reliability"], BLUE,
                     f"sampled before the deadline: slope {fit['slope']:.2f} [{fit['slope_ci'][0]:.2f}, {fit['slope_ci'][1]:.2f}]")
        ax.set_title(f"{title}, 1 day before", fontsize=10)
        ax.set_xlabel("price")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, loc="upper left", frameon=False)
    axes[0].set_ylabel("share that resolved YES (95% CI, event-clustered)")
    fig.suptitle("R1. Do 30-cent contracts win 30% of the time?", fontsize=11)
    fig.tight_layout()
    p = os.path.join(out, "calibration.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def crossvenue(results: str, out: str) -> str:
    d = pd.read_csv(os.path.join(results, "r2_pairs.csv"))
    d = d[d.verdict != "not_reviewed_never_positive"].copy()
    groups = {"false_match": ("different question (false match)", RED),
              "rules_differ": ("same question, different rules", ORANGE),
              "equivalent": ("same question, same rules", GREEN),
              "likely_equivalent": ("same question, same rules", GREEN)}
    d["label"] = d.verdict.map(lambda v: groups.get(v, ("unreviewed", GREY))[0])
    d = d.sort_values("max_net")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = np.arange(len(d))
    legend = {label: color for label, color in groups.values()} | {"unreviewed": GREY}
    for label, color in legend.items():
        sel = d.label.values == label
        if not sel.any():
            continue
        ax.barh(y[sel], d.max_net.values[sel] * 100, color=color, label=label, height=0.75)
    ax.set_yticks(y)
    ax.set_yticklabels([t[:48] for t in d.poly_text], fontsize=6)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("best executable gap after both venues' taker fees (cents per $1 contract)")
    ax.set_title("R2. Every cross-venue pair that ever looked profitable, after reading both venues' rules", fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    p = os.path.join(out, "crossvenue.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def marketmaking(results: str, out: str) -> str:
    r = json.load(open(os.path.join(results, "marketmaking.json")))
    venues = [(v, r[v]) for v in ("polymarket", "kalshi") if v in r] or [("polymarket", r)]
    names = [("queue_ahead=1.0", "back of queue
(realistic)"), ("queue_ahead=0.5", "half the level
ahead"),
             ("queue_ahead=0.0", "front of queue
(optimistic)")]
    fig, axes = plt.subplots(1, len(venues), figsize=(5.2 * len(venues), 3.9), squeeze=False)
    for ax, (venue, v) in zip(axes[0], venues):
        x = np.arange(len(names))
        pnl = [v[k]["pnl_per_contract"] * 100 for k, _ in names]
        mk = [v[k]["markout_5m_per_contract"] * 100 for k, _ in names]
        ax.bar(x - 0.18, pnl, 0.36, color=BLUE, label="P&L per contract (closed at the touch)")
        ax.bar(x + 0.18, mk, 0.36, color=ORANGE, label="5-minute markout per contract")
        for i, (k, _) in enumerate(names):
            ax.text(i - 0.18, pnl[i] / 2, f"{v[k]['fills']}
fills", ha="center", va="center", fontsize=6.5, color="white")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([n for _, n in names], fontsize=7.5)
        n = v.get("tokens", v.get("markets"))
        book = "streamed depth" if venue == "polymarket" else "book polled every ~30 s"
        ax.set_title(f"{venue.capitalize()}: {v['hours']:.1f} h, {n} markets, {book}", fontsize=9)
    axes[0][0].set_ylabel("cents per contract")
    axes[0][0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("R4. Passive market making, parameters chosen walk-forward", fontsize=11)
    fig.tight_layout()
    p = os.path.join(out, "marketmaking.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=os.path.join("docs", "img"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for f in (calibration, crossvenue, marketmaking):
        print(f(a.results, a.out))


if __name__ == "__main__":
    main()

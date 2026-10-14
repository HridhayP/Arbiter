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
    venues = [(v, r[v]) for v in ("polymarket", "kalshi") if v in r]
    names = [("queue_ahead=1.0", "back of queue\n(realistic)"), ("queue_ahead=0.5", "half the level\nahead"),
             ("queue_ahead=0.0", "front of queue\n(optimistic)")]
    parts = [("edge_per_contract", "spread captured at the fill", GREEN), ("drift_per_contract", "inventory drift", ORANGE),
             ("closeout_per_contract", "paying the spread to go flat", RED), ("fees_per_contract", "fees", GREY)]
    fig, axes = plt.subplots(1, len(venues), figsize=(5.4 * len(venues), 4.1), squeeze=False, sharey=True)
    for ax, (venue, v) in zip(axes[0], venues):
        x = np.arange(len(names))
        for i, (k, _) in enumerate(names):
            pos = neg = 0.0
            for key, label, color in parts:
                val = v[k][key] * 100 * (-1 if key == "fees_per_contract" else 1)
                base = pos if val >= 0 else neg
                ax.bar(i, val, 0.55, bottom=base, color=color, label=label if i == 0 else None)
                pos, neg = (pos + val, neg) if val >= 0 else (pos, neg + val)
            net = v[k]["pnl_per_contract"] * 100
            ax.plot([i - 0.32, i + 0.32], [net, net], color="black", lw=2, label="net P&L" if i == 0 else None)
            ax.text(i + 0.34, net, f"{net:+.1f}¢", va="center", fontsize=7)
            ax.text(i, 0.15 + pos, f"{v[k]['fills']} fills", ha="center", va="bottom", fontsize=6.5)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([n for _, n in names], fontsize=7.5)
        book = "streamed depth" if venue == "polymarket" else "book polled every ~30 s"
        ax.set_title(f"{venue.capitalize()}: {v['hours']:.0f} h, {v.get('tokens', v.get('markets'))} markets, {book}", fontsize=9)
    axes[0][0].set_ylabel("cents per contract")
    axes[0][0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("R4. Where a passive market maker's P&L goes (parameters chosen walk-forward)", fontsize=11)
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

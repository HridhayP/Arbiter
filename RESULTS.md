# Results

Every number below was measured by the code in this repo on the data described here. The raw reports are in [`results/`](results/), copied from `data/reports/`. Commands to reproduce each one are at the end.

**Short version:**
- Prices on both venues are close to right, and every apparent edge either disappeared once spreads and fees were paid or could only be traded in single contracts.
- Most of the work was finding apparent edges that were data artifacts. Each one is listed in [Artifacts caught](#artifacts-caught) with how it was found.

## Data

| | Kalshi | Polymarket |
|---|---|---|
| Markets collected | 525,190 (up to 300 per series, across 14,993 series) | 1,051,527 (closed, ≥ $1,000 volume, since 2020) |
| Resolved YES/NO | 519,256 | 1,046,373 |
| Markets with price history | 66,438 (hourly bid/ask candles, last 7 days before close; ≥1,000 volume, ≤15 per series) | 60,719 (hourly price, last 7 days; ≥$50K volume, ≤2 per event) |
| Price rows | 8,967,777 total | |
| Live recording | Full listing of every open market (~135K) every ~15 min; 7 complete snapshots | Full listing (~36K markets), 24 snapshots; WebSocket depth + trades for 150 tokens |
| Recording window | 2026-10-13 21:35 → 2026-10-14 05:19 UTC (7.7 h) | same; WebSocket book data 5.1 h (24,406 trades) |

The recording is short (one evening), so the cross-venue, consistency and market-making results are based on hours, not weeks. They describe that window only.

## R1. Calibration

**Method:**
- Price = midpoint 1 hour, 1 day or 1 week before each market's *scheduled* deadline, and only if the market was still open then.
- Kalshi quotes must be two-sided with a spread of at most 10¢ and be no more than 6 hours old. Polymarket uses its hourly history price.
- Logistic GLM of the outcome on logit(price), with event-clustered SEs. A slope above 1 is the favorite–longshot bias; below 1 means prices are too extreme.

| Venue / horizon | Snapshots | Events | Slope (95% CI) | Brier | Brier skill vs base rate |
|---|---|---|---|---|---|
| Kalshi 1 h | 48,999 | 17,399 | 1.061 (1.038–1.084) | 0.078 | 0.658 |
| Kalshi 1 d | 40,552 | 13,889 | **1.009 (0.986–1.033)** | 0.108 | 0.522 |
| Kalshi 1 w | 3,206 | 983 | 1.126 (1.017–1.235) | 0.081 | 0.602 |
| Polymarket 1 h | 48,480 | 46,044 | 0.786 (0.759–0.813) | 0.212 | 0.148 |
| Polymarket 1 d | 22,917 | 20,525 | **0.747 (0.718–0.775)** | 0.202 | 0.180 |
| Polymarket 1 w | 7,545 | 7,452 | 0.841 (0.779–0.903) | 0.220 | 0.120 |

**Kalshi is calibrated at 1 day.** In the reliability table, 7¢ contracts won 8.4% (95% CI 7.3–9.4%), 25¢ won 26.5%, 75¢ won 73.6% and 93¢ won 94.2%. There's a mild favorite–longshot pattern 1 hour and 1 week out. By category at 1 day, slopes run from 0.95 (Sports, n = 16,532) to 1.16 (Politics) and 1.38 (Science and Technology, n = 856).

**Buying favorites doesn't pay on Kalshi.** The strategy buys whichever side is above a threshold, at the ask, plus taker fees, holds to resolution, and picks the threshold walk-forward by month.

| Horizon | Trades | Events | Win rate | Mean cost | P&L per contract | Event-clustered t |
|---|---|---|---|---|---|---|
| 1 h | 30,456 | 10,163 | 98.8% | 99.2¢ | −0.51¢ | −7.5 |
| 1 d | 17,639 | 6,597 | 98.5% | 99.3¢ | −0.77¢ | −8.2 |
| 1 w | 2,484 | 878 | 95.2% | 95.1¢ | −0.16¢ | −0.3 |

**Polymarket prices look too extreme (slope ≈ 0.75), but this is not claimed as a tradable bias.** At 1 day, 7¢ contracts won 15.1% (12.3–18.0%) and 92¢ contracts won 85.4% (80.8–89.7%). The pattern holds in every volume tier: low 0.71, mid 0.76, high 0.78. Splitting by how far the actual close lands from the scheduled end puts most of the effect in markets resolving 0.05–1 day after `endDate`, which on Polymarket is mostly sports, where `endDate` is the scheduled start. Two reasons for caution:
- Polymarket's history has hourly prices with no bid or ask, so I can't filter stale quotes or thin books the way the Kalshi sample is filtered.
- Random error in the price variable biases the slope toward 0 even when the true price is calibrated.

Telling a real bias apart from noisy prices needs historical quotes, which Polymarket doesn't publish. The recorder is collecting them going forward.

**Sampling before the actual close leaks the outcome** (proven on synthetic data; see [Artifacts caught](#artifacts-caught), item 1), but on this data the 1-day difference is small. For comparison, with horizons measured from the actual close: Kalshi 1 d slope 1.027, Kalshi 1 h 1.083, Polymarket 1 h 1.033. See [Artifacts caught](#artifacts-caught), item 1.

## R2. Cross-venue mispricing

**Method:**
- Pairs come from the matcher: p ≥ 0.7, which gives 0.95 precision and 0.49 recall in 5-fold cross-validation on 300 hand-labeled pairs.
- In every complete snapshot pair (Kalshi and Polymarket within 30 minutes), the executable gap is the bid on one venue minus the ask on the other, net of both taker fees. Polymarket fees are charged at the highest published rate (0.07).
- **Every pair that was ever net-positive was checked by hand against both venues' rules** ([`labels/r2_review.csv`](labels/r2_review.csv)).

| | All matched pairs | After the rules review |
|---|---|---|
| Pairs quoted on both venues | 117 | 98 |
| Pair-snapshots | 694 | 580 |
| Median \|mid gap\| | 3.0¢ | 2.2¢ |
| 90th percentile \|mid gap\| | 24.4¢ | 14.1¢ |
| Share of snapshots net-positive after fees | 18.6% | 7.4% |
| Pairs ever net-positive | 29 | 10 |
| **Largest net gap** | **60.0¢** | **1.9¢** |

What the review found among the 29 pairs that were ever positive:
- **11 false matches:** album vs song, tour vs album, headliner vs performer, "above 29 seats" vs "35 or more".
- **8 pairs with different resolution rules.** In the Coachella headliner pairs, Kalshi pays only if the artist actually performs as a headliner, through Dec 31, 2027; Polymarket resolves on the lineup announcement. The Hawaii hurricane pair compares a season with a calendar year.
- **10 equivalent or likely equivalent pairs,** worth at most 1.9¢ net.

The largest gaps are where the matcher's errors are: two different questions naturally have different prices. That's a selection effect a precision number alone doesn't show.

## R3. Internal consistency

Bids, asks and fees come from the same listing snapshot, and every candidate violation was read by hand ([`labels/r3_review.csv`](labels/r3_review.csv)).

| Check | Scope | Apparent violations | After verification |
|---|---|---|---|
| Strike ladders (Kalshi): P(above X) must not rise with X | 7 snapshots | 16 events with a violation, median 1¢ gross | **0** profitable after fees |
| Sell every outcome when the bids sum above $1 (Kalshi mutually-exclusive events) | 5,378 events | 1 | Redistricting count: +0.12¢ net, thinnest bid 3 contracts |
| Sell every outcome (Polymarket neg-risk groups) | 3,010 groups | 3 groups (31 snapshots) | NHL most hat tricks: up to +6.5¢, thinnest bid **1 contract**. Ravens–Bengals: a cancelled season resolves all three outcomes 50-50, which turns the trade into a loss. VALORANT: +1.2¢, unchecked |
| Buy every outcome when the asks sum below $1 (Polymarket) | 553 candidate groups | **548 missing outcomes** from the listing | 5 complete sets. Maduro sentencing: 3.4¢ net, fee-free, rules exhaustive, but 10 contracts deep and ~15 months to resolution (≈2%/yr, below T-bills). Others 0.6–2.6¢ |
| Buy every outcome (Kalshi) | 513 candidate group-snapshots | Kalshi's "mutually exclusive" events are open fields (next Pope, primaries, House races listing only R and D) | none riskless |
| Combos vs the product of their legs (Kalshi) | 2 combos with resting quotes | ask 2.3–2.6% above the independence price | too few to conclude; combos are priced on request |

## R4. Market making

**Method:**
- Polymarket WebSocket capture (full-depth book snapshots and deltas, plus every trade) for the 31 YES tokens with at least 50 trades, replayed through the C++ simulator. Resting orders fill only when trades work through the queue ahead of them.
- The 5.1 h recording is cut into 6 blocks. Each block starts flat and closes any inventory at the touch.
- Parameters for block *i* come from a 12-point grid of risk aversion, k and minimum spread, chosen by P&L on blocks < *i* (5 traded blocks, 122 token-blocks).
- The tick is inferred per market. Fees use the worst-case taker rate; every fill was passive, so fees were $0.

Trade prints agree with the reconstructed book: 84.6% of taker buys printed at or above the prevailing ask (5.3% at or below the bid), and 84.3% of taker sells at or below the bid.

| Queue assumption | Fills | Contracts | P&L | P&L / contract | 1-min markout | 5-min markout | t (per block, n=5) | Tokens net positive |
|---|---|---|---|---|---|---|---|---|
| Back of queue (default) | 204 | 1,871 | −$15.36 | −0.82¢ | −0.72¢ | −0.78¢ | −5.1 | 19% |
| Half the level ahead | 206 | 1,893 | −$14.84 | −0.78¢ | −0.69¢ | −0.77¢ | −4.8 | 23% |
| Front of queue | 370 | 3,196 | −$12.21 | −0.38¢ | −0.48¢ | −0.46¢ | −5.6 | 35% |

**The quoter loses, and the loss per contract roughly equals the 5-minute markout.** It's being picked off: fills arrive just before the price moves against it. Even the unrealistic front-of-queue assumption doesn't rescue it. With 5 hours of data and 5 traded blocks this is evidence, not proof.

## R5. Matching and forecasting

**Matching:** with a logistic model on 12 features and 300 hand labels, 5-fold CV precision is 0.95 and recall 0.49 at p ≥ 0.7. 128 pairs were matched among the open markets.

**Forecasting:**
- Features at 1 day before the deadline: price (logit), moves over 6 h, 1 d and 3 d, spread, market age, category, venue, and price × category and price × venue interactions.
- Time-ordered folds: train on earlier markets, score later ones. 58,155 markets scored in 28,493 events.

| Forecast | Brier | Log loss | Brier − market (event-clustered 95% CI) | Brier skill vs market |
|---|---|---|---|---|
| Market price | 0.14931 | 0.4415 | | |
| Logistic | 0.14870 | 0.4398 | −0.00061 (−0.00096, −0.00025) | +0.41% |
| Gradient boosting | 0.14860 | 0.4388 | −0.00070 (−0.00112, −0.00028) | +0.47% |

By venue (gradient boosting vs price): Kalshi 0.12513 vs 0.12546, Polymarket 0.20736 vs 0.20900. The model beats the price by a statistically clear but economically tiny margin, mostly by recalibrating, for example shrinking Polymarket's too-extreme prices. A 0.0007 Brier gain is nowhere near a 1–2¢ spread plus fees.

## Simulator performance

Release build (`-O3`), Intel Core Ultra 9 185H under WSL2, Google Benchmark, median of 5 repetitions ([`results/bench.txt`](results/bench.txt)):

| Benchmark | Result |
|---|---|
| Replay 1M events with `LogitMarketMaker` attached (quotes, cancels, fills) | 47.3 ms → **21.2M events/s** |
| One book level update (`Book::set`) | **1.58 ns** |

An earlier run, while the data crawls were using the CPU, measured 16.1M events/s and 2.04 ns.

## Artifacts caught

Each of these produced a striking "result" before it was found:

1. **Outcome-dependent close times (R1, R5).** Both venues close "by date X" markets early when X happens. So "1 day before the actual close" samples YES markets just before the surprise and NO markets at the dead end: the close time carries the outcome. A synthetic test with perfectly calibrated by-date markets shows this sampling inflating the win rate by 20 points. The fix measures horizons from the scheduled deadline and requires the market to be open. On the real data, the effect at 1 day turned out small (Kalshi slope 1.03 naive vs 1.01 fixed), and it does **not** explain Polymarket's too-extreme prices, which remain with the fix. That's why that pattern is reported with a caveat instead of being dismissed as this artifact.
2. **Pager duplicates (R3).** Polymarket's gamma API caps pages at 100 and rejects offsets past ~2,000. The date-window crawler duplicated markets on window edges, which created 78 fake neg-risk arbitrages.
3. **Kalshi events aren't ladders (R3).** One event can hold different quantities (rain in several cities). Grouping strikes by event produced 4,933 fake ladder violations; keying on ticker minus strike left 16 events.
4. **Incomplete outcome sets (R3).** 548 of 553 sub-$1 Polymarket outcome sets were missing outcomes from the listing.
5. **False matches carry the biggest gaps (R2).** All of the 60¢ and 7–25¢ cross-venue gaps were different questions or different rules.
6. **Timestamp units.** pandas 3 reads microsecond timestamps; day differences computed as if they were nanoseconds were 1000× too small, which hid that most true cross-venue matches close more than 3 days apart.
7. **Partial listing snapshots.** Under rate limits a Kalshi listing pass sometimes finished early (52K of 135K markets). Snapshots now carry a completeness flag, and only complete ones are analysed (7 of 27).

## Reproduce

```bash
python -m arbiter.collect kalshi-series && python -m arbiter.collect kalshi-markets --per-series 300 --workers 6
python -m arbiter.collect polymarket-markets --min-volume 1000
python -m arbiter.collect kalshi-candles --min-volume 1000 --per-series 15 --horizon-days 7 --workers 6
python -m arbiter.collect polymarket-prices --workers 8 --min-volume 50000 --per-event 2 --max-markets 60000
python -m arbiter.collect tables
python -m arbiter.recorder --kalshi 40 --polymarket 150 --kalshi-rps 1.5     # leave running
python -m arbiter.research.calibration
python -m arbiter.research.crossvenue
python -m arbiter.research.consistency --verify
python -m arbiter.research.marketmaking
python -m arbiter.research.forecast
./build/arbiter_bench --benchmark_repetitions=5 --benchmark_report_aggregates_only=true
```

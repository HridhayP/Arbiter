# Interview notes

Likely questions about Arbiter, answered from the code. Every number is in [RESULTS.md](RESULTS.md).

### 1. What is this project, in one minute?

Prediction markets like Kalshi and Polymarket trade contracts that pay $1 if an event happens. A contract's price should be the probability of the event. Arbiter collects every resolved market both venues expose (about 1.56M) and records live order books, then asks whether the prices are right and whether any mistakes can be traded after fees. It tests five ideas:

1. calibration;
2. the same event on two venues;
3. internal consistency (ladders, outcome sets, combos);
4. market making, in a C++ simulator;
5. a model that tries to beat the market's price.

The headline is mostly "no tradable edge after fees and spread". The interesting part is how many apparent edges were data artifacts, and how each one was found.

### 2. What is the favorite–longshot bias, and how did you measure it?

In betting markets, longshots tend to be overpriced and favorites underpriced. Arbiter regresses the outcome on logit(price) with a logistic GLM. A slope above 1 means realized odds are more extreme than prices (the classic bias), and below 1 means the reverse. Standard errors are clustered by event, because the 11 outcomes of one Fed meeting aren't independent data points. Reliability tables use a clustered bootstrap, and prices are bid/ask midpoints with a spread of at most 10¢, sampled at fixed horizons before the deadline. A favorite-buying strategy is then tested walk-forward, paying the ask and fees, with the threshold chosen on earlier months only.

**Results.**
- **Kalshi** is well calibrated. The 1-day slope is 1.01 (95% CI 0.99–1.03) over 40,552 market snapshots in 13,889 events. Buying favorites loses 0.77¢ per contract after fees (event-clustered t = −8.2): the small edge left in favorites is smaller than the ask plus fees.
- **Polymarket** prices look too extreme: slope 0.75, with 7¢ contracts winning 15%. But the gap sits in sports markets sampled before kickoff, and Polymarket's history has only hourly prices with no bid or ask. Random error in a price variable pulls a regression slope below 1 even when the true price is calibrated (errors-in-variables). So this isn't claimed as a tradable bias.

### 3. What was the subtlest bug you found?

A look-ahead leak hiding in the close time. I sampled prices "1 day before close", but both venues close a "will X happen by date Y" market early when X happens. So YES markets were sampled just before the surprise, while NO markets were sampled at the dead end, near 0. The close time itself carries the outcome.

The fix samples before the *scheduled* deadline and keeps a market only if it was still open then, which is known at that moment. Polymarket keeps the deadline (`endDate`). Kalshi overwrites `close_time`, so the latest close in the event is used as a proxy. A synthetic test with perfectly calibrated by-date markets shows the naive sampling inflating the win rate by 20 points while the fixed version stays calibrated. The forecasting features had the same leak and got the same fix.

I'm careful about what this does and doesn't explain. I first suspected it caused Polymarket's too-extreme prices (7¢ contracts winning 15%), but that pattern survived the fix. On the real data the 1-day effect is small (Kalshi slope 1.03 vs 1.01). So the Polymarket result stays in the write-up as an open question with a stated caveat, rather than being explained away.

### 4. How do you match the same event across two venues?

Candidates come from sentence embeddings (MiniLM) of the titles, compared in blocks sorted by close date. A logistic model on 300 hand-labeled pairs then scores each candidate using:
- embedding and TF-IDF similarity, and the gap between close dates;
- conflict features that catch near-duplicates: different numbers, years, ranks ("top 5" vs "winner"), dates, stages ("nominee" vs "winner") or "touch" wording.

Cross-validated precision is 0.95 at p ≥ 0.7. A pandas 3 unit bug (microsecond timestamps read as nanoseconds) had made every date gap 1000× too small. Fixing it showed that 35 of 39 true matches close more than 3 days apart, so the close-date window had to widen to 400 days.

### 5. Were the cross-venue gaps real arbitrage?

Mostly not. An opportunity counts only if Polymarket's bid minus Kalshi's ask (or the reverse) beats both venues' taker fees. Over about a day of snapshots, 29 matched pairs showed that, with net gaps of up to 60¢. I read both venues' rules for all 29:
- **11 false matches.** The largest gaps come from different questions, which is a selection effect: a 95%-precise matcher still puts its 5% errors exactly where the gaps are.
- **9 with different resolution rules.** Kalshi's Coachella headliner market requires actually performing, while Polymarket's resolves on the poster announcement.
- **9 equivalent pairs** had a best net gap of 1.9¢. Those small gaps were stable across snapshots: real but tiny price differences between venues.

### 6. Tell me about the consistency checks.

There are three kinds:
- **Strike ladders:** P(above X) can't rise with X.
- **Mutually exclusive sets:** if the bids sum above $1, selling every outcome is riskless. If the asks sum below $1, buying every outcome is riskless only if the set is also exhaustive.
- **Combos:** compared with the product of their legs.

Every apparent violation turned out to have a mundane cause:
- **Ladders:** Kalshi puts different quantities in one event (rain in several cities), so grouping by event produced 4,933 fake ladder violations. Grouping by ticker minus strike left 107 violations in 18 events, none profitable after fees.
- **Polymarket buy-all candidates:** fetching each candidate's full event showed 586 of 591 were missing outcomes from the listing (no "draw" market, for instance).
- **Kalshi buy-all candidates:** these were open fields that aren't exhaustive (next Pope, primaries).
- **The real ones:** NHL "most hat tricks" bids summed to $1.16, and the Maduro sentencing asks summed to $0.976. But the thinnest level had about 1 contract, or the money was locked up for 15 months at less than the T-bill rate.

### 7. How does the C++ simulator work?

The book holds prices on a 0.001 grid, 1,001 levels per side, stored in a flat array, so a level update is an array write (about 1.6 ns). Events are book updates, trades and clears; the simulator replays about 21M events/s with the market maker attached.
- **Resting orders:** each one records the volume ahead of it at its price. Trades at that price use up the volume ahead first, then fill us. Cancels are assumed to come proportionally from ahead of us and behind us. A trade *through* our price fills us first (price priority).
- **Crossing orders:** immediate-or-cancel against displayed size, and they can't retake liquidity already consumed.
- **Fees:** both venues' formulas, shared with Python and tested for agreement.
- **Bindings:** pybind11, with a trampoline so strategies can be prototyped in Python.

### 8. Why quote in logit space?

A binary contract lives in (0, 1), and its price volatility shrinks near the bounds. Avellaneda–Stoikov assumes an unbounded price with constant volatility. In log-odds the price is unbounded and moves more evenly, so the reservation price (mid minus an inventory skew) and the half-spread are computed there and mapped back through the sigmoid. Quotes can never leave (0, 1), and spreads in price terms tighten near 0 and 1 automatically. The quoter also stops before close, because settlement is a jump to 0 or 1 that no spread pays for.

### 9. Did market making make money?

No, and the useful part is *why*. On 22 hours of recorded books (64 Polymarket markets with streamed depth, 146 Kalshi markets with books polled every ~30 s plus trade prints), parameters were chosen walk-forward on earlier time blocks. Inventory was closed at the touch at the end of each ~3.7 h block. I split P&L exactly into four parts: spread captured at the fill (vs the mid just before it), inventory drift afterwards, the cost of going flat, and fees.

- **Polymarket, back of the queue:** −2.1¢ per contract, made up of **0.0¢ spread captured**, −0.4¢ drift and −1.7¢ close-out. A quote at the back of the queue only fills once the price has come to it, so it earns nothing. That's adverse selection at the instant of the fill.
- **Kalshi:** the quoter captures **+1.7¢** per contract, because Kalshi's spreads are wide, but pays −3.5¢ to get flat across those same spreads, plus −0.6¢ drift.
- **Caveats:**
  - Without the forced close-out, Kalshi would be +1.1¢ per contract. A real desk works out of inventory passively, so the close-out overstates the cost.
  - The Kalshi book is 30 s stale, so its quotes are worse-informed than a live quoter's would be.
  - 5 traded blocks isn't enough for significance (per-block t between −1.0 and −1.7).

I also learned to distrust short samples. On the first 5 hours, the loss looked like pure 5-minute adverse selection; with 4× the data the decomposition told a different story. Two checks back the fill model. Trade prints match the reconstructed book: 83% of Polymarket taker buys print at or above the ask. And a unit test confirms the decomposition adds up exactly.

### 10. Can a model beat the market's price?

Barely, and not in a way you can trade. Features are the price 1 day before the deadline, its moves over the previous 6 hours, 1 day and 3 days, the spread, the market's age, category and venue, plus interactions that let the price's weight differ by category and venue. A logistic model and a gradient-boosted model are trained on earlier markets and scored on later ones (time-ordered folds).

On 58,155 scored markets, gradient boosting improves the Brier score by 0.0007 over the price alone, a 0.5% skill gain. The event-clustered 95% CI (−0.0011 to −0.0003) excludes zero. Most of the gain is recalibration: learning how much to shrink a venue's or category's prices. That's worth far less than a 1–2¢ spread plus fees, so the honest answer is that the price already contains almost everything public features can add.

### 11. What would you do with more time?

- **Record for weeks, not hours.** The market-making sample is small (5 traded blocks), and cross-venue persistence needs many more snapshots. The paper trader (`arbiter.papertrade`, the same quoter on live books) gives an out-of-sample check that accumulates on its own.
- **Stream Kalshi's book over its WebSocket** (which needs an API key) instead of polling every 30 s, so Kalshi queue positions are as exact as Polymarket's.
- **Manage inventory instead of dumping it.** Most measured losses are the forced close-out, so working out of positions passively is the obvious next strategy to test.
- **Model each market's information arrival,** so the quoter can pull before scheduled news (game starts, data releases), where most adverse selection happens.
- **Read rules automatically.** An LLM pass over both venues' rules text could flag non-equivalent matches before a human reviews them.

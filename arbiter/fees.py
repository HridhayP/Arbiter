"""Trading fees for binary contracts, read from venue metadata.

Both venues charge proportionally to p * (1 - p): fees peak at 50 cents and
vanish toward 0 and 1, so cheap longshots and near-certain favorites are
cheap to trade while coin flips are expensive.

Kalshi (fee schedule PDF, series `fee_type` / `fee_multiplier` in the API):
    taker = ceil_to_cent(0.07   * m * C * P * (1 - P))
    maker = ceil_to_cent(0.0175 * m * C * P * (1 - P))   only for fee_type
            "quadratic_with_maker_fees"; otherwise makers pay nothing.
Polymarket (docs.polymarket.com/trading/fees, per-market `feeSchedule`):
    taker = round(C * rate * (p * (1 - p)) ** exponent, 5)   (exponent is 1
            for every current schedule); makers never pay; markets with
            feesEnabled = false (and older markets) are free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

KALSHI_TAKER = 0.07
KALSHI_MAKER = 0.0175


def _ceil_cents(x: float) -> float:
    # Guard against float noise turning an exact cent into the next one.
    return math.ceil(round(x * 100, 9)) / 100


@dataclass(frozen=True)
class FeeModel:
    venue: str  # "kalshi" | "polymarket"
    taker_rate: float  # Kalshi: 0.07 * multiplier; Polymarket: schedule rate (0 if disabled)
    maker_rate: float  # Kalshi: 0.0175 * multiplier or 0; Polymarket: always 0
    exponent: float = 1.0  # Polymarket schedule exponent

    def fee(self, price: float, contracts: float, maker: bool) -> float:
        """Fee in dollars for one fill of `contracts` at `price` (0..1)."""
        if not 0.0 <= price <= 1.0:
            raise ValueError(f"price {price} outside [0, 1]")
        rate = self.maker_rate if maker else self.taker_rate
        if rate == 0 or contracts == 0:
            return 0.0
        if self.venue == "kalshi":
            return _ceil_cents(rate * contracts * price * (1 - price))
        raw = contracts * rate * (price * (1 - price)) ** self.exponent
        return round(raw, 5)

    def per_contract(self, price: float, maker: bool) -> float:
        """Marginal fee per contract, without per-order rounding (for research
        aggregates where order sizes are unknown)."""
        rate = self.maker_rate if maker else self.taker_rate
        if self.venue == "kalshi":
            return rate * price * (1 - price)
        return rate * (price * (1 - price)) ** self.exponent


def kalshi_fees(fee_type: str | None, multiplier: float | None) -> FeeModel:
    m = 1.0 if multiplier is None else float(multiplier)
    ft = (fee_type or "quadratic").lower()
    if ft in ("free", "none", "no_fee"):
        return FeeModel("kalshi", 0.0, 0.0)
    maker = KALSHI_MAKER * m if "maker" in ft else 0.0
    return FeeModel("kalshi", KALSHI_TAKER * m, maker)


def polymarket_fees(market: dict) -> FeeModel:
    """From a gamma-api market object."""
    sched = market.get("feeSchedule") or {}
    if not market.get("feesEnabled") or not sched:
        return FeeModel("polymarket", 0.0, 0.0)
    return FeeModel("polymarket", float(sched.get("rate", 0.0)), 0.0, float(sched.get("exponent", 1.0)))

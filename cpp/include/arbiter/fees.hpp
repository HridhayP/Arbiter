// Venue fee models for binary contracts; mirrors arbiter/fees.py.
#pragma once

#include <cmath>
#include <stdexcept>

namespace arbiter {

enum class Venue { Kalshi, Polymarket };

struct FeeModel {
    Venue venue = Venue::Kalshi;
    double taker_rate = 0.07;  // Kalshi: 0.07 * multiplier; Polymarket: schedule rate
    double maker_rate = 0.0;   // Kalshi: 0.0175 * multiplier on maker-fee series
    double exponent = 1.0;     // Polymarket schedule exponent

    // Fee in dollars for one fill of `contracts` at `price` in [0, 1].
    double fee(double price, double contracts, bool maker) const {
        if (price < 0.0 || price > 1.0) throw std::invalid_argument("price outside [0, 1]");
        const double rate = maker ? maker_rate : taker_rate;
        if (rate == 0.0 || contracts == 0.0) return 0.0;
        if (venue == Venue::Kalshi) {
            // Kalshi rounds each fill up to the next cent; round(.., 1e-9) first
            // so float noise doesn't push an exact cent to the next one.
            const double cents = std::round(rate * contracts * price * (1.0 - price) * 100.0 * 1e9) / 1e9;
            return std::ceil(cents) / 100.0;
        }
        const double raw = contracts * rate * std::pow(price * (1.0 - price), exponent);
        return std::round(raw * 1e5) / 1e5;
    }
};

}  // namespace arbiter

// Top-of-book and depth for one binary contract, quoted as YES prices.
//
// Prices live on a fixed grid of 0.001 (ticks 0..1000), so each side is a
// flat array indexed by tick: set/lookup are O(1) and nothing allocates.
// The best price is cached and only rescanned when the best level empties.
#pragma once

#include <array>
#include <cstdint>

namespace arbiter {

inline constexpr int kTicks = 1001;  // 0.000 .. 1.000
using Tick = int16_t;

inline constexpr double tick_to_price(int t) { return t / 1000.0; }
inline constexpr int price_to_tick(double p) { return static_cast<int>(p * 1000.0 + 0.5); }

enum class Side : uint8_t { Bid = 0, Ask = 1 };

class Book {
public:
    Book() { clear(); }

    void clear() {
        bids_.fill(0.0);
        asks_.fill(0.0);
        best_bid_ = -1;
        best_ask_ = kTicks;
    }

    // Absolute size at a level (0 removes it).
    void set(Side side, int tick, double size) {
        if (tick < 0 || tick >= kTicks) return;
        if (side == Side::Bid) {
            bids_[tick] = size;
            if (size > 0 && tick > best_bid_) best_bid_ = tick;
            else if (size <= 0 && tick == best_bid_) rescan_bid();
        } else {
            asks_[tick] = size;
            if (size > 0 && tick < best_ask_) best_ask_ = tick;
            else if (size <= 0 && tick == best_ask_) rescan_ask();
        }
    }

    double size(Side side, int tick) const {
        if (tick < 0 || tick >= kTicks) return 0.0;
        return side == Side::Bid ? bids_[tick] : asks_[tick];
    }

    int best_bid() const { return best_bid_; }  // -1 if empty
    int best_ask() const { return best_ask_; }  // kTicks if empty
    bool has_bid() const { return best_bid_ >= 0; }
    bool has_ask() const { return best_ask_ < kTicks; }
    bool two_sided() const { return has_bid() && has_ask(); }
    double mid() const { return two_sided() ? (best_bid_ + best_ask_) / 2000.0 : -1.0; }

private:
    void rescan_bid() {
        int t = best_bid_;
        while (t >= 0 && bids_[t] <= 0) --t;
        best_bid_ = t;
    }
    void rescan_ask() {
        int t = best_ask_;
        while (t < kTicks && asks_[t] <= 0) ++t;
        best_ask_ = t;
    }

    std::array<double, kTicks> bids_{};
    std::array<double, kTicks> asks_{};
    int best_bid_ = -1;
    int best_ask_ = kTicks;
};

}  // namespace arbiter

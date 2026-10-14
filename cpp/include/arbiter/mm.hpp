// Avellaneda-Stoikov market maker adapted to binary contracts.
//
// Prices of binary contracts live in (0, 1) and resolve to 0 or 1, so the
// quoter works in logit space x = log(p / (1 - p)): a symmetric spread there
// is narrow near 0.5 and widens naturally toward the bounds, and quotes can
// never leave (0, 1). With mid x, inventory q, risk aversion gamma,
// logit volatility sigma (per sqrt hour), fill-intensity decay k and
// remaining horizon tau (hours):
//
//     reservation  x_r   = x - q * gamma * sigma^2 * tau
//     half-spread  delta = gamma * sigma^2 * tau / 2 + ln(1 + gamma / k) / gamma
//     quotes       sigmoid(x_r -/+ delta), rounded outward to the venue tick
//
// tau shrinks toward the market's close and quoting stops a fixed time
// before it: a binary contract jumps to 0 or 1 at resolution, and a resting
// quote at that moment is a free option for better-informed traders.
// sigma is estimated online (EWMA of squared logit-mid changes per hour),
// seeded with params.sigma. Quotes are passive only and are moved only when
// the target moves by at least requote_ticks, to keep queue position.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "arbiter/sim.hpp"

namespace arbiter {

struct MMParams {
    double gamma = 0.1;              // risk aversion (per contract, logit units)
    double k = 1.5;                  // order-arrival decay with distance (per logit unit)
    double sigma = 0.5;              // initial logit volatility per sqrt(hour)
    double quote_size = 10;          // contracts per side
    double max_inventory = 100;      // stop adding beyond this many contracts
    int min_spread_ticks = 1;        // at least this many venue ticks between our quotes
    int64_t horizon_ns = 3600LL * 1'000'000'000LL;          // tau cap
    int64_t stop_before_close_ns = 3600LL * 1'000'000'000LL;  // pull quotes this long before close
    int64_t close_ts = INT64_MAX;    // market close (ns)
    int requote_ticks = 1;           // move a quote only if its target moved this far
    int tick_size = 10;              // venue tick in 0.001 units (Kalshi cents: 10)
};

class LogitMarketMaker : public Strategy {
public:
    explicit LogitMarketMaker(MMParams p) : p_(p), var_(p.sigma * p.sigma) {}

    void on_event(Simulator& sim, const Event& ev) override {
        if (ev.type == EventType::Trade) return;  // quotes react to the book, not prints
        const Book& b = sim.book();
        if (ev.ts >= p_.close_ts - p_.stop_before_close_ns) {
            if (!stopped_) {
                sim.cancel_all();
                stopped_ = true;
            }
            return;
        }
        if (!b.two_sided()) return;
        const double mid = std::clamp(b.mid(), 0.001, 0.999);
        const double x = std::log(mid / (1.0 - mid));
        update_vol(x, ev.ts);

        const double tau = std::max(1e-3, std::min<double>(p_.horizon_ns, p_.close_ts - p_.stop_before_close_ns - ev.ts) / 3.6e12);
        const double q = sim.position();
        const double xr = x - q * p_.gamma * var_ * tau;
        const double delta = 0.5 * p_.gamma * var_ * tau + std::log1p(p_.gamma / p_.k) / p_.gamma;
        int bid = round_down(sigmoid(xr - delta));
        int ask = round_up(sigmoid(xr + delta));
        if (ask - bid < p_.min_spread_ticks * p_.tick_size) {
            const int pad = (p_.min_spread_ticks * p_.tick_size - (ask - bid) + 1) / 2;
            bid -= pad;
            ask += pad;
            bid = round_down(bid / 1000.0);
            ask = round_up(ask / 1000.0);
        }
        // passive only: never cross the displayed book
        if (b.has_ask()) bid = std::min(bid, b.best_ask() - p_.tick_size);
        if (b.has_bid()) ask = std::max(ask, b.best_bid() + p_.tick_size);
        const bool can_buy = q + p_.quote_size <= p_.max_inventory;
        const bool can_sell = -(q - p_.quote_size) <= p_.max_inventory;
        bid_id_ = requote(sim, Side::Bid, bid_id_, can_buy ? bid : -1);
        ask_id_ = requote(sim, Side::Ask, ask_id_, can_sell ? ask : -1);
    }

    void on_fill(Simulator&, const Fill&) override { /* the next book event requotes */ }

    int64_t quotes_placed() const { return placed_; }

private:
    static double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
    int round_down(double p) const {
        const int t = static_cast<int>(std::floor(p * 1000.0 / p_.tick_size + 1e-9)) * p_.tick_size;
        return std::clamp(t, p_.tick_size, 1000 - p_.tick_size);
    }
    int round_up(double p) const {
        const int t = static_cast<int>(std::ceil(p * 1000.0 / p_.tick_size - 1e-9)) * p_.tick_size;
        return std::clamp(t, p_.tick_size, 1000 - p_.tick_size);
    }

    void update_vol(double x, int64_t ts) {
        if (last_ts_ > 0 && ts > last_ts_) {
            const double dt_h = (ts - last_ts_) / 3.6e12;
            if (dt_h > 1e-6) {
                const double inst = (x - last_x_) * (x - last_x_) / dt_h;
                const double w = 1.0 - std::exp(-dt_h / 6.0);  // ~6 hour memory
                var_ = (1 - w) * var_ + w * std::min(inst, 100.0);
            }
        }
        last_x_ = x;
        last_ts_ = ts;
    }

    int64_t requote(Simulator& sim, Side side, int64_t id, int target) {
        const auto& orders = sim.orders();
        auto it = id > 0 ? orders.find(id) : orders.end();
        if (target < 0) {
            if (it != orders.end()) sim.cancel(id);
            return 0;
        }
        if (it != orders.end() && std::abs(it->second.tick - target) < p_.requote_ticks * p_.tick_size) return id;
        if (it != orders.end()) sim.cancel(id);
        ++placed_;
        const int64_t nid = sim.place(side, target, p_.quote_size);
        return nid > 0 && sim.orders().count(nid) ? nid : 0;
    }

    MMParams p_;
    double var_;
    double last_x_ = 0.0;
    int64_t last_ts_ = 0;
    int64_t bid_id_ = 0;
    int64_t ask_id_ = 0;
    int64_t placed_ = 0;
    bool stopped_ = false;
};

}  // namespace arbiter

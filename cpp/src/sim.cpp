#include "arbiter/sim.hpp"

#include <algorithm>
#include <vector>

namespace arbiter {

namespace {
inline int idx(Side s, int tick) { return static_cast<int>(s) * kTicks + tick; }
}  // namespace

int64_t Simulator::place(Side side, int tick, double qty) {
    if (tick < 1 || tick > kTicks - 2 || qty <= 0) return -1;
    // Inventory cap, counting every resting order on that side as if filled.
    double pending = 0.0;
    for (const auto& [id, o] : orders_)
        if (o.side == side) pending += o.qty;
    const double worst = side == Side::Bid ? position_ + pending + qty : -(position_ - pending - qty);
    if (worst > cfg_.max_position) return -1;

    Order o{next_id_++, side, tick, qty, 0.0};
    const Side opp = side == Side::Bid ? Side::Ask : Side::Bid;
    const bool crosses = side == Side::Bid ? (book_.has_ask() && tick >= book_.best_ask())
                                           : (book_.has_bid() && tick <= book_.best_bid());
    if (crosses) {
        // Immediate-or-cancel against displayed size at the touch, net of what
        // we already took from that level since it last updated.
        const int touch = side == Side::Bid ? book_.best_ask() : book_.best_bid();
        const double avail = book_.size(opp, touch) - consumed_[idx(opp, touch)];
        const double q = std::min(qty, std::max(0.0, avail));
        if (q > 0) {
            consumed_[idx(opp, touch)] += q;
            fill(o, q, touch, /*maker=*/false);
        }
        return o.id;
    }
    // Rest at the back of the level: displayed size plus our own earlier
    // orders there are ahead of us.
    o.ahead = book_.size(side, tick);
    for (const auto& [id, other] : orders_)
        if (other.side == side && other.tick == tick) o.ahead += other.qty;
    orders_.emplace(o.id, o);
    return o.id;
}

bool Simulator::cancel(int64_t id) { return orders_.erase(id) > 0; }

void Simulator::cancel_all() { orders_.clear(); }

void Simulator::fill(Order& o, double qty, int tick, bool maker) {
    const double price = tick_to_price(tick);
    const double fee = cfg_.fees.fee(price, qty, maker);
    if (o.side == Side::Bid) {
        position_ += qty;
        cash_ -= price * qty + fee;
    } else {
        position_ -= qty;
        cash_ += price * qty - fee;
    }
    fees_paid_ += fee;
    o.qty -= qty;
    Fill f{now_, o.id, o.side, price, qty, maker, fee};
    fills_.push_back(f);
    if (strategy_) strategy_->on_fill(*this, f);
}

void Simulator::on_trade(const Event& ev) {
    // ev.side is the book side that was hit: Bid means a seller hit bids,
    // which can fill our resting bids; Ask means a buyer lifted offers.
    traded_[idx(ev.side, ev.tick)] += ev.size;
    std::vector<int64_t> ids;
    for (const auto& [id, o] : orders_)
        if (o.side == ev.side) ids.push_back(id);
    std::sort(ids.begin(), ids.end());  // time priority
    for (int64_t id : ids) {
        auto it = orders_.find(id);
        if (it == orders_.end()) continue;  // cancelled by a callback
        Order& o = it->second;
        const bool through = ev.side == Side::Bid ? ev.tick < o.tick : ev.tick > o.tick;
        const bool at = ev.tick == o.tick;
        if (!through && !at) continue;
        double q;
        if (through) {
            q = std::min(o.qty, ev.size);  // price priority: we are ahead of everyone at the worse price
        } else {
            q = std::min(o.qty, std::max(0.0, ev.size - o.ahead));
            o.ahead = std::max(0.0, o.ahead - ev.size);
        }
        if (q > 0) fill(o, q, o.tick, /*maker=*/true);
        if (o.qty <= 1e-12) orders_.erase(id);
    }
}

void Simulator::on_level_change(Side side, int tick, double old_size, double new_size) {
    const int k = idx(side, tick);
    // Shrinkage not explained by trades since the last update was cancelled,
    // from ahead of and behind our orders in proportion.
    const double unexplained = std::max(0.0, old_size - new_size - traded_[k]);
    if (unexplained > 0 && old_size > 0) {
        const double keep = 1.0 - unexplained / old_size;
        for (auto& [id, o] : orders_)
            if (o.side == side && o.tick == tick) o.ahead *= keep;
    }
    traded_[k] = 0.0;
    consumed_[k] = 0.0;  // the level was re-quoted
}

void Simulator::check_cross() {
    // Conservative fill rule for snapshot-only data: the opposite best quote
    // moved strictly through our price, so someone traded at least as well.
    std::vector<int64_t> done;
    for (auto& [id, o] : orders_) {
        const bool hit = o.side == Side::Bid ? (book_.has_ask() && book_.best_ask() < o.tick)
                                             : (book_.has_bid() && book_.best_bid() > o.tick);
        if (hit) {
            fill(o, o.qty, o.tick, /*maker=*/true);
            done.push_back(id);
        }
    }
    for (int64_t id : done) orders_.erase(id);
}

void Simulator::process(const Event& ev, Strategy& strategy) {
    strategy_ = &strategy;
    now_ = ev.ts;
    ++events_processed_;
    switch (ev.type) {
        case EventType::BookClear:
            book_.clear();
            std::fill(consumed_.begin(), consumed_.end(), 0.0);
            break;
        case EventType::BookSet: {
            const double old = book_.size(ev.side, ev.tick);
            book_.set(ev.side, ev.tick, ev.size);
            on_level_change(ev.side, ev.tick, old, ev.size);
            if (cfg_.fill_on_cross) check_cross();
            break;
        }
        case EventType::Trade:
            on_trade(ev);
            break;
        case EventType::Timer:
            break;
    }
    strategy.on_event(*this, ev);
}

void Simulator::run(const std::vector<Event>& events, Strategy& strategy) {
    for (const Event& ev : events) process(ev, strategy);
}

double Simulator::settle(bool yes) {
    cancel_all();
    cash_ += position_ * (yes ? 1.0 : 0.0);
    position_ = 0.0;
    return cash_;
}

double Simulator::mark_to_market() const {
    const double m = book_.mid();
    return cash_ + position_ * (m >= 0 ? m : 0.5);
}

}  // namespace arbiter

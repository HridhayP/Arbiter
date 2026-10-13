// Event-driven backtester for one binary contract.
//
// Market data arrives as a time-ordered stream of events (book level sets,
// book resets for snapshots, trade prints, timers). A Strategy reacts to
// each event and places or cancels limit orders through the simulator.
//
// Fill model
// - A resting order joins the back of its price level: `ahead` is the size
//   already resting there when it was placed (zero for a new, better price).
// - A trade at our price consumes `ahead` first; whatever is left fills us.
//   A trade at a worse price (through our level) fills us first, because
//   price priority puts our order ahead of everything at that worse price.
// - A level that shrinks without a trade lost size to cancellations, which
//   are assumed to come from ahead of and behind us in proportion.
// - With snapshot-only data (no trade prints), `fill_on_cross` fills a
//   resting order only when the opposite best quote moves strictly through
//   its price, a deliberately conservative rule.
// - An order priced to cross the book is a taker: it fills immediately
//   against displayed size at the touch (one level, then rests).
// Fees come from the venue FeeModel (maker for resting fills, taker for
// crossing ones). At the end the position settles at 1 (YES) or 0 (NO).
#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "arbiter/book.hpp"
#include "arbiter/fees.hpp"

namespace arbiter {

enum class EventType : uint8_t { BookSet = 0, BookClear = 1, Trade = 2, Timer = 3 };

struct Event {
    int64_t ts;       // nanoseconds since epoch
    EventType type;
    Side side;        // BookSet: level side; Trade: side of the book that was hit (Bid = sell aggressor)
    int16_t tick;     // price tick
    double size;      // BookSet: absolute size; Trade: traded size
};

struct Order {
    int64_t id;
    Side side;  // Bid = buy YES, Ask = sell YES
    int tick;
    double qty;      // remaining
    double ahead;    // size queued in front of us at our level
};

struct Fill {
    int64_t ts;
    int64_t order_id;
    Side side;
    double price;
    double qty;
    bool maker;
    double fee;
};

class Simulator;

class Strategy {
public:
    virtual ~Strategy() = default;
    virtual void on_event(Simulator& sim, const Event& ev) = 0;
    virtual void on_fill(Simulator& /*sim*/, const Fill& /*fill*/) {}
};

struct SimConfig {
    FeeModel fees;
    bool fill_on_cross = false;  // for data without trade prints
    double max_position = 1e18;  // hard inventory cap enforced at order entry
};

class Simulator {
public:
    explicit Simulator(SimConfig cfg) : cfg_(cfg) {}

    // ---- strategy API ----
    int64_t place(Side side, int tick, double qty);  // returns order id, or -1 if rejected
    bool cancel(int64_t id);
    void cancel_all();
    const Book& book() const { return book_; }
    double position() const { return position_; }
    double cash() const { return cash_; }
    int64_t now() const { return now_; }
    const std::unordered_map<int64_t, Order>& orders() const { return orders_; }

    // ---- driver ----
    void run(const std::vector<Event>& events, Strategy& strategy);
    void process(const Event& ev, Strategy& strategy);
    double settle(bool yes);  // pays out the position; returns final P&L
    double mark_to_market() const;
    const std::vector<Fill>& fills() const { return fills_; }
    double fees_paid() const { return fees_paid_; }
    int64_t events_processed() const { return events_processed_; }

private:
    void fill(Order& o, double qty, int tick, bool maker);
    void on_trade(const Event& ev);
    void on_level_change(Side side, int tick, double old_size, double new_size);
    void check_cross();

    SimConfig cfg_;
    Book book_;
    std::unordered_map<int64_t, Order> orders_;
    std::vector<Fill> fills_;
    Strategy* strategy_ = nullptr;
    int64_t next_id_ = 1;
    int64_t now_ = 0;
    int64_t events_processed_ = 0;
    double position_ = 0.0;
    double cash_ = 0.0;
    double fees_paid_ = 0.0;
    // Per (side, tick): volume traded since the level last updated, and size
    // our crossing orders took from it (so it can't be taken twice).
    std::vector<double> traded_ = std::vector<double>(2 * kTicks, 0.0);
    std::vector<double> consumed_ = std::vector<double>(2 * kTicks, 0.0);
};

}  // namespace arbiter

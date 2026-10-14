#include <gtest/gtest.h>

#include <vector>

#include "arbiter/book.hpp"
#include "arbiter/sim.hpp"

using namespace arbiter;

namespace {

Event set(int64_t ts, Side s, int tick, double size) { return {ts, EventType::BookSet, s, static_cast<int16_t>(tick), size}; }
Event trade(int64_t ts, Side hit, int tick, double size) { return {ts, EventType::Trade, hit, static_cast<int16_t>(tick), size}; }

struct Idle : Strategy {
    void on_event(Simulator&, const Event&) override {}
};

SimConfig no_fees(bool cross = false) {
    SimConfig c;
    c.fees = FeeModel{Venue::Kalshi, 0.0, 0.0, 1.0};
    c.fill_on_cross = cross;
    return c;
}

// A book with 100 bid at 0.500 and 100 ask at 0.520.
Simulator seeded(SimConfig cfg, Idle& s) {
    Simulator sim(cfg);
    sim.process(set(1, Side::Bid, 500, 100), s);
    sim.process(set(1, Side::Ask, 520, 100), s);
    return sim;
}

}  // namespace

TEST(Book, TracksBestAndRescans) {
    Book b;
    EXPECT_FALSE(b.two_sided());
    b.set(Side::Bid, 400, 5);
    b.set(Side::Bid, 450, 7);
    b.set(Side::Ask, 470, 3);
    EXPECT_EQ(b.best_bid(), 450);
    EXPECT_EQ(b.best_ask(), 470);
    EXPECT_DOUBLE_EQ(b.mid(), 0.46);
    b.set(Side::Bid, 450, 0);  // best level empties: rescan finds 400
    EXPECT_EQ(b.best_bid(), 400);
    b.set(Side::Ask, 470, 0);
    EXPECT_FALSE(b.has_ask());
    b.clear();
    EXPECT_FALSE(b.has_bid());
}

TEST(Sim, QueuePositionMustTradeThroughBeforeWeFill) {
    Idle s;
    Simulator sim = seeded(no_fees(), s);
    const int64_t id = sim.place(Side::Bid, 500, 10);  // 100 ahead of us
    sim.process(trade(2, Side::Bid, 500, 60), s);
    EXPECT_EQ(sim.fills().size(), 0u);
    EXPECT_DOUBLE_EQ(sim.orders().at(id).ahead, 40);
    sim.process(trade(3, Side::Bid, 500, 50), s);  // 40 ahead, then 10 to us
    ASSERT_EQ(sim.fills().size(), 1u);
    EXPECT_DOUBLE_EQ(sim.fills()[0].qty, 10);
    EXPECT_TRUE(sim.fills()[0].maker);
    EXPECT_DOUBLE_EQ(sim.position(), 10);
    EXPECT_DOUBLE_EQ(sim.cash(), -5.0);
    EXPECT_TRUE(sim.orders().empty());
}

TEST(Sim, QueueAheadShareSetsStartingPosition) {
    Idle s;
    SimConfig cfg = no_fees();
    cfg.queue_ahead = 0.25;  // assume only a quarter of the level is ahead of us
    Simulator sim = seeded(cfg, s);
    const int64_t id = sim.place(Side::Bid, 500, 10);
    EXPECT_DOUBLE_EQ(sim.orders().at(id).ahead, 25);
    sim.process(trade(2, Side::Bid, 500, 30), s);  // 25 ahead, then 5 to us
    ASSERT_EQ(sim.fills().size(), 1u);
    EXPECT_DOUBLE_EQ(sim.fills()[0].qty, 5);
}

TEST(Sim, TradeThroughOurPriceFillsUsFirst) {
    Idle s;
    Simulator sim = seeded(no_fees(), s);
    sim.place(Side::Bid, 500, 10);
    sim.process(trade(2, Side::Bid, 490, 4), s);  // a seller went below our bid
    ASSERT_EQ(sim.fills().size(), 1u);
    EXPECT_DOUBLE_EQ(sim.fills()[0].qty, 4);
    EXPECT_DOUBLE_EQ(sim.fills()[0].price, 0.5);  // at our limit, not the trade price
}

TEST(Sim, CancelsAheadOfUsAreProportional) {
    Idle s;
    Simulator sim = seeded(no_fees(), s);
    const int64_t id = sim.place(Side::Bid, 500, 10);
    sim.process(set(2, Side::Bid, 500, 50), s);  // half the level cancelled, no trades
    EXPECT_DOUBLE_EQ(sim.orders().at(id).ahead, 50);
    sim.process(trade(3, Side::Bid, 500, 30), s);
    sim.process(set(3, Side::Bid, 500, 20), s);  // shrink fully explained by the trade
    EXPECT_DOUBLE_EQ(sim.orders().at(id).ahead, 20);
}

TEST(Sim, OurOwnEarlierOrdersAreAheadOfUs) {
    Idle s;
    Simulator sim = seeded(no_fees(), s);
    sim.place(Side::Bid, 510, 5);             // new best level: nothing ahead
    const int64_t b = sim.place(Side::Bid, 510, 5);
    EXPECT_DOUBLE_EQ(sim.orders().at(b).ahead, 5);
    sim.process(trade(2, Side::Bid, 510, 7), s);  // first order fills 5, second gets 2
    ASSERT_EQ(sim.fills().size(), 2u);
    EXPECT_DOUBLE_EQ(sim.fills()[0].qty, 5);
    EXPECT_DOUBLE_EQ(sim.fills()[1].qty, 2);
}

TEST(Sim, CrossingOrderIsImmediateOrCancelAndCannotRetakeLiquidity) {
    Idle s;
    SimConfig cfg = no_fees();
    cfg.fees = FeeModel{Venue::Kalshi, 0.07, 0.0, 1.0};
    Simulator sim = seeded(cfg, s);
    sim.place(Side::Bid, 530, 60);  // lifts 60 of the 100 offered at 0.520
    ASSERT_EQ(sim.fills().size(), 1u);
    EXPECT_FALSE(sim.fills()[0].maker);
    EXPECT_DOUBLE_EQ(sim.fills()[0].price, 0.52);
    EXPECT_DOUBLE_EQ(sim.fills()[0].fee, 1.05);  // 0.07*60*0.52*0.48 = 1.048, rounded up to the cent
    sim.place(Side::Bid, 530, 60);  // only 40 left at that level
    EXPECT_DOUBLE_EQ(sim.fills()[1].qty, 40);
    EXPECT_TRUE(sim.orders().empty());  // IOC: nothing rests
    sim.process(set(2, Side::Ask, 520, 100), s);  // level re-quoted
    sim.place(Side::Bid, 530, 10);
    EXPECT_DOUBLE_EQ(sim.fills()[2].qty, 10);
}

TEST(Sim, SettlementAndFees) {
    Idle s;
    SimConfig cfg = no_fees();
    cfg.fees = FeeModel{Venue::Kalshi, 0.07, 0.0175, 1.0};
    Simulator sim = seeded(cfg, s);
    sim.place(Side::Bid, 500, 100);
    sim.process(trade(2, Side::Bid, 500, 200), s);  // 100 ahead, then our 100
    EXPECT_DOUBLE_EQ(sim.fees_paid(), 0.44);          // maker fee on 100 @ 0.50
    EXPECT_NEAR(sim.settle(true), 100 * (1 - 0.5) - 0.44, 1e-9);
}

TEST(Sim, FillOnCrossForSnapshotData) {
    Idle s;
    Simulator sim = seeded(no_fees(true), s);
    sim.place(Side::Bid, 505, 10);
    sim.process(set(2, Side::Ask, 505, 50), s);  // ask moves to our price: not strictly through
    EXPECT_TRUE(sim.fills().empty());
    sim.process(set(3, Side::Ask, 504, 50), s);  // strictly through
    ASSERT_EQ(sim.fills().size(), 1u);
    EXPECT_DOUBLE_EQ(sim.fills()[0].price, 0.505);
}

TEST(Sim, InventoryCapRejectsOrders) {
    Idle s;
    SimConfig cfg = no_fees();
    cfg.max_position = 50;
    Simulator sim = seeded(cfg, s);
    EXPECT_GT(sim.place(Side::Bid, 500, 40), 0);
    EXPECT_EQ(sim.place(Side::Bid, 499, 20), -1);  // 40 pending + 20 > 50
    EXPECT_GT(sim.place(Side::Ask, 530, 50), 0);   // short side checked separately
    EXPECT_EQ(sim.place(Side::Bid, 0, 1), -1);     // ticks 0 and 1000 are not quotable
}

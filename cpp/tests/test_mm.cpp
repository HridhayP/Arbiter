#include <gtest/gtest.h>

#include "arbiter/mm.hpp"

using namespace arbiter;

namespace {
Event set(int64_t ts, Side s, int tick, double size) { return {ts, EventType::BookSet, s, static_cast<int16_t>(tick), size}; }
constexpr int64_t H = 3600LL * 1'000'000'000LL;

struct Quotes { int bid = -1, ask = -1; };
Quotes quotes(const Simulator& sim) {
    Quotes q;
    for (const auto& [id, o] : sim.orders()) (o.side == Side::Bid ? q.bid : q.ask) = o.tick;
    return q;
}
SimConfig cfg() { SimConfig c; c.fees = FeeModel{Venue::Kalshi, 0.0, 0.0, 1.0}; return c; }
MMParams params() { MMParams p; p.close_ts = 100 * H; p.stop_before_close_ns = H; return p; }
}  // namespace

TEST(MarketMaker, SymmetricInLogitSpaceWhenFlat) {
    Simulator sim(cfg());
    LogitMarketMaker mm(params());
    sim.process(set(H, Side::Bid, 400, 100), mm);
    sim.process(set(H, Side::Ask, 600, 100), mm);
    const Quotes q = quotes(sim);
    ASSERT_GT(q.bid, 0);
    ASSERT_GT(q.ask, 0);
    EXPECT_EQ(q.bid + q.ask, 1000);  // mid 0.5: logit-symmetric quotes are price-symmetric
    EXPECT_LT(q.bid, 600);
    EXPECT_GT(q.ask, 400);
}

TEST(MarketMaker, SkewsAgainstInventory) {
    Simulator flat(cfg()), longer(cfg());
    LogitMarketMaker a(params()), b(params());
    for (auto* pair : {&flat, &longer}) {
        LogitMarketMaker& m = pair == &flat ? a : b;
        pair->process(set(H, Side::Bid, 300, 100), m);
        pair->process(set(H, Side::Ask, 700, 100), m);
    }
    // give `longer` a long position by lifting the offer, then re-trigger quoting
    longer.cancel_all();
    longer.place(Side::Bid, 700, 50);
    ASSERT_DOUBLE_EQ(longer.position(), 50);
    longer.process(set(2 * H, Side::Ask, 700, 100), b);
    flat.process(set(2 * H, Side::Ask, 700, 100), a);
    EXPECT_LT(quotes(longer).bid, quotes(flat).bid);  // long: bid lower...
    EXPECT_LE(quotes(longer).ask, quotes(flat).ask);  // ...and offer at or below
}

TEST(MarketMaker, NeverCrossesAndStopsBeforeClose) {
    Simulator sim(cfg());
    MMParams p = params();
    p.gamma = 0.01;  // very tight quotes would cross a 1-tick book if allowed
    LogitMarketMaker mm(p);
    sim.process(set(H, Side::Bid, 490, 100), mm);
    sim.process(set(H, Side::Ask, 500, 100), mm);
    const Quotes q = quotes(sim);
    EXPECT_LE(q.bid, 490);
    EXPECT_GE(q.ask, 500);
    sim.process(set(99 * H + 1, Side::Bid, 490, 100), mm);  // within an hour of close
    EXPECT_TRUE(sim.orders().empty());
}

TEST(MarketMaker, RespectsInventoryLimit) {
    Simulator sim(cfg());
    MMParams p = params();
    p.max_inventory = 10;
    p.quote_size = 10;
    LogitMarketMaker mm(p);
    sim.process(set(H, Side::Bid, 400, 100), mm);
    sim.process(set(H, Side::Ask, 600, 100), mm);
    sim.cancel_all();
    sim.place(Side::Bid, 600, 10);  // now long 10 = the limit
    sim.process(set(2 * H, Side::Ask, 600, 100), mm);
    EXPECT_EQ(quotes(sim).bid, -1);  // no more buying
    EXPECT_GT(quotes(sim).ask, 0);
}

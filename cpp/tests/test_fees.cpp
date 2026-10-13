#include <gtest/gtest.h>

#include "arbiter/fees.hpp"

using arbiter::FeeModel;
using arbiter::Venue;

TEST(KalshiFees, PublishedExamples) {
    FeeModel taker{Venue::Kalshi, 0.07, 0.0175, 1.0};
    // 100 contracts at 50c: 0.07 * 100 * 0.25 = 1.75 exactly (must not round to 1.76)
    EXPECT_DOUBLE_EQ(taker.fee(0.50, 100, false), 1.75);
    // 1 contract at 50c: 0.0175 -> rounds up to 2 cents
    EXPECT_DOUBLE_EQ(taker.fee(0.50, 1, false), 0.02);
    // 10 contracts at 10c: 0.07 * 10 * 0.09 = 0.063 -> 0.07
    EXPECT_DOUBLE_EQ(taker.fee(0.10, 10, false), 0.07);
    // maker on a maker-fee series: 0.0175 * 100 * 0.25 = 0.4375 -> 0.44
    EXPECT_DOUBLE_EQ(taker.fee(0.50, 100, true), 0.44);
    EXPECT_DOUBLE_EQ(taker.fee(1.0, 100, false), 0.0);
}

TEST(KalshiFees, SymmetricInPrice) {
    FeeModel f{Venue::Kalshi, 0.07, 0.0, 1.0};
    for (int c = 1; c < 100; ++c) {
        const double p = c / 100.0;
        EXPECT_DOUBLE_EQ(f.fee(p, 37, false), f.fee(1.0 - p, 37, false)) << p;
    }
}

TEST(PolymarketFees, ScheduleFormula) {
    FeeModel f{Venue::Polymarket, 0.05, 0.0, 1.0};
    EXPECT_DOUBLE_EQ(f.fee(0.50, 100, false), 1.25);  // 100 * 0.05 * 0.25
    EXPECT_DOUBLE_EQ(f.fee(0.50, 100, true), 0.0);    // makers never pay
    FeeModel free{Venue::Polymarket, 0.0, 0.0, 1.0};
    EXPECT_DOUBLE_EQ(free.fee(0.3, 1000, false), 0.0);
}

TEST(Fees, RejectsBadPrice) {
    FeeModel f;
    EXPECT_THROW(f.fee(1.5, 1, false), std::invalid_argument);
}

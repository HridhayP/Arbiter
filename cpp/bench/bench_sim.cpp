// Replay throughput: a synthetic random-walk book with trade prints,
// processed by the simulator with the logit market maker attached.
#include <benchmark/benchmark.h>

#include <random>
#include <vector>

#include "arbiter/mm.hpp"
#include "arbiter/sim.hpp"

using namespace arbiter;

static std::vector<Event> synthetic(size_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> step(0.0, 1.0);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<Event> ev;
    ev.reserve(n);
    int mid = 500;
    int64_t ts = 1'700'000'000'000'000'000LL;
    while (ev.size() < n) {
        ts += 50'000'000;  // 50 ms
        if (u(rng) < 0.1) mid = std::clamp(mid + (step(rng) > 0 ? 10 : -10), 50, 950);
        const double r = u(rng);
        if (r < 0.45) {
            ev.push_back({ts, EventType::BookSet, Side::Bid, static_cast<int16_t>(mid - 10 * (1 + static_cast<int>(u(rng) * 3))), 50 + 500 * u(rng)});
        } else if (r < 0.9) {
            ev.push_back({ts, EventType::BookSet, Side::Ask, static_cast<int16_t>(mid + 10 * (1 + static_cast<int>(u(rng) * 3))), 50 + 500 * u(rng)});
        } else {
            const bool hit_bid = u(rng) < 0.5;
            ev.push_back({ts, EventType::Trade, hit_bid ? Side::Bid : Side::Ask,
                          static_cast<int16_t>(hit_bid ? mid - 10 : mid + 10), 1 + 50 * u(rng)});
        }
    }
    return ev;
}

static void BM_ReplayWithMarketMaker(benchmark::State& state) {
    const auto events = synthetic(static_cast<size_t>(state.range(0)), 42);
    int64_t fills = 0;
    for (auto _ : state) {
        SimConfig cfg;
        cfg.fees = FeeModel{Venue::Kalshi, 0.07, 0.0175, 1.0};
        Simulator sim(cfg);
        MMParams p;
        p.close_ts = INT64_MAX;
        p.stop_before_close_ns = 0;
        LogitMarketMaker mm(p);
        sim.run(events, mm);
        fills = static_cast<int64_t>(sim.fills().size());
        benchmark::DoNotOptimize(sim.cash());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(events.size()));
    state.counters["fills"] = static_cast<double>(fills);
}
BENCHMARK(BM_ReplayWithMarketMaker)->Arg(1'000'000)->Unit(benchmark::kMillisecond);

static void BM_BookSet(benchmark::State& state) {
    Book b;
    int i = 0;
    for (auto _ : state) {
        b.set(i & 1 ? Side::Bid : Side::Ask, 400 + (i % 200), (i % 7) * 10.0);
        benchmark::DoNotOptimize(b.best_bid());
        ++i;
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_BookSet);

BENCHMARK_MAIN();

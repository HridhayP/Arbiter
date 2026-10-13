# Arbiter

Arbiter is a research lab for prediction markets (Kalshi and Polymarket). It asks one question: **are the prices right, and where they aren't, can the error be traded after fees?** It collects every resolved market both venues expose, records live order books, and measures five things:

1. **Calibration (R1).** Do 90¢ contracts win 90% of the time, or are long shots overpriced?
2. **Cross-venue mispricing (R2).** Gaps between Kalshi and Polymarket prices for the same event, counted only when they are executable after fees.
3. **Internal consistency (R3).** Strike ladders, mutually exclusive outcome sets and combo (parlay) prices, with every violation classified as risk-free or not.
4. **Market making (R4).** A logit-space Avellaneda–Stoikov quoter, backtested in a C++20 event-driven simulator with queue-position fills.
5. **Forecasting (R5b).** Can a model beat the market's own price, out of sample?

It is research and paper trading only. No orders are ever placed. Results: [RESULTS.md](RESULTS.md).

## Architecture

```mermaid
flowchart LR
    subgraph SRC["Public APIs (no keys)"]
        KR[Kalshi REST<br/>live + /historical]
        PR[Polymarket gamma + CLOB REST]
        PW[Polymarket market WebSocket]
    end
    subgraph ING["Python ingest"]
        C["collectors<br/>per-series / date-window crawls<br/>resumable, raw responses kept"]
        REC["recorder<br/>WS books + REST polling<br/>full-listing snapshots every ~15 min"]
    end
    subgraph DATA["Storage"]
        RAW[(raw JSON.gz)]
        PQ[(Parquet tables<br/>+ DuckDB)]
    end
    subgraph RES["Research (Python)"]
        R1[R1 calibration]
        R2[R2 cross-venue]
        R3[R3 consistency]
        R5["R5 matching (embeddings + rules)<br/>and forecasting"]
    end
    subgraph SIM["arbiter._sim (C++20 + pybind11)"]
        BK["Book<br/>flat 0.001 tick grid"]
        SM["Simulator<br/>queue-position fills · fees · settlement"]
        MM["LogitMarketMaker"]
    end
    KR & PR --> C --> RAW --> PQ
    PW & KR & PR --> REC --> PQ
    PQ --> R1 & R2 & R3 & R5
    R5 -->|matched pairs| R2
    PQ --> SIM --> R4[R4 market making]
```

| Path | Role |
|---|---|
| `arbiter/venues/` | Kalshi and Polymarket clients and normalizers |
| `arbiter/collect.py` | historical crawls (series, markets, candles, price history, trades) |
| `arbiter/recorder.py` | live depth, trades and full-listing top-of-book capture |
| `arbiter/fees.py`, `cpp/include/arbiter/fees.hpp` | fee models read from venue metadata (Python and C++, tested for agreement) |
| `arbiter/research/` | calibration, matching, consistency, forecasting |
| `arbiter/backtest.py` | events from candles/recordings into the simulator; walk-forward evaluation |
| `cpp/` | order book, simulator, market maker, GoogleTest, Google Benchmark, pybind11 bindings |
| `labels/match_labels.csv` | 300 hand-labeled cross-venue pairs |

## Quickstart

Linux or WSL with g++ ≥ 12 and CMake; Python ≥ 3.10.

```bash
pip install -e ".[dev]"                     # builds the C++ extension
pytest -q                                   # Python tests (C++ bindings included)
cmake -S . -B build && cmake --build build -j && ctest --test-dir build   # C++ tests
./build/arbiter_bench                       # simulator throughput

python -m arbiter.collect kalshi-series     # then kalshi-markets, polymarket-markets, tables, ...
python -m arbiter.recorder                  # leave running; writes data/recorder/
python -m arbiter.research.calibration      # and consistency, forecast; see RESULTS.md for commands
```

## Benchmarks and results

__RESULTS_TABLE__

## Design decisions & tradeoffs

__DESIGN__

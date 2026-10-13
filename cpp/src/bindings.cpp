// Python bindings for the arbiter C++ core.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

#include "arbiter/book.hpp"
#include "arbiter/fees.hpp"
#include "arbiter/mm.hpp"
#include "arbiter/sim.hpp"

namespace py = pybind11;
using namespace arbiter;

// Lets Python subclasses of Strategy receive callbacks from C++.
class PyStrategy : public Strategy {
public:
    using Strategy::Strategy;
    void on_event(Simulator& sim, const Event& ev) override {
        PYBIND11_OVERRIDE_PURE(void, Strategy, on_event, sim, ev);
    }
    void on_fill(Simulator& sim, const Fill& f) override { PYBIND11_OVERRIDE(void, Strategy, on_fill, sim, f); }
};

static std::vector<Event> to_events(py::array_t<int64_t> ts, py::array_t<uint8_t> type, py::array_t<uint8_t> side,
                                    py::array_t<int16_t> tick, py::array_t<double> size) {
    const auto n = static_cast<size_t>(ts.size());
    if (type.size() != ts.size() || side.size() != ts.size() || tick.size() != ts.size() || size.size() != ts.size())
        throw std::invalid_argument("event arrays must have equal length");
    auto t = ts.unchecked<1>();
    auto ty = type.unchecked<1>();
    auto s = side.unchecked<1>();
    auto k = tick.unchecked<1>();
    auto z = size.unchecked<1>();
    std::vector<Event> ev(n);
    for (size_t i = 0; i < n; ++i) {
        if (ty(i) > 3) throw std::invalid_argument("bad event type");
        ev[i] = Event{t(i), static_cast<EventType>(ty(i)), static_cast<Side>(s(i)), k(i), z(i)};
    }
    return ev;
}

PYBIND11_MODULE(_sim, m) {
    m.doc() = "arbiter C++ core: fees, order book and backtest simulator";

    py::enum_<Venue>(m, "Venue").value("kalshi", Venue::Kalshi).value("polymarket", Venue::Polymarket);
    py::class_<FeeModel>(m, "FeeModel")
        .def(py::init([](Venue v, double taker, double maker, double exponent) {
                 return FeeModel{v, taker, maker, exponent};
             }),
             py::arg("venue"), py::arg("taker_rate"), py::arg("maker_rate"), py::arg("exponent") = 1.0)
        .def("fee", &FeeModel::fee, py::arg("price"), py::arg("contracts"), py::arg("maker"));

    py::enum_<Side>(m, "Side").value("bid", Side::Bid).value("ask", Side::Ask);
    py::enum_<EventType>(m, "EventType")
        .value("book_set", EventType::BookSet)
        .value("book_clear", EventType::BookClear)
        .value("trade", EventType::Trade)
        .value("timer", EventType::Timer);

    py::class_<Event>(m, "Event")
        .def_readonly("ts", &Event::ts)
        .def_readonly("type", &Event::type)
        .def_readonly("side", &Event::side)
        .def_readonly("tick", &Event::tick)
        .def_readonly("size", &Event::size);
    py::class_<Fill>(m, "Fill")
        .def_readonly("ts", &Fill::ts)
        .def_readonly("order_id", &Fill::order_id)
        .def_readonly("side", &Fill::side)
        .def_readonly("price", &Fill::price)
        .def_readonly("qty", &Fill::qty)
        .def_readonly("maker", &Fill::maker)
        .def_readonly("fee", &Fill::fee);
    py::class_<Order>(m, "Order")
        .def_readonly("id", &Order::id)
        .def_readonly("side", &Order::side)
        .def_readonly("tick", &Order::tick)
        .def_readonly("qty", &Order::qty)
        .def_readonly("ahead", &Order::ahead);
    py::class_<Book>(m, "Book")
        .def_property_readonly("best_bid", &Book::best_bid)
        .def_property_readonly("best_ask", &Book::best_ask)
        .def_property_readonly("mid", &Book::mid)
        .def_property_readonly("two_sided", &Book::two_sided)
        .def("size", &Book::size);

    py::class_<Strategy, PyStrategy>(m, "Strategy")
        .def(py::init<>())
        .def("on_event", &Strategy::on_event)
        .def("on_fill", &Strategy::on_fill);

    py::class_<SimConfig>(m, "SimConfig")
        .def(py::init([](FeeModel fees, bool fill_on_cross, double max_position) {
                 SimConfig c;
                 c.fees = fees;
                 c.fill_on_cross = fill_on_cross;
                 c.max_position = max_position;
                 return c;
             }),
             py::arg("fees"), py::arg("fill_on_cross") = false, py::arg("max_position") = 1e18);

    py::class_<Simulator>(m, "Simulator")
        .def(py::init<SimConfig>())
        .def("place", &Simulator::place, py::arg("side"), py::arg("tick"), py::arg("qty"))
        .def("cancel", &Simulator::cancel)
        .def("cancel_all", &Simulator::cancel_all)
        .def_property_readonly("book", &Simulator::book, py::return_value_policy::reference_internal)
        .def_property_readonly("position", &Simulator::position)
        .def_property_readonly("cash", &Simulator::cash)
        .def_property_readonly("now", &Simulator::now)
        .def_property_readonly("fees_paid", &Simulator::fees_paid)
        .def_property_readonly("events_processed", &Simulator::events_processed)
        .def_property_readonly("orders", [](const Simulator& s) {
            std::vector<Order> out;
            for (const auto& [id, o] : s.orders()) out.push_back(o);
            return out;
        })
        .def_property_readonly("fills", &Simulator::fills)
        .def("mark_to_market", &Simulator::mark_to_market)
        .def("settle", &Simulator::settle, py::arg("yes"))
        .def("run",
             [](Simulator& s, py::array_t<int64_t> ts, py::array_t<uint8_t> type, py::array_t<uint8_t> side,
                py::array_t<int16_t> tick, py::array_t<double> size, Strategy& strategy) {
                 auto ev = to_events(ts, type, side, tick, size);
                 s.run(ev, strategy);
             },
             py::arg("ts"), py::arg("type"), py::arg("side"), py::arg("tick"), py::arg("size"), py::arg("strategy"));

    py::class_<MMParams>(m, "MMParams")
        .def(py::init<>())
        .def_readwrite("gamma", &MMParams::gamma)
        .def_readwrite("k", &MMParams::k)
        .def_readwrite("sigma", &MMParams::sigma)
        .def_readwrite("quote_size", &MMParams::quote_size)
        .def_readwrite("max_inventory", &MMParams::max_inventory)
        .def_readwrite("min_spread_ticks", &MMParams::min_spread_ticks)
        .def_readwrite("horizon_ns", &MMParams::horizon_ns)
        .def_readwrite("stop_before_close_ns", &MMParams::stop_before_close_ns)
        .def_readwrite("close_ts", &MMParams::close_ts)
        .def_readwrite("requote_ticks", &MMParams::requote_ticks)
        .def_readwrite("tick_size", &MMParams::tick_size);
    py::class_<LogitMarketMaker, Strategy>(m, "LogitMarketMaker")
        .def(py::init<MMParams>())
        .def_property_readonly("quotes_placed", &LogitMarketMaker::quotes_placed);
}

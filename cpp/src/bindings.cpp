// Python bindings for the arbiter C++ core.
#include <pybind11/pybind11.h>

#include "arbiter/fees.hpp"

namespace py = pybind11;
using namespace arbiter;

PYBIND11_MODULE(_sim, m) {
    m.doc() = "arbiter C++ core: fees, order book and backtest simulator";
    py::enum_<Venue>(m, "Venue").value("kalshi", Venue::Kalshi).value("polymarket", Venue::Polymarket);
    py::class_<FeeModel>(m, "FeeModel")
        .def(py::init([](Venue v, double taker, double maker, double exponent) {
                 return FeeModel{v, taker, maker, exponent};
             }),
             py::arg("venue"), py::arg("taker_rate"), py::arg("maker_rate"), py::arg("exponent") = 1.0)
        .def("fee", &FeeModel::fee, py::arg("price"), py::arg("contracts"), py::arg("maker"));
}

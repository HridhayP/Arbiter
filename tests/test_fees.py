import itertools

import pytest

from arbiter.fees import FeeModel, kalshi_fees, polymarket_fees


def test_kalshi_published_values():
    f = kalshi_fees("quadratic_with_maker_fees", 1)
    assert f.fee(0.50, 100, maker=False) == 1.75
    assert f.fee(0.50, 1, maker=False) == 0.02
    assert f.fee(0.10, 10, maker=False) == 0.07
    assert f.fee(0.50, 100, maker=True) == 0.44
    assert kalshi_fees("quadratic", 1).fee(0.5, 100, maker=True) == 0.0  # no maker fee on plain series
    assert kalshi_fees("quadratic", 0.5).fee(0.5, 100, maker=False) == 0.88  # multiplier halves: 0.875 -> 0.88


def test_polymarket_from_market_object():
    m = {"feesEnabled": True, "feeSchedule": {"exponent": 1, "rate": 0.05, "takerOnly": True}}
    f = polymarket_fees(m)
    assert f.fee(0.5, 100, maker=False) == 1.25
    assert f.fee(0.5, 100, maker=True) == 0.0
    assert polymarket_fees({"feesEnabled": False}).fee(0.5, 100, maker=False) == 0.0


def test_python_matches_cpp():
    sim = pytest.importorskip("arbiter._sim")
    cases = [("kalshi", 0.07, 0.0175, 1.0), ("polymarket", 0.05, 0.0, 1.0)]
    for (venue, t, mk, e), p, c, maker in itertools.product(cases, [0.01, 0.1, 0.37, 0.5, 0.93], [1, 7, 100, 2500], [False, True]):
        py = FeeModel(venue, t, mk, e).fee(p, c, maker)
        cc = sim.FeeModel(getattr(sim.Venue, venue), t, mk, e).fee(p, c, maker)
        assert py == pytest.approx(cc, abs=1e-12), (venue, p, c, maker)

import httpx
import pytest

from arbiter import http as H


def client_with(handler, **kw):
    c = H.RateLimitedClient(rates={"example.test": 1000.0}, **kw)
    c._client = httpx.Client(transport=httpx.MockTransport(handler))
    return c


def test_rides_out_a_network_outage(monkeypatch):
    monkeypatch.setattr(H.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] <= 8:  # more failures than the HTTP-error retry budget
            raise httpx.ConnectError("Temporary failure in name resolution")
        return httpx.Response(200, json={"ok": True})

    assert client_with(handler).get_json("https://example.test/x") == {"ok": True}
    assert calls["n"] == 9


def test_gives_up_after_max_outage(monkeypatch):
    monkeypatch.setattr(H.time, "sleep", lambda s: None)
    clock = {"t": 0.0}
    monkeypatch.setattr(H.time, "monotonic", lambda: clock.__setitem__("t", clock["t"] + 100) or clock["t"])

    def handler(request):
        raise httpx.ConnectError("down")

    with pytest.raises(httpx.ConnectError):
        client_with(handler, max_outage=1000).get_json("https://example.test/x")


def test_http_errors_still_have_a_budget(monkeypatch):
    monkeypatch.setattr(H.time, "sleep", lambda s: None)
    with pytest.raises(httpx.HTTPStatusError):
        client_with(lambda r: httpx.Response(503), max_retries=2).get_json("https://example.test/x")
    assert client_with(lambda r: httpx.Response(404)).get_json("https://example.test/x") is None

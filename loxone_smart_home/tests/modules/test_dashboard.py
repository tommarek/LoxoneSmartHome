"""Unit tests for the monitoring dashboard's pure helpers.

The dashboard (modules/growatt/dashboard.py) is the public production page on
port 5555. Most of it is aiohttp wiring, but the price-row builder and the
actuals query helper contain real arithmetic/timezone/cache logic worth
locking down so a refactor of the shared helpers can't silently change output.
"""
from datetime import datetime
from types import SimpleNamespace

import pytest

from modules.growatt import dashboard


class _FakeConfig:
    # Mirror the real settings defaults (D57d: VT+system / NT+system).
    distribution_tariff_high = 0.919
    distribution_tariff_low = 0.281
    low_tariff_hours = "0-10,11-12,13-14,15-17,18-24"


def _fake_ctrl():
    return SimpleNamespace(config=_FakeConfig())


def test_build_price_rows_basic_economics_and_status():
    ctrl = _fake_ctrl()
    price_items = {
        ("10:00", "10:15"): 3.0,
        ("11:00", "11:15"): 5.0,
    }.items()
    rows = dashboard._build_price_rows(
        ctrl, price_items, "today",
        charging={("10:00", "10:15")},
        pre_discharge=set(),
        discharge={("11:00", "11:15")},
        sell_production=set(),
        soc_lookup={},
        sell_fee=0.5,
        batt_amort=2.0,
        inv_off_threshold=-2.1,
        cur_block=("10:00", "10:15"),
    )
    assert [r["start"] for r in rows] == ["10:00", "11:00"]
    charging_row, discharge_row = rows
    # Status reflects the per-action block-sets.
    assert charging_row["status"] == "charging"
    assert charging_row["is_current"] is True
    assert discharge_row["status"] == "discharge"
    assert discharge_row["is_current"] is False
    # net_sell = price - sell_fee - batt_amort. Export pays NO distribution, so
    # net_sell must NOT subtract the distribution tariff (which is import-only).
    # Hour 10 is a HIGH-tariff hour (ranges are half-open: 0-10 excludes 10),
    # so the reference distribution must be exactly the high tariff (0.919→0.92).
    assert charging_row["distribution_czk"] == 0.92
    assert charging_row["net_sell_czk"] == round(3.0 - 0.5 - 2.0, 2)
    assert charging_row["day"] == "today"
    # Planned inverter powerRate is surfaced per row (None when no projection).
    assert "projected_power_rate" in charging_row


def test_build_price_rows_inverter_off_only_when_not_charging():
    ctrl = _fake_ctrl()
    price_items = {
        ("02:00", "02:15"): -3.0,  # below threshold, not charging → inverter_off
        ("03:00", "03:15"): -3.0,  # below threshold but charging → charging wins
    }.items()
    rows = dashboard._build_price_rows(
        ctrl, price_items, "today",
        charging={("03:00", "03:15")},
        pre_discharge=set(), discharge=set(), sell_production=set(),
        soc_lookup={}, sell_fee=0.5, batt_amort=2.0,
        inv_off_threshold=-2.1, cur_block=None,
    )
    assert rows[0]["status"] == "inverter_off"
    assert rows[0]["is_inverter_off"] is True
    assert rows[1]["status"] == "charging"
    assert rows[1]["is_inverter_off"] is False


def test_build_price_rows_projection_lookup_uses_day_prefix():
    ctrl = _fake_ctrl()
    soc_lookup = {"tomorrow:10:00": {"soc": 55.0, "action": "charge"}}
    rows = dashboard._build_price_rows(
        ctrl, {("10:00", "10:15"): 1.0}.items(), "tomorrow",
        charging=set(), pre_discharge=set(), discharge=set(), sell_production=set(),
        soc_lookup=soc_lookup, sell_fee=0.5, batt_amort=2.0,
        inv_off_threshold=-2.1, cur_block=None,
    )
    assert rows[0]["projected_soc"] == 55.0
    assert rows[0]["projected_action"] == "charge"


class _FakeRecord:
    def __init__(self, t, v):
        self._t, self._v = t, v

    def get_time(self):
        return self._t

    def get_value(self):
        return self._v


class _FakeTable:
    def __init__(self, records):
        self.records = records


class _FakeInflux:
    def __init__(self, records):
        self._records = records

    async def query(self, q):
        return [_FakeTable(self._records)]


def _actuals_ctrl(records):
    return SimpleNamespace(
        influxdb_client=_FakeInflux(records),
        settings=SimpleNamespace(influxdb=SimpleNamespace(bucket_solar="solar")),
    )


@pytest.mark.asyncio
async def test_today_local_actuals_buckets_today_and_applies_transform():
    today = datetime.now().date()
    # Naive timestamps (no tzinfo) are used verbatim; one today, one yesterday.
    rec_today = _FakeRecord(datetime(today.year, today.month, today.day, 9, 0), 2000.0)
    rec_old = _FakeRecord(datetime(2000, 1, 1, 9, 0), 9999.0)
    ctrl = _actuals_ctrl([rec_today, rec_old])
    payload = await dashboard._today_local_actuals(
        ctrl, field="InputPower", every="1h", agg_fn="mean", time_fmt="%H:00",
        value_fn=lambda w: round(w / 1000.0, 3),
        result_key="hourly", cache_attr="_test_cache", ttl=300,
    )
    # Only today's record kept; 2000 W → 2.0 kWh; yesterday dropped.
    assert payload == {"hourly": {"09:00": 2.0}}


@pytest.mark.asyncio
async def test_today_local_actuals_no_client_returns_empty():
    payload = await dashboard._today_local_actuals(
        SimpleNamespace(influxdb_client=None),
        field="SOC", every="15m", agg_fn="last", time_fmt="%H:%M",
        value_fn=lambda s: round(s, 1),
        result_key="blocks", cache_attr="_c", ttl=90,
    )
    assert payload == {"blocks": {}}


@pytest.mark.asyncio
async def test_today_local_actuals_uses_cache_within_ttl():
    ctrl = _actuals_ctrl([])
    ctrl._test_cache = (datetime.now(), {"hourly": {"cached": 1.0}})
    payload = await dashboard._today_local_actuals(
        ctrl, field="InputPower", every="1h", agg_fn="mean", time_fmt="%H:00",
        value_fn=lambda w: w, result_key="hourly", cache_attr="_test_cache", ttl=300,
    )
    assert payload == {"hourly": {"cached": 1.0}}


# ---------------------------------------------------------------------------
# Mobile-app / PWA structure
# ---------------------------------------------------------------------------
import json as _json

from aiohttp.test_utils import TestClient, TestServer


def test_dashboard_html_has_tab_app_shell():
    html = dashboard.DASHBOARD_HTML
    # Bottom tab bar + the four tab pages.
    assert 'class="tabbar"' in html
    for tab in ("tab-home", "tab-chart", "tab-insights", "tab-control"):
        assert f'id="{tab}"' in html
    # Exactly four tab pages, balanced with their closing tags.
    assert html.count('class="tab-page') == 4
    assert html.count("</section>") == 4
    # showTab nav JS present.
    assert "function showTab(" in html


def test_dashboard_html_has_pwa_head_and_scroll_chart():
    html = dashboard.DASHBOARD_HTML
    assert 'rel="manifest" href="/manifest.webmanifest"' in html
    assert 'name="theme-color"' in html
    assert 'rel="apple-touch-icon"' in html
    assert "serviceWorker" in html and "/sw.js" in html
    # Horizontal-scroll chart wrappers + shared content-width for overlays.
    assert 'class="chart-scroll"' in html
    assert "dataset.contentW" in html


def test_manifest_json_is_standalone_pwa():
    m = _json.loads(dashboard.MANIFEST_JSON)
    assert m["display"] == "standalone"
    assert m["start_url"] == "/"
    assert m["theme_color"] == "#0f1117"
    assert any(i["src"] == "/icon-512.png" for i in m["icons"])


def test_sw_and_icon_assets_present():
    assert dashboard.SW_JS.strip()
    assert "addEventListener('fetch'" in dashboard.SW_JS  # installable
    assert dashboard.ICON_SVG.strip().startswith("<svg")
    # Committed PNG icons load (180 is the iOS apple-touch-icon).
    assert dashboard._ICON_180[:8] == b"\x89PNG\r\n\x1a\n"
    assert dashboard._ICON_512[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize("path,ctype", [
    ("/manifest.webmanifest", "application/manifest+json"),
    ("/sw.js", "application/javascript"),
    ("/icon.svg", "image/svg+xml"),
    ("/icon-180.png", "image/png"),
    ("/icon-512.png", "image/png"),
])
async def test_pwa_routes_serve_200(path, ctype):
    app = dashboard.create_dashboard_app(_fake_ctrl())
    async with TestClient(TestServer(app)) as client:
        resp = await client.get(path)
        assert resp.status == 200
        assert resp.headers["Content-Type"].startswith(ctype)


async def test_dashboard_root_serves_html():
    app = dashboard.create_dashboard_app(_fake_ctrl())
    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/")
        assert resp.status == 200
        body = await resp.text()
        assert 'class="tabbar"' in body


def test_dashboard_has_live_header_strip_and_ptr():
    html = dashboard.DASHBOARD_HTML
    # Always-visible app bar with the live stat strip (solar/load/battery/grid).
    assert 'class="appbar"' in html
    assert 'class="statstrip"' in html
    for sid in ("ssSolar", "ssLoad", "ssBatt", "ssGrid"):
        assert f'id="{sid}"' in html
    # Status dot (green/red/gray) + the renderer + dot state setter.
    assert 'id="statusDot"' in html
    assert "function setDot(" in html and "renderHeaderStrip" in html
    # Pull-to-refresh element + handler + a manual refresh button.
    assert 'class="ptr"' in html and "refreshAll" in html
    assert 'id="refreshBtn"' in html
    assert "touchmove" in html and "Release to refresh" in html


# --- API/pages split (web-container decoupling) ---------------------------

def _route_paths(app):
    return {r.resource.canonical for r in app.router.routes()}


def test_api_app_has_no_page_routes():
    """The controller-backed API app must not serve pages — those live in the
    standalone web container so it can restart without the controller."""
    paths = _route_paths(dashboard.create_api_app(_fake_ctrl()))
    assert "/api/settings" in paths and "/api/status" in paths
    assert "/" not in paths and "/settings" not in paths


def test_pages_app_has_no_direct_api_handlers():
    """The pages app serves pages and proxies /api/* — it must NOT register the
    real controller-backed handlers (it has no controller)."""
    app = dashboard.create_pages_app("http://main:5556")
    paths = _route_paths(app)
    assert "/" in paths and "/settings" in paths
    # The only /api route is the catch-all proxy, not the concrete endpoints.
    assert "/api/status" not in paths
    assert app["api_upstream"] == "http://main:5556"


async def test_pages_app_proxies_api_to_upstream():
    """End-to-end: GET + POST under /api on the pages app are forwarded to the
    upstream and the response streamed back; method, body and status survive."""
    from aiohttp import web as _web

    async def _echo(request):
        body = await request.text() if request.body_exists else ""
        return _web.json_response(
            {"method": request.method, "path": request.path, "body": body}
        )

    upstream = _web.Application()
    upstream.router.add_route("*", "/api/{tail:.*}", _echo)

    async with TestServer(upstream) as up_server:
        up_url = f"http://127.0.0.1:{up_server.port}"
        pages = dashboard.create_pages_app(up_url)
        async with TestClient(TestServer(pages)) as client:
            r = await client.get("/api/status")
            assert r.status == 200
            assert (await r.json()) == {
                "method": "GET", "path": "/api/status", "body": ""
            }
            r2 = await client.post("/api/settings", json={"x": 1})
            j2 = await r2.json()
            assert j2["method"] == "POST" and j2["path"] == "/api/settings"
            assert '"x": 1' in j2["body"]

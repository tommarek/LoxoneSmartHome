#!/usr/bin/env python3
"""Standalone Solcast -> InfluxDB PV-forecast writer.

The Rust mpc-brain reads PV forecasts from InfluxDB:

    bucket=solar  measurement=solar_forecast_history
    tag   forecast_date="YYYY-MM-DD" (local)
    field hourly_json='{"<hour>": <kWh>, ...}' (local-hour buckets)
    field total_kwh=<float>
    field source="solcast"      <-- the brain filters on exactly this

That data used to be produced inside the Growatt controller. With the controller
disabled (GROWATT_CONTROLLER_ENABLED=false, so the Python optimizer doesn't fight
the Rust mpc-growatt), the only Solcast producer died and the brain falls back to
its clear-sky model. This standalone service restores it: it reuses the tested
SolcastForecast client to fetch + aggregate, and writes the points the brain reads.
It touches no inverter / MQTT / control path -- telemetry write only.
"""
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None

sys.path.insert(0, "/app")
from modules.growatt.solcast_forecast import SolcastForecast  # noqa: E402
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync  # noqa: E402
from influxdb_client import Point  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [solcast-writer] %(levelname)s %(message)s",
)
log = logging.getLogger("solcast-writer")

API_KEY = os.environ.get("GROWATT_SOLCAST_API_KEY", "")
ROOFTOP = os.environ.get("GROWATT_SOLCAST_ROOFTOP_ID", "")
QUANTILE = os.environ.get("GROWATT_SOLCAST_QUANTILE", "p50")
INFLUX_URL = os.environ.get("INFLUXDB_HOST", "http://influxdb:8086")
INFLUX_TOKEN = os.environ.get("INFLUXDB_TOKEN", "")
INFLUX_ORG = os.environ.get("INFLUXDB_ORG", "loxone")
BUCKET = os.environ.get("SOLCAST_BUCKET", "solar")
INTERVAL_H = float(os.environ.get("SOLCAST_INTERVAL_HOURS", "6"))
TZ_NAME = os.environ.get("TZ", "Europe/Prague")

# Fail closed on timezone: bucketing into the wrong (UTC-shifted) local hours/date
# would silently feed the brain bad PV. Refuse rather than mis-bucket.
if ZoneInfo is None:
    log.error("zoneinfo/tzdata unavailable -- cannot bucket into %s local hours; refusing. Install tzdata.", TZ_NAME)
    raise RuntimeError(f"zoneinfo/tzdata unavailable; cannot resolve TZ {TZ_NAME!r}")
try:
    LOCAL_TZ = ZoneInfo(TZ_NAME)
except Exception as e:  # noqa: BLE001 (ZoneInfoNotFoundError subclasses KeyError)
    log.error("Invalid/unavailable TZ %r (%s) -- refusing to publish UTC-shifted PV forecasts", TZ_NAME, e)
    raise
QUOTA_PATH = Path(os.environ.get("SOLCAST_QUOTA_PATH", "/app/data/solcast_quota.json"))


async def write_once(solcast: SolcastForecast, influx: InfluxDBClientAsync) -> int:
    # {date_str: {hour: kWh}}, summed across all rooftop sites, local-hour buckets.
    hourly_by_date = await solcast.fetch_hourly_today_tomorrow(force=True, local_tz=LOCAL_TZ)
    if not hourly_by_date:
        log.warning("No Solcast data returned (quota throttle or transient error) -- nothing written this cycle")
        return 0
    write_api = influx.write_api()
    written = 0
    for date_str, hourly in sorted(hourly_by_date.items()):
        total = round(sum(hourly.values()), 3)
        if total < 0.01:
            continue
        hourly_json = json.dumps({str(h): round(float(k), 3) for h, k in sorted(hourly.items())})
        point = (
            Point("solar_forecast_history")
            .tag("forecast_date", date_str)
            .field("hourly_json", hourly_json)
            .field("total_kwh", float(total))
            .field("source", "solcast")
        )
        try:
            await write_api.write(bucket=BUCKET, org=INFLUX_ORG, record=point)
        except Exception as e:  # noqa: BLE001
            log.error("write failed for %s: %s -- continuing with next date", date_str, e)
            continue
        log.info("wrote %s: total=%.2f kWh across %d hours", date_str, total, len(hourly))
        written += 1
    return written


async def main() -> None:
    if not API_KEY or not ROOFTOP:
        log.error("GROWATT_SOLCAST_API_KEY / GROWATT_SOLCAST_ROOFTOP_ID missing -- exiting")
        return
    QUOTA_PATH.parent.mkdir(parents=True, exist_ok=True)
    solcast = SolcastForecast(
        api_key=API_KEY, rooftop_id=ROOFTOP, logger=log,
        quantile=QUANTILE, quota_path=QUOTA_PATH,
    )
    if not solcast.enabled:
        log.error("Solcast client not enabled (key/rooftop) -- exiting")
        return
    log.info(
        "Solcast writer up: %d site(s), quantile=%s, every %sh -> bucket=%s measurement=solar_forecast_history source=solcast",
        len(solcast.rooftop_ids), solcast.quantile, INTERVAL_H, BUCKET,
    )
    while True:
        try:
            async with InfluxDBClientAsync(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG) as influx:
                n = await write_once(solcast, influx)
                log.info("cycle complete: %d day(s) written", n)
        except Exception as e:  # noqa: BLE001
            log.error("cycle failed: %s", e, exc_info=True)
        await asyncio.sleep(INTERVAL_H * 3600)


if __name__ == "__main__":
    asyncio.run(main())

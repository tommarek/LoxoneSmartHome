r"""Read-only economics diagnostic — run in the controller container.

Cross-checks the dashboard's per-block import-cost integration against the raw
InfluxDB meter counters and a hand-computed sum, and prints the config-derived
VT/NT distribution-tariff hour split. Confirms (or rules out) the "import cost
reads too low" bug before/after the dashboard.py economics fix.

Run from the dev machine (per CLAUDE.md — token can't be inlined, the container
has no curl, so read it from the env inside the container):

    ssh -p 2222 tom@192.168.0.201 bash -l << 'ENDSSH'
    docker exec -i -w /app loxone_smart_home python3 - < \
      scripts/diagnose_economics.py 2>&1 | sed 's/\x1b\[[0-9;]*m//g'
    ENDSSH

(or tar this file to the server first, then `docker exec ... python3 scripts/diagnose_economics.py`).
"""
import asyncio
import zoneinfo
from datetime import datetime


async def main() -> None:
    from utils.async_influxdb_client import AsyncInfluxDBClient
    from config.settings import Settings
    from modules.growatt.decision_engine import GrowattDecisionEngine, PriceThresholds

    settings = Settings()           # reads .env / env exactly like the controller
    cfg = settings.growatt          # GrowattConfig (distribution tariff, etc.)
    bucket = "solar"                # CLAUDE.md: inverter telemetry bucket
    client = AsyncInfluxDBClient(settings)

    tz = zoneinfo.ZoneInfo("Europe/Prague")
    now = datetime.now(tz)
    today = now.date()
    print(f"\n=== Economics diagnostic {today} {now:%H:%M} (Europe/Prague) ===\n")

    # 1) Raw cumulative counters (latest value today)
    q_raw = f'''
from(bucket: "{bucket}")
  |> range(start: -18h)
  |> filter(fn: (r) => r._measurement == "solar" and
      (r._field == "EnergyToUserToday" or r._field == "EnergyToGridToday"))
  |> last()
'''
    raw = await client.query(q_raw)
    counters = {}
    for table in raw:
        for rec in table.records:
            counters[rec.get_field()] = rec.get_value()
    print("Raw meter counters (cumulative today):")
    for k, v in sorted(counters.items()):
        print(f"  {k}: {v:.3f} kWh")

    # 2) Per-block deltas (what the integration sees), with drop accounting
    q_blocks = f'''
from(bucket: "{bucket}")
  |> range(start: -18h)
  |> filter(fn: (r) => r._measurement == "solar" and
      (r._field == "EnergyToUserToday" or r._field == "EnergyToGridToday"))
  |> aggregateWindow(every: 15m, fn: last, createEmpty: false)
'''
    blocks = await client.query(q_blocks)
    series = {"EnergyToUserToday": [], "EnergyToGridToday": []}
    for table in blocks:
        for rec in table.records:
            t = rec.get_time()
            if getattr(t, "tzinfo", None) is not None:
                t = t.astimezone(tz)
            f, v = rec.get_field(), rec.get_value()
            if v is not None and f in series:
                series[f].append((t, float(v)))
    for f, pts in series.items():
        pts.sort()
        total, neg = 0.0, 0
        for i in range(1, len(pts)):
            de = pts[i][1] - pts[i - 1][1]
            if pts[i][0].date() != today:
                continue
            if de > 0:
                total += de
            else:
                neg += 1
        counter = pts[-1][1] if pts else 0.0
        print(f"\n{f}: counter={counter:.3f} kWh, sum-of-deltas={total:.3f} kWh, "
              f"non-positive-deltas-skipped={neg}")
        if abs(counter - total) > 0.2:
            print("  *** MISMATCH counter vs deltas — investigate ***")

    # 3) Distribution tariff VT/NT split from live config
    th = PriceThresholds(
        charge_price_max=getattr(cfg, "charge_price_max", 1.5),
        export_price_min=getattr(cfg, "export_price_min", 1.0),
        discharge_price_min=getattr(cfg, "discharge_price_min", 5.0),
        discharge_profit_margin=getattr(cfg, "discharge_profit_margin", 4.0),
        battery_efficiency=getattr(cfg, "battery_efficiency", 0.85),
        distribution_tariff_high=cfg.distribution_tariff_high,
        distribution_tariff_low=cfg.distribution_tariff_low,
        low_tariff_hours=cfg.low_tariff_hours,
    )
    vt = [h for h in range(24)
          if GrowattDecisionEngine._get_distribution_tariff(h, th)
          == cfg.distribution_tariff_high]
    nt = [h for h in range(24) if h not in vt]
    print(f"\nDistribution tariff from config (low_tariff_hours='{cfg.low_tariff_hours}'):")
    print(f"  VT (high {cfg.distribution_tariff_high}) hours: {vt}")
    print(f"  NT (low  {cfg.distribution_tariff_low}) hours: {nt}")


if __name__ == "__main__":
    asyncio.run(main())

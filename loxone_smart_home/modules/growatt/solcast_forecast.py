"""Solcast PV production forecast client.

Free-tier Solcast provides 10 API requests per day per rooftop site with
30-min granularity. We fetch once per refresh (the same cadence the
existing forecast.solar source uses) and convert their response into our
DailyForecast shape so the consumption code is source-agnostic.

Disabled (returns nothing, no exception raised) when no API key is
configured. Falls through to the existing forecast.solar + ML model
consensus when Solcast is unavailable or rate-limited.

API docs:
- https://docs.solcast.com.au/#0a6da91d-3a07-4f5f-849f-9aa19c8d2614
"""

import logging
from datetime import datetime, date, timedelta, timezone
from typing import Any, Dict, List, Optional

import aiohttp


class SolcastForecast:
    """Thin async client for Solcast's rooftop-PV forecast endpoint."""

    BASE_URL = "https://api.solcast.com.au/rooftop_sites"

    def __init__(
        self,
        api_key: Optional[str],
        rooftop_id: Optional[str],
        logger: Optional[logging.Logger] = None,
    ):
        self.api_key = (api_key or "").strip() or None
        self.rooftop_id = (rooftop_id or "").strip() or None
        self.logger = logger or logging.getLogger(__name__)
        # Track last ATTEMPT (success or failure) so we throttle every call,
        # not just successful ones — the free tier hard cap is 10 req/day and
        # a failing endpoint must not be hammered. UTC-aware so the throttle
        # stays monotonic across DST transitions.
        self._last_attempt: Optional[datetime] = None
        self._cached: Dict[str, Dict[int, float]] = {}
        # Absolute per-UTC-day request counter as a hard backstop under the
        # 10/day cap, independent of the interval throttle.
        self._req_day: Optional[date] = None
        self._req_count: int = 0
        self._max_requests_per_day: int = 9

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.rooftop_id)

    async def fetch_hourly_today_tomorrow(
        self,
        force: bool = False,
        min_refresh_interval_hours: int = 3,
        local_tz: Any = None,
    ) -> Dict[str, Dict[int, float]]:
        """Return {date_str: {hour: kWh}} for today + tomorrow.

        Throttled to stay under the Solcast free-tier hard cap of 10 req/day
        per rooftop via two guards that BOTH count every attempt (success or
        failure, so a failing endpoint can't be hammered):
        - a per-call interval (default 3h ⇒ ≤8/day), and
        - an absolute per-UTC-day counter (≤9/day).
        On any error returns the last good cache if available, else {} (the
        controller then falls back to its model+API consensus).

        `local_tz`: timezone to bucket the forecast into. Solcast returns
        UTC `period_end` timestamps; the rest of the controller works in
        local wall-clock hours, so we must convert before bucketing or the
        forecast lands 1-2 hours off.
        """
        if not self.enabled:
            return {}

        now_utc = datetime.now(timezone.utc)

        # Interval throttle (monotonic UTC) — gates EVERY attempt, not just
        # successes, so a failing endpoint can't be hammered.
        if (
            not force
            and self._last_attempt is not None
            and (now_utc - self._last_attempt).total_seconds()
                < min_refresh_interval_hours * 3600
        ):
            return dict(self._cached)

        # Absolute per-UTC-day backstop under the 10/day cap.
        today_utc = now_utc.date()
        if self._req_day != today_utc:
            self._req_day = today_utc
            self._req_count = 0
        if not force and self._req_count >= self._max_requests_per_day:
            self.logger.debug(
                "Solcast daily request budget exhausted — using cache"
            )
            return dict(self._cached)

        # Count this attempt against both throttles up front.
        self._last_attempt = now_utc
        self._req_count += 1

        url = f"{self.BASE_URL}/{self.rooftop_id}/forecasts"
        params = {"format": "json", "hours": 48}
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params, headers=headers) as resp:
                    if resp.status != 200:
                        # 401/403 = misconfig; 429 = rate limited; others
                        # transient. Visible at WARNING; return stale cache.
                        self.logger.warning(
                            f"Solcast returned {resp.status}: {await resp.text()}"
                        )
                        return dict(self._cached)
                    payload = await resp.json()
        except Exception as e:
            # Network/JSON errors: warn (rate-limited by the interval throttle)
            # so a broken paid feature is visible rather than silently disabled.
            self.logger.warning(f"Solcast fetch failed: {e}")
            return dict(self._cached)

        result = self._parse_forecasts(payload, local_tz)

        if result:
            self._cached = result
            self.logger.info(
                f"☀️  Solcast forecast: {sum(sum(h.values()) for h in result.values()):.1f} "
                f"kWh across {len(result)} day(s)"
            )
            return dict(result)

        # Parsed empty (unexpected payload shape) — keep prior cache.
        return dict(self._cached)

    @staticmethod
    def _parse_forecasts(
        payload: Any, local_tz: Any = None
    ) -> Dict[str, Dict[int, float]]:
        """Convert a Solcast forecasts payload into {date_str: {hour: kWh}}.

        Response shape: {"forecasts": [{"period_end": iso8601,
        "pv_estimate": kW, ...}, ...]}. `period_end` marks the END of each
        30-min interval (UTC); `pv_estimate` is the average kW over that
        interval, so kWh = kW * 0.5. Times are converted to `local_tz`
        before bucketing so hours align with the optimizer's local grid.
        Malformed entries are skipped individually.
        """
        forecasts = payload.get("forecasts", []) if isinstance(payload, dict) else []
        result: Dict[str, Dict[int, float]] = {}
        for entry in forecasts:
            try:
                end_iso = entry.get("period_end")
                kw = float(entry.get("pv_estimate", 0.0))
                if not end_iso:
                    continue
                end_dt = SolcastForecast._parse_iso8601_utc(end_iso)
                # Convert to local wall-clock so hour bucketing aligns with
                # the optimizer (which uses local hours).
                if local_tz is not None and end_dt.tzinfo is not None:
                    end_dt = end_dt.astimezone(local_tz)
                # Bucket by the interval START's hour (period_end is exclusive).
                start_dt = end_dt - timedelta(minutes=30)
                key = start_dt.date().strftime("%Y-%m-%d")
                hour = start_dt.hour
                result.setdefault(key, {})
                result[key][hour] = result[key].get(hour, 0.0) + kw * 0.5
            except Exception:
                continue
        return result

    @staticmethod
    def _parse_iso8601_utc(value: str) -> datetime:
        """Parse Solcast's ISO-8601 timestamps robustly.

        Solcast emits a trailing ``Z`` and 7-digit fractional seconds
        (e.g. ``2026-05-29T10:00:00.0000000Z``). Python <3.11's
        ``datetime.fromisoformat`` rejects both the ``Z`` and any fractional
        precision other than 3/6 digits, so we normalise first: drop the
        fractional part entirely (we only bucket to the hour) and convert a
        trailing ``Z`` to ``+00:00``.
        """
        v = value.strip()
        if v.endswith("Z"):
            v = v[:-1] + "+00:00"
        # Strip fractional seconds: "...:00.0000000+00:00" -> "...:00+00:00".
        if "." in v:
            head, _, tail = v.partition(".")
            # tail = fractional digits + optional offset; keep the offset.
            offset = ""
            for marker in ("+", "-"):
                idx = tail.find(marker)
                if idx != -1:
                    offset = tail[idx:]
                    break
            v = head + offset
        return datetime.fromisoformat(v)

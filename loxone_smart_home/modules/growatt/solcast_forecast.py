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

    # Solcast returns three PV estimates per interval; map our config token to
    # the response field. p10 is the conservative (cloudy-biased) estimate,
    # useful for risk-aware reserve sizing; p90 the optimistic one.
    _QUANTILE_FIELD = {
        "p10": "pv_estimate10",
        "p50": "pv_estimate",
        "p90": "pv_estimate90",
    }

    def __init__(
        self,
        api_key: Optional[str],
        rooftop_id: Optional[str],
        logger: Optional[logging.Logger] = None,
        quantile: str = "p50",
    ):
        self.api_key = (api_key or "").strip() or None
        # One or more rooftop sites (comma/semicolon/space separated). A home
        # with multiple roof orientations (e.g. SW + SE arrays) registers one
        # Solcast site per orientation; we query each and SUM them into total
        # production. `rooftop_id` keeps the first id for back-compat/logging.
        self.rooftop_ids: List[str] = [
            s.strip()
            for s in (rooftop_id or "").replace(";", ",").replace(" ", ",").split(",")
            if s.strip()
        ]
        self.rooftop_id = self.rooftop_ids[0] if self.rooftop_ids else None
        self.logger = logger or logging.getLogger(__name__)
        # Which PV-estimate quantile to read (falls back to p50/pv_estimate for
        # any unrecognised value).
        self.quantile = quantile if quantile in self._QUANTILE_FIELD else "p50"
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
        # Set when a permanent auth error (401/403) is seen, so we stop
        # hammering the API and burning the daily budget on a bad key.
        # (No lock is needed for the throttle: the check-and-increment below
        # has no await between gate and counter bump, so asyncio's cooperative
        # scheduling makes it atomic across concurrent callers.)
        self._auth_failed: bool = False

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.rooftop_ids)

    async def fetch_hourly_today_tomorrow(
        self,
        force: bool = False,
        min_refresh_interval_hours: int = 3,
        local_tz: Any = None,
    ) -> Dict[str, Dict[int, float]]:
        """Return {date_str: {hour: kWh}} for today + tomorrow, summed across
        ALL configured rooftop sites (e.g. SW + SE arrays → total home output).

        Throttled to stay under the Solcast free-tier hard cap of 10 req/day
        (shared across the account, so every site call counts) via two guards:
        - a per-batch interval (default 3h), and
        - an absolute per-UTC-day counter (≤9 total calls). A refresh is only
          started if there's budget for ALL sites, so we never spend the budget
          on a half-updated (one-array) forecast.
        On error returns the last good cache if available, else {} (the
        controller then falls back to its model+API consensus).

        `local_tz`: timezone to bucket the forecast into. Solcast returns
        UTC `period_end` timestamps; the rest of the controller works in
        local wall-clock hours, so we must convert before bucketing or the
        forecast lands 1-2 hours off.
        """
        if not self.enabled:
            return {}

        # A permanent auth failure won't fix itself — stop attempting so we
        # don't drain the daily budget on a bad key/rooftop id.
        if self._auth_failed:
            return dict(self._cached)

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

        # Absolute per-UTC-day backstop under the 10/day cap. With N sites a
        # refresh costs N calls, so only start one if the whole batch fits.
        today_utc = now_utc.date()
        if self._req_day != today_utc:
            self._req_day = today_utc
            self._req_count = 0
        n_sites = len(self.rooftop_ids)
        # This day cap is absolute and is NOT bypassed by `force` (which only
        # overrides the interval throttle). The free-tier 10/day quota is a hard
        # account limit — letting a forced refresh blow past it would burn the
        # budget and get later calls rejected by the API.
        if self._req_count + n_sites > self._max_requests_per_day:
            self.logger.debug(
                "Solcast daily request budget would be exceeded by a "
                f"{n_sites}-site refresh — using cache"
            )
            return dict(self._cached)

        # Reserve the interval slot AND the whole batch's budget up front, before
        # any await, so two concurrent refreshes can't both pass the gate and
        # then over-spend the daily quota. Every site below counts (success or
        # not), so no refund is needed.
        self._last_attempt = now_utc
        self._req_count += n_sites

        params = {"format": "json", "hours": 48}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        combined: Dict[str, Dict[int, float]] = {}
        per_site_totals = []
        any_success = False

        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for rid in self.rooftop_ids:
                # Budget already reserved atomically above (self._req_count +=
                # n_sites), so do NOT increment per-site here (double-count).
                url = f"{self.BASE_URL}/{rid}/forecasts"
                try:
                    async with session.get(url, params=params, headers=headers) as resp:
                        if resp.status != 200:
                            body = await resp.text()
                            if resp.status in (401, 403):
                                # Bad key — permanent for the whole account.
                                self._auth_failed = True
                                self.logger.error(
                                    f"Solcast auth failed ({resp.status}) for "
                                    f"site {rid} — disabling until restart. "
                                    f"Check solcast_api_key. Body: {body}"
                                )
                                return dict(self._cached)
                            # 429 / transient: skip this site, keep others.
                            self.logger.warning(
                                f"Solcast site {rid} returned {resp.status}: {body}"
                            )
                            continue
                        payload = await resp.json()
                except Exception as e:
                    self.logger.warning(f"Solcast fetch failed for site {rid}: {e}")
                    continue

                parsed = self._parse_forecasts(payload, local_tz, self.quantile)
                if not parsed:
                    continue
                any_success = True
                site_total = sum(sum(h.values()) for h in parsed.values())
                per_site_totals.append(f"{rid[:4]}…={site_total:.1f}")
                for date_str, hours in parsed.items():
                    day = combined.setdefault(date_str, {})
                    for h, kwh in hours.items():
                        day[h] = day.get(h, 0.0) + kwh

        if any_success and combined:
            self._cached = combined
            total = sum(sum(h.values()) for h in combined.values())
            self.logger.info(
                f"☀️  Solcast forecast: {total:.1f} kWh across "
                f"{len(combined)} day(s), {n_sites} site(s) "
                f"[{', '.join(per_site_totals)}]"
            )
            return dict(combined)

        # Everything failed / parsed empty — keep prior cache.
        return dict(self._cached)

    @staticmethod
    def _parse_forecasts(
        payload: Any, local_tz: Any = None, quantile: str = "p50"
    ) -> Dict[str, Dict[int, float]]:
        """Convert a Solcast forecasts payload into {date_str: {hour: kWh}}.

        Response shape: {"forecasts": [{"period_end": iso8601,
        "pv_estimate": kW, "pv_estimate10": kW, "pv_estimate90": kW, ...}, ...]}.
        `period_end` marks the END of each 30-min interval (UTC); the chosen
        estimate is the average kW over that interval, so kWh = kW * 0.5. Times
        are converted to `local_tz` before bucketing so hours align with the
        optimizer's local grid. `quantile` selects p10/p50/p90; it falls back
        to pv_estimate (p50) when the requested field is absent. Malformed
        entries are skipped individually.
        """
        field = SolcastForecast._QUANTILE_FIELD.get(quantile, "pv_estimate")
        forecasts = payload.get("forecasts", []) if isinstance(payload, dict) else []
        result: Dict[str, Dict[int, float]] = {}
        for entry in forecasts:
            try:
                end_iso = entry.get("period_end")
                # Fall back to the median estimate if the quantile field is
                # missing (older payloads / partial responses).
                raw = entry.get(field)
                if raw is None:
                    raw = entry.get("pv_estimate", 0.0)
                kw = float(raw)
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

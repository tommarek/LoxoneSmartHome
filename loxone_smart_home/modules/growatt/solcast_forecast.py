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

import asyncio
import hashlib
import json
import logging
import os
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp


def _default_quota_path() -> Path:
    """Sibling of the persisted config overrides (same docker volume).

    Mirrors ``config.settings_overrides.DEFAULT_OVERRIDES_PATH``: the quota
    file must survive container restarts (the UI restart button, deploys)
    or every restart resets the ≤9/day Solcast budget and re-spends it.
    Overridable via the CONFIG_OVERRIDES_PATH env var for local runs/tests.
    """
    overrides = Path(
        os.environ.get(
            "CONFIG_OVERRIDES_PATH", "/app/config_state/config_overrides.json"
        )
    )
    return overrides.parent / "solcast_quota.json"


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
        quota_path: Optional[Path] = None,
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
        # Persist the throttle state across container restarts (UI restart
        # button, deploys) — otherwise every restart resets the ≤9/day budget
        # and re-spends requests / re-hammers a bad key. Path is injectable
        # for tests; the default lives next to config_overrides.json on the
        # loxone_config_state volume.
        self._quota_path: Path = Path(quota_path) if quota_path else _default_quota_path()
        self._load_quota_state()

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.rooftop_ids)

    def _api_key_fingerprint(self) -> Optional[str]:
        """Short non-reversible fingerprint of the API key.

        Stored alongside the persisted ``auth_failed`` latch so the latch is
        only honoured while the SAME key is configured — fixing the key and
        restarting must re-enable Solcast, not stay latched forever.
        """
        if not self.api_key:
            return None
        return hashlib.sha256(self.api_key.encode("utf-8")).hexdigest()[:16]

    def _load_quota_state(self) -> None:
        """Restore the persisted throttle state, tolerating a missing or
        corrupt file (fresh defaults already set by __init__)."""
        try:
            raw = json.loads(self._quota_path.read_text())
        except FileNotFoundError:
            return  # first run — nothing persisted yet
        except (OSError, ValueError) as e:
            self.logger.warning(
                f"Could not read Solcast quota state at {self._quota_path} "
                f"({e}) — starting with a fresh throttle state"
            )
            return
        if not isinstance(raw, dict):
            return
        try:
            req_day = (
                date.fromisoformat(raw["utc_day"]) if raw.get("utc_day") else None
            )
            req_count = int(raw.get("req_count", 0))
            last_attempt = (
                datetime.fromisoformat(raw["last_attempt_iso"])
                if raw.get("last_attempt_iso")
                else None
            )
            auth_failed = bool(raw.get("auth_failed", False))
            key_fp = raw.get("api_key_fingerprint")
        except (KeyError, TypeError, ValueError):
            self.logger.warning(
                f"Solcast quota state at {self._quota_path} is corrupt — "
                f"starting with a fresh throttle state"
            )
            return
        # All-or-nothing: only apply a fully-parsed state.
        self._req_day = req_day
        self._req_count = max(0, req_count)
        if last_attempt is not None and last_attempt.tzinfo is not None:
            self._last_attempt = last_attempt
        # The auth latch only applies while the key it was recorded against
        # is still in use; a changed key gets a fresh chance.
        if auth_failed and key_fp == self._api_key_fingerprint():
            self._auth_failed = True
            self.logger.warning(
                "Solcast auth-failure latch restored from persisted state — "
                "the configured API key was previously rejected (401/403); "
                "change solcast_api_key to re-enable"
            )

    def _save_quota_state(self) -> None:
        """Persist the throttle state atomically (write temp + replace).

        Best-effort: persistence failing (read-only FS in local dev/tests)
        must never break forecasting — the in-memory state stays authoritative
        for this process lifetime.
        """
        payload = {
            "utc_day": self._req_day.isoformat() if self._req_day else None,
            "req_count": self._req_count,
            "last_attempt_iso": (
                self._last_attempt.isoformat() if self._last_attempt else None
            ),
            "auth_failed": self._auth_failed,
            "api_key_fingerprint": (
                self._api_key_fingerprint() if self._auth_failed else None
            ),
        }
        try:
            self._quota_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._quota_path.with_suffix(self._quota_path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
            tmp.replace(self._quota_path)
        except OSError as e:
            self.logger.warning(
                f"Could not persist Solcast quota state to {self._quota_path}: {e}"
            )

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
            self.logger.warning(
                "Solcast daily request budget would be exceeded by a "
                f"{n_sites}-site refresh ({self._req_count}/"
                f"{self._max_requests_per_day} used) — using cache"
            )
            return dict(self._cached)

        # Reserve the interval slot AND the whole batch's budget up front, before
        # any await, so two concurrent refreshes can't both pass the gate and
        # then over-spend the daily quota. Every site that gets an HTTP response
        # counts (success or error status); only connect-level failures where
        # the request provably never went out (ClientConnectorError) are
        # refunded below — they can't have touched Solcast's real quota.
        self._last_attempt = now_utc
        self._req_count += n_sites
        self._save_quota_state()

        params = {"format": "json", "hours": 48}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        combined: Dict[str, Dict[int, float]] = {}
        per_site_totals = []
        any_success = False

        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for idx, rid in enumerate(self.rooftop_ids):
                # Budget already reserved atomically above (self._req_count +=
                # n_sites), so do NOT increment per-site here (double-count).
                url = f"{self.BASE_URL}/{rid}/forecasts"
                got_response = False
                try:
                    async with session.get(url, params=params, headers=headers) as resp:
                        got_response = True
                        if resp.status != 200:
                            body = await resp.text()
                            if resp.status in (401, 403):
                                # Bad key — permanent for the whole account.
                                # Refund the sites AFTER this one: the batch
                                # reserved n_sites up front, but we bail here
                                # without ever sending their requests, so they
                                # never touched Solcast's ledger. (This site
                                # DID get a response, so it stays counted.)
                                self._auth_failed = True
                                self._req_count = max(
                                    0, self._req_count - (n_sites - (idx + 1))
                                )
                                self._save_quota_state()
                                self.logger.error(
                                    f"Solcast auth failed ({resp.status}) for "
                                    f"site {rid} — disabling until the API key "
                                    f"changes. Check solcast_api_key. Body: {body}"
                                )
                                return dict(self._cached)
                            # 429 / transient: skip this site, keep others.
                            self.logger.warning(
                                f"Solcast site {rid} returned {resp.status}: {body}"
                            )
                            continue
                        payload = await resp.json()
                except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
                    # Refund ONLY when the request provably never went out
                    # (DNS failure / connection refused = ClientConnectorError,
                    # raised before anything was sent) — a LAN/WAN outage must
                    # not zero the day's budget. A TIMEOUT is NOT refunded: the
                    # request may have reached Solcast and been counted by its
                    # real ledger, and refunding locally could let us exceed
                    # the actual quota. Anything after a received response
                    # (body read failure) always counts.
                    refund = not got_response and isinstance(
                        e, aiohttp.ClientConnectorError
                    )
                    if refund:
                        self._req_count = max(0, self._req_count - 1)
                        self._save_quota_state()
                    self.logger.warning(
                        f"Solcast connection failed for site {rid}: "
                        f"{type(e).__name__}: {e}"
                        + (" (request refunded)" if refund else "")
                    )
                    continue
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

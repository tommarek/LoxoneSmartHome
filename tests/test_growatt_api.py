"""Unit tests for Growatt API endpoints."""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "loxone_smart_home"))

import pytest  # noqa: E402
from aiohttp import web  # noqa: E402
from aiohttp.test_utils import AioHTTPTestCase  # noqa: E402

from modules.growatt.api import GROWATT_CONTROLLER_KEY, create_growatt_api  # noqa: E402


class TestGrowattAPI(AioHTTPTestCase):
    """Test Growatt API endpoints."""

    async def get_application(self) -> web.Application:
        """Create test application."""
        app = web.Application()

        # Create mock controller
        self.mock_controller = MagicMock()
        self.mock_controller._running = True
        self.mock_controller._season_mode = "winter"
        self.mock_controller._season_mode_updated = datetime.now()
        self.mock_controller._ac_enabled = True
        self.mock_controller._export_enabled = False
        self.mock_controller._scheduled_periods = []
        self.mock_controller._optional_config = {
            "simulation_mode": False,
            "summer_temp_threshold": 15.0,
            "temperature_avg_days": 3,
            "eur_czk_rate": 25.0
        }
        self.mock_controller._eur_czk_rate = 25.0
        self.mock_controller._eur_czk_rate_updated = datetime.now()
        
        # Mock price data
        self.mock_controller._current_prices = {
            ("00:00", "01:00"): 60.0,
            ("01:00", "02:00"): 55.0,
            ("02:00", "03:00"): 50.0,
            ("03:00", "04:00"): 48.0,
            ("04:00", "05:00"): 52.0,
            ("05:00", "06:00"): 58.0,
            ("06:00", "07:00"): 65.0,
            ("07:00", "08:00"): 75.0,
            ("08:00", "09:00"): 85.0,
            ("09:00", "10:00"): 90.0,
            ("10:00", "11:00"): 88.0,
            ("11:00", "12:00"): 86.0,
            ("12:00", "13:00"): 84.0,
            ("13:00", "14:00"): 82.0,
            ("14:00", "15:00"): 80.0,
            ("15:00", "16:00"): 78.0,
            ("16:00", "17:00"): 82.0,
            ("17:00", "18:00"): 95.0,
            ("18:00", "19:00"): 100.0,
            ("19:00", "20:00"): 98.0,
            ("20:00", "21:00"): 90.0,
            ("21:00", "22:00"): 75.0,
            ("22:00", "23:00"): 65.0,
            ("23:00", "24:00"): 60.0,
        }
        self.mock_controller._prices_date = "2024-12-17"
        self.mock_controller._prices_updated = datetime.now()
        
        self.mock_controller._get_local_now = MagicMock(
            return_value=datetime.now()
        )
        self.mock_controller._select_primary_mode = MagicMock(
            return_value="load_first"
        )
        self.mock_controller._get_inverter_time = AsyncMock(
            return_value=datetime.now()
        )
        self.mock_controller._sync_inverter_time = AsyncMock(
            return_value=True
        )
        self.mock_controller._set_battery_first = AsyncMock()
        self.mock_controller._set_grid_first = AsyncMock()
        self.mock_controller._set_load_first = AsyncMock()

        # Mock config
        self.mock_controller.config = MagicMock()
        self.mock_controller.config.battery_capacity = 10.0
        self.mock_controller.config.max_charge_power = 3.0
        self.mock_controller.config.min_soc = 20
        self.mock_controller.config.max_soc = 90
        self.mock_controller.config.export_price_threshold = 1.0
        self.mock_controller.config.battery_charge_hours = 2
        self.mock_controller.config.individual_cheapest_hours = 6
        self.mock_controller.config.summer_price_threshold = 1.0
        self.mock_controller.config.device_serial = "ABC123"
        self.mock_controller.config.schedule_hour = 23
        self.mock_controller.config.schedule_minute = 59

        # Mock MQTT client
        self.mock_controller.mqtt_client = AsyncMock()

        # Register API with mock controller
        create_growatt_api(app, self.mock_controller)

        return app

    async def test_get_status(self) -> None:
        """Test GET /api/growatt/status endpoint."""
        resp = await self.client.request("GET", "/api/growatt/status")
        self.assertEqual(resp.status, 200)

        data = await resp.json()
        self.assertIn("running", data)
        self.assertIn("current_mode", data)
        self.assertIn("season_mode", data)
        self.assertTrue(data["running"])
        self.assertEqual(data["season_mode"], "winter")
        self.assertTrue(data["ac_enabled"])
        self.assertFalse(data["export_enabled"])

    async def test_get_status_no_controller(self) -> None:
        """Test GET /api/growatt/status without controller."""
        # Remove controller
        self.app[GROWATT_CONTROLLER_KEY] = None

        resp = await self.client.request("GET", "/api/growatt/status")
        self.assertEqual(resp.status, 503)

        data = await resp.json()
        self.assertIn("error", data)
        self.assertEqual(data["error"], "Controller not initialized")

    async def test_get_schedule(self) -> None:
        """Test GET /api/growatt/schedule endpoint."""
        resp = await self.client.request("GET", "/api/growatt/schedule")
        self.assertEqual(resp.status, 200)

        data = await resp.json()
        self.assertIn("schedule", data)
        self.assertIn("active_now", data)
        self.assertIn("total_periods", data)
        self.assertEqual(data["total_periods"], 0)

    async def test_get_prices(self) -> None:
        """Test GET /api/growatt/prices endpoint."""
        resp = await self.client.request("GET", "/api/growatt/prices")
        self.assertEqual(resp.status, 200)

        data = await resp.json()
        self.assertIn("has_data", data)
        self.assertTrue(data["has_data"])
        self.assertIn("eur_czk_rate", data)
        self.assertEqual(data["eur_czk_rate"], 25.0)
        self.assertIn("hourly_prices", data)
        self.assertEqual(len(data["hourly_prices"]), 24)
        self.assertIn("average_today_czk_kwh", data)
        self.assertIn("min_price_czk_kwh", data)
        self.assertIn("max_price_czk_kwh", data)

    async def test_get_config(self) -> None:
        """Test GET /api/growatt/config endpoint."""
        resp = await self.client.request("GET", "/api/growatt/config")
        self.assertEqual(resp.status, 200)

        data = await resp.json()
        self.assertIn("battery_capacity", data)
        self.assertIn("max_charge_power", data)
        self.assertIn("min_soc", data)
        self.assertIn("max_soc", data)
        self.assertEqual(data["battery_capacity"], 10.0)
        self.assertEqual(data["max_charge_power"], 3.0)
        self.assertEqual(data["min_soc"], 20)
        self.assertEqual(data["max_soc"], 90)

    async def test_set_mode_battery_first(self) -> None:
        """Test POST /api/growatt/mode for battery_first."""
        payload = {
            "mode": "battery_first",
            "start": "08:00",
            "stop": "10:00",
            "params": {
                "stop_soc": 85,
                "power_rate": 95
            }
        }

        resp = await self.client.request(
            "POST",
            "/api/growatt/mode",
            json=payload
        )
        self.assertEqual(resp.status, 200)

        data = await resp.json()
        self.assertIn("success", data)
        self.assertIn("mode", data)
        self.assertTrue(data["success"])
        self.assertEqual(data["mode"], "battery_first")

        # Verify controller method was called
        self.mock_controller._set_battery_first.assert_called_once_with(
            "08:00", "10:00",
            stop_soc=85,
            power_rate=95
        )

    async def test_set_mode_grid_first(self) -> None:
        """Test POST /api/growatt/mode for grid_first."""
        payload = {
            "mode": "grid_first",
            "start": "17:00",
            "stop": "20:00",
            "params": {
                "stop_soc": 25,
                "power_rate": 15
            }
        }

        resp = await self.client.request(
            "POST",
            "/api/growatt/mode",
            json=payload
        )
        self.assertEqual(resp.status, 200)

        data = await resp.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["mode"], "grid_first")

        # Verify controller method was called
        self.mock_controller._set_grid_first.assert_called_once_with(
            "17:00", "20:00",
            stop_soc=25,
            power_rate=15
        )

    async def test_set_mode_load_first(self) -> None:
        """Test POST /api/growatt/mode for load_first."""
        payload = {"mode": "load_first"}

        resp = await self.client.request(
            "POST",
            "/api/growatt/mode",
            json=payload
        )
        self.assertEqual(resp.status, 200)

        data = await resp.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["mode"], "load_first")

        # Verify controller method was called
        self.mock_controller._set_load_first.assert_called_once()

    async def test_set_mode_invalid(self) -> None:
        """Test POST /api/growatt/mode with invalid mode."""
        payload = {"mode": "invalid_mode"}

        resp = await self.client.request(
            "POST",
            "/api/growatt/mode",
            json=payload
        )
        self.assertEqual(resp.status, 400)

        data = await resp.json()
        self.assertIn("error", data)
        self.assertEqual(data["error"], "Invalid mode")

    async def test_sync_time(self) -> None:
        """Test POST /api/growatt/sync-time endpoint."""
        payload = {"force": True}

        resp = await self.client.request(
            "POST",
            "/api/growatt/sync-time",
            json=payload
        )
        self.assertEqual(resp.status, 200)

        data = await resp.json()
        self.assertIn("success", data)
        self.assertIn("inverter_time", data)
        self.assertIn("server_time", data)
        self.assertIn("drift_seconds", data)
        self.assertIn("synced", data)
        self.assertTrue(data["synced"])

        # Verify controller methods were called
        self.mock_controller._get_inverter_time.assert_called()
        self.mock_controller._sync_inverter_time.assert_called_once()

    async def test_sync_time_no_force(self) -> None:
        """Test POST /api/growatt/sync-time without force."""
        # No payload means force=False
        resp = await self.client.request(
            "POST",
            "/api/growatt/sync-time",
            data=""
        )
        self.assertEqual(resp.status, 200)

        data = await resp.json()
        self.assertIn("success", data)
        # With small drift and no force, sync should not happen
        self.assertFalse(data["synced"])

        # Verify controller methods were called
        self.mock_controller._get_inverter_time.assert_called()
        # Sync should not be called with small drift
        self.mock_controller._sync_inverter_time.assert_not_called()

    async def test_sync_time_no_inverter_time(self) -> None:
        """Test POST /api/growatt/sync-time when inverter time unavailable."""
        self.mock_controller._get_inverter_time = AsyncMock(
            return_value=None
        )

        payload = {"force": True}
        resp = await self.client.request(
            "POST",
            "/api/growatt/sync-time",
            json=payload
        )
        self.assertEqual(resp.status, 500)

        data = await resp.json()
        self.assertIn("success", data)
        self.assertIn("message", data)
        self.assertFalse(data["success"])
        self.assertEqual(data["message"], "Could not read inverter time")

    @patch('modules.growatt.api._telemetry_cache')
    async def test_get_status_with_inverter_data(
        self,
        mock_cache: MagicMock
    ) -> None:
        """Test GET /api/growatt/status with cached telemetry data."""
        # Mock telemetry cache with data
        mock_cache.__bool__.return_value = True
        mock_cache.get.return_value = 50  # ActivePowerRate

        resp = await self.client.request("GET", "/api/growatt/status")
        self.assertEqual(resp.status, 200)

        data = await resp.json()
        self.assertIn("inverter_data", data)
        inverter = data["inverter_data"]
        # Check we have active_power_rate from telemetry cache
        self.assertIn("active_power_rate", inverter)
        self.assertEqual(inverter["active_power_rate"], 50)
        # Check we have mode status inferred from periods
        self.assertIn("battery_first_status", inverter)
        self.assertIn("grid_first_status", inverter)
        # Since no periods are active, both should be disabled
        self.assertFalse(inverter["battery_first_status"]["enabled"])
        self.assertFalse(inverter["grid_first_status"]["enabled"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Test Growatt logging functionality."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from config.settings import GrowattConfig, Settings
from modules.growatt.types import GrowattLogLevel
from modules.growatt_controller import GrowattController


class TestGrowattLogLevel:
    """Test GrowattLogLevel enum functionality."""

    def test_log_level_comparison(self):
        """Test log level comparison operators."""
        # Test greater than or equal
        assert GrowattLogLevel.DEBUG >= GrowattLogLevel.SUMMARY
        assert GrowattLogLevel.VERBOSE >= GrowattLogLevel.DETAIL
        assert GrowattLogLevel.DETAIL >= GrowattLogLevel.DETAIL
        assert GrowattLogLevel.DETAIL >= GrowattLogLevel.SUMMARY

        # Test greater than
        assert GrowattLogLevel.DEBUG > GrowattLogLevel.VERBOSE
        assert GrowattLogLevel.VERBOSE > GrowattLogLevel.DETAIL
        assert GrowattLogLevel.DETAIL > GrowattLogLevel.SUMMARY
        assert not (GrowattLogLevel.SUMMARY > GrowattLogLevel.VERBOSE)
        assert not (GrowattLogLevel.DETAIL > GrowattLogLevel.DETAIL)

    def test_log_level_ordering(self):
        """Test that log levels are properly ordered."""
        levels = [
            GrowattLogLevel.SUMMARY,
            GrowattLogLevel.DETAIL,
            GrowattLogLevel.VERBOSE,
            GrowattLogLevel.DEBUG
        ]

        for i, level in enumerate(levels):
            # Check that each level is greater than all previous levels
            for j in range(i):
                assert level > levels[j], f"{level} should be > {levels[j]}"
            # Check that each level is not greater than itself or later levels
            for j in range(i, len(levels)):
                assert not (levels[j] < level), f"{levels[j]} should not be < {level}"


class TestGrowattControllerLogging:
    """Test Growatt controller logging features."""

    @pytest.fixture
    def controller(self):
        """Create a Growatt controller with mocked dependencies."""
        # Create mock settings with required attributes
        settings = MagicMock(spec=Settings)
        settings.growatt = GrowattConfig(
            simulation_mode=False,
            export_price_min=2.5,
            battery_charge_blocks=8,
            log_level="DETAIL"
        )
        settings.log_level = "INFO"
        settings.log_timezone = "Europe/Prague"
        settings.weather = MagicMock()
        settings.weather.latitude = 50.0
        settings.weather.longitude = 14.0

        mqtt_client = MagicMock()
        influxdb_client = MagicMock()

        controller = GrowattController(mqtt_client, influxdb_client, settings)
        controller.logger = MagicMock()

        return controller

    def test_should_log(self, controller):
        """Test _should_log method with different log levels."""
        # Set log level to DETAIL
        controller._log_level = GrowattLogLevel.DETAIL

        assert controller._should_log(GrowattLogLevel.SUMMARY)
        assert controller._should_log(GrowattLogLevel.DETAIL)
        assert not controller._should_log(GrowattLogLevel.VERBOSE)
        assert not controller._should_log(GrowattLogLevel.DEBUG)

        # Set log level to VERBOSE
        controller._log_level = GrowattLogLevel.VERBOSE

        assert controller._should_log(GrowattLogLevel.SUMMARY)
        assert controller._should_log(GrowattLogLevel.DETAIL)
        assert controller._should_log(GrowattLogLevel.VERBOSE)
        assert not controller._should_log(GrowattLogLevel.DEBUG)

    def test_format_price_summary_single_block(self, controller):
        """Test price summary formatting for a single block."""
        blocks = [
            (datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 11, 0), 100.0)
        ]
        eur_czk_rate = 25.0

        summary = controller._format_price_summary(blocks, eur_czk_rate)

        assert "10:00-11:00" in summary
        assert "1 blocks" in summary
        assert "2.50 CZK/kWh" in summary

    def test_format_price_summary_multiple_blocks(self, controller):
        """Test price summary formatting for multiple non-consecutive blocks."""
        blocks = [
            (datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 11, 0), 100.0),
            (datetime(2024, 1, 1, 14, 0), datetime(2024, 1, 1, 15, 0), 120.0),
            (datetime(2024, 1, 1, 18, 0), datetime(2024, 1, 1, 19, 0), 80.0)
        ]
        eur_czk_rate = 25.0

        summary = controller._format_price_summary(blocks, eur_czk_rate)

        assert "3 blocks" in summary
        assert "3 periods" in summary
        assert "min" in summary
        assert "max" in summary
        assert "avg" in summary

    def test_format_price_summary_consecutive_blocks(self, controller):
        """Test price summary formatting for consecutive blocks."""
        blocks = [
            (datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 11, 0), 100.0),
            (datetime(2024, 1, 1, 11, 0), datetime(2024, 1, 1, 12, 0), 100.0),
            (datetime(2024, 1, 1, 12, 0), datetime(2024, 1, 1, 13, 0), 100.0)
        ]
        eur_czk_rate = 25.0

        summary = controller._format_price_summary(blocks, eur_czk_rate)

        assert "10:00-13:00" in summary
        assert "3 blocks" in summary
        assert "2.50 CZK/kWh" in summary

    def test_format_price_summary_empty(self, controller):
        """Test price summary formatting with no blocks."""
        blocks = []
        eur_czk_rate = 25.0

        summary = controller._format_price_summary(blocks, eur_czk_rate)

        assert summary == "No blocks"

    @patch('modules.growatt_controller.datetime')
    async def test_log_periodic_summary(self, mock_datetime, controller):
        """Test periodic summary logging."""
        # Setup mock datetime
        mock_now = datetime(2024, 1, 1, 14, 30)
        mock_datetime.now.return_value = mock_now

        # Setup controller state
        controller._current_mode = "battery_first"
        controller._battery_soc = 75.5
        controller._commands_sent_count = 10
        controller._commands_skipped_count = 2
        controller._high_loads_active = False
        controller._manual_override_period = None
        controller._current_prices = {
            ("14:30", "14:45"): 120.0
        }
        controller._eur_czk_rate = 25.0

        # Enable SUMMARY level logging
        controller._log_level = GrowattLogLevel.SUMMARY

        await controller._log_periodic_summary()

        # Verify that a summary was logged
        controller.logger.info.assert_called()
        call_args = controller.logger.info.call_args[0][0]

        assert "Status:" in call_args
        assert "Mode: battery_first" in call_args
        assert "Battery: 76%" in call_args
        assert "Price: 3.00 CZK/kWh" in call_args
        assert "Commands: 10 sent, 2 skipped" in call_args

    @patch('modules.growatt_controller.datetime')
    async def test_log_periodic_summary_with_high_load(self, mock_datetime, controller):
        """Test periodic summary logging with high load active."""
        # Setup mock datetime
        mock_now = datetime(2024, 1, 1, 14, 30)
        mock_datetime.now.return_value = mock_now

        # Setup controller state with high load
        controller._current_mode = "load_first"
        controller._battery_soc = 50.0
        controller._high_loads_active = True
        controller._log_level = GrowattLogLevel.SUMMARY

        await controller._log_periodic_summary()

        # Verify high load status is included
        call_args = controller.logger.info.call_args[0][0]
        assert "⚡ HIGH LOAD ACTIVE" in call_args

    @patch('modules.growatt_controller.datetime')
    async def test_log_periodic_summary_with_manual_override(
        self, mock_datetime, controller
    ):
        """Test periodic summary logging with manual override active."""
        # Setup mock datetime
        mock_now = datetime(2024, 1, 1, 14, 30)
        mock_datetime.now.return_value = mock_now

        # Setup controller state with manual override
        controller._current_mode = "battery_first"
        controller._battery_soc = 60.0
        controller._manual_override_period = MagicMock()
        controller._manual_override_period.kind = "ac_charge"
        controller._manual_override_end_time = mock_now + timedelta(minutes=45)
        controller._log_level = GrowattLogLevel.SUMMARY

        await controller._log_periodic_summary()

        # Verify manual override status is included
        call_args = controller.logger.info.call_args[0][0]
        assert "Override: ac_charge" in call_args
        assert "45min remaining" in call_args

    def test_log_level_initialization(self):
        """Test that log level is properly initialized from config."""
        # Test with valid log level
        settings = MagicMock(spec=Settings)
        settings.growatt = GrowattConfig(
            simulation_mode=False,
            export_price_min=2.5,
            battery_charge_blocks=8,
            log_level="VERBOSE"
        )
        settings.log_level = "INFO"
        settings.log_timezone = "Europe/Prague"
        settings.weather = MagicMock()
        settings.weather.latitude = 50.0
        settings.weather.longitude = 14.0

        controller = GrowattController(MagicMock(), MagicMock(), settings)
        assert controller._log_level == GrowattLogLevel.VERBOSE

        # Test with invalid log level (should default to DETAIL)
        settings2 = MagicMock(spec=Settings)
        settings2.growatt = GrowattConfig(
            simulation_mode=False,
            export_price_min=2.5,
            battery_charge_blocks=8,
            log_level="INVALID"
        )
        settings2.log_level = "INFO"
        settings2.log_timezone = "Europe/Prague"
        settings2.weather = MagicMock()
        settings2.weather.latitude = 50.0
        settings2.weather.longitude = 14.0

        controller = GrowattController(MagicMock(), MagicMock(), settings2)
        assert controller._log_level == GrowattLogLevel.DETAIL

    def test_log_compact_schedule(self, controller):
        """Test compact schedule logging."""
        charging_today = [
            (datetime(2024, 1, 1, 2, 0), datetime(2024, 1, 1, 3, 0), 50.0),
            (datetime(2024, 1, 1, 3, 0), datetime(2024, 1, 1, 4, 0), 55.0)
        ]
        charging_tomorrow = [
            (datetime(2024, 1, 2, 1, 0), datetime(2024, 1, 2, 2, 0), 45.0)
        ]
        discharge_today = [
            (datetime(2024, 1, 1, 18, 0), datetime(2024, 1, 1, 19, 0), 200.0)
        ]
        discharge_tomorrow = []
        eur_czk_rate = 25.0

        controller._log_compact_schedule(
            charging_today,
            charging_tomorrow,
            discharge_today,
            discharge_tomorrow,
            eur_czk_rate
        )

        # Verify all schedules were logged
        assert controller.logger.info.call_count >= 4

        # Check that appropriate summaries were generated
        calls = [call[0][0] for call in controller.logger.info.call_args_list]

        assert any("Schedule Summary" in call for call in calls)
        assert any("Charging today" in call for call in calls)
        assert any("Charging tomorrow" in call for call in calls)
        assert any("Discharge today" in call for call in calls)

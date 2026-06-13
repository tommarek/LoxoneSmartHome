"""Test the settings module."""

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from config.settings import GrowattConfig, InfluxDBConfig, MQTTConfig, Settings, WeatherConfig


class TestSettings:
    """Test the Settings class."""

    def test_default_settings(self) -> None:
        """Test default settings values."""
        with patch.dict("os.environ", {"INFLUXDB_TOKEN": "test-token"}, clear=True):
            settings = Settings(influxdb_token="test-token")

            assert settings.log_level == "INFO"
            assert settings.mqtt_broker in ["mqtt", "localhost"]  # Allow both defaults
            assert settings.mqtt_port == 1883
            assert settings.influxdb_token == "test-token"

    def test_settings_from_env(self) -> None:
        """Test loading settings from environment variables."""
        env_vars = {
            "LOG_LEVEL": "DEBUG",
            "MQTT_BROKER": "test-broker",
            "MQTT_PORT": "1234",
            "INFLUXDB_TOKEN": "my-token",
            "UDP_LISTENER_ENABLED": "false",
        }

        with patch.dict("os.environ", env_vars):
            settings = Settings(influxdb_token="my-token")

            assert settings.log_level == "DEBUG"
            assert settings.mqtt_broker == "test-broker"
            assert settings.mqtt_port == 1234
            assert settings.influxdb_token == "my-token"
            assert settings.udp_listener_enabled is False

    def test_invalid_log_level(self) -> None:
        """Test invalid log level validation."""
        with patch.dict("os.environ", {"LOG_LEVEL": "INVALID", "INFLUXDB_TOKEN": "test"}):
            with pytest.raises(ValidationError):
                Settings(influxdb_token="test")

    def test_port_validation(self) -> None:
        """Test port number validation."""
        with patch.dict("os.environ", {"MQTT_PORT": "99999", "INFLUXDB_TOKEN": "test"}):
            with pytest.raises(ValidationError):
                Settings(influxdb_token="test")


class TestMQTTConfig:
    """Test the MQTTConfig class."""

    def test_default_values(self) -> None:
        """Test default MQTT configuration values."""
        config = MQTTConfig()

        assert config.broker == "mqtt"
        assert config.port == 1883
        assert config.client_id == "loxone-smart-home"
        assert config.username is None
        assert config.password is None

    def test_port_validation(self) -> None:
        """Test MQTT port validation."""
        with pytest.raises(ValidationError):
            MQTTConfig(port=0)

        with pytest.raises(ValidationError):
            MQTTConfig(port=70000)


class TestInfluxDBConfig:
    """Test the InfluxDBConfig class."""

    def test_required_fields(self) -> None:
        """Test required fields for InfluxDB config."""
        with pytest.raises(ValidationError):
            InfluxDBConfig(url="http://localhost:8086", org="test", token="")  # Empty token

    def test_batch_size_validation(self) -> None:
        """Test batch size validation."""
        with pytest.raises(ValidationError):
            InfluxDBConfig(
                url="http://localhost:8086",
                token="test-token",
                org="test",
                batch_size=0,
            )


class TestWeatherConfig:
    """Test the WeatherConfig class."""

    def test_coordinate_validation(self) -> None:
        """Test latitude/longitude validation."""
        # Valid coordinates
        config = WeatherConfig(latitude=45.0, longitude=90.0)
        assert config.latitude == 45.0
        assert config.longitude == 90.0

        # Invalid latitude
        with pytest.raises(ValidationError):
            WeatherConfig(latitude=91.0, longitude=0.0)

        # Invalid longitude
        with pytest.raises(ValidationError):
            WeatherConfig(latitude=0.0, longitude=181.0)

    def test_update_interval_validation(self) -> None:
        """Test update interval validation."""
        with pytest.raises(ValidationError):
            WeatherConfig(latitude=0.0, longitude=0.0, update_interval=30)  # Too short


class TestGrowattConfig:
    """Test the GrowattConfig class."""

    def test_low_tariff_hours_validation(self) -> None:
        """low_tariff_hours must parse and contain at least one range."""
        # Valid multi-range value passes through unchanged.
        cfg = GrowattConfig(low_tariff_hours="0-10,15-17")
        assert cfg.low_tariff_hours == "0-10,15-17"

        # An empty / all-whitespace value would silently bill every import hour
        # at the high VT tariff — must be rejected, not accepted.
        with pytest.raises(ValidationError):
            GrowattConfig(low_tariff_hours="")
        with pytest.raises(ValidationError):
            GrowattConfig(low_tariff_hours="  ,  ")

        # Malformed ranges still raise.
        with pytest.raises(ValidationError):
            GrowattConfig(low_tariff_hours="10-5")  # start >= end

    def test_soc_validation(self) -> None:
        """Test state of charge validation."""
        # Valid SOC values
        config = GrowattConfig(min_soc=20.0, max_soc=90.0)
        assert config.min_soc == 20.0
        assert config.max_soc == 90.0

        # Test invalid cases
        with pytest.raises(ValidationError):
            GrowattConfig(min_soc=50.0, max_soc=40.0)

        with pytest.raises(ValidationError):
            GrowattConfig(min_soc=20.0, max_soc=101.0)

    def test_positive_values(self) -> None:
        """Test positive value constraints."""
        with pytest.raises(ValidationError):
            GrowattConfig(battery_capacity=-10.0)


class TestDeferrableLoadsJsonValidator:
    """Test the deferrable_loads_json field validator (fail-fast at startup)."""

    def test_valid_passes_through_unchanged(self) -> None:
        raw = (
            '[{"name": "ev", "energy_required_kwh": 8.0, "power_kw": 3.0, '
            '"earliest_start": "22:00", "latest_end": "06:00", '
            '"mqtt_topic_on": "ev/on", "mqtt_topic_off": "ev/off"}]'
        )
        config = GrowattConfig(deferrable_loads_json=raw)
        assert config.deferrable_loads_json == raw

    def test_empty_array_is_valid(self) -> None:
        config = GrowattConfig(deferrable_loads_json="[]")
        assert config.deferrable_loads_json == "[]"

    def test_load_with_only_required_fields_passes(self) -> None:
        config = GrowattConfig(
            deferrable_loads_json='[{"name": "wp", "energy_required_kwh": 2, "power_kw": 1}]'
        )
        assert "wp" in config.deferrable_loads_json

    @pytest.mark.parametrize(
        "raw",
        [
            "not json",                                   # invalid JSON
            '{"name": "ev"}',                             # not a list
            "[42]",                                       # entry not an object
            '[{"energy_required_kwh": 1, "power_kw": 1}]',  # missing name
            '[{"name": "", "energy_required_kwh": 1, "power_kw": 1}]',  # empty name
            '[{"name": "ev", "power_kw": 1}]',            # missing energy_required_kwh
            '[{"name": "ev", "energy_required_kwh": 1}]',  # missing power_kw
            '[{"name": "ev", "energy_required_kwh": "x", "power_kw": 1}]',  # non-numeric
            '[{"name": "ev", "energy_required_kwh": 0, "power_kw": 1}]',  # non-positive
            '[{"name": "ev", "energy_required_kwh": 1, "power_kw": -1}]',  # non-positive
            '[{"name": "ev", "energy_required_kwh": 1, "power_kw": 1, "earliest_start": "25:00"}]',  # bad time
            # mqtt_topic_on without mqtt_topic_off — cannot be turned off
            '[{"name": "ev", "energy_required_kwh": 1, "power_kw": 1, "mqtt_topic_on": "ev/on"}]',
            # zero-width window — in_window() would always be False
            '[{"name": "ev", "energy_required_kwh": 1, "power_kw": 1, '
            '"earliest_start": "08:00", "latest_end": "08:00"}]',
        ],
    )
    def test_malformed_entries_raise(self, raw: str) -> None:
        with pytest.raises(ValidationError):
            GrowattConfig(deferrable_loads_json=raw)

    def test_zero_width_window_error_suggests_full_day(self) -> None:
        # earliest_start == latest_end never schedules (in_window always
        # False); the error must say so and point at the 00:00→23:59 idiom.
        raw = (
            '[{"name": "ev", "energy_required_kwh": 1, "power_kw": 1, '
            '"earliest_start": "08:00", "latest_end": "08:00"}]'
        )
        with pytest.raises(ValidationError, match="zero-width"):
            GrowattConfig(deferrable_loads_json=raw)

    def test_zero_width_window_via_default_is_rejected(self) -> None:
        # A single-key entry must be checked against the controller's
        # defaults (00:00 / 23:59): {"latest_end": "00:00"} defaults
        # earliest_start to 00:00 → zero-width → silently never schedules.
        raw = (
            '[{"name": "ev", "energy_required_kwh": 1, "power_kw": 1, '
            '"latest_end": "00:00"}]'
        )
        with pytest.raises(ValidationError, match="zero-width"):
            GrowattConfig(deferrable_loads_json=raw)

    def test_full_day_window_is_valid(self) -> None:
        raw = (
            '[{"name": "ev", "energy_required_kwh": 1, "power_kw": 1, '
            '"earliest_start": "00:00", "latest_end": "23:59"}]'
        )
        config = GrowattConfig(deferrable_loads_json=raw)
        assert "ev" in config.deferrable_loads_json

    def test_midnight_wrap_window_is_valid(self) -> None:
        # 22:00 -> 06:00 wraps midnight; the mod-24h width (480 min) must
        # not be misread as negative/zero by the width check.
        raw = (
            '[{"name": "ev", "energy_required_kwh": 1, "power_kw": 1, '
            '"earliest_start": "22:00", "latest_end": "06:00"}]'
        )
        config = GrowattConfig(deferrable_loads_json=raw)
        assert "ev" in config.deferrable_loads_json

    def test_exactly_one_block_window_is_valid(self) -> None:
        # 15 minutes = exactly one block; schedulable, must pass.
        raw = (
            '[{"name": "ev", "energy_required_kwh": 0.1, "power_kw": 1, '
            '"earliest_start": "10:00", "latest_end": "10:15"}]'
        )
        config = GrowattConfig(deferrable_loads_json=raw)
        assert "ev" in config.deferrable_loads_json

    def test_sub_block_window_is_rejected(self) -> None:
        # 9-minute window (23:50 -> default 23:59) can contain no block
        # start unless grid-aligned — same silent never-schedules class.
        raw = (
            '[{"name": "ev", "energy_required_kwh": 1, "power_kw": 1, '
            '"earliest_start": "23:50"}]'
        )
        with pytest.raises(ValidationError, match="15-minute block"):
            GrowattConfig(deferrable_loads_json=raw)

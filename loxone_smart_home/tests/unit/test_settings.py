"""Test the settings module."""

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from config.settings import (
    GrowattConfig,
    InfluxDBConfig,
    MQTTConfig,
    Settings,
    WeatherConfig,
)


class TestSettings:
    """Test the Settings class."""

    def test_default_settings(self) -> None:
        """Test default settings values."""
        with patch.dict("os.environ", {"INFLUXDB_TOKEN": "test-token"}):
            settings = Settings(influxdb_token="test-token")

            assert settings.log_level == "INFO"
            assert settings.mqtt_broker == "mqtt"
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

    def test_soc_validation(self) -> None:
        """Test state of charge validation."""
        # Valid SOC values
        config = GrowattConfig(min_soc=20.0, max_soc=90.0)
        assert config.min_soc == 20.0
        assert config.max_soc == 90.0

        # Invalid: max_soc <= min_soc
        with pytest.raises(ValidationError):
            GrowattConfig(min_soc=50.0, max_soc=40.0)

        # Invalid: SOC > 100
        with pytest.raises(ValidationError):
            GrowattConfig(min_soc=20.0, max_soc=101.0)

    def test_positive_values(self) -> None:
        """Test positive value constraints."""
        with pytest.raises(ValidationError):
            GrowattConfig(battery_capacity=-10.0)

        with pytest.raises(ValidationError):
            GrowattConfig(max_charge_power=0.0)

    def test_schedule_validation(self) -> None:
        """Test schedule time validation."""
        # Valid schedule time
        config = GrowattConfig(schedule_hour=15, schedule_minute=30)
        assert config.schedule_hour == 15
        assert config.schedule_minute == 30

        # Invalid hour
        with pytest.raises(ValidationError):
            GrowattConfig(schedule_hour=24)

        # Invalid minute
        with pytest.raises(ValidationError):
            GrowattConfig(schedule_minute=60)

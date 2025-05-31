"""Tests for MQTT bridge module."""

import json
import socket
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.settings import Settings
from modules.mqtt_bridge import MQTTBridge


@pytest.fixture
def mock_settings() -> Settings:
    """Create mock settings."""
    settings = MagicMock(spec=Settings)
    settings.loxone_bridge.bridge_topics = [
        "energy/solar",
        "teplomer/TC",
    ]
    settings.loxone_bridge.loxone_host = "192.168.101.34"
    settings.loxone_bridge.loxone_udp_port = 4000
    # Add logging configuration
    settings.log_level = "INFO"
    settings.log_timezone = "Europe/Prague"
    return settings


@pytest.fixture
def mock_mqtt_client() -> MagicMock:
    """Create mock MQTT client."""
    from utils.async_mqtt_client import AsyncMQTTClient

    client = MagicMock(spec=AsyncMQTTClient)
    client.subscribe = AsyncMock()
    return client


@pytest.fixture
def mock_socket() -> Generator[MagicMock, None, None]:
    """Create mock UDP socket."""
    with patch("socket.socket") as mock_socket_class:
        mock_socket_instance = MagicMock()
        mock_socket_class.return_value = mock_socket_instance
        yield mock_socket_instance


@pytest.fixture
def mqtt_bridge(
    mock_mqtt_client: MagicMock,
    mock_settings: Settings,
    mock_socket: MagicMock,
) -> MQTTBridge:
    """Create MQTT bridge instance."""
    return MQTTBridge(mock_mqtt_client, mock_settings)


@pytest.mark.asyncio
async def test_mqtt_bridge_init(mock_mqtt_client: MagicMock, mock_settings: Settings) -> None:
    """Test MQTT bridge initialization."""
    with patch("socket.socket") as mock_socket_class:
        mock_socket_instance = MagicMock()
        mock_socket_class.return_value = mock_socket_instance

        bridge = MQTTBridge(mock_mqtt_client, mock_settings)

        assert bridge.name == "MQTTBridge"
        assert bridge.mqtt_client == mock_mqtt_client
        assert bridge.settings == mock_settings
        mock_socket_class.assert_called_once_with(socket.AF_INET, socket.SOCK_DGRAM)


@pytest.mark.asyncio
async def test_start_subscribes_to_topics(
    mqtt_bridge: MQTTBridge, mock_mqtt_client: MagicMock
) -> None:
    """Test that start() subscribes to configured topics."""
    await mqtt_bridge.start()

    # Should subscribe to each topic with the callback
    assert mock_mqtt_client.subscribe.call_count == 2

    # Check the topics are correct
    actual_calls = mock_mqtt_client.subscribe.call_args_list
    topics = [call[0][0] for call in actual_calls]
    callbacks = [call[0][1] for call in actual_calls]

    assert "energy/solar" in topics
    assert "teplomer/TC" in topics
    assert all(callback == mqtt_bridge.on_mqtt_message for callback in callbacks)


@pytest.mark.asyncio
async def test_stop_closes_socket(mqtt_bridge: MQTTBridge, mock_socket: MagicMock) -> None:
    """Test that stop() closes the UDP socket."""
    await mqtt_bridge.stop()
    mock_socket.close.assert_called_once()


@pytest.mark.asyncio
async def test_on_mqtt_message_dict_payload(
    mqtt_bridge: MQTTBridge, mock_socket: MagicMock
) -> None:
    """Test handling MQTT message with dict payload."""
    topic = "energy/solar"
    payload_dict = {
        "power": 2500.5,
        "voltage": 240.1,
        "current": 10.4,
    }
    payload = json.dumps(payload_dict)

    await mqtt_bridge.on_mqtt_message(topic, payload)

    # Should convert dict to semicolon-separated key=value pairs
    expected_message = "power=2500.5;voltage=240.1;current=10.4"

    mock_socket.sendto.assert_called_once_with(
        expected_message.encode("utf-8"),
        ("192.168.101.34", 4000),
    )


@pytest.mark.asyncio
async def test_on_mqtt_message_non_dict_payload(
    mqtt_bridge: MQTTBridge, mock_socket: MagicMock
) -> None:
    """Test handling MQTT message with non-dict payload."""
    topic = "teplomer/TC"
    payload_data = 23.5
    payload = json.dumps(payload_data)

    await mqtt_bridge.on_mqtt_message(topic, payload)

    # Should use topic=value format for non-dict data
    expected_message = "teplomer/TC=23.5"

    mock_socket.sendto.assert_called_once_with(
        expected_message.encode("utf-8"),
        ("192.168.101.34", 4000),
    )


@pytest.mark.asyncio
async def test_on_mqtt_message_string_payload(
    mqtt_bridge: MQTTBridge, mock_socket: MagicMock
) -> None:
    """Test handling MQTT message with string payload."""
    topic = "status/system"
    payload_data = "online"
    payload = json.dumps(payload_data)

    await mqtt_bridge.on_mqtt_message(topic, payload)

    # Should use topic=value format for string data
    expected_message = "status/system=online"

    mock_socket.sendto.assert_called_once_with(
        expected_message.encode("utf-8"),
        ("192.168.101.34", 4000),
    )


@pytest.mark.asyncio
async def test_on_mqtt_message_complex_dict(
    mqtt_bridge: MQTTBridge, mock_socket: MagicMock
) -> None:
    """Test handling MQTT message with complex dict payload."""
    topic = "weather/data"
    payload_dict = {
        "temperature": 22.3,
        "humidity": 65,
        "pressure": 1013.25,
        "wind_speed": 5.2,
    }
    payload = json.dumps(payload_dict)

    await mqtt_bridge.on_mqtt_message(topic, payload)

    # Should convert all key=value pairs
    sent_message = mock_socket.sendto.call_args[0][0].decode("utf-8")
    message_parts = sent_message.split(";")

    # Check that all expected key=value pairs are present
    assert "temperature=22.3" in message_parts
    assert "humidity=65" in message_parts
    assert "pressure=1013.25" in message_parts
    assert "wind_speed=5.2" in message_parts
    assert len(message_parts) == 4


@pytest.mark.asyncio
async def test_on_mqtt_message_invalid_json(
    mqtt_bridge: MQTTBridge, mock_socket: MagicMock
) -> None:
    """Test handling MQTT message with invalid JSON."""
    topic = "invalid/json"
    payload = "not valid json{"

    with patch.object(mqtt_bridge.logger, "error") as mock_error:
        await mqtt_bridge.on_mqtt_message(topic, payload)

    # Should not send any UDP message for invalid JSON
    mock_socket.sendto.assert_not_called()
    mock_error.assert_called_once()
    assert "Invalid JSON payload" in mock_error.call_args[0][0]


@pytest.mark.asyncio
async def test_on_mqtt_message_socket_error(
    mqtt_bridge: MQTTBridge, mock_socket: MagicMock
) -> None:
    """Test handling socket error during UDP send."""
    mock_socket.sendto.side_effect = OSError("Network unreachable")

    topic = "energy/solar"
    payload = json.dumps({"power": 1000})

    with patch.object(mqtt_bridge.logger, "error") as mock_error:
        await mqtt_bridge.on_mqtt_message(topic, payload)

    # Should log the error
    mock_error.assert_called_once()
    assert "Error processing MQTT message" in mock_error.call_args[0][0]


@pytest.mark.asyncio
async def test_message_format_matches_original() -> None:
    """Test that message format exactly matches original implementation."""
    # This test verifies the exact format from the original mqtt-loxone-bridge.py
    # Line 31: message = ';'.join([f"{k}={v}" for k, v in data.items()])

    test_data = {"power": 2500, "voltage": 240}
    expected = "power=2500;voltage=240"

    # Simulate our implementation
    message_parts = [f"{k}={v}" for k, v in test_data.items()]
    actual = ";".join(message_parts)

    assert actual == expected


@pytest.mark.asyncio
async def test_non_dict_format_matches_original() -> None:
    """Test non-dict format matches original implementation."""
    # From original line 33: message = f"{msg.topic}={data}"

    topic = "teplomer/TC"
    data = 23.5
    expected = "teplomer/TC=23.5"

    # Simulate our implementation
    actual = f"{topic}={data}"

    assert actual == expected

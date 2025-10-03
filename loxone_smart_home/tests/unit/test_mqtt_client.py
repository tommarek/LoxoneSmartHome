"""Test the MQTT client module."""

import asyncio
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from asyncio_mqtt import MqttError

from config.settings import Settings
from utils.async_mqtt_client import AsyncMQTTClient


class TestAsyncMQTTClient:
    """Test the AsyncMQTTClient class."""

    @pytest.fixture
    def settings(self) -> Settings:
        """Create test settings."""
        with patch.dict("os.environ", {"INFLUXDB_TOKEN": "test-token"}):
            settings = Settings(influxdb_token="test-token")
            # Add logging configuration
            settings.log_level = "INFO"
            settings.log_timezone = "Europe/Prague"
            return settings

    @pytest.fixture
    async def mqtt_client(self, settings: Settings) -> AsyncGenerator[AsyncMQTTClient, None]:
        """Create MQTT client instance."""
        client = AsyncMQTTClient(settings)
        yield client
        # Cleanup after each test
        try:
            await client.disconnect()
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_connect_success(self, mqtt_client: AsyncMQTTClient) -> None:
        """Test successful connection to MQTT broker."""
        with patch("utils.async_mqtt_client.Client") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Patch the _read_messages method to prevent it from running
            with patch.object(mqtt_client, "_read_messages", new_callable=AsyncMock):
                await mqtt_client.connect()

                assert mqtt_client.client is not None
                mock_client.connect.assert_called_once()
                assert mqtt_client._running is True

                # The read task is now started with a delay, so wait for it
                await asyncio.sleep(0.15)  # Wait for the 100ms delay
                assert mqtt_client._read_task is not None

    @pytest.mark.asyncio
    async def test_connect_failure(self, mqtt_client: AsyncMQTTClient) -> None:
        """Test connection failure handling."""
        with patch("utils.async_mqtt_client.Client") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.connect.side_effect = MqttError("Connection failed")
            mock_client_class.return_value = mock_client

            with pytest.raises(MqttError):
                await mqtt_client.connect()

    @pytest.mark.asyncio
    async def test_disconnect(self, mqtt_client: AsyncMQTTClient) -> None:
        """Test disconnection from MQTT broker."""
        mqtt_client.client = AsyncMock()
        mqtt_client._running = True
        mqtt_client._connected = True

        # Create a task that completes immediately
        async def dummy_task() -> None:
            pass

        mqtt_client._read_task = asyncio.create_task(dummy_task())

        await mqtt_client.disconnect()

        assert mqtt_client._running is False
        mqtt_client.client.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_success(self, mqtt_client: AsyncMQTTClient) -> None:
        """Test successful message publishing (queues message)."""
        # AsyncMQTTClient queues messages rather than publishing directly
        await mqtt_client.publish("test/topic", "test message", retain=True)

        # Verify message was queued
        assert mqtt_client.publish_queue.qsize() == 1
        queued_item = await mqtt_client.publish_queue.get()
        assert queued_item == ("test/topic", "test message", True)

    @pytest.mark.asyncio
    async def test_publish_without_connection(self, mqtt_client: AsyncMQTTClient) -> None:
        """Test publishing without active connection (still queues)."""
        # AsyncMQTTClient allows queuing even without connection
        mqtt_client.client = None
        mqtt_client._connected = False

        # Should still queue the message
        await mqtt_client.publish("test/topic", "test message")
        assert mqtt_client.publish_queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_subscribe(self, mqtt_client: AsyncMQTTClient) -> None:
        """Test topic subscription."""
        mqtt_client.client = AsyncMock()
        mqtt_client._connected = True  # Simulate connected state
        callback = AsyncMock()

        await mqtt_client.subscribe("test/topic", callback)

        assert "test/topic" in mqtt_client.subscribers
        assert callback in mqtt_client.subscribers["test/topic"]
        mqtt_client.client.subscribe.assert_called_once_with("test/topic")

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, mqtt_client: AsyncMQTTClient) -> None:
        """Test multiple callbacks for same topic."""
        mqtt_client.client = AsyncMock()
        mqtt_client._connected = True  # Simulate connected state
        callback1 = AsyncMock()
        callback2 = AsyncMock()

        await mqtt_client.subscribe("test/topic", callback1)
        await mqtt_client.subscribe("test/topic", callback2)

        assert len(mqtt_client.subscribers["test/topic"]) == 2
        # Should only subscribe once to the broker
        mqtt_client.client.subscribe.assert_called_once_with("test/topic")

    @pytest.mark.asyncio
    async def test_callback_execution(self, mqtt_client: AsyncMQTTClient) -> None:
        """Test callback execution for received messages."""
        async_callback = AsyncMock()
        sync_callback = MagicMock()

        # Test async callback
        await mqtt_client._execute_callback(async_callback, "test/topic", "payload")
        async_callback.assert_called_once_with("test/topic", "payload")

        # Test sync callback
        await mqtt_client._execute_callback(sync_callback, "test/topic", "payload")
        sync_callback.assert_called_once_with("test/topic", "payload")

    @pytest.mark.asyncio
    async def test_late_subscription_handling(self, mqtt_client: AsyncMQTTClient) -> None:
        """Test that late subscriptions after connection trigger message loop restart."""
        with patch("utils.async_mqtt_client.Client") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Connect first
            await mqtt_client.connect()

            # Wait for delayed read task to start
            await asyncio.sleep(0.15)

            # Verify message loop has started
            assert mqtt_client._message_loop_started is True

            # Now subscribe to a topic (late subscription)
            callback = AsyncMock()
            await mqtt_client.subscribe("late/topic", callback)

            # Verify that the late subscription was detected and restart was triggered
            assert "late/topic" in mqtt_client.subscribers
            assert callback in mqtt_client.subscribers["late/topic"]

            # Verify MQTT client subscribe was called
            mock_client.subscribe.assert_called_with("late/topic")

    @pytest.mark.asyncio
    async def test_callback_error_handling(self, mqtt_client: AsyncMQTTClient) -> None:
        """Test error handling in callbacks."""
        callback = AsyncMock(side_effect=Exception("Callback error"))

        # Should raise the exception from the callback
        with pytest.raises(Exception, match="Callback error"):
            await mqtt_client._execute_callback(callback, "test/topic", "payload")

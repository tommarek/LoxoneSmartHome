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

                # The read task is now started immediately with pre-registered subscriptions
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

        # Verify message was queued. Items are
        # (topic, payload, retain, qos, enqueue_monotonic).
        assert mqtt_client.publish_queue.qsize() == 1
        topic, payload, retain, qos, ts = await mqtt_client.publish_queue.get()
        assert (topic, payload, retain) == ("test/topic", "test message", True)
        assert qos == 1
        assert isinstance(ts, float)

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
    async def test_pre_registration(self, mqtt_client: AsyncMQTTClient) -> None:
        """Test that pre-registered subscriptions are applied on connection."""
        with patch("utils.async_mqtt_client.Client") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Pre-register subscriptions before connection
            callback1 = AsyncMock()
            callback2 = AsyncMock()
            mqtt_client.register_subscription("topic1", callback1)
            mqtt_client.register_subscription("topic2", callback2)

            # Connect
            await mqtt_client.connect()

            # Verify pre-registered subscriptions were applied
            assert "topic1" in mqtt_client.subscribers
            assert "topic2" in mqtt_client.subscribers
            assert callback1 in mqtt_client.subscribers["topic1"]
            assert callback2 in mqtt_client.subscribers["topic2"]

            # Verify MQTT broker subscriptions were made
            calls = mock_client.subscribe.call_args_list
            topics_subscribed = [call[0][0] for call in calls]
            assert "topic1" in topics_subscribed
            assert "topic2" in topics_subscribed

    @pytest.mark.asyncio
    async def test_late_subscription_warning(self, mqtt_client: AsyncMQTTClient) -> None:
        """Test that late subscriptions after connection show a warning."""
        with patch("utils.async_mqtt_client.Client") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Connect first
            await mqtt_client.connect()

            # Now subscribe to a topic (late subscription)
            callback = AsyncMock()

            # Capture warning logs to verify late subscription warning
            with patch.object(mqtt_client.logger, 'warning') as mock_warning:
                await mqtt_client.subscribe("late/topic", callback)

                # Verify warning about runtime subscription
                mock_warning.assert_called_once()
                assert "Runtime subscription" in mock_warning.call_args[0][0]
                assert "register_subscription" in mock_warning.call_args[0][0]

            # Verify subscription was still processed
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

    @pytest.mark.asyncio
    async def test_publish_queue_bounded_drops_oldest(
        self, mqtt_client: AsyncMQTTClient
    ) -> None:
        """Under a sustained outage the queue must stay bounded by dropping the
        OLDEST message, keeping the newest intent."""
        mqtt_client._max_publish_queue = 3
        for i in range(3):
            await mqtt_client.publish(f"topic/{i}", f"msg{i}")
        assert mqtt_client.publish_queue.qsize() == 3

        # One more over the bound drops the oldest (topic/0), not the newest.
        await mqtt_client.publish("topic/3", "msg3")
        assert mqtt_client.publish_queue.qsize() == 3
        topics = []
        while not mqtt_client.publish_queue.empty():
            topic, *_ = mqtt_client.publish_queue.get_nowait()
            topics.append(topic)
        assert topics == ["topic/1", "topic/2", "topic/3"]

    @pytest.mark.asyncio
    async def test_publish_loop_drops_stale_message(
        self, mqtt_client: AsyncMQTTClient
    ) -> None:
        """A message older than the TTL is dropped, not published."""
        mqtt_client._publish_ttl_s = 120.0
        mqtt_client._running = True
        mqtt_client._connected = True
        mqtt_client._publish_with_retry = AsyncMock()

        # Enqueue a message whose enqueue time is 200s in the past (stale).
        import time as _time
        stale_ts = _time.monotonic() - 200.0
        await mqtt_client.publish_queue.put(("t", "p", False, 1, stale_ts))

        # Run a single loop iteration, then stop.
        async def _stop_soon() -> None:
            await asyncio.sleep(0.05)
            mqtt_client._running = False

        with patch.object(mqtt_client.logger, "warning") as mock_warning:
            await asyncio.gather(mqtt_client._publish_loop(), _stop_soon())

        # The stale message is dropped (never published) and a warning is logged.
        mqtt_client._publish_with_retry.assert_not_called()
        assert any("stale" in str(c).lower() for c in mock_warning.call_args_list)

    @pytest.mark.asyncio
    async def test_publish_loop_requeue_preserves_enqueue_time(
        self, mqtt_client: AsyncMQTTClient
    ) -> None:
        """When disconnected, a message is re-queued with its ORIGINAL enqueue
        timestamp so the TTL keeps counting (not reset on each retry)."""
        mqtt_client._publish_ttl_s = 120.0
        mqtt_client._running = True
        mqtt_client._connected = False  # force the re-queue branch

        import time as _time
        original_ts = _time.monotonic() - 10.0  # 10s old, still within TTL
        await mqtt_client.publish_queue.put(("t", "p", False, 1, original_ts))

        async def _stop_soon() -> None:
            await asyncio.sleep(0.05)
            mqtt_client._running = False

        await asyncio.gather(mqtt_client._publish_loop(), _stop_soon())

        # Message is back on the queue with the SAME ts (TTL not reset).
        assert not mqtt_client.publish_queue.empty()
        _, _, _, _, ts = mqtt_client.publish_queue.get_nowait()
        assert ts == original_ts

    @pytest.mark.asyncio
    async def test_is_connected_property(self, mqtt_client: AsyncMQTTClient) -> None:
        """is_connected reflects the live connection flag (used to gate
        fire-and-forget deferrable-load completion accounting)."""
        mqtt_client._connected = False
        assert mqtt_client.is_connected is False
        mqtt_client._connected = True
        assert mqtt_client.is_connected is True

"""Async MQTT client with thread-safe operations and connection management."""

import asyncio
import logging
from typing import Any, Callable, Dict, Optional, Set

from asyncio_mqtt import Client, MqttError

from config.settings import Settings


class AsyncMQTTClient:
    """Thread-safe async MQTT client with connection management and retry logic."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the async MQTT client."""
        self.settings = settings
        self.logger = logging.getLogger(f"{__name__}.AsyncMQTT")

        # Connection state
        self.client: Optional[Client] = None
        self.connection_lock = asyncio.Lock()
        self._connected = False
        self._running = False

        # Subscribers with thread-safe access
        self.subscribers: Dict[str, Set[Callable]] = {}
        self.subscribers_lock = asyncio.Lock()

        # Publish queue for reliability
        self.publish_queue: asyncio.Queue[tuple[str, Any, bool]] = asyncio.Queue()

        # Background tasks
        self._read_task: Optional[asyncio.Task[None]] = None
        self._publish_task: Optional[asyncio.Task[None]] = None
        self._reconnect_task: Optional[asyncio.Task[None]] = None

        # Metrics
        self.messages_published = 0
        self.messages_received = 0
        self.reconnect_attempts = 0

    async def connect(self) -> None:
        """Connect to the MQTT broker with retry logic."""
        async with self.connection_lock:
            if self._connected:
                return

            self._running = True
            await self._connect_with_retry()

            # Start background tasks
            self._read_task = asyncio.create_task(self._read_messages())
            self._publish_task = asyncio.create_task(self._publish_loop())
            self._reconnect_task = asyncio.create_task(self._monitor_connection())

    async def _connect_with_retry(self, max_retries: int = 5) -> None:
        """Connect to MQTT broker with exponential backoff."""
        for attempt in range(max_retries):
            try:
                self.client = Client(
                    hostname=self.settings.mqtt.broker,
                    port=self.settings.mqtt.port,
                    username=self.settings.mqtt.username,
                    password=self.settings.mqtt.password,
                    client_id=self.settings.mqtt.client_id,
                    keepalive=30,  # Keep connection alive
                )
                await self.client.connect()
                self._connected = True

                # Re-subscribe to all topics
                async with self.subscribers_lock:
                    for topic in self.subscribers:
                        await self.client.subscribe(topic)

                self.logger.info(
                    f"Connected to MQTT broker at {self.settings.mqtt.broker}:"
                    f"{self.settings.mqtt.port}"
                )
                return

            except Exception as e:
                self.reconnect_attempts += 1
                if attempt < max_retries - 1:
                    wait_time = min(2**attempt, 30)  # Max 30s wait
                    self.logger.warning(
                        f"Failed to connect to MQTT broker, " f"retrying in {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(
                        f"Failed to connect to MQTT broker after " f"{max_retries} attempts: {e}"
                    )
                    raise

    async def disconnect(self) -> None:
        """Disconnect from the MQTT broker."""
        if not self._running:
            return  # Already disconnected
        self._running = False

        # Cancel background tasks and wait for completion
        for task in [self._read_task, self._publish_task, self._reconnect_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=1.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                except Exception:
                    pass  # Ignore other exceptions during cleanup

        # Disconnect client
        try:
            async with self.connection_lock:
                if self.client and self._connected:
                    await self.client.disconnect()
                    self._connected = False
        except Exception:
            pass  # Ignore exceptions during disconnect

        # Reset task references
        self._read_task = None
        self._publish_task = None
        self._reconnect_task = None

        self.logger.info(
            f"Disconnected from MQTT broker. "
            f"Messages: published={self.messages_published}, "
            f"received={self.messages_received}, "
            f"reconnects={self.reconnect_attempts}"
        )

    async def publish(self, topic: str, payload: Any, retain: bool = False, qos: int = 1) -> None:
        """Queue a message for publishing with reliability."""
        await self.publish_queue.put((topic, payload, retain))

    async def _publish_loop(self) -> None:
        """Background task to publish queued messages."""
        while self._running:
            try:
                # Wait for message with timeout to allow periodic checks
                topic, payload, retain = await asyncio.wait_for(
                    self.publish_queue.get(), timeout=1.0
                )

                # Ensure we're connected
                if not self._connected:
                    # Re-queue the message
                    await self.publish_queue.put((topic, payload, retain))
                    await asyncio.sleep(0.1)
                    continue

                # Publish with retry
                await self._publish_with_retry(topic, payload, retain)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error in publish loop: {e}", exc_info=True)

    async def _publish_with_retry(
        self, topic: str, payload: Any, retain: bool, max_retries: int = 3
    ) -> None:
        """Publish a message with retry logic."""
        for attempt in range(max_retries):
            try:
                async with self.connection_lock:
                    if self.client and self._connected:
                        await self.client.publish(topic, payload, retain=retain)

                self.messages_published += 1
                self.logger.debug(f"Published to {topic}: {payload}")
                return

            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Failed to publish to {topic}, retrying: {e}")
                    await asyncio.sleep(0.5)
                else:
                    self.logger.error(
                        f"Failed to publish to {topic} after " f"{max_retries} attempts: {e}"
                    )

    async def subscribe(self, topic: str, callback: Callable[[str, Any], Any]) -> None:
        """Subscribe to a topic with a callback."""
        async with self.subscribers_lock:
            if topic not in self.subscribers:
                self.subscribers[topic] = set()
                # Subscribe if connected
                if self.client and self._connected:
                    await self.client.subscribe(topic)
                    self.logger.info(f"Subscribed to topic: {topic}")

            self.subscribers[topic].add(callback)

    async def unsubscribe(self, topic: str, callback: Optional[Callable] = None) -> None:
        """Unsubscribe from a topic."""
        async with self.subscribers_lock:
            if topic in self.subscribers:
                if callback:
                    self.subscribers[topic].discard(callback)
                    if not self.subscribers[topic]:
                        del self.subscribers[topic]
                else:
                    del self.subscribers[topic]

                # Unsubscribe if no more callbacks
                if topic not in self.subscribers and self.client and self._connected:
                    await self.client.unsubscribe(topic)
                    self.logger.info(f"Unsubscribed from topic: {topic}")

    async def _read_messages(self) -> None:
        """Read messages from subscribed topics."""
        while self._running:
            try:
                if not self.client or not self._connected:
                    await asyncio.sleep(1)
                    continue

                async with self.client.messages() as messages:
                    async for message in messages:
                        self.messages_received += 1
                        await self._handle_message(message)

            except MqttError as e:
                self.logger.error(f"MQTT read error: {e}")
                self._connected = False
                await asyncio.sleep(1)
            except Exception as e:
                self.logger.error(f"Unexpected error in read loop: {e}", exc_info=True)
                await asyncio.sleep(1)

    async def _handle_message(self, message: Any) -> None:
        """Handle incoming MQTT message."""
        topic = str(message.topic)
        payload = (
            message.payload.decode("utf-8")
            if isinstance(message.payload, (bytes, bytearray))
            else str(message.payload)
        )

        # Get callbacks with lock
        async with self.subscribers_lock:
            callbacks = list(self.subscribers.get(topic, []))

        # Execute callbacks concurrently
        if callbacks:
            tasks = []
            for callback in callbacks:
                task = asyncio.create_task(self._execute_callback(callback, topic, payload))
                tasks.append(task)

            # Wait for all callbacks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Log any errors
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Error in callback {callbacks[i]} for {topic}: {result}")

    async def _execute_callback(self, callback: Callable, topic: str, payload: str) -> None:
        """Execute a callback safely."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(topic, payload)
            else:
                # Run sync callbacks in executor to not block
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, callback, topic, payload)
        except Exception as e:
            self.logger.error(f"Error executing callback for {topic}: {e}", exc_info=True)
            raise

    async def _monitor_connection(self) -> None:
        """Monitor connection health and reconnect if needed."""
        while self._running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                if not self._connected:
                    self.logger.info("Connection lost, attempting to reconnect...")
                    async with self.connection_lock:
                        await self._connect_with_retry()

            except Exception as e:
                self.logger.error(f"Error in connection monitor: {e}", exc_info=True)

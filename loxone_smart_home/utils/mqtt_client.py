"""Shared MQTT client for all modules."""

import asyncio
import logging
from typing import Any, Callable, Dict, Optional, Union

from asyncio_mqtt import Client, MqttError

from ..config.settings import Settings


class SharedMQTTClient:
    """Shared MQTT client that manages a single connection for all modules."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the MQTT client."""
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.client: Optional[Client] = None
        self.subscribers: Dict[
            str, list[Union[Callable[[str, Any], None], Callable[[str, Any], Any]]]
        ] = {}
        self._running = False
        self._read_task: Optional[asyncio.Task[None]] = None

    async def connect(self) -> None:
        """Connect to the MQTT broker."""
        try:
            self.client = Client(
                hostname=self.settings.mqtt.broker,
                port=self.settings.mqtt.port,
                username=self.settings.mqtt.username,
                password=self.settings.mqtt.password,
                client_id=self.settings.mqtt.client_id,
            )
            await self.client.connect()
            self._running = True
            self._read_task = asyncio.create_task(self._read_messages())
            self.logger.info(
                f"Connected to MQTT broker at {self.settings.mqtt.broker}:"
                f"{self.settings.mqtt.port}"
            )
        except MqttError as e:
            self.logger.error(f"Failed to connect to MQTT broker: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from the MQTT broker."""
        self._running = False
        if self._read_task:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass

        if self.client:
            await self.client.disconnect()
            self.logger.info("Disconnected from MQTT broker")

    async def publish(self, topic: str, payload: Any, retain: bool = False) -> None:
        """Publish a message to a topic."""
        if not self.client:
            raise RuntimeError("MQTT client not connected")

        try:
            await self.client.publish(topic, payload, retain=retain)
            self.logger.debug(f"Published to {topic}: {payload}")
        except MqttError as e:
            self.logger.error(f"Failed to publish to {topic}: {e}")
            raise

    async def subscribe(
        self, topic: str, callback: Union[Callable[[str, Any], None], Callable[[str, Any], Any]]
    ) -> None:
        """Subscribe to a topic with a callback."""
        if not self.client:
            raise RuntimeError("MQTT client not connected")

        if topic not in self.subscribers:
            self.subscribers[topic] = []
            await self.client.subscribe(topic)
            self.logger.info(f"Subscribed to topic: {topic}")

        self.subscribers[topic].append(callback)

    async def _read_messages(self) -> None:
        """Read messages from subscribed topics."""
        if not self.client:
            return

        async with self.client.messages() as messages:
            async for message in messages:
                topic = str(message.topic)
                payload = (
                    message.payload.decode("utf-8")
                    if isinstance(message.payload, (bytes, bytearray))
                    else str(message.payload)
                )

                if topic in self.subscribers:
                    for callback in self.subscribers[topic]:
                        try:
                            await asyncio.create_task(
                                self._handle_callback(callback, topic, payload)
                            )
                        except Exception as e:
                            self.logger.error(f"Error in callback for {topic}: {e}", exc_info=True)

    async def _handle_callback(
        self,
        callback: Union[Callable[[str, Any], None], Callable[[str, Any], Any]],
        topic: str,
        payload: str,
    ) -> None:
        """Handle callback execution."""
        if asyncio.iscoroutinefunction(callback):
            await callback(topic, payload)
        else:
            callback(topic, payload)

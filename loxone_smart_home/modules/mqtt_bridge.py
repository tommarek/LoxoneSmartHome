"""MQTT to Loxone bridge module."""

import json
import socket

from config.settings import Settings
from modules.base import BaseModule
from utils.async_mqtt_client import AsyncMQTTClient


class MQTTBridge(BaseModule):
    """Bridge that forwards MQTT messages to Loxone via UDP."""

    def __init__(self, mqtt_client: AsyncMQTTClient, settings: Settings) -> None:
        """Initialize the MQTT bridge."""
        super().__init__(
            name="MQTTBridge",
            service_name="BRIDGE",
            mqtt_client=mqtt_client,
            settings=settings,
        )
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    async def start(self) -> None:
        """Start the MQTT bridge."""
        if self.mqtt_client is None:
            self.logger.error("MQTT client not available")
            return
        # Subscribe to configured topics
        for topic in self.settings.loxone_bridge.bridge_topics:
            await self.mqtt_client.subscribe(topic, self.on_mqtt_message)
            self.logger.info(f"Subscribed to topic: {topic}")

        self.logger.info("MQTT to Loxone bridge started")

    async def stop(self) -> None:
        """Stop the MQTT bridge."""
        self.udp_socket.close()
        self.logger.info("MQTT to Loxone bridge stopped")

    async def on_mqtt_message(self, topic: str, payload: str) -> None:
        """Handle incoming MQTT messages and forward to Loxone."""
        try:
            # Parse JSON payload
            data = json.loads(payload)
            self.logger.debug(f"Received MQTT message on {topic}: {data}")

            # Convert to message format for Loxone
            if isinstance(data, dict):
                # Convert dict to semicolon-separated key=value pairs (matching original)
                message_parts = [f"{k}={v}" for k, v in data.items()]
                message = ";".join(message_parts)
            else:
                # For non-dict data, use topic=value format
                message = f"{topic}={data}"

            # Send to Loxone via UDP
            self.udp_socket.sendto(
                message.encode("utf-8"),
                (
                    self.settings.loxone_bridge.loxone_host,
                    self.settings.loxone_bridge.loxone_udp_port,
                ),
            )
            self.logger.debug(f"Sent to Loxone: {message}")

        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON payload: {payload}")
        except Exception as e:
            self.logger.error(f"Error processing MQTT message: {e}", exc_info=True)

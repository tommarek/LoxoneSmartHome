"""Base module class for all service modules."""

import asyncio
import os
from abc import ABC, abstractmethod
from typing import Optional

from config.settings import Settings
from utils.async_influxdb_client import AsyncInfluxDBClient
from utils.async_mqtt_client import AsyncMQTTClient
from utils.logging import configure_module_logger


class BaseModule(ABC):
    """Base class for all service modules."""

    def __init__(
        self,
        name: str,
        service_name: str,
        mqtt_client: Optional[AsyncMQTTClient] = None,
        influxdb_client: Optional[AsyncInfluxDBClient] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize the base module."""
        self.name = name
        self.service_name = service_name
        self.mqtt_client = mqtt_client
        self.influxdb_client = influxdb_client
        self.settings = settings or Settings(influxdb_token=os.getenv("INFLUXDB_TOKEN", ""))

        # Configure service-specific logger
        self.logger = configure_module_logger(
            f"{__name__}.{name}", service_name, self.settings.log_timezone, self.settings.log_level
        )
        self._running = False

    @abstractmethod
    async def start(self) -> None:
        """Start the module."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the module."""
        pass

    async def run(self, shutdown_event: asyncio.Event) -> None:
        """Run the module until shutdown is requested."""
        self._running = True
        await self.start()

        try:
            await shutdown_event.wait()
        finally:
            self._running = False
            await self.stop()

"""Base module class for all service modules."""

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

from config.settings import Settings
from utils.influxdb_client import SharedInfluxDBClient
from utils.mqtt_client import SharedMQTTClient


class BaseModule(ABC):
    """Base class for all service modules."""

    def __init__(
        self,
        name: str,
        mqtt_client: Optional[SharedMQTTClient] = None,
        influxdb_client: Optional[SharedInfluxDBClient] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize the base module."""
        self.name = name
        self.mqtt_client = mqtt_client
        self.influxdb_client = influxdb_client
        self.settings = settings or Settings(influxdb_token=os.getenv("INFLUXDB_TOKEN", ""))
        self.logger = logging.getLogger(f"{__name__}.{name}")
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

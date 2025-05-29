"""Growatt controller module - manages solar battery based on energy prices."""

import asyncio
from typing import Optional

from loxone_smart_home.config.settings import Settings
from loxone_smart_home.modules.base import BaseModule
from loxone_smart_home.utils.influxdb_client import SharedInfluxDBClient
from loxone_smart_home.utils.mqtt_client import SharedMQTTClient


class GrowattController(BaseModule):
    """Growatt controller that manages battery charging based on energy prices."""

    def __init__(
        self,
        mqtt_client: SharedMQTTClient,
        influxdb_client: SharedInfluxDBClient,
        settings: Settings,
    ) -> None:
        """Initialize the Growatt controller."""
        super().__init__(
            name="GrowattController",
            mqtt_client=mqtt_client,
            influxdb_client=influxdb_client,
            settings=settings,
        )
        self._task: Optional[asyncio.Task[None]] = None

    async def start(self) -> None:
        """Start the Growatt controller."""
        self._task = asyncio.create_task(self._run_periodic())
        self.logger.info("Growatt controller started")

    async def stop(self) -> None:
        """Stop the Growatt controller."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.logger.info("Growatt controller stopped")

    async def _run_periodic(self) -> None:
        """Run periodic control logic."""
        while self._running:
            try:
                # Placeholder for control logic
                self.logger.debug("Running Growatt control logic...")
                # TODO: Implement actual control logic
                await asyncio.sleep(60)  # Run every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in Growatt controller: {e}", exc_info=True)
                await asyncio.sleep(60)

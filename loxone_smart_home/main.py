#!/usr/bin/env python3
"""Loxone Smart Home - Consolidated Service.

Combines all individual services into a single async Python application.
"""

import asyncio
import logging
import os
import signal
import sys
from typing import Any, List, Optional

import colorlog
from dotenv import load_dotenv

from config.settings import Settings
from modules.growatt_controller import GrowattController
from modules.mqtt_bridge import MQTTBridge
from modules.ote_price_collector import OTEPriceCollector
from modules.udp_listener import UDPListener
from modules.weather_scraper import WeatherScraper
from utils.async_influxdb_client import AsyncInfluxDBClient
from utils.async_mqtt_client import AsyncMQTTClient
from utils.logging import TimezoneAwareFormatter


class LoxoneSmartHome:
    """Main application class that manages all modules."""

    def __init__(self) -> None:
        """Initialize the application."""
        self.settings = Settings(influxdb_token=os.getenv("INFLUXDB_TOKEN", ""))
        self.setup_logging()

        # Shared clients
        self.mqtt_client = AsyncMQTTClient(self.settings)
        self.influxdb_client = AsyncInfluxDBClient(self.settings)

        # Modules
        self.modules: List[asyncio.Task[None]] = []
        self.udp_listener: Optional[UDPListener] = None
        self.mqtt_bridge: Optional[MQTTBridge] = None
        self.weather_scraper: Optional[WeatherScraper] = None
        self.growatt_controller: Optional[GrowattController] = None
        self.ote_collector: Optional[OTEPriceCollector] = None

        # Shutdown event
        self.shutdown_event = asyncio.Event()

    def setup_logging(self) -> None:
        """Configure colored logging with timezone support."""
        handler = colorlog.StreamHandler()
        handler.setFormatter(
            TimezoneAwareFormatter(
                fmt="%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                timezone=self.settings.log_timezone,
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "red,bg_white",
                },
            )
        )

        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, self.settings.log_level))

    async def initialize_modules(self) -> None:
        """Initialize all modules."""
        logger = logging.getLogger(__name__)

        # Initialize shared clients
        await self.mqtt_client.connect()
        logger.info("MQTT client connected")

        await self.influxdb_client.start()
        logger.info("InfluxDB client started")

        # Initialize modules based on configuration
        if self.settings.modules.udp_listener_enabled:
            self.udp_listener = UDPListener(self.influxdb_client, self.settings)
            logger.info("UDP Listener module initialized")

        if self.settings.modules.mqtt_bridge_enabled:
            self.mqtt_bridge = MQTTBridge(self.mqtt_client, self.settings)
            logger.info("MQTT Bridge module initialized")

        if self.settings.modules.weather_scraper_enabled:
            self.weather_scraper = WeatherScraper(
                self.mqtt_client, self.influxdb_client, self.settings
            )
            logger.info("Weather Scraper module initialized")

        if self.settings.modules.growatt_controller_enabled:
            self.growatt_controller = GrowattController(
                self.mqtt_client, self.influxdb_client, self.settings
            )
            logger.info("Growatt Controller module initialized")

        if self.settings.ote_collector_enabled:
            self.ote_collector = OTEPriceCollector(self.influxdb_client, self.settings)
            logger.info("OTE Price Collector module initialized")

    async def start_modules(self) -> None:
        """Start all enabled modules."""
        logger = logging.getLogger(__name__)

        # Start UDP Listener
        if self.udp_listener:
            task = asyncio.create_task(self.udp_listener.run(self.shutdown_event))
            self.modules.append(task)
            logger.info("UDP Listener started")

        # Start MQTT Bridge
        if self.mqtt_bridge:
            task = asyncio.create_task(self.mqtt_bridge.run(self.shutdown_event))
            self.modules.append(task)
            logger.info("MQTT Bridge started")

        # Start Weather Scraper
        if self.weather_scraper:
            task = asyncio.create_task(self.weather_scraper.run(self.shutdown_event))
            self.modules.append(task)
            logger.info("Weather Scraper started")

        # Start Growatt Controller
        if self.growatt_controller:
            task = asyncio.create_task(self.growatt_controller.run(self.shutdown_event))
            self.modules.append(task)
            logger.info("Growatt Controller started")

        # Start OTE Price Collector
        if self.ote_collector:
            task = asyncio.create_task(self.ote_collector.run(self.shutdown_event))
            self.modules.append(task)
            logger.info("OTE Price Collector started")

    async def shutdown(self) -> None:
        """Gracefully shutdown all modules."""
        logger = logging.getLogger(__name__)
        logger.info("Initiating shutdown...")

        # Set shutdown event
        self.shutdown_event.set()

        # Wait for all modules to complete
        if self.modules:
            await asyncio.gather(*self.modules, return_exceptions=True)

        # Disconnect shared clients
        await self.mqtt_client.disconnect()
        await self.influxdb_client.stop()

        logger.info("Shutdown complete")

    def handle_signal(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        logger = logging.getLogger(__name__)
        logger.info(f"Received signal {signum}")
        asyncio.create_task(self.shutdown())

    async def run(self) -> None:
        """Run the main application loop."""
        logger = logging.getLogger(__name__)

        try:
            # Register signal handlers
            signal.signal(signal.SIGINT, self.handle_signal)
            signal.signal(signal.SIGTERM, self.handle_signal)

            # Initialize modules
            await self.initialize_modules()

            # Start all modules
            await self.start_modules()

            logger.info("Loxone Smart Home is running...")

            # Wait for shutdown
            await self.shutdown_event.wait()

        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            await self.shutdown()
            sys.exit(1)


async def main() -> None:
    """Run the main entry point."""
    # Load environment variables
    load_dotenv()

    # Create and run application
    app = LoxoneSmartHome()
    await app.run()


if __name__ == "__main__":
    asyncio.run(main())

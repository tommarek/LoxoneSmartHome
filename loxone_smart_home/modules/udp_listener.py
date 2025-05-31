"""UDP Listener module - receives data from Loxone and stores in InfluxDB."""

import asyncio
from datetime import datetime
from typing import Optional, Tuple

import pytz

from config.settings import Settings
from modules.base import BaseModule
from utils.influxdb_client import SharedInfluxDBClient


class UDPListener(BaseModule):
    """UDP Listener that receives data from Loxone and stores it in InfluxDB."""

    def __init__(self, influxdb_client: SharedInfluxDBClient, settings: Settings) -> None:
        """Initialize the UDP listener."""
        super().__init__(
            name="UDPListener",
            influxdb_client=influxdb_client,
            settings=settings,
        )
        self.transport: Optional[asyncio.DatagramTransport] = None
        self.protocol: Optional[asyncio.DatagramProtocol] = None
        self.buffer_size = 2048  # Match Rust implementation

    async def start(self) -> None:
        """Start the UDP listener."""
        loop = asyncio.get_event_loop()
        self.transport, self.protocol = await loop.create_datagram_endpoint(
            lambda: UDPProtocol(self),
            local_addr=(
                self.settings.udp_listener.host,
                self.settings.udp_listener.port,
            ),
        )
        self.logger.info(
            f"UDP listener started on {self.settings.udp_listener.host}:"
            f"{self.settings.udp_listener.port}"
        )
        self.logger.info("Starting to accept data.")
        delim = self.settings.udp_listener.delimiter
        self.logger.info(
            f"Expected format: timestamp{delim}measurement_name{delim}value{delim}"
            f"room[optional]{delim}measurement_type[optional]{delim}"
            f"tag1[optional]{delim}tag2[optional]"
        )

    async def stop(self) -> None:
        """Stop the UDP listener."""
        if self.transport:
            self.transport.close()
            self.logger.info("UDP listener stopped")

    async def process_data(self, data: bytes, addr: Tuple[str, int]) -> None:
        """Process received UDP data matching Rust implementation logic."""
        try:
            # Decode the data
            message = data.decode("utf-8").strip()

            # Parse the data structure:
            # timestamp;measurement_name;value;room_name[optional];measurement_type[optional];tag1[optional];tag2[optional]
            parts = message.split(self.settings.udp_listener.delimiter)

            if len(parts) < 3:
                self.logger.error(f"Failed to parse incoming data: not enough fields in {message}")
                return

            # Required fields
            timestamp_str = parts[0]
            measurement_name = (
                parts[1].replace(" ", "_").lower()
            )  # Match Rust: replace spaces, lowercase
            try:
                value = float(parts[2])
            except ValueError as e:
                self.logger.error(f"Failed to parse incoming data: invalid value {parts[2]}: {e}")
                return

            # Optional fields with defaults matching Rust
            room_name = parts[3] if len(parts) > 3 else "_"
            measurement_type = parts[4].strip() if len(parts) > 4 else "default"
            tag1 = parts[5].strip() if len(parts) > 5 else "_"
            tag2 = parts[6].strip() if len(parts) > 6 else "_"

            # Convert timestamp from Prague to UTC (matching Rust behavior)
            try:
                prague_tz = pytz.timezone(self.settings.udp_listener.timezone)
                local_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                local_time = prague_tz.localize(local_time)
                utc_time = local_time.astimezone(pytz.UTC)
            except Exception as e:
                self.logger.error(f"Failed to parse timestamp {timestamp_str}: {e}")
                return

            # Log the parsed data (matching Rust debug output)
            self.logger.debug(
                f"{int(utc_time.timestamp() * 1000)};{measurement_name};{value};"
                f"{room_name};{measurement_type};{tag1};{tag2}"
            )

            # Write to InfluxDB
            # Note: In Rust, measurement_type is used as the measurement name
            if self.influxdb_client is None:
                self.logger.error("InfluxDB client not available")
                return
            await self.influxdb_client.write_point(
                bucket=self.settings.influxdb.bucket_loxone,
                measurement=measurement_type,  # Use measurement_type as measurement (like Rust)
                fields={measurement_name: value},  # measurement_name becomes the field name
                tags={
                    "room": room_name,
                    "tag1": tag1,
                    "tag2": tag2,
                },
                timestamp=utc_time,
            )

            # Log detailed information about what was stored
            self.logger.info(
                f"Stored: {measurement_type}.{measurement_name}={value} "
                f"(room: {room_name}, from: {addr[0]}, time: {utc_time.strftime('%H:%M:%S')})"
            )

        except Exception as e:
            self.logger.error(f"Failed to process incoming data: {e}", exc_info=True)


class UDPProtocol(asyncio.DatagramProtocol):
    """UDP protocol handler."""

    def __init__(self, listener: UDPListener) -> None:
        """Initialize the protocol."""
        self.listener = listener

    def datagram_received(self, data: bytes, addr: Tuple[str, int]) -> None:
        """Handle received datagram."""
        asyncio.create_task(self.listener.process_data(data, addr))

"""UDP Listener module - receives data from Loxone and stores in InfluxDB."""

import asyncio
from collections import defaultdict
from datetime import datetime
from typing import Dict, Optional, Tuple

import pytz
from config.settings import Settings
from modules.base import BaseModule
from utils.async_influxdb_client import AsyncInfluxDBClient


class UDPListener(BaseModule):
    """UDP Listener that receives data from Loxone and stores it in InfluxDB."""

    def __init__(
        self, influxdb_client: AsyncInfluxDBClient, settings: Settings
    ) -> None:
        """Initialize the UDP listener."""
        super().__init__(
            name="UDPListener",
            service_name="UDP",
            influxdb_client=influxdb_client,
            settings=settings,
        )
        self.transport: Optional[asyncio.DatagramTransport] = None
        self.protocol: Optional[asyncio.DatagramProtocol] = None
        self.buffer_size = 2048  # Match Rust implementation

        # Statistics for periodic logging
        self.stats_lock = asyncio.Lock()
        self.packet_count = 0
        self.error_count = 0
        self.measurement_counts: Dict[str, int] = defaultdict(int)
        self.room_counts: Dict[str, int] = defaultdict(int)
        self.last_stats_log = datetime.now()
        self.stats_interval = 60  # Log stats every 60 seconds
        self._stats_task: Optional[asyncio.Task[None]] = None

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

        # Start statistics logging task
        self._stats_task = asyncio.create_task(self._stats_logger())

    async def stop(self) -> None:
        """Stop the UDP listener."""
        # Stop statistics task
        if self._stats_task and not self._stats_task.done():
            self._stats_task.cancel()
            try:
                await self._stats_task
            except asyncio.CancelledError:
                pass

        if self.transport:
            self.transport.close()
            # Log final statistics
            await self._log_statistics(final=True)
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
                self.logger.error(
                    f"Failed to parse incoming data: not enough fields in {message}"
                )
                async with self.stats_lock:
                    self.error_count += 1
                return

            # Required fields
            timestamp_str = parts[0]
            measurement_name = (
                parts[1].replace(" ", "_").lower()
            )  # Match Rust: replace spaces, lowercase
            try:
                value = float(parts[2])
            except ValueError as e:
                self.logger.error(
                    f"Failed to parse incoming data: invalid value {parts[2]}: {e}"
                )
                async with self.stats_lock:
                    self.error_count += 1
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
                async with self.stats_lock:
                    self.error_count += 1
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
                fields={
                    measurement_name: value
                },  # measurement_name becomes the field name
                tags={
                    "room": room_name,
                    "tag1": tag1,
                    "tag2": tag2,
                },
                timestamp=utc_time,
            )

            # Update statistics instead of logging each packet
            async with self.stats_lock:
                self.packet_count += 1
                self.measurement_counts[f"{measurement_type}.{measurement_name}"] += 1
                self.room_counts[room_name] += 1

            # Log detailed information only at DEBUG level
            self.logger.debug(
                f"Stored: {measurement_type}.{measurement_name}={value} "
                f"(room: {room_name}, from: {addr[0]}, time: {utc_time.strftime('%H:%M:%S')})"
            )

        except Exception as e:
            self.logger.error(f"Failed to process incoming data: {e}", exc_info=True)
            async with self.stats_lock:
                self.error_count += 1

    async def _stats_logger(self) -> None:
        """Periodically log statistics summary."""
        while self._running:
            try:
                await asyncio.sleep(self.stats_interval)
                await self._log_statistics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in stats logger: {e}")

    async def _log_statistics(self, final: bool = False) -> None:
        """Log current statistics."""
        async with self.stats_lock:
            if self.packet_count == 0 and not final:
                return  # No activity to report

            # Calculate time since last log
            now = datetime.now()
            time_elapsed = (now - self.last_stats_log).total_seconds()
            rate = self.packet_count / time_elapsed if time_elapsed > 0 else 0

            status = "Final" if final else "Periodic"
            self.logger.info(
                f"{status} UDP Stats: {self.packet_count} packets processed "
                f"({rate:.1f}/sec), {self.error_count} errors"
            )

            # Log top 5 most active measurements
            if self.measurement_counts:
                top_measurements = sorted(
                    self.measurement_counts.items(), key=lambda x: x[1], reverse=True
                )[:5]
                measurement_summary = ", ".join(
                    [f"{name}({count})" for name, count in top_measurements]
                )
                self.logger.info(f"Top measurements: {measurement_summary}")

            # Log room distribution if more than one room
            if len(self.room_counts) > 1:
                room_summary = ", ".join(
                    [f"{room}({count})" for room, count in self.room_counts.items()]
                )
                self.logger.info(f"Room distribution: {room_summary}")

            if not final:
                # Reset counters for next period
                self.packet_count = 0
                self.error_count = 0
                self.measurement_counts.clear()
                self.room_counts.clear()
                self.last_stats_log = now


class UDPProtocol(asyncio.DatagramProtocol):
    """UDP protocol handler."""

    def __init__(self, listener: UDPListener) -> None:
        """Initialize the protocol."""
        self.listener = listener

    def datagram_received(self, data: bytes, addr: Tuple[str, int]) -> None:
        """Handle received datagram."""
        asyncio.create_task(self.listener.process_data(data, addr))

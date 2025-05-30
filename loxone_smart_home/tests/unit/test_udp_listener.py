"""Test the UDP listener module."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytz

from loxone_smart_home.config.settings import Settings
from loxone_smart_home.modules.udp_listener import UDPListener, UDPProtocol
from loxone_smart_home.utils.influxdb_client import SharedInfluxDBClient


class TestUDPListener:
    """Test the UDPListener class."""

    @pytest.fixture
    def settings(self) -> Settings:
        """Create test settings."""
        with patch.dict("os.environ", {"INFLUXDB_TOKEN": "test-token"}):
            return Settings(influxdb_token="test-token")

    @pytest.fixture
    def mock_influxdb(self) -> AsyncMock:
        """Create mock InfluxDB client."""
        client = AsyncMock()
        client.write_point = AsyncMock()
        return client

    @pytest.fixture
    def udp_listener(self, mock_influxdb: AsyncMock, settings: Settings) -> UDPListener:
        """Create UDP listener instance."""
        return UDPListener(mock_influxdb, settings)

    @pytest.mark.asyncio
    async def test_start(self, udp_listener: UDPListener) -> None:
        """Test starting the UDP listener."""
        with patch("asyncio.get_event_loop") as mock_loop:
            mock_transport = AsyncMock()
            mock_protocol = AsyncMock()
            mock_loop.return_value.create_datagram_endpoint = AsyncMock(
                return_value=(mock_transport, mock_protocol)
            )

            await udp_listener.start()

            assert udp_listener.transport is not None
            mock_loop.return_value.create_datagram_endpoint.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop(self, udp_listener: UDPListener) -> None:
        """Test stopping the UDP listener."""
        mock_transport = MagicMock()
        udp_listener.transport = mock_transport

        await udp_listener.stop()

        mock_transport.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_valid_data(
        self, udp_listener: UDPListener, mock_influxdb: SharedInfluxDBClient
    ) -> None:
        """Test processing valid UDP data."""
        data = b"2024-01-15 10:30:00;temperature;21.5;living_room;sensor;tag1_value;tag2_value"
        addr = ("192.168.1.100", 12345)

        await udp_listener.process_data(data, addr)

        # Verify InfluxDB write was called
        mock_influxdb.write_point.assert_called_once()  # type: ignore[attr-defined]
        call_args = mock_influxdb.write_point.call_args  # type: ignore[attr-defined]

        assert call_args.kwargs["bucket"] == "loxone"
        assert call_args.kwargs["measurement"] == "sensor"  # measurement_type is the measurement
        assert call_args.kwargs["fields"] == {"temperature": 21.5}  # measurement_name is the field
        assert call_args.kwargs["tags"]["room"] == "living_room"
        assert call_args.kwargs["tags"]["tag1"] == "tag1_value"
        assert call_args.kwargs["tags"]["tag2"] == "tag2_value"

    @pytest.mark.asyncio
    async def test_process_minimal_data(
        self, udp_listener: UDPListener, mock_influxdb: SharedInfluxDBClient
    ) -> None:
        """Test processing minimal UDP data (only required fields)."""
        data = b"2024-01-15 10:30:00;temperature;21.5"
        addr = ("192.168.1.100", 12345)

        await udp_listener.process_data(data, addr)

        mock_influxdb.write_point.assert_called_once()  # type: ignore[attr-defined]
        call_args = mock_influxdb.write_point.call_args  # type: ignore[attr-defined]

        assert call_args.kwargs["measurement"] == "default"  # Default measurement_type
        assert call_args.kwargs["fields"] == {"temperature": 21.5}
        assert call_args.kwargs["tags"]["room"] == "_"  # Default room
        assert call_args.kwargs["tags"]["tag1"] == "_"  # Default tag1
        assert call_args.kwargs["tags"]["tag2"] == "_"  # Default tag2

    @pytest.mark.asyncio
    async def test_process_invalid_data_format(
        self, udp_listener: UDPListener, mock_influxdb: SharedInfluxDBClient
    ) -> None:
        """Test processing invalid data format."""
        data = b"invalid;data"  # Not enough fields
        addr = ("192.168.1.100", 12345)

        await udp_listener.process_data(data, addr)

        # Should not write to InfluxDB
        mock_influxdb.write_point.assert_not_called()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_process_invalid_value(
        self, udp_listener: UDPListener, mock_influxdb: SharedInfluxDBClient
    ) -> None:
        """Test processing data with invalid numeric value."""
        data = b"2024-01-15 10:30:00;temperature;not_a_number"
        addr = ("192.168.1.100", 12345)

        await udp_listener.process_data(data, addr)

        # Should not write to InfluxDB
        mock_influxdb.write_point.assert_not_called()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_timezone_conversion(
        self, udp_listener: UDPListener, mock_influxdb: SharedInfluxDBClient
    ) -> None:
        """Test proper timezone conversion from Prague to UTC."""
        data = b"2024-01-15 10:30:00;temperature;21.5"
        addr = ("192.168.1.100", 12345)

        await udp_listener.process_data(data, addr)

        call_args = mock_influxdb.write_point.call_args  # type: ignore[attr-defined]
        timestamp = call_args.kwargs["timestamp"]

        # Verify timestamp is in UTC
        assert timestamp.tzinfo == pytz.UTC

        # Verify correct conversion (Prague is UTC+1 in winter)
        prague_tz = pytz.timezone("Europe/Prague")
        prague_time = prague_tz.localize(datetime(2024, 1, 15, 10, 30, 0))
        expected_utc = prague_time.astimezone(pytz.UTC)
        assert timestamp == expected_utc

    @pytest.mark.asyncio
    async def test_measurement_name_normalization(
        self, udp_listener: UDPListener, mock_influxdb: SharedInfluxDBClient
    ) -> None:
        """Test measurement name normalization (spaces to underscores, lowercase)."""
        data = b"2024-01-15 10:30:00;Room Temperature;21.5;living_room;sensor"
        addr = ("192.168.1.100", 12345)

        await udp_listener.process_data(data, addr)

        call_args = mock_influxdb.write_point.call_args  # type: ignore[attr-defined]
        assert call_args.kwargs["fields"] == {"room_temperature": 21.5}  # Normalized


class TestUDPProtocol:
    """Test the UDPProtocol class."""

    def test_datagram_received(self) -> None:
        """Test datagram reception."""
        mock_listener = AsyncMock()
        protocol = UDPProtocol(mock_listener)

        data = b"test data"
        addr = ("192.168.1.100", 12345)

        with patch("asyncio.create_task") as mock_create_task:
            protocol.datagram_received(data, addr)
            mock_create_task.assert_called_once()

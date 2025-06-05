"""Tests for the OTE price collector module."""

from datetime import datetime, timedelta
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytz

from config.settings import Settings
from modules.ote_price_collector import OTEPriceCollector


@pytest.fixture
def mock_settings() -> Settings:
    """Create mock settings."""
    settings = MagicMock(spec=Settings)
    settings.log_level = "INFO"
    settings.log_timezone = "Europe/Prague"

    # Mock OTE config
    ote_config = MagicMock()
    ote_config.base_url = (
        "https://www.ote-cr.cz/cs/kratkodobe-trhy/elektrina/denni-trh/" "@@chart-data"
    )
    ote_config.time_resolution = "PT60M"
    ote_config.update_hour = 15
    ote_config.update_minute = 0
    ote_config.load_historical_days = 1095
    ote_config.request_delay = 0.5
    ote_config.error_delay = 2.0
    ote_config.eur_czk_rate = 25.0
    settings.ote = ote_config

    return settings


@pytest.fixture
def mock_influxdb_client() -> AsyncMock:
    """Create mock InfluxDB client."""
    client = AsyncMock()
    client.write_point = AsyncMock()
    client.query = AsyncMock(return_value=[])
    return client


@pytest.fixture
def ote_collector(
    mock_influxdb_client: AsyncMock,
    mock_settings: Settings,
) -> OTEPriceCollector:
    """Create OTE collector instance."""
    return OTEPriceCollector(mock_influxdb_client, mock_settings)


@pytest.fixture
def mock_ote_response() -> Dict[str, Any]:
    """Mock OTE API response."""
    return {
        "data": {
            "dataLine": [
                {},
                {
                    "point": [
                        {"x": "1", "y": "45.50"},  # 00:00-01:00
                        {"x": "2", "y": "43.25"},  # 01:00-02:00
                        {"x": "3", "y": "41.00"},  # 02:00-03:00
                        {"x": "4", "y": "42.75"},  # 03:00-04:00
                    ]
                },
            ]
        }
    }


@pytest.mark.asyncio
async def test_fetch_prices_for_date(
    ote_collector: OTEPriceCollector,
    mock_ote_response: Dict[str, Any],
) -> None:
    """Test fetching prices for a specific date."""
    # Mock HTTP session
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=mock_ote_response)

    mock_session = AsyncMock()
    mock_session.get.return_value.__aenter__.return_value = mock_response

    ote_collector._session = mock_session

    # Fetch prices
    prices = await ote_collector._fetch_prices_for_date("2025-06-06")

    # Verify results
    assert prices is not None
    assert len(prices) == 4
    assert prices[("00:00", "01:00")] == 45.50
    assert prices[("01:00", "02:00")] == 43.25
    assert prices[("02:00", "03:00")] == 41.00
    assert prices[("03:00", "04:00")] == 42.75


@pytest.mark.asyncio
async def test_fetch_prices_for_date_http_error(
    ote_collector: OTEPriceCollector,
) -> None:
    """Test handling HTTP errors when fetching prices."""
    # Mock HTTP session with error response
    mock_response = AsyncMock()
    mock_response.status = 404

    mock_session = AsyncMock()
    mock_session.get.return_value.__aenter__.return_value = mock_response

    ote_collector._session = mock_session

    # Fetch prices
    prices = await ote_collector._fetch_prices_for_date("2025-06-06")

    # Verify error handling
    assert prices is None


@pytest.mark.asyncio
async def test_fetch_prices_for_date_invalid_data(
    ote_collector: OTEPriceCollector,
) -> None:
    """Test handling invalid data structure."""
    # Mock HTTP session with invalid response
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"invalid": "data"})

    mock_session = AsyncMock()
    mock_session.get.return_value.__aenter__.return_value = mock_response

    ote_collector._session = mock_session

    # Fetch prices
    prices = await ote_collector._fetch_prices_for_date("2025-06-06")

    # Verify error handling
    assert prices is None


@pytest.mark.asyncio
async def test_store_prices(
    ote_collector: OTEPriceCollector,
    mock_influxdb_client: AsyncMock,
) -> None:
    """Test storing prices in InfluxDB."""
    prices = {
        ("00:00", "01:00"): 45.50,
        ("01:00", "02:00"): 43.25,
    }

    await ote_collector._store_prices(prices, "2025-06-06")

    # Verify InfluxDB writes
    assert mock_influxdb_client.write_point.call_count == 2

    # Check first call
    first_call = mock_influxdb_client.write_point.call_args_list[0]
    assert first_call.kwargs["bucket"] == "ote_prices"
    assert first_call.kwargs["measurement"] == "electricity_prices"
    assert first_call.kwargs["fields"]["price"] == 45.50
    assert first_call.kwargs["fields"]["price_czk_kwh"] == 45.50 * 25.0 / 1000
    assert first_call.kwargs["tags"]["market"] == "day_ahead"
    assert first_call.kwargs["tags"]["currency"] == "EUR_MWh"
    assert first_call.kwargs["tags"]["source"] == "OTE"
    assert first_call.kwargs["tags"]["resolution"] == "PT60M"

    # Check timestamp conversion
    timestamp = first_call.kwargs["timestamp"]
    assert timestamp.tzinfo == pytz.UTC


@pytest.mark.asyncio
async def test_needs_historical_load_no_data(
    ote_collector: OTEPriceCollector,
    mock_influxdb_client: AsyncMock,
) -> None:
    """Test checking if historical load is needed when no data exists."""
    mock_influxdb_client.query = AsyncMock(return_value=[])

    needs_load = await ote_collector._needs_historical_load()

    assert needs_load is True


@pytest.mark.asyncio
async def test_needs_historical_load_recent_data(
    ote_collector: OTEPriceCollector,
    mock_influxdb_client: AsyncMock,
) -> None:
    """Test checking if historical load is needed when recent data exists."""
    # Mock query result with recent timestamp
    mock_table = MagicMock()
    mock_record = MagicMock()
    mock_record.get_time.return_value = datetime.now(pytz.UTC) - timedelta(hours=12)
    mock_table.records = [mock_record]
    mock_influxdb_client.query = AsyncMock(return_value=[mock_table])

    needs_load = await ote_collector._needs_historical_load()

    assert needs_load is False


@pytest.mark.asyncio
async def test_needs_historical_load_old_data(
    ote_collector: OTEPriceCollector,
    mock_influxdb_client: AsyncMock,
) -> None:
    """Test checking if historical load is needed when old data exists."""
    # Mock query result with old timestamp
    mock_table = MagicMock()
    mock_record = MagicMock()
    mock_record.get_time.return_value = datetime.now(pytz.UTC) - timedelta(days=3)
    mock_table.records = [mock_record]
    mock_influxdb_client.query = AsyncMock(return_value=[mock_table])

    needs_load = await ote_collector._needs_historical_load()

    assert needs_load is True


@pytest.mark.asyncio
async def test_has_data_for_date_with_data(
    ote_collector: OTEPriceCollector,
    mock_influxdb_client: AsyncMock,
) -> None:
    """Test checking if data exists for a specific date."""
    # Mock query result with count
    mock_table = MagicMock()
    mock_record = MagicMock()
    mock_record.get_value.return_value = 24  # 24 hours of data
    mock_table.records = [mock_record]
    mock_influxdb_client.query = AsyncMock(return_value=[mock_table])

    has_data = await ote_collector._has_data_for_date("2025-06-06")

    assert has_data is True


@pytest.mark.asyncio
async def test_has_data_for_date_insufficient_data(
    ote_collector: OTEPriceCollector,
    mock_influxdb_client: AsyncMock,
) -> None:
    """Test checking if data exists for a specific date with insufficient data."""
    # Mock query result with low count
    mock_table = MagicMock()
    mock_record = MagicMock()
    mock_record.get_value.return_value = 10  # Only 10 hours of data
    mock_table.records = [mock_record]
    mock_influxdb_client.query = AsyncMock(return_value=[mock_table])

    has_data = await ote_collector._has_data_for_date("2025-06-06")

    assert has_data is False


@pytest.mark.asyncio
async def test_get_prices_for_date(
    ote_collector: OTEPriceCollector,
    mock_influxdb_client: AsyncMock,
) -> None:
    """Test getting prices for a specific date from database."""
    # Mock query result with price data
    prague_tz = pytz.timezone("Europe/Prague")

    mock_table = MagicMock()
    mock_records = []

    # Create mock records for a few hours
    for hour in range(3):
        mock_record = MagicMock()
        mock_time = prague_tz.localize(datetime(2025, 6, 6, hour, 0)).astimezone(pytz.UTC)
        mock_record.get_time.return_value = mock_time
        mock_record.get_value.return_value = 50.0 + hour  # Different price
        mock_records.append(mock_record)

    mock_table.records = mock_records
    mock_influxdb_client.query = AsyncMock(return_value=[mock_table])

    date = datetime(2025, 6, 6)
    prices = await ote_collector.get_prices_for_date(date)

    # Verify results
    assert prices is not None
    assert len(prices) == 3
    assert prices["00:00"] == 50.0
    assert prices["01:00"] == 51.0
    assert prices["02:00"] == 52.0


@pytest.mark.asyncio
async def test_get_prices_for_date_no_data(
    ote_collector: OTEPriceCollector,
    mock_influxdb_client: AsyncMock,
) -> None:
    """Test getting prices when no data exists."""
    mock_influxdb_client.query = AsyncMock(return_value=[])

    date = datetime(2025, 6, 6)
    prices = await ote_collector.get_prices_for_date(date)

    assert prices is None


@pytest.mark.asyncio
async def test_start_with_historical_load(
    ote_collector: OTEPriceCollector,
    mock_influxdb_client: AsyncMock,
) -> None:
    """Test starting the collector with historical data load."""
    # Mock that we need historical load
    ote_collector._needs_historical_load = AsyncMock(  # type: ignore[attr-defined]
        return_value=True
    )
    ote_collector._load_historical_data = AsyncMock()  # type: ignore[attr-defined]

    with patch("aiohttp.ClientSession") as mock_session_class:
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session

        await ote_collector.start()

        # Verify session creation
        mock_session_class.assert_called_once()

        # Verify historical load was called
        ote_collector._load_historical_data.assert_called_once()  # type: ignore[attr-defined]  # noqa: E501

        # Verify daily update task was created
        assert ote_collector._daily_update_task is not None


@pytest.mark.asyncio
async def test_start_without_historical_load(
    ote_collector: OTEPriceCollector,
    mock_influxdb_client: AsyncMock,
) -> None:
    """Test starting the collector without historical data load."""
    # Mock that we don't need historical load
    ote_collector._needs_historical_load = AsyncMock(  # type: ignore[attr-defined]
        return_value=False
    )
    ote_collector._load_historical_data = AsyncMock()  # type: ignore[attr-defined]

    with patch("aiohttp.ClientSession") as mock_session_class:
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session

        await ote_collector.start()

        # Verify session creation
        mock_session_class.assert_called_once()

        # Verify historical load was NOT called
        ote_collector._load_historical_data.assert_not_called()  # type: ignore[attr-defined]  # noqa: E501

        # Verify daily update task was created
        assert ote_collector._daily_update_task is not None


@pytest.mark.asyncio
async def test_stop(
    ote_collector: OTEPriceCollector,
    mock_influxdb_client: AsyncMock,
) -> None:
    """Test stopping the collector."""
    # Create mock session and task
    mock_session = AsyncMock()
    mock_task = AsyncMock()
    mock_task.done.return_value = False

    ote_collector._session = mock_session
    ote_collector._daily_update_task = mock_task

    await ote_collector.stop()

    # Verify task cancellation
    mock_task.cancel.assert_called_once()

    # Verify session closure
    mock_session.close.assert_called_once()


@pytest.mark.asyncio
async def test_update_today_and_tomorrow_before_2pm(
    ote_collector: OTEPriceCollector,
    mock_influxdb_client: AsyncMock,
) -> None:
    """Test updating prices before 2 PM (only today)."""
    # Mock current time before 2 PM
    prague_tz = pytz.timezone("Europe/Prague")
    mock_now = prague_tz.localize(datetime(2025, 6, 6, 13, 0))  # 1 PM

    ote_collector._fetch_prices_for_date = AsyncMock(  # type: ignore[attr-defined]
        return_value={("00:00", "01:00"): 50.0}
    )
    ote_collector._store_prices = AsyncMock()  # type: ignore[attr-defined]

    with patch("modules.ote_price_collector.datetime") as mock_datetime:
        mock_datetime.now.return_value = mock_now
        mock_datetime.strftime = datetime.strftime

        await ote_collector._update_today_and_tomorrow()

        # Verify only today's prices were fetched
        assert ote_collector._fetch_prices_for_date.call_count == 1  # type: ignore[attr-defined]
        ote_collector._fetch_prices_for_date.assert_called_with(  # type: ignore[attr-defined]
            "2025-06-06"
        )


@pytest.mark.asyncio
async def test_update_today_and_tomorrow_after_2pm(
    ote_collector: OTEPriceCollector,
    mock_influxdb_client: AsyncMock,
) -> None:
    """Test updating prices after 2 PM (today and tomorrow)."""
    # Mock current time after 2 PM
    prague_tz = pytz.timezone("Europe/Prague")
    mock_now = prague_tz.localize(datetime(2025, 6, 6, 15, 0))  # 3 PM

    ote_collector._fetch_prices_for_date = AsyncMock(  # type: ignore[attr-defined]
        return_value={("00:00", "01:00"): 50.0}
    )
    ote_collector._store_prices = AsyncMock()  # type: ignore[attr-defined]

    with patch("modules.ote_price_collector.datetime") as mock_datetime:
        mock_datetime.now.return_value = mock_now
        mock_datetime.strftime = datetime.strftime

        await ote_collector._update_today_and_tomorrow()

        # Verify both today's and tomorrow's prices were fetched
        assert ote_collector._fetch_prices_for_date.call_count == 2  # type: ignore[attr-defined]
        calls = ote_collector._fetch_prices_for_date.call_args_list  # type: ignore[attr-defined]
        assert calls[0][0][0] == "2025-06-06"  # Today
        assert calls[1][0][0] == "2025-06-07"  # Tomorrow

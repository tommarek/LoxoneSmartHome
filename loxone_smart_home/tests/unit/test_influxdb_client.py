"""Tests for AsyncInfluxDBClient failure-handling paths.

Covers the branches added on the battery-optimization branch: the bounded
re-queue of points after a terminal write failure (OOM protection), and the
hard query timeout. Both are correctness-sensitive failure paths with no other
coverage.
"""
import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from config.settings import Settings
from utils.async_influxdb_client import AsyncInfluxDBClient


@pytest.fixture
def client() -> AsyncInfluxDBClient:
    with patch.dict("os.environ", {"INFLUXDB_TOKEN": "test-token"}):
        return AsyncInfluxDBClient(Settings(influxdb_token="test-token"))


@pytest.mark.asyncio
async def test_write_failure_requeues_newest_points_bounded(client) -> None:
    """On terminal failure the newest points are re-queued up to the buffer
    bound; the overflow (oldest) is dropped with a warning."""
    client.max_buffer_size = 5
    # Pre-fill with 3 existing points → room for 2 more.
    for i in range(3):
        client.write_buffer.append(("solar", f"existing{i}"))

    client._get_client = AsyncMock(side_effect=Exception("InfluxDB down"))

    points = ["p0", "p1", "p2", "p3"]  # 4 points, only 2 fit
    with patch.object(client.logger, "warning") as mock_warning:
        await client._write_batch_with_retry("solar", points, max_retries=1)

    # Buffer is capped at max_buffer_size and the NEWEST two points (p2, p3) are
    # re-queued at the front; p0/p1 (oldest of the batch) are dropped.
    assert len(client.write_buffer) == 5
    front = [client.write_buffer[0], client.write_buffer[1]]
    assert front == [("solar", "p2"), ("solar", "p3")]
    assert any("Dropped 2" in str(c) for c in mock_warning.call_args_list)


@pytest.mark.asyncio
async def test_write_failure_full_buffer_drops_all(client) -> None:
    """When the buffer is already full, a failed batch is dropped entirely
    (no unbounded growth)."""
    client.max_buffer_size = 3
    for i in range(3):
        client.write_buffer.append(("solar", f"existing{i}"))

    client._get_client = AsyncMock(side_effect=Exception("InfluxDB down"))

    with patch.object(client.logger, "warning") as mock_warning:
        await client._write_batch_with_retry("solar", ["p0", "p1"], max_retries=1)

    # Nothing added; buffer unchanged.
    assert len(client.write_buffer) == 3
    assert all(p[1].startswith("existing") for p in client.write_buffer)
    assert any("InfluxDB unreachable" in str(c) for c in mock_warning.call_args_list)


@pytest.mark.asyncio
async def test_query_times_out(client) -> None:
    """query() is hard-bounded by asyncio.wait_for and re-raises TimeoutError."""

    class _HangingQueryApi:
        def query(self, query, org):  # noqa: ARG002
            async def _hang():
                await asyncio.sleep(5)
            return _hang()

    class _FakeClient:
        def query_api(self):
            return _HangingQueryApi()

    client._get_client = AsyncMock(return_value=_FakeClient())

    with pytest.raises(asyncio.TimeoutError):
        await client.query("from(bucket: \"x\")", timeout=0.05)

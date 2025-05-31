"""Async InfluxDB client with connection pooling and batch processing."""

import asyncio
import logging
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

from influxdb_client import Point, WritePrecision
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync

from config.settings import Settings


class AsyncInfluxDBClient:
    """Async InfluxDB client with connection pooling, batching, and retry logic."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the async InfluxDB client."""
        self.settings = settings
        self.logger = logging.getLogger(f"{__name__}.AsyncInfluxDB")

        # Connection pool
        self.client_pool: List[InfluxDBClientAsync] = []
        self.pool_size = 5  # Configurable pool size
        self.pool_lock = asyncio.Lock()

        # Write buffer for batching
        self.write_buffer: Deque[tuple[str, Point]] = deque()
        self.buffer_lock = asyncio.Lock()
        self.max_buffer_size = 5000
        self.flush_interval = 1.0  # seconds

        # Background tasks
        self._flush_task: Optional[asyncio.Task[None]] = None
        self._running = False

        # Metrics
        self.writes_queued = 0
        self.writes_completed = 0
        self.write_errors = 0

    async def start(self) -> None:
        """Start the async InfluxDB client and background tasks."""
        self._running = True

        # Initialize connection pool
        await self._initialize_pool()

        # Start background flush task
        self._flush_task = asyncio.create_task(self._flush_loop())

        self.logger.info(f"Async InfluxDB client started with pool size {self.pool_size}")

    async def stop(self) -> None:
        """Stop the client and flush remaining data."""
        if not self._running:
            return  # Already stopped
        self._running = False

        # Flush any remaining data
        try:
            await self._flush_buffer(force=True)
        except Exception:
            pass  # Ignore flush errors during shutdown

        # Cancel background task
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass  # Ignore other exceptions during cleanup

        # Close all connections in pool
        try:
            async with self.pool_lock:
                for client in self.client_pool:
                    try:
                        await client.close()
                    except Exception:
                        pass  # Ignore individual close errors
                self.client_pool.clear()
        except Exception:
            pass  # Ignore pool errors

        self.logger.info(
            f"Async InfluxDB client stopped. "
            f"Writes: queued={self.writes_queued}, "
            f"completed={self.writes_completed}, errors={self.write_errors}"
        )

    async def _initialize_pool(self) -> None:
        """Initialize the connection pool."""
        async with self.pool_lock:
            for _ in range(self.pool_size):
                client = InfluxDBClientAsync(
                    url=self.settings.influxdb.url,
                    token=self.settings.influxdb.token,
                    org=self.settings.influxdb.org,
                )
                self.client_pool.append(client)

    async def _get_client(self) -> InfluxDBClientAsync:
        """Get a client from the pool."""
        async with self.pool_lock:
            if not self.client_pool:
                raise RuntimeError("No clients available in pool")
            # Simple round-robin
            client = self.client_pool.pop(0)
            self.client_pool.append(client)
            return client

    async def write_point(
        self,
        bucket: str,
        measurement: str,
        fields: Dict[str, Any],
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Queue a point for writing to InfluxDB."""
        point = Point(measurement)

        # Add tags
        if tags:
            for key, value in tags.items():
                point = point.tag(key, value)

        # Add fields
        for key, value in fields.items():
            point = point.field(key, value)

        # Add timestamp
        if timestamp:
            point = point.time(timestamp, WritePrecision.NS)

        # Add to buffer
        async with self.buffer_lock:
            if len(self.write_buffer) >= self.max_buffer_size:
                # Buffer full, flush immediately
                await self._flush_buffer()

            self.write_buffer.append((bucket, point))
            self.writes_queued += 1

        self.logger.debug(
            f"Queued point for {bucket}/{measurement}, buffer size: {len(self.write_buffer)}"
        )

    async def _flush_loop(self) -> None:
        """Background task to periodically flush the write buffer."""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_buffer()
            except Exception as e:
                self.logger.error(f"Error in flush loop: {e}", exc_info=True)

    async def _flush_buffer(self, force: bool = False) -> None:
        """Flush the write buffer to InfluxDB."""
        async with self.buffer_lock:
            if not self.write_buffer and not force:
                return

            # Group points by bucket
            bucket_points: Dict[str, List[Point]] = {}
            while self.write_buffer:
                bucket, point = self.write_buffer.popleft()
                if bucket not in bucket_points:
                    bucket_points[bucket] = []
                bucket_points[bucket].append(point)

        # Write points for each bucket
        for bucket, points in bucket_points.items():
            await self._write_batch_with_retry(bucket, points)

    async def _write_batch_with_retry(
        self, bucket: str, points: List[Point], max_retries: int = 3
    ) -> None:
        """Write a batch of points with retry logic."""
        for attempt in range(max_retries):
            try:
                client = await self._get_client()
                write_api = client.write_api()
                await write_api.write(bucket=bucket, record=points)

                self.writes_completed += len(points)
                self.logger.debug(f"Wrote {len(points)} points to {bucket}")
                return

            except Exception as e:
                self.write_errors += 1
                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff
                    self.logger.warning(
                        f"Failed to write batch to {bucket}, " f"retrying in {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(
                        f"Failed to write batch to {bucket} after " f"{max_retries} attempts: {e}"
                    )
                    # Could implement dead letter queue here

    async def query(self, query: str) -> Any:
        """Execute a query against InfluxDB."""
        client = await self._get_client()
        try:
            return await client.query_api().query(query=query, org=self.settings.influxdb.org)
        except Exception as e:
            self.logger.error(f"Failed to query InfluxDB: {e}")
            raise

"""Shared InfluxDB client for all modules."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from config.settings import Settings
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS


class SharedInfluxDBClient:
    """Shared InfluxDB client that manages a single connection for all modules."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the InfluxDB client."""
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.client: Optional[InfluxDBClient] = None
        self.write_api: Optional[Any] = None
        self._connect()

    def _connect(self) -> None:
        """Connect to InfluxDB."""
        try:
            self.client = InfluxDBClient(
                url=self.settings.influxdb.url,
                token=self.settings.influxdb.token,
                org=self.settings.influxdb.org,
            )
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            self.logger.info(f"Connected to InfluxDB at {self.settings.influxdb.url}")
        except Exception as e:
            self.logger.error(f"Failed to connect to InfluxDB: {e}")
            raise

    async def write_point(
        self,
        bucket: str,
        measurement: str,
        fields: Dict[str, Any],
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Write a single point to InfluxDB."""
        if not self.write_api:
            raise RuntimeError("InfluxDB client not connected")

        try:
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

            self.write_api.write(bucket=bucket, record=point)
            self.logger.debug(f"Wrote point to {bucket}/{measurement}")
        except Exception as e:
            self.logger.error(f"Failed to write to InfluxDB: {e}")
            raise

    async def write_points(self, bucket: str, points: List[Point]) -> None:
        """Write multiple points to InfluxDB."""
        if not self.write_api:
            raise RuntimeError("InfluxDB client not connected")

        try:
            self.write_api.write(bucket=bucket, record=points)
            self.logger.debug(f"Wrote {len(points)} points to {bucket}")
        except Exception as e:
            self.logger.error(f"Failed to write batch to InfluxDB: {e}")
            raise

    async def query(self, query: str) -> Any:
        """Execute a query against InfluxDB."""
        if not self.client:
            raise RuntimeError("InfluxDB client not connected")

        try:
            query_api = self.client.query_api()
            result = query_api.query(query=query, org=self.settings.influxdb.org)
            return result
        except Exception as e:
            self.logger.error(f"Failed to query InfluxDB: {e}")
            raise

    async def close(self) -> None:
        """Close the InfluxDB connection."""
        if self.client:
            self.client.close()
            self.logger.info("Closed InfluxDB connection")

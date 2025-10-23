"""Web service configuration."""

from typing import List

from pydantic import BaseModel, Field


class WebConfig(BaseModel):
    """Web service configuration."""

    # Server settings
    enabled: bool = Field(default=True, description="Enable web service")
    host: str = Field(default="0.0.0.0", description="Bind address")
    port: int = Field(default=8080, description="Port number")

    # CORS settings
    cors_origins: List[str] = Field(
        default=["*"],
        description="Allowed CORS origins (* for all)"
    )

    # Cache settings
    cache_ttl_current: int = Field(
        default=10,
        description="Cache TTL for current data in seconds"
    )
    cache_ttl_historical: int = Field(
        default=300,
        description="Cache TTL for historical data in seconds (5 minutes)"
    )
    cache_ttl_analytics: int = Field(
        default=3600,
        description="Cache TTL for analytics in seconds (1 hour)"
    )

    # WebSocket settings
    websocket_interval: int = Field(
        default=5,
        description="WebSocket broadcast interval in seconds"
    )

    # Aggregation settings
    aggregation_enabled: bool = Field(
        default=True,
        description="Enable data aggregation service"
    )
    aggregation_15min: bool = Field(
        default=True,
        description="Enable 15-minute aggregation"
    )
    aggregation_hourly: bool = Field(
        default=True,
        description="Enable hourly aggregation"
    )
    aggregation_daily: bool = Field(
        default=True,
        description="Enable daily aggregation"
    )

    # Feature flags
    enable_api_docs: bool = Field(
        default=True,
        description="Enable OpenAPI documentation at /docs"
    )
    enable_metrics: bool = Field(
        default=False,
        description="Enable Prometheus metrics endpoint"
    )
    enable_export: bool = Field(
        default=True,
        description="Enable data export features (CSV, PDF)"
    )

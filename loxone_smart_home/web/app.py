"""FastAPI web service for Loxone Smart Home monitoring and analytics."""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from config.settings import Settings
from utils.async_influxdb_client import AsyncInfluxDBClient
from utils.async_mqtt_client import AsyncMQTTClient

from .api import energy, prices, weather, analytics, websocket
from .auth import RequireAuth
from .services.data_aggregator import DataAggregator
from .services.cache import CacheService


logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"


class WebService:
    """Web service for monitoring and analytics."""

    def __init__(
        self,
        mqtt_client: Optional[AsyncMQTTClient] = None,
        influxdb_client: Optional[AsyncInfluxDBClient] = None,
        settings: Optional[Settings] = None
    ):
        """Initialize web service."""
        self.mqtt_client = mqtt_client
        self.influxdb_client = influxdb_client
        self.settings = settings or Settings(influxdb_token="")

        # Initialize services
        self.cache = CacheService()
        self.aggregator = DataAggregator(influxdb_client, self.cache)

        # WebSocket connections manager
        self.websocket_manager = websocket.ConnectionManager()

        # Background tasks
        self._aggregation_task: Optional[asyncio.Task[None]] = None
        self._websocket_broadcast_task: Optional[asyncio.Task[None]] = None

    async def start(self) -> None:
        """Start background services."""
        # Start aggregation service
        self._aggregation_task = asyncio.create_task(self.aggregator.run())

        # Start WebSocket broadcast service
        self._websocket_broadcast_task = asyncio.create_task(
            self._broadcast_updates()
        )

        logger.info("Web service started")

    async def stop(self) -> None:
        """Stop background services."""
        # Cancel background tasks
        for task in [self._aggregation_task, self._websocket_broadcast_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        logger.info("Web service stopped")

    async def _broadcast_updates(self) -> None:
        """Broadcast updates to WebSocket clients."""
        while True:
            try:
                await asyncio.sleep(5)  # Update every 5 seconds

                # Get current data from cache
                current_data = await self.cache.get_current_data()

                if current_data and self.websocket_manager.active_connections:
                    await self.websocket_manager.broadcast(current_data)

            except Exception as e:
                logger.error(f"Error broadcasting updates: {e}")


# Global web service instance
web_service: Optional[WebService] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle."""
    global web_service

    # Startup
    settings = Settings(influxdb_token="")

    # Initialize clients (shared with main application if running together)
    mqtt_client = AsyncMQTTClient(settings)
    influxdb_client = AsyncInfluxDBClient(settings)

    # Start clients (they don't have start methods, they start on first use)
    # await mqtt_client.start()
    # await influxdb_client.start()

    # Initialize web service
    web_service = WebService(mqtt_client, influxdb_client, settings)
    await web_service.start()

    # Store in app state for access in endpoints
    app.state.web_service = web_service
    app.state.mqtt_client = mqtt_client
    app.state.influxdb_client = influxdb_client

    yield

    # Shutdown
    if web_service:
        await web_service.stop()

    # Clients don't have stop methods, they clean up on deletion
    # await mqtt_client.stop()
    # await influxdb_client.stop()


# Create FastAPI app
app = FastAPI(
    title="Loxone Smart Home Monitor",
    description="Web service for monitoring and analyzing smart home data",
    version="1.0.0",
    lifespan=lifespan
)


def configure_app() -> None:
    """Configure app with CORS, static files, and routers.

    This is separated to allow initialization after Settings are available.
    """
    # Get settings for configuration
    try:
        settings = Settings(influxdb_token="")
    except Exception:
        # If settings fail to load (e.g., in tests), use defaults
        from config.settings import WebServiceConfig

        class DefaultSettings:
            web_service = WebServiceConfig()
        settings = DefaultSettings()  # type: ignore[assignment]

    # Configure CORS with settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.web_service.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "WebSocket"],
        allow_headers=["*"],
    )

    # Mount static files with proper path
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Include API routers with authentication
    # Note: Authentication is handled via dependency injection. If auth is enabled,
    # API endpoints can use RequireAuth dependency
    app.include_router(
        energy.router,
        prefix="/api/energy",
        tags=["Energy"],
        dependencies=[RequireAuth] if settings.web_service.enable_auth else []
    )
    app.include_router(
        prices.router,
        prefix="/api/prices",
        tags=["Prices"],
        dependencies=[RequireAuth] if settings.web_service.enable_auth else []
    )
    app.include_router(
        weather.router,
        prefix="/api/weather",
        tags=["Weather"],
        dependencies=[RequireAuth] if settings.web_service.enable_auth else []
    )
    app.include_router(
        analytics.router,
        prefix="/api/analytics",
        tags=["Analytics"],
        dependencies=[RequireAuth] if settings.web_service.enable_auth else []
    )


# Configure the app
configure_app()


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    """Serve the main dashboard."""
    template_path = TEMPLATES_DIR / "index.html"
    if template_path.exists():
        with open(template_path, "r") as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="<h1>Dashboard template not found</h1>", status_code=404)


@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time updates."""
    if not web_service:
        await websocket.close()
        return

    await web_service.websocket_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            _ = await websocket.receive_text()
            # Process any client commands if needed
    except WebSocketDisconnect:
        web_service.websocket_manager.disconnect(websocket)


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "services": {
            "mqtt": web_service.mqtt_client is not None if web_service else False,
            "influxdb": web_service.influxdb_client is not None if web_service else False,
            "cache": web_service.cache is not None if web_service else False,
        }
    }

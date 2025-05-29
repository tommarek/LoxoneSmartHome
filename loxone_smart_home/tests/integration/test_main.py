"""Integration tests for the main application."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from loxone_smart_home.main import LoxoneSmartHome, main


class TestLoxoneSmartHome:
    """Test the main LoxoneSmartHome application."""

    @pytest.fixture
    def app(self) -> LoxoneSmartHome:
        """Create application instance."""
        with patch.dict("os.environ", {"INFLUXDB_TOKEN": "test-token"}):
            return LoxoneSmartHome()

    @pytest.mark.asyncio
    async def test_initialize_modules(self, app: LoxoneSmartHome) -> None:
        """Test module initialization."""
        # Mock the MQTT client connection
        app.mqtt_client.connect = AsyncMock()

        await app.initialize_modules()

        app.mqtt_client.connect.assert_called_once()

        # Check modules are initialized based on settings
        if app.settings.modules.udp_listener_enabled:
            assert app.udp_listener is not None
        if app.settings.modules.mqtt_bridge_enabled:
            assert app.mqtt_bridge is not None
        if app.settings.modules.weather_scraper_enabled:
            assert app.weather_scraper is not None
        if app.settings.modules.growatt_controller_enabled:
            assert app.growatt_controller is not None

    @pytest.mark.asyncio
    async def test_start_modules(self, app: LoxoneSmartHome) -> None:
        """Test starting all modules."""
        # Create mock modules
        app.udp_listener = AsyncMock()
        app.udp_listener.run = AsyncMock()
        app.mqtt_bridge = AsyncMock()
        app.mqtt_bridge.run = AsyncMock()
        app.weather_scraper = AsyncMock()
        app.weather_scraper.run = AsyncMock()
        app.growatt_controller = AsyncMock()
        app.growatt_controller.run = AsyncMock()

        await app.start_modules()

        # Verify tasks were created
        assert len(app.modules) == 4

    @pytest.mark.asyncio
    async def test_shutdown(self, app: LoxoneSmartHome) -> None:
        """Test graceful shutdown."""
        # Mock clients
        app.mqtt_client.disconnect = AsyncMock()
        app.influxdb_client.close = AsyncMock()

        # Create a mock task
        mock_task = asyncio.create_task(asyncio.sleep(0))
        app.modules = [mock_task]

        await app.shutdown()

        assert app.shutdown_event.is_set()
        app.mqtt_client.disconnect.assert_called_once()
        app.influxdb_client.close.assert_called_once()

    def test_signal_handling(self, app: LoxoneSmartHome) -> None:
        """Test signal handler."""
        with patch("asyncio.create_task") as mock_create_task:
            app.handle_signal(15, None)  # SIGTERM
            mock_create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_success(self, app: LoxoneSmartHome) -> None:
        """Test successful application run."""
        app.initialize_modules = AsyncMock()
        app.start_modules = AsyncMock()
        app.shutdown = AsyncMock()

        # Set shutdown event after a short delay
        async def set_shutdown():
            await asyncio.sleep(0.1)
            app.shutdown_event.set()

        asyncio.create_task(set_shutdown())

        await app.run()

        app.initialize_modules.assert_called_once()
        app.start_modules.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_error(self, app: LoxoneSmartHome) -> None:
        """Test application run with initialization error."""
        app.initialize_modules = AsyncMock(side_effect=Exception("Init error"))
        app.shutdown = AsyncMock()

        with pytest.raises(SystemExit):
            await app.run()

        app.shutdown.assert_called_once()


@pytest.mark.asyncio
async def test_main_entry_point() -> None:
    """Test the main entry point."""
    with patch("loxone_smart_home.main.load_dotenv") as mock_load_dotenv:
        with patch("loxone_smart_home.main.LoxoneSmartHome") as mock_app_class:
            mock_app = AsyncMock()
            mock_app_class.return_value = mock_app

            await main()

            mock_load_dotenv.assert_called_once()
            mock_app.run.assert_called_once()

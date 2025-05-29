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
    @patch('loxone_smart_home.utils.mqtt_client.SharedMQTTClient.connect')
    async def test_initialize_modules(self, mock_connect: AsyncMock, app: LoxoneSmartHome) -> None:
        """Test module initialization."""
        await app.initialize_modules()

        mock_connect.assert_called_once()

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
        with patch.object(app, 'udp_listener', AsyncMock()) as mock_udp, \
             patch.object(app, 'mqtt_bridge', AsyncMock()) as mock_bridge, \
             patch.object(app, 'weather_scraper', AsyncMock()) as mock_weather, \
             patch.object(app, 'growatt_controller', AsyncMock()) as mock_growatt:

            mock_udp.run = AsyncMock()
            mock_bridge.run = AsyncMock()
            mock_weather.run = AsyncMock()
            mock_growatt.run = AsyncMock()

            await app.start_modules()

            # Verify tasks were created
            assert len(app.modules) == 4

    @pytest.mark.asyncio
    @patch('loxone_smart_home.utils.mqtt_client.SharedMQTTClient.disconnect')
    @patch('loxone_smart_home.utils.influxdb_client.SharedInfluxDBClient.close')
    async def test_shutdown(
        self, mock_influx_close: AsyncMock, mock_mqtt_disconnect: AsyncMock, app: LoxoneSmartHome
    ) -> None:
        """Test graceful shutdown."""
        # Create a mock task
        mock_task = asyncio.create_task(asyncio.sleep(0))
        app.modules = [mock_task]

        await app.shutdown()

        assert app.shutdown_event.is_set()
        mock_mqtt_disconnect.assert_called_once()
        mock_influx_close.assert_called_once()

    def test_signal_handling(self, app: LoxoneSmartHome) -> None:
        """Test signal handler."""
        with patch("asyncio.create_task") as mock_create_task:
            app.handle_signal(15, None)  # SIGTERM
            mock_create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_success(self, app: LoxoneSmartHome) -> None:
        """Test successful application run."""
        with patch.object(app, 'initialize_modules', new_callable=AsyncMock) as mock_init, \
             patch.object(app, 'start_modules', new_callable=AsyncMock) as mock_start:

            # Set shutdown event after a short delay
            async def set_shutdown() -> None:
                await asyncio.sleep(0.1)
                app.shutdown_event.set()

            asyncio.create_task(set_shutdown())

            await app.run()

            mock_init.assert_called_once()
            mock_start.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_error(self, app: LoxoneSmartHome) -> None:
        """Test application run with initialization error."""
        with patch.object(app, 'initialize_modules', new_callable=AsyncMock) as mock_init, \
             patch.object(app, 'shutdown', new_callable=AsyncMock) as mock_shutdown:

            mock_init.side_effect = Exception("Init error")

            with pytest.raises(SystemExit):
                await app.run()

            mock_shutdown.assert_called_once()


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

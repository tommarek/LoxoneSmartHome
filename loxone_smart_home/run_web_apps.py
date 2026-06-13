#!/usr/bin/env python3
"""Standalone web container entry point.

Runs ONLY the web-facing apps — no controller, no async modules — so it can be
restarted freely while the main service (controller, models, inverter) keeps
running:

- the monitoring dashboard PAGES on port 5555 (HTML/JS/PWA assets + the
  settings editor), with everything under ``/api/`` reverse-proxied to the main
  container's controller-backed API app (``DASHBOARD_API_UPSTREAM``);
- optionally the FastAPI web service on port 8080 when WEB_SERVICE_ENABLED=true
  (off by default, as in production).

The split keeps all controller-coupled logic in the main process; this process
only serves static pages and proxies API calls, so editing the UI never bounces
the controller. See modules/growatt/dashboard.py (create_pages_app).
"""

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Where the controller-backed API app lives (the main container). Defaults to
# the compose service name + internal API port.
DEFAULT_API_UPSTREAM = "http://loxone_smart_home:5556"


def _maybe_start_fastapi() -> None:
    """Start the FastAPI web service in a child process if enabled."""
    from config.settings import Settings

    settings = Settings(influxdb_token=os.getenv("INFLUXDB_TOKEN", ""))
    if not settings.web_service.enabled:
        return
    from multiprocessing import Process

    import uvicorn

    def _run() -> None:
        uvicorn.run(
            "web.app:app",
            host=settings.web_service.host,
            port=settings.web_service.port,
            log_level="info",
            access_log=False,
            reload=False,
        )

    proc = Process(target=_run, daemon=True)
    proc.start()
    logging.getLogger(__name__).info(
        f"FastAPI web service started on port {settings.web_service.port}"
    )


async def _serve_pages() -> None:
    from modules.growatt.dashboard import start_pages_dashboard

    api_upstream = os.getenv("DASHBOARD_API_UPSTREAM", DEFAULT_API_UPSTREAM)
    port = int(os.getenv("DASHBOARD_PAGES_PORT", "5555"))
    runner = await start_pages_dashboard(api_upstream, port=port)
    logging.getLogger(__name__).info(
        f"Dashboard pages on :{port}, proxying /api -> {api_upstream}"
    )
    try:
        # Run forever until the process is signalled.
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()


def main() -> None:
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger(__name__).info("Starting Loxone web apps (pages + API proxy)")
    _maybe_start_fastapi()
    try:
        asyncio.run(_serve_pages())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

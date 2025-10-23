#!/usr/bin/env python3
"""Integrated launcher for Loxone Smart Home with Web Monitoring Service.

This script runs both the main application and the FastAPI web service together.
"""

import asyncio
import logging
import os
import signal
import sys
import threading
from multiprocessing import Process

import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_web_service():
    """Run the FastAPI web service in a separate process."""
    import web.app
    from config.settings import Settings

    settings = Settings(influxdb_token=os.getenv("INFLUXDB_TOKEN", ""))

    # Configure uvicorn to run the FastAPI app
    uvicorn.run(
        "web.app:app",
        host=settings.web_service.host,
        port=settings.web_service.port,
        log_level="info",
        access_log=False,  # Reduce noise
        reload=False
    )


def run_main_application():
    """Run the main application."""
    import asyncio
    from main import main

    # Run the main application using asyncio
    asyncio.run(main())


def main():
    """Main entry point for integrated service."""
    from config.settings import Settings

    settings = Settings(influxdb_token=os.getenv("INFLUXDB_TOKEN", ""))

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger = logging.getLogger(__name__)

    # Check if web service is enabled
    if settings.web_service.enabled:
        logger.info("Starting integrated Loxone Smart Home with Web Monitoring Service")

        # Start web service in a separate process
        web_process = Process(target=run_web_service, daemon=True)
        web_process.start()
        logger.info(f"Web monitoring service process started on port {settings.web_service.port}")

        # Handle signals to clean up both processes
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            if web_process.is_alive():
                web_process.terminate()
                web_process.join(timeout=5)
                if web_process.is_alive():
                    web_process.kill()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            # Run main application
            run_main_application()
        finally:
            # Clean up web process
            if web_process.is_alive():
                web_process.terminate()
                web_process.join(timeout=5)
                if web_process.is_alive():
                    web_process.kill()
    else:
        logger.info("Starting Loxone Smart Home (web service disabled)")
        # Just run the main application
        run_main_application()


if __name__ == "__main__":
    main()
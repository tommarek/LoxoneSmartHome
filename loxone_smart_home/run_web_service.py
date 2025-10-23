#!/usr/bin/env python3
"""Run the Loxone Smart Home web monitoring service."""

import logging
import sys
from pathlib import Path

import uvicorn
from config.settings import Settings


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Set specific logger levels
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def main():
    """Main entry point for the web service."""
    setup_logging()

    # Load settings
    settings = Settings()

    logger = logging.getLogger(__name__)

    # Check if web service is enabled
    if not settings.web_service.enabled:
        logger.error("Web service is disabled in configuration. Set WEB_SERVICE_ENABLED=true")
        sys.exit(1)

    logger.info(
        f"Starting Loxone Smart Home Web Service on "
        f"{settings.web_service.host}:{settings.web_service.port}"
    )

    if settings.web_service.enable_auth:
        if not settings.web_service.api_key:
            logger.error("Authentication enabled but no API key configured")
            sys.exit(1)
        logger.info("API authentication is enabled")
    else:
        logger.warning("API authentication is disabled - not recommended for production")

    # Configure and run the server
    uvicorn.run(
        "web.app:app",
        host=settings.web_service.host,
        port=settings.web_service.port,
        reload=False,  # Set to True for development
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    main()

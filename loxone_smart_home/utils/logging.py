"""Enhanced logging utilities with timezone support and service prefixes."""

import logging
import time
import zoneinfo
from typing import Optional

import colorlog


class TimezoneAwareFormatter(colorlog.ColoredFormatter):
    """Custom formatter that displays timestamps in local timezone with service prefixes."""

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        timezone: str = "Europe/Prague",
        service_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize the timezone-aware formatter.

        Args:
            fmt: Log format string
            datefmt: Date format string
            timezone: Timezone name (e.g., 'Europe/Prague')
            service_name: Service name for prefix (e.g., 'UDP', 'MQTT')
            **kwargs: Additional arguments passed to ColoredFormatter
        """
        # Add service prefix to format if provided
        if service_name and fmt:
            fmt = fmt.replace("%(name)s", f"[{service_name.upper()}] %(name)s")
        elif service_name:
            fmt = (
                f"%(log_color)s%(asctime)s - [{service_name.upper()}] "
                f"%(name)s - %(levelname)s - %(message)s"
            )

        super().__init__(fmt, datefmt, **kwargs)
        self.timezone = zoneinfo.ZoneInfo(timezone)

    def formatTime(
        self, record: logging.LogRecord, datefmt: Optional[str] = None
    ) -> str:
        """Format time in the specified timezone."""
        # Convert the record timestamp to local timezone
        dt = time.localtime(record.created)
        local_time = time.struct_time(dt)

        # Create a datetime object and convert to our timezone
        import datetime

        utc_time = datetime.datetime.fromtimestamp(
            record.created, tz=datetime.timezone.utc
        )
        local_time = utc_time.astimezone(self.timezone)

        if datefmt:
            return local_time.strftime(datefmt)
        else:
            return local_time.strftime("%Y-%m-%d %H:%M:%S")


def setup_service_logger(
    service_name: str, timezone: str = "Europe/Prague", log_level: str = "INFO"
) -> logging.Logger:
    """Set up a logger for a specific service with timezone support.

    Args:
        service_name: Name of the service (e.g., 'UDP', 'MQTT', 'WEATHER', 'GROWATT')
        timezone: Timezone for timestamps
        log_level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(service_name)

    # Don't add handlers if they already exist
    if logger.handlers:
        return logger

    handler = colorlog.StreamHandler()
    formatter = TimezoneAwareFormatter(
        fmt="%(log_color)s%(asctime)s - [%(name)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        timezone=timezone,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, log_level.upper()))

    return logger


def configure_module_logger(
    module_name: str,
    service_name: str,
    timezone: str = "Europe/Prague",
    log_level: str = "INFO",
) -> logging.Logger:
    """Configure a logger for a module with service prefix.

    Args:
        module_name: Full module name (e.g., 'modules.udp_listener')
        service_name: Service name for prefix (e.g., 'UDP')
        timezone: Timezone for timestamps
        log_level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(module_name)

    # Create a custom handler for this module
    handler = colorlog.StreamHandler()
    formatter = TimezoneAwareFormatter(
        fmt=(
            f"%(log_color)s%(asctime)s - [{service_name.upper()}] "
            f"%(name)s - %(levelname)s - %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
        timezone=timezone,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )

    handler.setFormatter(formatter)

    # Clear existing handlers and add our custom one
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    return logger

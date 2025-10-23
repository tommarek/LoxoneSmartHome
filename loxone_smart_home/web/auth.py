"""Authentication middleware for web service."""

from typing import Optional

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN

from config.settings import Settings

# API key header configuration
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_settings() -> Settings:
    """Get settings instance."""
    return Settings()


async def verify_api_key(
    api_key: Optional[str] = Security(api_key_header),
    settings: Settings = Depends(get_settings)
) -> bool:
    """Verify API key if authentication is enabled.

    Args:
        api_key: The API key from the request header
        settings: Application settings

    Returns:
        True if authentication passed

    Raises:
        HTTPException: If authentication fails
    """
    # If authentication is disabled, allow all requests
    if not settings.web_service.enable_auth:
        return True

    # If authentication is enabled, check the API key
    if not api_key or api_key != settings.web_service.api_key:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key"
        )

    return True


# Dependency for protected endpoints
RequireAuth = Depends(verify_api_key)

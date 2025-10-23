"""In-memory cache service with TTL support."""

import asyncio
import time
from typing import Any, Dict, Optional


class CacheEntry:
    """Single cache entry with TTL."""

    def __init__(self, value: Any, ttl: int):
        """Initialize cache entry."""
        self.value = value
        self.expires_at = time.time() + ttl if ttl > 0 else float('inf')

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() > self.expires_at


class CacheService:
    """Simple in-memory cache with TTL support."""

    def __init__(self) -> None:
        """Initialize cache service."""
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task[None]] = None
        self._running = False

    async def start(self) -> None:
        """Start cache cleanup task."""
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """Stop cache cleanup task."""
        self._running = False
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                return None

            if entry.is_expired():
                del self._cache[key]
                return None

            return entry.value

    async def set(self, key: str, value: Any, ttl: int = 300) -> None:
        """Set value in cache with TTL in seconds."""
        async with self._lock:
            self._cache[key] = CacheEntry(value, ttl)

    async def delete(self, key: str) -> None:
        """Delete value from cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]

    async def clear(self) -> None:
        """Clear entire cache."""
        async with self._lock:
            self._cache.clear()

    async def get_current_data(self) -> Dict[str, Any]:
        """Get all current data for WebSocket broadcast."""
        data = {}

        # Get energy data
        energy_current = await self.get("energy:current")
        if energy_current:
            data["energy"] = energy_current

        # Get battery status
        battery_status = await self.get("battery:status")
        if battery_status:
            data["battery"] = battery_status

        # Get current price
        current_price = await self.get("prices:current")
        if current_price:
            data["price"] = current_price

        # Get weather
        weather = await self.get("weather:current")
        if weather:
            data["weather"] = weather

        return data

    async def _cleanup_loop(self) -> None:
        """Periodically clean up expired entries."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception:
                pass  # Ignore cleanup errors

    async def _cleanup_expired(self) -> None:
        """Remove expired entries from cache."""
        async with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]

            for key in expired_keys:
                del self._cache[key]

            if expired_keys:
                # Log cleanup if needed
                pass

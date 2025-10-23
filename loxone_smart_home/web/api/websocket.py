"""WebSocket connection management."""

import logging
from typing import Any, Dict, List, Set

from fastapi import WebSocket


logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manager for WebSocket connections."""

    def __init__(self) -> None:
        """Initialize connection manager."""
        self.active_connections: List[WebSocket] = []
        self.subscriptions: Dict[WebSocket, Set[str]] = {}

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and track a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.subscriptions[websocket] = {"all"}  # Default subscription
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.subscriptions:
            del self.subscriptions[websocket]
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket) -> None:
        """Send a message to a specific WebSocket."""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)

    async def send_json(self, data: Dict[str, Any], websocket: WebSocket) -> None:
        """Send JSON data to a specific WebSocket."""
        try:
            await websocket.send_json(data)
        except Exception as e:
            logger.error(f"Error sending JSON: {e}")
            self.disconnect(websocket)

    async def broadcast(self, data: Dict[str, Any], topic: str = "all") -> None:
        """Broadcast data to all connected WebSockets subscribed to the topic."""
        if not self.active_connections:
            return

        # Prepare message
        message = {
            "topic": topic,
            "timestamp": data.get("timestamp"),
            "data": data
        }

        disconnected = []
        for connection in self.active_connections:
            # Check if connection is subscribed to this topic
            subscriptions = self.subscriptions.get(connection, {"all"})
            if topic == "all" or topic in subscriptions or "all" in subscriptions:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to connection: {e}")
                    disconnected.append(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    async def subscribe(self, websocket: WebSocket, topics: List[str]) -> None:
        """Subscribe a WebSocket to specific topics."""
        if websocket in self.subscriptions:
            self.subscriptions[websocket].update(topics)
        else:
            self.subscriptions[websocket] = set(topics)

        # Send confirmation
        await self.send_json({
            "type": "subscription_confirmed",
            "topics": list(self.subscriptions[websocket])
        }, websocket)

    async def unsubscribe(self, websocket: WebSocket, topics: List[str]) -> None:
        """Unsubscribe a WebSocket from specific topics."""
        if websocket in self.subscriptions:
            self.subscriptions[websocket].difference_update(topics)

            # Send confirmation
            await self.send_json({
                "type": "subscription_updated",
                "topics": list(self.subscriptions[websocket])
            }, websocket)

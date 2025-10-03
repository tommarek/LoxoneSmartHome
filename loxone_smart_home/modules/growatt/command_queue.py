"""Command queue system for managing MQTT command/response patterns."""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import uuid4


@dataclass
class PendingCommand:
    """Represents a pending command awaiting response."""

    command_id: str
    command_type: str
    future: asyncio.Future[Dict[str, Any]]
    timeout_task: asyncio.Task[None]
    sent_at: datetime
    timeout_seconds: float


class CommandQueue:
    """Manages queued commands and matches responses to pending requests."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """Initialize the command queue."""
        self.logger = logger or logging.getLogger(__name__)
        self._pending_commands: Dict[str, PendingCommand] = {}
        self._lock = asyncio.Lock()

    async def add_command(
        self, command_type: str, timeout: float = 10.0
    ) -> tuple[str, asyncio.Future[Dict[str, Any]]]:
        """Add a command to the queue and return its ID and future.

        Args:
            command_type: Type of command (e.g., "datetime/get", "batteryfirst/set/timeslot")
            timeout: Timeout in seconds

        Returns:
            Tuple of (command_id, future) where future will contain the response
        """
        command_id = str(uuid4())
        future: asyncio.Future[Dict[str, Any]] = asyncio.get_running_loop().create_future()

        # Create timeout task
        timeout_task = asyncio.create_task(self._timeout_command(command_id, command_type, timeout))

        async with self._lock:
            pending = PendingCommand(
                command_id=command_id,
                command_type=command_type,
                future=future,
                timeout_task=timeout_task,
                sent_at=datetime.now(),
                timeout_seconds=timeout
            )
            self._pending_commands[command_id] = pending

        self.logger.debug(f"Queued command {command_type} with ID {command_id}, timeout={timeout}s")
        return command_id, future

    async def _timeout_command(self, command_id: str, command_type: str, timeout: float) -> None:
        """Handle command timeout."""
        await asyncio.sleep(timeout)

        async with self._lock:
            if command_id in self._pending_commands:
                pending = self._pending_commands[command_id]
                if not pending.future.done():
                    self.logger.warning(
                        f"⏱️ Command {command_type} (ID: {command_id}) timed out after {timeout}s"
                    )
                    # Set exception on the future
                    pending.future.set_exception(
                        asyncio.TimeoutError(f"Command {command_type} timed out after {timeout}s")
                    )
                # Remove from pending
                del self._pending_commands[command_id]

    async def resolve_command(self, command_type: str, result: Dict[str, Any]) -> bool:
        """Resolve a pending command with its result.

        Finds the first matching command by type and resolves it.

        Args:
            command_type: Type of command to resolve
            result: The result data to return

        Returns:
            True if a command was resolved, False if no matching command found
        """
        async with self._lock:
            # Find first matching command that's not done
            for command_id, pending in list(self._pending_commands.items()):
                if pending.command_type == command_type and not pending.future.done():
                    # Cancel timeout task
                    pending.timeout_task.cancel()

                    # Set result on future
                    pending.future.set_result(result)

                    # Log resolution
                    elapsed = (datetime.now() - pending.sent_at).total_seconds()
                    self.logger.debug(
                        f"Resolved command {command_type} (ID: {command_id}) after {elapsed:.2f}s"
                    )

                    # Remove from pending
                    del self._pending_commands[command_id]
                    return True

        # No matching command found
        self.logger.warning(
            f"Received response for {command_type} but no pending command found "
            f"(orphaned response or duplicate)"
        )
        return False

    async def cancel_all(self) -> None:
        """Cancel all pending commands."""
        async with self._lock:
            for command_id, pending in list(self._pending_commands.items()):
                if not pending.future.done():
                    pending.future.cancel()
                pending.timeout_task.cancel()
            self._pending_commands.clear()

    def get_pending_count(self) -> int:
        """Get number of pending commands."""
        return len(self._pending_commands)

    def get_pending_types(self) -> list[str]:
        """Get list of pending command types."""
        return [cmd.command_type for cmd in self._pending_commands.values()]

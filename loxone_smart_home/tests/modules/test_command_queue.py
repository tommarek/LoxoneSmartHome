"""Test the command queue module for concurrent command handling."""

import asyncio
import pytest
from datetime import datetime
from typing import Dict, Any

from modules.growatt.command_queue import CommandQueue


class TestCommandQueue:
    """Test command queue functionality."""

    @pytest.mark.asyncio
    async def test_single_command(self) -> None:
        """Test single command queueing and resolution."""
        queue = CommandQueue()

        # Add a command
        command_id, future = await queue.add_command("test/command", timeout=5.0)

        # Verify command is pending
        assert queue.get_pending_count() == 1
        assert "test/command" in queue.get_pending_types()

        # Resolve the command
        result = {"success": True, "data": "test"}
        resolved = await queue.resolve_command("test/command", result)

        # Verify resolution
        assert resolved is True
        assert await future == result
        assert queue.get_pending_count() == 0

    @pytest.mark.asyncio
    async def test_multiple_concurrent_commands(self) -> None:
        """Test multiple concurrent commands of different types."""
        queue = CommandQueue()

        # Add multiple commands
        cmd1_id, future1 = await queue.add_command("battery/get", timeout=5.0)
        cmd2_id, future2 = await queue.add_command("grid/get", timeout=5.0)
        cmd3_id, future3 = await queue.add_command("datetime/get", timeout=5.0)

        # Verify all are pending
        assert queue.get_pending_count() == 3
        pending_types = queue.get_pending_types()
        assert "battery/get" in pending_types
        assert "grid/get" in pending_types
        assert "datetime/get" in pending_types

        # Resolve in different order
        result2 = {"success": True, "type": "grid"}
        assert await queue.resolve_command("grid/get", result2) is True

        result3 = {"success": True, "type": "datetime"}
        assert await queue.resolve_command("datetime/get", result3) is True

        result1 = {"success": True, "type": "battery"}
        assert await queue.resolve_command("battery/get", result1) is True

        # Verify results
        assert await future1 == result1
        assert await future2 == result2
        assert await future3 == result3
        assert queue.get_pending_count() == 0

    @pytest.mark.asyncio
    async def test_multiple_same_type_commands(self) -> None:
        """Test multiple commands of the same type (e.g., retries)."""
        queue = CommandQueue()

        # Add multiple commands of same type
        cmd1_id, future1 = await queue.add_command("battery/set", timeout=5.0)
        cmd2_id, future2 = await queue.add_command("battery/set", timeout=5.0)

        # Verify both are pending
        assert queue.get_pending_count() == 2

        # First response resolves first command
        result1 = {"success": False, "error": "retry"}
        assert await queue.resolve_command("battery/set", result1) is True
        assert await future1 == result1
        assert queue.get_pending_count() == 1

        # Second response resolves second command
        result2 = {"success": True, "data": "ok"}
        assert await queue.resolve_command("battery/set", result2) is True
        assert await future2 == result2
        assert queue.get_pending_count() == 0

    @pytest.mark.asyncio
    async def test_command_timeout(self) -> None:
        """Test command timeout handling."""
        queue = CommandQueue()

        # Add command with very short timeout
        command_id, future = await queue.add_command("slow/command", timeout=0.1)

        # Wait for timeout
        with pytest.raises(asyncio.TimeoutError):
            await future

        # Verify command was removed
        assert queue.get_pending_count() == 0

        # Attempting to resolve after timeout should return False
        result = {"success": True}
        resolved = await queue.resolve_command("slow/command", result)
        assert resolved is False

    @pytest.mark.asyncio
    async def test_orphaned_response(self) -> None:
        """Test handling of response with no pending command."""
        queue = CommandQueue()

        # Try to resolve without any pending commands
        result = {"success": True}
        resolved = await queue.resolve_command("orphan/command", result)

        # Should return False for no match
        assert resolved is False
        assert queue.get_pending_count() == 0

    @pytest.mark.asyncio
    async def test_cancel_all(self) -> None:
        """Test cancelling all pending commands."""
        queue = CommandQueue()

        # Add multiple commands
        cmd1_id, future1 = await queue.add_command("cmd1", timeout=5.0)
        cmd2_id, future2 = await queue.add_command("cmd2", timeout=5.0)
        cmd3_id, future3 = await queue.add_command("cmd3", timeout=5.0)

        assert queue.get_pending_count() == 3

        # Cancel all
        await queue.cancel_all()

        # Verify all cancelled
        assert queue.get_pending_count() == 0
        assert future1.cancelled()
        assert future2.cancelled()
        assert future3.cancelled()

    @pytest.mark.asyncio
    async def test_concurrent_resolution_and_timeout(self) -> None:
        """Test race condition between resolution and timeout."""
        queue = CommandQueue()

        # Add command with moderate timeout
        command_id, future = await queue.add_command("race/command", timeout=0.5)

        # Start resolution task slightly before timeout
        async def resolve_late():
            await asyncio.sleep(0.4)  # Just before timeout
            return await queue.resolve_command("race/command", {"success": True})

        # Run resolution
        resolution_task = asyncio.create_task(resolve_late())

        # Try to get result
        try:
            result = await future
            # If we get here, resolution won
            assert result == {"success": True}
            assert await resolution_task is True
        except asyncio.TimeoutError:
            # If we get here, timeout won
            assert await resolution_task is False  # Resolution should fail

        # Either way, queue should be empty
        assert queue.get_pending_count() == 0
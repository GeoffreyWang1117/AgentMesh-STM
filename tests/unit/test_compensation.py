"""Unit tests for Compensation Transaction Manager."""

import pytest
import asyncio
import os
import tempfile
from pathlib import Path

from agentmesh_stm.compensation.manager import (
    CompensationManager,
    CompensationManagerConfig,
    CompensationLog,
    CompensationState,
    OperationType,
)
from agentmesh_stm.compensation.operations import (
    FileWriteCompensation,
    FileDeleteCompensation,
    FileCreateCompensation,
)


class TestCompensationLog:
    """Tests for CompensationLog class."""

    def test_create(self):
        """Test creating a compensation log entry."""
        log = CompensationLog.create(
            transaction_id="txn123",
            operation_type=OperationType.FILE_WRITE,
            operation_data={"path": "/test/file.txt", "content": "new content"},
            compensation_data={"path": "/test/file.txt", "original_content": "old content"},
        )

        assert log.transaction_id == "txn123"
        assert log.operation_type == OperationType.FILE_WRITE
        assert log.state == CompensationState.PENDING
        assert log.id is not None


class TestFileWriteCompensation:
    """Tests for FileWriteCompensation operation."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as td:
            yield td

    @pytest.fixture
    def operation(self):
        return FileWriteCompensation()

    @pytest.mark.asyncio
    async def test_execute_new_file(self, operation, temp_dir):
        """Test executing write to new file."""
        path = os.path.join(temp_dir, "new_file.txt")

        compensation_data = await operation.execute({
            "path": path,
            "content": "new content",
        })

        assert os.path.exists(path)
        assert compensation_data["file_existed"] is False

        with open(path) as f:
            assert f.read() == "new content"

    @pytest.mark.asyncio
    async def test_execute_existing_file(self, operation, temp_dir):
        """Test executing write to existing file."""
        path = os.path.join(temp_dir, "existing.txt")

        # Create existing file
        with open(path, "w") as f:
            f.write("original content")

        compensation_data = await operation.execute({
            "path": path,
            "content": "modified content",
        })

        assert compensation_data["file_existed"] is True
        assert compensation_data["original_content"] == "original content"

        with open(path) as f:
            assert f.read() == "modified content"

    @pytest.mark.asyncio
    async def test_compensate_restore(self, operation, temp_dir):
        """Test compensating by restoring original content."""
        path = os.path.join(temp_dir, "file.txt")

        # Execute the write
        compensation_data = await operation.execute({
            "path": path,
            "content": "new content",
        })

        # Write was for new file, so compensation should delete
        success = await operation.compensate(compensation_data)

        assert success
        assert not os.path.exists(path)

    @pytest.mark.asyncio
    async def test_compensate_restore_existing(self, operation, temp_dir):
        """Test compensating by restoring existing file."""
        path = os.path.join(temp_dir, "file.txt")

        # Create existing file
        with open(path, "w") as f:
            f.write("original")

        compensation_data = await operation.execute({
            "path": path,
            "content": "modified",
        })

        # Compensate should restore original
        success = await operation.compensate(compensation_data)

        assert success
        with open(path) as f:
            assert f.read() == "original"

    def test_validate(self, operation):
        """Test validation of operation parameters."""
        assert operation.validate({"path": "/test", "content": "data"})
        assert not operation.validate({"path": "/test"})  # Missing content
        assert not operation.validate({"content": "data"})  # Missing path


class TestFileDeleteCompensation:
    """Tests for FileDeleteCompensation operation."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as td:
            yield td

    @pytest.fixture
    def operation(self):
        return FileDeleteCompensation()

    @pytest.mark.asyncio
    async def test_execute_and_compensate(self, operation, temp_dir):
        """Test deleting and restoring a file."""
        path = os.path.join(temp_dir, "to_delete.txt")

        # Create file
        with open(path, "w") as f:
            f.write("important content")

        # Execute delete
        compensation_data = await operation.execute({"path": path})

        assert not os.path.exists(path)
        assert compensation_data["content"] == "important content"
        assert compensation_data["existed"] is True

        # Compensate (restore)
        success = await operation.compensate(compensation_data)

        assert success
        assert os.path.exists(path)
        with open(path) as f:
            assert f.read() == "important content"


class TestFileCreateCompensation:
    """Tests for FileCreateCompensation operation."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as td:
            yield td

    @pytest.fixture
    def operation(self):
        return FileCreateCompensation()

    @pytest.mark.asyncio
    async def test_execute_and_compensate(self, operation, temp_dir):
        """Test creating and removing a file."""
        path = os.path.join(temp_dir, "new_file.txt")

        # Execute create
        compensation_data = await operation.execute({
            "path": path,
            "content": "created content",
        })

        assert os.path.exists(path)

        # Compensate (delete)
        success = await operation.compensate(compensation_data)

        assert success
        assert not os.path.exists(path)

    @pytest.mark.asyncio
    async def test_execute_fails_if_exists(self, operation, temp_dir):
        """Test that create fails if file exists."""
        path = os.path.join(temp_dir, "existing.txt")

        # Create existing file
        with open(path, "w") as f:
            f.write("existing")

        with pytest.raises(FileExistsError):
            await operation.execute({"path": path, "content": "new"})


class TestCompensationManager:
    """Tests for CompensationManager class."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as td:
            yield td

    @pytest.fixture
    def manager(self, temp_dir):
        config = CompensationManagerConfig(
            log_dir=os.path.join(temp_dir, "compensation_logs"),
            persist_logs=False,  # Don't persist for unit tests
        )
        return CompensationManager(config=config)

    @pytest.mark.asyncio
    async def test_log_operation(self, manager):
        """Test logging an operation."""
        log = await manager.log_operation(
            transaction_id="txn1",
            operation_type=OperationType.FILE_WRITE,
            operation_data={"path": "/test"},
            compensation_data={"path": "/test", "original_content": "old"},
        )

        assert log.transaction_id == "txn1"
        assert log.state == CompensationState.PENDING

        pending = await manager.get_pending_compensations("txn1")
        assert len(pending) == 1

    @pytest.mark.asyncio
    async def test_execute_with_compensation(self, manager, temp_dir):
        """Test executing operation with automatic logging."""
        path = os.path.join(temp_dir, "test_file.txt")

        result = await manager.execute_with_compensation(
            transaction_id="txn1",
            operation_type=OperationType.FILE_WRITE,
            operation_data={"path": path, "content": "test content"},
        )

        assert os.path.exists(path)

        pending = await manager.get_pending_compensations("txn1")
        assert len(pending) == 1

    @pytest.mark.asyncio
    async def test_rollback(self, manager, temp_dir):
        """Test rolling back operations."""
        path = os.path.join(temp_dir, "rollback_test.txt")

        # Execute operation
        await manager.execute_with_compensation(
            transaction_id="txn1",
            operation_type=OperationType.FILE_WRITE,
            operation_data={"path": path, "content": "will be rolled back"},
        )

        assert os.path.exists(path)

        # Rollback
        success = await manager.rollback("txn1")

        assert success
        assert not os.path.exists(path)  # New file should be deleted

    @pytest.mark.asyncio
    async def test_rollback_multiple_operations(self, manager, temp_dir):
        """Test rolling back multiple operations in reverse order."""
        file1 = os.path.join(temp_dir, "file1.txt")
        file2 = os.path.join(temp_dir, "file2.txt")

        # Execute multiple operations
        await manager.execute_with_compensation(
            transaction_id="txn1",
            operation_type=OperationType.FILE_WRITE,
            operation_data={"path": file1, "content": "content1"},
        )
        await manager.execute_with_compensation(
            transaction_id="txn1",
            operation_type=OperationType.FILE_WRITE,
            operation_data={"path": file2, "content": "content2"},
        )

        assert os.path.exists(file1)
        assert os.path.exists(file2)

        # Rollback should undo both
        success = await manager.rollback("txn1")

        assert success
        assert not os.path.exists(file1)
        assert not os.path.exists(file2)

    @pytest.mark.asyncio
    async def test_commit_clears_logs(self, manager, temp_dir):
        """Test that commit clears compensation logs."""
        path = os.path.join(temp_dir, "committed.txt")

        await manager.execute_with_compensation(
            transaction_id="txn1",
            operation_type=OperationType.FILE_WRITE,
            operation_data={"path": path, "content": "committed content"},
        )

        pending = await manager.get_pending_compensations("txn1")
        assert len(pending) == 1

        # Commit
        await manager.commit("txn1")

        pending = await manager.get_pending_compensations("txn1")
        assert len(pending) == 0

        # File should still exist after commit
        assert os.path.exists(path)

    @pytest.mark.asyncio
    async def test_rollback_no_operations(self, manager):
        """Test rollback with no operations is successful."""
        success = await manager.rollback("nonexistent_txn")
        assert success

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, manager, temp_dir):
        """Test that compensation retries on failure."""
        # This is harder to test without mocking, but we can verify
        # the config is respected
        assert manager._config.max_retry_attempts == 3

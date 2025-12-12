"""Unit tests for Transaction Abstraction Layer."""

import pytest
import asyncio
from datetime import datetime

from agentmesh_stm.core.transaction import (
    Transaction,
    TransactionConfig,
    TransactionError,
    TransactionManager,
    TransactionState,
    ReadSet,
    WriteSet,
)
from agentmesh_stm.storage.mvcc import MVCCStorage, InMemoryBackend


class TestReadSet:
    """Tests for ReadSet class."""

    def test_add_and_get(self):
        """Test adding and retrieving read set entries."""
        read_set = ReadSet()
        timestamp = datetime.utcnow()

        read_set.add("resource1", version=1, timestamp=timestamp, content_hash="abc123")

        entry = read_set.get("resource1")
        assert entry is not None
        assert entry.resource_id == "resource1"
        assert entry.version == 1
        assert entry.content_hash == "abc123"

    def test_contains(self):
        """Test checking if resource is in read set."""
        read_set = ReadSet()
        read_set.add("resource1", version=1, timestamp=datetime.utcnow())

        assert read_set.contains("resource1")
        assert not read_set.contains("resource2")

    def test_get_all(self):
        """Test getting all entries."""
        read_set = ReadSet()
        timestamp = datetime.utcnow()

        read_set.add("r1", version=1, timestamp=timestamp)
        read_set.add("r2", version=2, timestamp=timestamp)

        entries = read_set.get_all()
        assert len(entries) == 2

    def test_clear(self):
        """Test clearing the read set."""
        read_set = ReadSet()
        read_set.add("r1", version=1, timestamp=datetime.utcnow())
        read_set.clear()

        assert len(read_set.get_all()) == 0


class TestWriteSet:
    """Tests for WriteSet class."""

    def test_add_and_get(self):
        """Test adding and retrieving write set entries."""
        write_set = WriteSet()

        write_set.add("resource1", old_content="old", new_content="new", operation_type="write")

        entry = write_set.get("resource1")
        assert entry is not None
        assert entry.old_content == "old"
        assert entry.new_content == "new"
        assert entry.operation_type == "write"

    def test_get_resource_ids(self):
        """Test getting resource IDs."""
        write_set = WriteSet()
        write_set.add("r1", old_content=None, new_content="content1")
        write_set.add("r2", old_content=None, new_content="content2")

        ids = write_set.get_resource_ids()
        assert ids == {"r1", "r2"}


class TestTransaction:
    """Tests for Transaction class."""

    @pytest.fixture
    def storage(self):
        """Create in-memory storage."""
        return MVCCStorage(backend=InMemoryBackend())

    @pytest.fixture
    def transaction(self, storage):
        """Create a transaction with storage."""
        return Transaction(storage=storage)

    @pytest.mark.asyncio
    async def test_begin_transaction(self, transaction):
        """Test beginning a transaction."""
        assert transaction.state == TransactionState.CREATED

        await transaction.begin()

        assert transaction.state == TransactionState.ACTIVE
        assert transaction.snapshot_version is not None

    @pytest.mark.asyncio
    async def test_cannot_begin_twice(self, transaction):
        """Test that transaction cannot be begun twice."""
        await transaction.begin()

        with pytest.raises(TransactionError):
            await transaction.begin()

    @pytest.mark.asyncio
    async def test_read_and_write(self, storage, transaction):
        """Test reading and writing within transaction."""
        # First write something to storage
        await storage.write("test_file", "initial content")

        await transaction.begin()

        # Read should track in read set
        content = await transaction.read("test_file")
        assert content == "initial content"
        assert transaction.read_set.contains("test_file")

        # Write should buffer in write set
        await transaction.write("test_file", "modified content")
        assert transaction.write_set.contains("test_file")

        # Subsequent reads should see buffered write (read-your-writes)
        content = await transaction.read("test_file")
        assert content == "modified content"

    @pytest.mark.asyncio
    async def test_commit_success(self, storage, transaction):
        """Test successful commit."""
        await transaction.begin()
        await transaction.write("new_file", "new content")

        success = await transaction.commit()

        assert success
        assert transaction.state == TransactionState.COMMITTED

        # Verify write was applied
        resource = await storage.read("new_file")
        assert resource is not None
        assert resource.content == "new content"

    @pytest.mark.asyncio
    async def test_abort(self, transaction):
        """Test aborting a transaction."""
        await transaction.begin()
        await transaction.write("file", "content")

        await transaction.abort()

        assert transaction.state == TransactionState.ABORTED

    @pytest.mark.asyncio
    async def test_conflict_detection(self, storage):
        """Test conflict detection between transactions."""
        # Write initial content
        await storage.write("shared_file", "initial")

        # Transaction 1 reads
        txn1 = Transaction(storage=storage)
        await txn1.begin()
        await txn1.read("shared_file")

        # Transaction 2 writes and commits
        txn2 = Transaction(storage=storage)
        await txn2.begin()
        await txn2.write("shared_file", "modified by txn2")
        assert await txn2.commit()

        # Transaction 1 tries to commit - should fail due to conflict
        await txn1.write("shared_file", "modified by txn1")
        success = await txn1.commit()

        assert not success

    @pytest.mark.asyncio
    async def test_transaction_duration(self, transaction):
        """Test transaction duration tracking."""
        assert transaction.duration_seconds is None

        await transaction.begin()
        await asyncio.sleep(0.1)

        duration = transaction.duration_seconds
        assert duration is not None
        assert duration >= 0.1


class TestTransactionManager:
    """Tests for TransactionManager class."""

    @pytest.fixture
    def storage(self):
        """Create in-memory storage."""
        return MVCCStorage(backend=InMemoryBackend())

    @pytest.fixture
    def manager(self, storage):
        """Create transaction manager."""
        return TransactionManager(storage=storage)

    @pytest.mark.asyncio
    async def test_create_transaction(self, manager):
        """Test creating a transaction."""
        txn = await manager.create_transaction()

        assert txn is not None
        assert txn.state == TransactionState.CREATED

    @pytest.mark.asyncio
    async def test_execute_function(self, manager, storage):
        """Test executing a function within a transaction."""
        # Write initial data
        await storage.write("counter", "0")

        async def increment(txn):
            value = await txn.read("counter")
            new_value = str(int(value) + 1)
            await txn.write("counter", new_value)
            return {"new_value": new_value}

        result = await manager.execute(increment)

        assert result["new_value"] == "1"

        # Verify change was committed
        resource = await storage.read("counter")
        assert resource.content == "1"

    @pytest.mark.asyncio
    async def test_context_manager(self, manager, storage):
        """Test using transaction as context manager."""
        async with manager.transaction() as txn:
            await txn.write("ctx_file", "content via context manager")

        resource = await storage.read("ctx_file")
        assert resource is not None
        assert resource.content == "content via context manager"

    @pytest.mark.asyncio
    async def test_active_transaction_count(self, manager):
        """Test tracking active transactions."""
        assert await manager.get_active_transaction_count() == 0

        txn = await manager.create_transaction()
        await txn.begin()

        assert await manager.get_active_transaction_count() == 1

    @pytest.mark.asyncio
    async def test_abort_all(self, manager):
        """Test aborting all transactions."""
        txn1 = await manager.create_transaction()
        txn2 = await manager.create_transaction()
        await txn1.begin()
        await txn2.begin()

        await manager.abort_all()

        assert await manager.get_active_transaction_count() == 0


class TestTransactionRetry:
    """Tests for transaction retry behavior."""

    @pytest.fixture
    def storage(self):
        return MVCCStorage(backend=InMemoryBackend())

    @pytest.mark.asyncio
    async def test_retry_on_conflict(self, storage):
        """Test automatic retry on conflict."""
        await storage.write("data", "0")

        config = TransactionConfig(max_retries=3, retry_delay_ms=10)
        manager = TransactionManager(storage=storage, default_config=config)

        execution_count = 0

        async def conflicting_operation(txn):
            nonlocal execution_count
            execution_count += 1

            value = await txn.read("data")

            # Simulate external modification on first attempt
            if execution_count == 1:
                await storage.write("data", "external_change")

            await txn.write("data", f"attempt_{execution_count}")
            return {"count": execution_count}

        result = await manager.execute(conflicting_operation)

        # Should have retried
        assert execution_count > 1
        assert result["count"] == execution_count

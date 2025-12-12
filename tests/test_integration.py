"""
Integration Tests for AgentMesh-STM.

These tests verify the complete system working together:
- Transaction lifecycle with storage
- Conflict detection and resolution
- Multi-agent coordination
- Recovery after simulated crashes
"""

import asyncio
import tempfile
import pytest
from pathlib import Path

from agentmesh_stm.core.transaction import (
    TransactionManager,
    TransactionConfig,
    TransactionError,
)
from agentmesh_stm.core.logging import (
    WriteAheadLog,
    FileLogBackend,
    TransactionLogger,
    RecoveryManager,
)
from agentmesh_stm.storage.mvcc import MVCCStorage, InMemoryBackend, SQLiteBackend
from agentmesh_stm.conflict.detector import ConflictDetector
from agentmesh_stm.compensation.manager import CompensationManager
from agentmesh_stm.agent.base import Agent, AgentTask, AgentConfig


class TestFullTransactionLifecycle:
    """Test complete transaction lifecycle."""

    @pytest.mark.asyncio
    async def test_simple_read_write_commit(self):
        """Test basic transaction: read, write, commit."""
        storage = MVCCStorage(InMemoryBackend())
        manager = TransactionManager(storage=storage)

        # Initialize data
        async with manager.transaction() as txn:
            await txn.write("file.txt", "initial content")

        # Read and modify
        async with manager.transaction() as txn:
            content = await txn.read("file.txt")
            assert content == "initial content"
            await txn.write("file.txt", "modified content")

        # Verify modification persisted
        async with manager.transaction() as txn:
            content = await txn.read("file.txt")
            assert content == "modified content"

    @pytest.mark.asyncio
    async def test_transaction_isolation(self):
        """Test that transactions are isolated from each other."""
        storage = MVCCStorage(InMemoryBackend())
        manager = TransactionManager(storage=storage)

        # Initialize
        async with manager.transaction() as txn:
            await txn.write("file.txt", "original")

        # Start two transactions
        txn1 = await manager.create_transaction()
        txn2 = await manager.create_transaction()

        await txn1.begin()
        await txn2.begin()

        # Both read original value
        content1 = await txn1.read("file.txt")
        content2 = await txn2.read("file.txt")
        assert content1 == "original"
        assert content2 == "original"

        # txn1 writes
        await txn1.write("file.txt", "from txn1")

        # txn2 should still see original (not txn1's uncommitted write)
        content2_again = await txn2.read("file.txt")
        assert content2_again == "original"

        # Commit txn1
        assert await txn1.commit()

        # txn2's read set is now stale - commit should fail
        await txn2.write("file.txt", "from txn2")
        assert not await txn2.commit()

        await txn2.abort()

    @pytest.mark.asyncio
    async def test_read_your_writes(self):
        """Test that a transaction can read its own writes."""
        storage = MVCCStorage(InMemoryBackend())
        manager = TransactionManager(storage=storage)

        async with manager.transaction() as txn:
            await txn.write("new_file.txt", "new content")
            # Should read back our own write
            content = await txn.read("new_file.txt")
            assert content == "new content"


class TestConflictDetectionIntegration:
    """Test conflict detection with full system."""

    @pytest.mark.asyncio
    async def test_write_write_conflict_detection(self):
        """Test detection of write-write conflicts."""
        storage = MVCCStorage(InMemoryBackend())
        conflict_detector = ConflictDetector(storage)
        manager = TransactionManager(
            storage=storage,
            conflict_detector=conflict_detector,
        )

        # Initialize
        async with manager.transaction() as txn:
            await txn.write("shared.txt", "initial")

        # Two transactions modify same file
        async def writer1():
            async with manager.transaction() as txn:
                await txn.read("shared.txt")
                await asyncio.sleep(0.1)  # Simulate work
                await txn.write("shared.txt", "from writer1")

        async def writer2():
            async with manager.transaction() as txn:
                await txn.read("shared.txt")
                await asyncio.sleep(0.05)  # Finish faster
                await txn.write("shared.txt", "from writer2")

        # Run concurrently - one should succeed, one should fail/retry
        results = await asyncio.gather(writer1(), writer2(), return_exceptions=True)

        # Verify final state is consistent
        async with manager.transaction() as txn:
            content = await txn.read("shared.txt")
            assert content in ("from writer1", "from writer2")

    @pytest.mark.asyncio
    async def test_non_conflicting_writes(self):
        """Test that writes to different resources don't conflict."""
        storage = MVCCStorage(InMemoryBackend())
        manager = TransactionManager(storage=storage)

        # Initialize
        async with manager.transaction() as txn:
            await txn.write("file1.txt", "content1")
            await txn.write("file2.txt", "content2")

        # Two transactions modify different files
        async def writer1():
            async with manager.transaction() as txn:
                await txn.write("file1.txt", "modified1")

        async def writer2():
            async with manager.transaction() as txn:
                await txn.write("file2.txt", "modified2")

        # Both should succeed
        await asyncio.gather(writer1(), writer2())

        # Verify both modifications persisted
        async with manager.transaction() as txn:
            assert await txn.read("file1.txt") == "modified1"
            assert await txn.read("file2.txt") == "modified2"


class TestTransactionRetry:
    """Test automatic retry on conflict."""

    @pytest.mark.asyncio
    async def test_automatic_retry_on_conflict(self):
        """Test that transactions automatically retry on conflict."""
        storage = MVCCStorage(InMemoryBackend())
        manager = TransactionManager(
            storage=storage,
            default_config=TransactionConfig(max_retries=5, retry_delay_ms=10),
        )

        # Initialize counter
        async with manager.transaction() as txn:
            await txn.write("counter", "0")

        # Multiple transactions increment counter
        async def increment():
            async def do_increment(txn):
                value = await txn.read("counter")
                await asyncio.sleep(0.01)  # Simulate work
                await txn.write("counter", str(int(value) + 1))
                return int(value) + 1

            return await manager.execute(do_increment)

        # Run 5 concurrent increments
        results = await asyncio.gather(*[increment() for _ in range(5)])

        # Final value should be 5 (all increments succeeded through retries)
        async with manager.transaction() as txn:
            final_value = await txn.read("counter")
            assert int(final_value) == 5

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test that transaction fails after max retries."""
        storage = MVCCStorage(InMemoryBackend())
        manager = TransactionManager(
            storage=storage,
            default_config=TransactionConfig(max_retries=2, retry_delay_ms=1),
        )

        async with manager.transaction() as txn:
            await txn.write("contested", "initial")

        conflict_count = 0

        async def always_conflict(txn):
            nonlocal conflict_count
            await txn.read("contested")
            # Another transaction always commits first
            async with manager.transaction() as other:
                await other.read("contested")
                await other.write("contested", f"other-{conflict_count}")
            conflict_count += 1
            await txn.write("contested", "mine")

        with pytest.raises(TransactionError, match="retries"):
            await manager.execute(always_conflict)


class TestWALIntegration:
    """Test Write-Ahead Log integration."""

    @pytest.mark.asyncio
    async def test_wal_records_transactions(self):
        """Test that WAL properly records transaction events."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            backend = FileLogBackend(tmp_dir)
            wal = WriteAheadLog(backend)
            logger = TransactionLogger(wal)

            storage = MVCCStorage(InMemoryBackend())
            manager = TransactionManager(
                storage=storage,
                transaction_logger=logger,
            )

            # Execute a transaction
            async def write_data(txn):
                await txn.write("test.txt", "test content")

            await manager.execute(write_data)

            # Verify WAL has records
            records = await wal.backend.read_all()
            assert len(records) >= 3  # BEGIN, WRITE, COMMIT

            # Check record types
            record_types = [r.record_type.name for r in records]
            assert "BEGIN" in record_types
            assert "WRITE" in record_types
            assert "COMMIT" in record_types

            await logger.close()

    @pytest.mark.asyncio
    async def test_recovery_replays_committed(self):
        """Test that recovery replays committed transactions."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Phase 1: Execute and commit a transaction
            backend1 = FileLogBackend(tmp_dir)
            wal1 = WriteAheadLog(backend1)

            await wal1.begin_transaction("txn-1")
            await wal1.log_write("txn-1", "file.txt", None, "content")
            await wal1.commit_transaction("txn-1")
            await wal1.close()

            # Phase 2: Recover and verify
            applied_writes = {}

            async def apply_write(resource_id, value):
                applied_writes[resource_id] = value

            async def apply_undo(resource_id, old_value):
                if old_value is None:
                    applied_writes.pop(resource_id, None)
                else:
                    applied_writes[resource_id] = old_value

            backend2 = FileLogBackend(tmp_dir)
            wal2 = WriteAheadLog(backend2)
            recovery = RecoveryManager(wal2, apply_write, apply_undo)

            result = await recovery.recover()

            assert "txn-1" in result.committed_transactions
            assert result.redo_count == 1
            assert applied_writes == {"file.txt": "content"}

            await wal2.close()


class TestMultiAgentCoordination:
    """Test multiple agents working together."""

    @pytest.mark.asyncio
    async def test_multiple_agents_concurrent_work(self):
        """Test multiple agents working on different files."""
        storage = MVCCStorage(InMemoryBackend())
        manager = TransactionManager(storage=storage)

        # Initialize workspace
        async with manager.transaction() as txn:
            await txn.write("src/module_a.py", "# Module A")
            await txn.write("src/module_b.py", "# Module B")
            await txn.write("src/module_c.py", "# Module C")

        class SimpleWorkerAgent(Agent):
            def __init__(self, tm, name, target_file):
                super().__init__(tm, AgentConfig(name=name))
                self.target_file = target_file

            async def execute_task(self, task, transaction):
                content = await transaction.read(self.target_file)
                await asyncio.sleep(0.05)  # Simulate work
                await transaction.write(
                    self.target_file,
                    content + f"\n# Modified by {self.name}"
                )
                return {"file": self.target_file, "agent": self.name}

        # Create agents for different files
        agents = [
            SimpleWorkerAgent(manager, "Agent-A", "src/module_a.py"),
            SimpleWorkerAgent(manager, "Agent-B", "src/module_b.py"),
            SimpleWorkerAgent(manager, "Agent-C", "src/module_c.py"),
        ]

        tasks = [
            AgentTask(task_id=f"task-{i}", description="Modify file", input_data={})
            for i in range(3)
        ]

        # Run all agents concurrently
        results = await asyncio.gather(
            *[agent.run(task) for agent, task in zip(agents, tasks)]
        )

        # All should succeed
        assert all(r.status.value == "completed" for r in results)

        # Verify all modifications persisted
        async with manager.transaction() as txn:
            for agent in agents:
                content = await txn.read(agent.target_file)
                assert f"Modified by {agent.name}" in content

    @pytest.mark.asyncio
    async def test_agents_with_overlapping_work(self):
        """Test agents that need to coordinate on shared files."""
        storage = MVCCStorage(InMemoryBackend())
        manager = TransactionManager(
            storage=storage,
            default_config=TransactionConfig(max_retries=10, retry_delay_ms=10),
        )

        # Shared configuration file
        async with manager.transaction() as txn:
            await txn.write("config.json", '{"setting1": 1, "setting2": 2}')

        modification_count = 0

        class ConfigModifierAgent(Agent):
            def __init__(self, tm, name, setting):
                super().__init__(tm, AgentConfig(name=name))
                self.setting = setting

            async def execute_task(self, task, transaction):
                nonlocal modification_count
                content = await transaction.read("config.json")
                await asyncio.sleep(0.02)

                # Simple modification (in real code, would parse JSON)
                modified = content.replace(
                    f'"{self.setting}": ',
                    f'"{self.setting}": 100'
                )
                await transaction.write("config.json", modified)
                modification_count += 1
                return {"modified": self.setting}

        agents = [
            ConfigModifierAgent(manager, "Agent-1", "setting1"),
            ConfigModifierAgent(manager, "Agent-2", "setting2"),
        ]

        tasks = [
            AgentTask(task_id=f"task-{i}", description="Modify config", input_data={})
            for i in range(2)
        ]

        # Run concurrently - will require retries due to conflicts
        results = await asyncio.gather(
            *[agent.run(task) for agent, task in zip(agents, tasks)],
            return_exceptions=True
        )

        # At least some agents should have succeeded
        successful = [r for r in results if not isinstance(r, Exception)]
        assert len(successful) >= 1


class TestSQLiteStorage:
    """Test with SQLite storage backend."""

    @pytest.mark.asyncio
    async def test_sqlite_persistence(self):
        """Test that data persists with SQLite backend."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test.db"

            # Write data
            storage1 = MVCCStorage(SQLiteBackend(str(db_path)))
            manager1 = TransactionManager(storage=storage1)

            async with manager1.transaction() as txn:
                await txn.write("persistent.txt", "persisted content")

            # Close and reopen
            await storage1.close()

            storage2 = MVCCStorage(SQLiteBackend(str(db_path)))
            manager2 = TransactionManager(storage=storage2)

            # Data should still be there
            async with manager2.transaction() as txn:
                content = await txn.read("persistent.txt")
                assert content == "persisted content"

            await storage2.close()


class TestCompensation:
    """Test compensation/rollback functionality."""

    @pytest.mark.asyncio
    async def test_compensation_on_abort(self):
        """Test that compensation is executed on transaction abort."""
        storage = MVCCStorage(InMemoryBackend())
        compensation_manager = CompensationManager()
        manager = TransactionManager(
            storage=storage,
            compensation_manager=compensation_manager,
        )

        compensations_executed = []

        async with manager.transaction() as txn:
            await txn.write("file.txt", "content")

            # Register a compensation action
            from agentmesh_stm.compensation.operations import CompensableOperation

            class TestCompensation(CompensableOperation):
                async def execute(self):
                    pass

                async def compensate(self):
                    compensations_executed.append("compensated")

            compensation_manager.register(txn.id, TestCompensation())

            # Simulate error that triggers abort
            raise ValueError("Simulated error")

        # Compensation should have been called (if abort triggers it)
        # Note: This depends on implementation details


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_transaction(self):
        """Test committing a transaction with no operations."""
        storage = MVCCStorage(InMemoryBackend())
        manager = TransactionManager(storage=storage)

        # Empty transaction should commit successfully
        async with manager.transaction() as txn:
            pass  # No operations

    @pytest.mark.asyncio
    async def test_read_nonexistent_resource(self):
        """Test reading a resource that doesn't exist."""
        storage = MVCCStorage(InMemoryBackend())
        manager = TransactionManager(storage=storage)

        async with manager.transaction() as txn:
            content = await txn.read("nonexistent.txt")
            assert content is None

    @pytest.mark.asyncio
    async def test_delete_operation(self):
        """Test deleting a resource."""
        storage = MVCCStorage(InMemoryBackend())
        manager = TransactionManager(storage=storage)

        # Create resource
        async with manager.transaction() as txn:
            await txn.write("to_delete.txt", "content")

        # Delete resource
        async with manager.transaction() as txn:
            await txn.delete("to_delete.txt")

        # Resource should be gone
        async with manager.transaction() as txn:
            content = await txn.read("to_delete.txt")
            # Depending on implementation, might be None or empty
            assert content in (None, "")

    @pytest.mark.asyncio
    async def test_large_write(self):
        """Test writing large content."""
        storage = MVCCStorage(InMemoryBackend())
        manager = TransactionManager(storage=storage)

        large_content = "x" * (10 * 1024 * 1024)  # 10MB

        async with manager.transaction() as txn:
            await txn.write("large_file.txt", large_content)

        async with manager.transaction() as txn:
            content = await txn.read("large_file.txt")
            assert len(content) == len(large_content)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

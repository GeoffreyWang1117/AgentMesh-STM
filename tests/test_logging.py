"""Tests for transaction logging and recovery."""

import asyncio
import tempfile
import pytest
from pathlib import Path

from agentmesh_stm.core.logging import (
    LogRecord,
    LogRecordType,
    FileLogBackend,
    InMemoryLogBackend,
    WriteAheadLog,
    RecoveryManager,
    TransactionLogger,
)


class TestLogRecord:
    """Test LogRecord serialization."""

    def test_serialize_deserialize(self):
        """Test record serialization round-trip."""
        record = LogRecord(
            lsn=1,
            timestamp=1234567890.123,
            record_type=LogRecordType.WRITE,
            transaction_id="txn-123",
            data={"resource_id": "file.txt", "new_value": "content"},
            prev_lsn=0,
        )

        serialized = record.serialize()
        deserialized = LogRecord.deserialize(serialized)

        assert deserialized.lsn == record.lsn
        assert deserialized.timestamp == record.timestamp
        assert deserialized.record_type == record.record_type
        assert deserialized.transaction_id == record.transaction_id
        assert deserialized.data == record.data
        assert deserialized.prev_lsn == record.prev_lsn

    def test_checksum_verification(self):
        """Test that corrupted data is detected."""
        record = LogRecord(
            lsn=1,
            timestamp=1234567890.123,
            record_type=LogRecordType.BEGIN,
            transaction_id="txn-123",
        )

        serialized = bytearray(record.serialize())
        # Corrupt the data
        serialized[10] ^= 0xFF

        with pytest.raises(ValueError, match="checksum"):
            LogRecord.deserialize(bytes(serialized))


class TestInMemoryLogBackend:
    """Test in-memory log backend."""

    @pytest.mark.asyncio
    async def test_append_and_read(self):
        """Test appending and reading records."""
        backend = InMemoryLogBackend()

        record1 = LogRecord(
            lsn=0,
            timestamp=1.0,
            record_type=LogRecordType.BEGIN,
            transaction_id="txn-1",
        )
        record2 = LogRecord(
            lsn=0,
            timestamp=2.0,
            record_type=LogRecordType.WRITE,
            transaction_id="txn-1",
            data={"key": "value"},
        )

        lsn1 = await backend.append(record1)
        lsn2 = await backend.append(record2)

        assert lsn1 == 1
        assert lsn2 == 2

        read1 = await backend.read(1)
        read2 = await backend.read(2)

        assert read1.transaction_id == "txn-1"
        assert read2.data == {"key": "value"}

    @pytest.mark.asyncio
    async def test_read_all(self):
        """Test reading all records."""
        backend = InMemoryLogBackend()

        for i in range(5):
            record = LogRecord(
                lsn=0,
                timestamp=float(i),
                record_type=LogRecordType.WRITE,
                transaction_id=f"txn-{i}",
            )
            await backend.append(record)

        all_records = await backend.read_all()
        assert len(all_records) == 5

        from_lsn = await backend.read_all(from_lsn=3)
        assert len(from_lsn) == 3

    @pytest.mark.asyncio
    async def test_truncate(self):
        """Test log truncation."""
        backend = InMemoryLogBackend()

        for i in range(5):
            record = LogRecord(
                lsn=0,
                timestamp=float(i),
                record_type=LogRecordType.WRITE,
                transaction_id=f"txn-{i}",
            )
            await backend.append(record)

        await backend.truncate(3)

        remaining = await backend.read_all()
        assert len(remaining) == 2
        assert all(r.lsn > 3 for r in remaining)


class TestFileLogBackend:
    """Test file-based log backend."""

    @pytest.mark.asyncio
    async def test_append_and_read(self):
        """Test file-based append and read."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            backend = FileLogBackend(tmp_dir)

            record = LogRecord(
                lsn=0,
                timestamp=1.0,
                record_type=LogRecordType.BEGIN,
                transaction_id="txn-1",
            )
            lsn = await backend.append(record)

            assert lsn == 1

            read = await backend.read(1)
            assert read is not None
            assert read.transaction_id == "txn-1"

            await backend.close()

    @pytest.mark.asyncio
    async def test_persistence(self):
        """Test that data persists across restarts."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Write some records
            backend1 = FileLogBackend(tmp_dir)
            for i in range(3):
                record = LogRecord(
                    lsn=0,
                    timestamp=float(i),
                    record_type=LogRecordType.WRITE,
                    transaction_id=f"txn-{i}",
                )
                await backend1.append(record)
            await backend1.sync()
            await backend1.close()

            # Create new backend and verify data
            backend2 = FileLogBackend(tmp_dir)
            await backend2.initialize()

            records = await backend2.read_all()
            assert len(records) == 3

            await backend2.close()


class TestWriteAheadLog:
    """Test Write-Ahead Log."""

    @pytest.mark.asyncio
    async def test_transaction_lifecycle(self):
        """Test complete transaction lifecycle."""
        wal = WriteAheadLog()

        # Begin transaction
        begin_lsn = await wal.begin_transaction("txn-1")
        assert begin_lsn > 0

        # Log writes
        write_lsn = await wal.log_write(
            "txn-1", "file.txt", None, "content"
        )
        assert write_lsn > begin_lsn

        # Commit
        commit_lsn = await wal.commit_transaction("txn-1")
        assert commit_lsn > write_lsn

        await wal.close()

    @pytest.mark.asyncio
    async def test_abort_transaction(self):
        """Test transaction abort."""
        wal = WriteAheadLog()

        await wal.begin_transaction("txn-1")
        await wal.log_write("txn-1", "file.txt", "old", "new")
        abort_lsn = await wal.abort_transaction("txn-1")

        assert abort_lsn > 0

        await wal.close()

    @pytest.mark.asyncio
    async def test_checkpoint(self):
        """Test checkpoint creation."""
        wal = WriteAheadLog(checkpoint_interval=10000)

        await wal.begin_transaction("txn-1")
        checkpoint_lsn = await wal.checkpoint()

        assert checkpoint_lsn > 0

        # Check checkpoint record
        records = await wal.backend.read_all()
        checkpoint_records = [
            r for r in records
            if r.record_type == LogRecordType.CHECKPOINT
        ]
        assert len(checkpoint_records) == 1
        assert "active_transactions" in checkpoint_records[0].data
        assert "txn-1" in checkpoint_records[0].data["active_transactions"]

        # Cleanup
        await wal.abort_transaction("txn-1")
        await wal.close()

    @pytest.mark.asyncio
    async def test_automatic_checkpoint(self):
        """Test automatic checkpoint on threshold."""
        wal = WriteAheadLog(checkpoint_interval=5)

        await wal.begin_transaction("txn-1")

        # Create enough writes to trigger checkpoint
        for i in range(6):
            await wal.log_write("txn-1", f"file_{i}.txt", None, f"content_{i}")

        records = await wal.backend.read_all()
        checkpoint_records = [
            r for r in records
            if r.record_type == LogRecordType.CHECKPOINT
        ]
        assert len(checkpoint_records) >= 1

        await wal.abort_transaction("txn-1")
        await wal.close()


class TestRecoveryManager:
    """Test recovery manager."""

    @pytest.mark.asyncio
    async def test_recovery_committed_transaction(self):
        """Test recovery of committed transaction."""
        wal = WriteAheadLog()
        applied_writes = {}

        async def apply_write(resource_id: str, value: str):
            applied_writes[resource_id] = value

        async def apply_undo(resource_id: str, old_value: str):
            if old_value is None:
                applied_writes.pop(resource_id, None)
            else:
                applied_writes[resource_id] = old_value

        # Simulate committed transaction
        await wal.begin_transaction("txn-1")
        await wal.log_write("txn-1", "file1.txt", None, "content1")
        await wal.log_write("txn-1", "file2.txt", None, "content2")
        await wal.commit_transaction("txn-1")

        # Recover
        recovery = RecoveryManager(wal, apply_write, apply_undo)
        result = await recovery.recover()

        assert "txn-1" in result.committed_transactions
        assert result.redo_count == 2
        assert result.undo_count == 0
        assert applied_writes == {"file1.txt": "content1", "file2.txt": "content2"}

        await wal.close()

    @pytest.mark.asyncio
    async def test_recovery_uncommitted_transaction(self):
        """Test recovery of uncommitted transaction (undo)."""
        wal = WriteAheadLog()
        applied_writes = {"file1.txt": "original"}

        async def apply_write(resource_id: str, value: str):
            applied_writes[resource_id] = value

        async def apply_undo(resource_id: str, old_value: str):
            if old_value is None:
                applied_writes.pop(resource_id, None)
            else:
                applied_writes[resource_id] = old_value

        # Simulate uncommitted transaction (crash before commit)
        await wal.begin_transaction("txn-1")
        await wal.log_write("txn-1", "file1.txt", "original", "modified")

        # Don't commit - simulate crash

        # Recover
        recovery = RecoveryManager(wal, apply_write, apply_undo)
        result = await recovery.recover()

        assert "txn-1" in result.aborted_transactions
        assert result.undo_count == 1
        assert applied_writes == {"file1.txt": "original"}

        await wal.close()

    @pytest.mark.asyncio
    async def test_recovery_mixed_transactions(self):
        """Test recovery with mix of committed and uncommitted."""
        wal = WriteAheadLog()
        applied_writes = {}

        async def apply_write(resource_id: str, value: str):
            applied_writes[resource_id] = value

        async def apply_undo(resource_id: str, old_value: str):
            if old_value is None:
                applied_writes.pop(resource_id, None)
            else:
                applied_writes[resource_id] = old_value

        # Committed transaction
        await wal.begin_transaction("txn-1")
        await wal.log_write("txn-1", "file1.txt", None, "content1")
        await wal.commit_transaction("txn-1")

        # Uncommitted transaction
        await wal.begin_transaction("txn-2")
        await wal.log_write("txn-2", "file2.txt", None, "should_be_undone")

        # Recover
        recovery = RecoveryManager(wal, apply_write, apply_undo)
        result = await recovery.recover()

        assert "txn-1" in result.committed_transactions
        assert "txn-2" in result.aborted_transactions
        assert "file1.txt" in applied_writes
        assert "file2.txt" not in applied_writes

        await wal.close()


class TestTransactionLogger:
    """Test transaction logger integration."""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test logger through full transaction workflow."""
        wal = WriteAheadLog()
        logger = TransactionLogger(wal)

        # Simulate transaction workflow
        await logger.on_transaction_begin("txn-1")
        await logger.on_write("txn-1", "file.txt", None, "content")
        await logger.on_transaction_commit("txn-1")

        # Verify logs
        records = await wal.backend.read_all()

        types = [r.record_type for r in records]
        assert LogRecordType.BEGIN in types
        assert LogRecordType.WRITE in types
        assert LogRecordType.COMMIT in types

        await logger.close()

    @pytest.mark.asyncio
    async def test_disabled_logger(self):
        """Test that disabled logger doesn't write."""
        wal = WriteAheadLog()
        logger = TransactionLogger(wal, enabled=False)

        await logger.on_transaction_begin("txn-1")
        await logger.on_write("txn-1", "file.txt", None, "content")

        records = await wal.backend.read_all()
        assert len(records) == 0

        await logger.close()

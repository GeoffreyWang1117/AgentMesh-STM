"""
Transaction Logging and Recovery for AgentMesh-STM.

This module provides Write-Ahead Logging (WAL) and recovery mechanisms
to ensure transaction durability and crash recovery.
"""

from __future__ import annotations

import asyncio
import json
import os
import struct
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Callable
from abc import ABC, abstractmethod

from agentmesh_stm.utils.logging import get_logger

logger = get_logger(__name__)


class LogRecordType(Enum):
    """Types of log records."""

    BEGIN = auto()       # Transaction start
    WRITE = auto()       # Write operation
    COMMIT = auto()      # Transaction commit
    ABORT = auto()       # Transaction abort
    CHECKPOINT = auto()  # Checkpoint marker
    COMPENSATE = auto()  # Compensation action


@dataclass
class LogRecord:
    """A single log record."""

    lsn: int  # Log Sequence Number
    timestamp: float
    record_type: LogRecordType
    transaction_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    prev_lsn: Optional[int] = None  # Previous LSN for this transaction

    def serialize(self) -> bytes:
        """Serialize record to bytes."""
        payload = {
            "lsn": self.lsn,
            "timestamp": self.timestamp,
            "record_type": self.record_type.name,
            "transaction_id": self.transaction_id,
            "data": self.data,
            "prev_lsn": self.prev_lsn,
        }
        data = json.dumps(payload).encode("utf-8")
        # Format: [length: 4 bytes][data: variable][checksum: 4 bytes]
        checksum = self._compute_checksum(data)
        return struct.pack(">I", len(data)) + data + struct.pack(">I", checksum)

    @classmethod
    def deserialize(cls, data: bytes) -> "LogRecord":
        """Deserialize record from bytes."""
        length = struct.unpack(">I", data[:4])[0]
        payload_data = data[4:4 + length]
        checksum = struct.unpack(">I", data[4 + length:8 + length])[0]

        # Verify checksum
        expected_checksum = cls._compute_checksum(payload_data)
        if checksum != expected_checksum:
            raise ValueError("Log record checksum mismatch")

        payload = json.loads(payload_data.decode("utf-8"))
        return cls(
            lsn=payload["lsn"],
            timestamp=payload["timestamp"],
            record_type=LogRecordType[payload["record_type"]],
            transaction_id=payload["transaction_id"],
            data=payload.get("data", {}),
            prev_lsn=payload.get("prev_lsn"),
        )

    @staticmethod
    def _compute_checksum(data: bytes) -> int:
        """Compute CRC32-like checksum."""
        checksum = 0
        for byte in data:
            checksum = ((checksum << 5) - checksum + byte) & 0xFFFFFFFF
        return checksum

    @property
    def record_size(self) -> int:
        """Get size of serialized record."""
        return len(self.serialize())


class LogBackend(ABC):
    """Abstract base class for log storage backends."""

    @abstractmethod
    async def append(self, record: LogRecord) -> int:
        """Append a record and return the LSN."""
        pass

    @abstractmethod
    async def read(self, lsn: int) -> Optional[LogRecord]:
        """Read a record by LSN."""
        pass

    @abstractmethod
    async def read_all(self, from_lsn: int = 0) -> List[LogRecord]:
        """Read all records from a given LSN."""
        pass

    @abstractmethod
    async def sync(self) -> None:
        """Force sync to durable storage."""
        pass

    @abstractmethod
    async def truncate(self, to_lsn: int) -> None:
        """Truncate log up to (and including) the given LSN."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the log backend."""
        pass


class FileLogBackend(LogBackend):
    """File-based log backend."""

    def __init__(self, log_dir: str, max_file_size: int = 64 * 1024 * 1024):
        self.log_dir = Path(log_dir)
        self.max_file_size = max_file_size
        self._current_file: Optional[Any] = None
        self._current_file_path: Optional[Path] = None
        self._current_lsn = 0
        self._file_index = 0
        self._lock = asyncio.Lock()
        self._lsn_to_file: Dict[int, Path] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the log backend."""
        if self._initialized:
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Find existing log files and recover state
        log_files = sorted(self.log_dir.glob("wal_*.log"))
        if log_files:
            # Read existing files to find last LSN
            for log_file in log_files:
                records = await self._read_file(log_file)
                for record in records:
                    self._lsn_to_file[record.lsn] = log_file
                    self._current_lsn = max(self._current_lsn, record.lsn)

            # Extract file index from last file
            last_file = log_files[-1]
            self._file_index = int(last_file.stem.split("_")[1])
            self._current_file_path = last_file

        self._initialized = True
        logger.info(
            "Log backend initialized",
            log_dir=str(self.log_dir),
            current_lsn=self._current_lsn,
        )

    async def _ensure_current_file(self) -> None:
        """Ensure we have a current file open."""
        if self._current_file is None or (
            self._current_file_path
            and self._current_file_path.stat().st_size >= self.max_file_size
        ):
            await self._rotate_file()

    async def _rotate_file(self) -> None:
        """Rotate to a new log file."""
        if self._current_file:
            self._current_file.close()

        self._file_index += 1
        self._current_file_path = self.log_dir / f"wal_{self._file_index:06d}.log"
        self._current_file = open(self._current_file_path, "ab")
        logger.debug("Rotated to new log file", file=str(self._current_file_path))

    async def _read_file(self, file_path: Path) -> List[LogRecord]:
        """Read all records from a file."""
        records = []
        try:
            with open(file_path, "rb") as f:
                while True:
                    length_data = f.read(4)
                    if not length_data or len(length_data) < 4:
                        break

                    length = struct.unpack(">I", length_data)[0]
                    data = length_data + f.read(length + 4)  # +4 for checksum
                    if len(data) < length + 8:
                        break

                    record = LogRecord.deserialize(data)
                    records.append(record)
        except Exception as e:
            logger.error("Error reading log file", file=str(file_path), error=str(e))

        return records

    async def append(self, record: LogRecord) -> int:
        """Append a record to the log."""
        async with self._lock:
            await self.initialize()
            await self._ensure_current_file()

            self._current_lsn += 1
            record.lsn = self._current_lsn

            data = record.serialize()
            self._current_file.write(data)
            self._lsn_to_file[record.lsn] = self._current_file_path

            return record.lsn

    async def read(self, lsn: int) -> Optional[LogRecord]:
        """Read a record by LSN."""
        file_path = self._lsn_to_file.get(lsn)
        if not file_path:
            return None

        records = await self._read_file(file_path)
        for record in records:
            if record.lsn == lsn:
                return record

        return None

    async def read_all(self, from_lsn: int = 0) -> List[LogRecord]:
        """Read all records from a given LSN."""
        await self.initialize()

        all_records = []
        log_files = sorted(self.log_dir.glob("wal_*.log"))

        for log_file in log_files:
            records = await self._read_file(log_file)
            for record in records:
                if record.lsn >= from_lsn:
                    all_records.append(record)

        return sorted(all_records, key=lambda r: r.lsn)

    async def sync(self) -> None:
        """Force sync to disk."""
        async with self._lock:
            if self._current_file:
                self._current_file.flush()
                os.fsync(self._current_file.fileno())

    async def truncate(self, to_lsn: int) -> None:
        """Truncate log up to the given LSN."""
        async with self._lock:
            files_to_delete = set()
            lsns_to_remove = []

            for lsn, file_path in self._lsn_to_file.items():
                if lsn <= to_lsn:
                    files_to_delete.add(file_path)
                    lsns_to_remove.append(lsn)

            # Don't delete the current file
            if self._current_file_path:
                files_to_delete.discard(self._current_file_path)

            for file_path in files_to_delete:
                try:
                    file_path.unlink()
                    logger.debug("Deleted log file", file=str(file_path))
                except OSError as e:
                    logger.error("Failed to delete log file", file=str(file_path), error=str(e))

            for lsn in lsns_to_remove:
                self._lsn_to_file.pop(lsn, None)

    async def close(self) -> None:
        """Close the log backend."""
        async with self._lock:
            if self._current_file:
                self._current_file.close()
                self._current_file = None


class InMemoryLogBackend(LogBackend):
    """In-memory log backend for testing."""

    def __init__(self):
        self._records: Dict[int, LogRecord] = {}
        self._current_lsn = 0
        self._lock = asyncio.Lock()

    async def append(self, record: LogRecord) -> int:
        async with self._lock:
            self._current_lsn += 1
            record.lsn = self._current_lsn
            self._records[record.lsn] = record
            return record.lsn

    async def read(self, lsn: int) -> Optional[LogRecord]:
        return self._records.get(lsn)

    async def read_all(self, from_lsn: int = 0) -> List[LogRecord]:
        return sorted(
            [r for r in self._records.values() if r.lsn >= from_lsn],
            key=lambda r: r.lsn,
        )

    async def sync(self) -> None:
        pass  # No-op for in-memory

    async def truncate(self, to_lsn: int) -> None:
        async with self._lock:
            self._records = {
                lsn: record
                for lsn, record in self._records.items()
                if lsn > to_lsn
            }

    async def close(self) -> None:
        pass


@dataclass
class TransactionState:
    """State of a transaction during recovery."""

    transaction_id: str
    status: str  # "active", "committed", "aborted"
    writes: List[Dict[str, Any]] = field(default_factory=list)
    start_lsn: int = 0
    last_lsn: int = 0


class WriteAheadLog:
    """
    Write-Ahead Log for transaction durability.

    Implements ARIES-style logging with:
    - Write-ahead logging (WAL)
    - Checkpointing
    - Recovery (Analysis, Redo, Undo)
    """

    def __init__(
        self,
        backend: Optional[LogBackend] = None,
        checkpoint_interval: int = 1000,  # Records between checkpoints
    ):
        self.backend = backend or InMemoryLogBackend()
        self.checkpoint_interval = checkpoint_interval
        self._active_transactions: Dict[str, TransactionState] = {}
        self._records_since_checkpoint = 0
        self._last_checkpoint_lsn = 0
        self._lock = asyncio.Lock()

    async def begin_transaction(self, transaction_id: str) -> int:
        """Log transaction begin."""
        record = LogRecord(
            lsn=0,
            timestamp=time.time(),
            record_type=LogRecordType.BEGIN,
            transaction_id=transaction_id,
        )
        lsn = await self.backend.append(record)

        async with self._lock:
            self._active_transactions[transaction_id] = TransactionState(
                transaction_id=transaction_id,
                status="active",
                start_lsn=lsn,
                last_lsn=lsn,
            )
            self._records_since_checkpoint += 1

        logger.debug("Transaction began", txn_id=transaction_id, lsn=lsn)
        return lsn

    async def log_write(
        self,
        transaction_id: str,
        resource_id: str,
        old_value: Optional[str],
        new_value: str,
    ) -> int:
        """Log a write operation."""
        async with self._lock:
            txn_state = self._active_transactions.get(transaction_id)
            if not txn_state:
                raise ValueError(f"Unknown transaction: {transaction_id}")

            prev_lsn = txn_state.last_lsn

        record = LogRecord(
            lsn=0,
            timestamp=time.time(),
            record_type=LogRecordType.WRITE,
            transaction_id=transaction_id,
            data={
                "resource_id": resource_id,
                "old_value": old_value,
                "new_value": new_value,
            },
            prev_lsn=prev_lsn,
        )
        lsn = await self.backend.append(record)

        async with self._lock:
            txn_state.last_lsn = lsn
            txn_state.writes.append(record.data)
            self._records_since_checkpoint += 1

        await self._maybe_checkpoint()
        return lsn

    async def commit_transaction(self, transaction_id: str) -> int:
        """Log transaction commit."""
        async with self._lock:
            txn_state = self._active_transactions.get(transaction_id)
            if not txn_state:
                raise ValueError(f"Unknown transaction: {transaction_id}")

            prev_lsn = txn_state.last_lsn

        record = LogRecord(
            lsn=0,
            timestamp=time.time(),
            record_type=LogRecordType.COMMIT,
            transaction_id=transaction_id,
            prev_lsn=prev_lsn,
        )
        lsn = await self.backend.append(record)

        # Force sync for durability
        await self.backend.sync()

        async with self._lock:
            txn_state.status = "committed"
            txn_state.last_lsn = lsn
            del self._active_transactions[transaction_id]
            self._records_since_checkpoint += 1

        logger.debug("Transaction committed", txn_id=transaction_id, lsn=lsn)
        return lsn

    async def abort_transaction(self, transaction_id: str) -> int:
        """Log transaction abort."""
        async with self._lock:
            txn_state = self._active_transactions.get(transaction_id)
            if not txn_state:
                raise ValueError(f"Unknown transaction: {transaction_id}")

            prev_lsn = txn_state.last_lsn

        record = LogRecord(
            lsn=0,
            timestamp=time.time(),
            record_type=LogRecordType.ABORT,
            transaction_id=transaction_id,
            prev_lsn=prev_lsn,
        )
        lsn = await self.backend.append(record)

        async with self._lock:
            txn_state.status = "aborted"
            txn_state.last_lsn = lsn
            del self._active_transactions[transaction_id]
            self._records_since_checkpoint += 1

        logger.debug("Transaction aborted", txn_id=transaction_id, lsn=lsn)
        return lsn

    async def log_compensate(
        self,
        transaction_id: str,
        resource_id: str,
        compensation_action: str,
    ) -> int:
        """Log a compensation action."""
        record = LogRecord(
            lsn=0,
            timestamp=time.time(),
            record_type=LogRecordType.COMPENSATE,
            transaction_id=transaction_id,
            data={
                "resource_id": resource_id,
                "action": compensation_action,
            },
        )
        lsn = await self.backend.append(record)

        async with self._lock:
            self._records_since_checkpoint += 1

        return lsn

    async def checkpoint(self) -> int:
        """Create a checkpoint."""
        async with self._lock:
            active_txns = {
                txn_id: {
                    "start_lsn": state.start_lsn,
                    "last_lsn": state.last_lsn,
                }
                for txn_id, state in self._active_transactions.items()
            }

        record = LogRecord(
            lsn=0,
            timestamp=time.time(),
            record_type=LogRecordType.CHECKPOINT,
            transaction_id="SYSTEM",
            data={
                "active_transactions": active_txns,
            },
        )
        lsn = await self.backend.append(record)
        await self.backend.sync()

        async with self._lock:
            self._last_checkpoint_lsn = lsn
            self._records_since_checkpoint = 0

        logger.info("Checkpoint created", lsn=lsn, active_txns=len(active_txns))
        return lsn

    async def _maybe_checkpoint(self) -> None:
        """Create checkpoint if needed."""
        if self._records_since_checkpoint >= self.checkpoint_interval:
            await self.checkpoint()

    async def close(self) -> None:
        """Close the WAL."""
        await self.backend.close()


@dataclass
class RecoveryResult:
    """Result of recovery operation."""

    committed_transactions: List[str]
    aborted_transactions: List[str]
    redo_count: int
    undo_count: int
    recovery_time_ms: float


class RecoveryManager:
    """
    Manages crash recovery using ARIES-style algorithm.

    Recovery phases:
    1. Analysis: Scan log to identify transactions
    2. Redo: Replay committed transactions
    3. Undo: Rollback uncommitted transactions
    """

    def __init__(
        self,
        wal: WriteAheadLog,
        apply_write: Callable[[str, str], Any],  # (resource_id, value) -> None
        apply_undo: Callable[[str, Optional[str]], Any],  # (resource_id, old_value) -> None
    ):
        self.wal = wal
        self.apply_write = apply_write
        self.apply_undo = apply_undo

    async def recover(self) -> RecoveryResult:
        """Perform crash recovery."""
        start_time = time.time()

        logger.info("Starting recovery")

        # Phase 1: Analysis
        transactions, start_lsn = await self._analysis_phase()

        # Phase 2: Redo
        redo_count = await self._redo_phase(transactions, start_lsn)

        # Phase 3: Undo
        undo_count = await self._undo_phase(transactions)

        recovery_time = (time.time() - start_time) * 1000

        committed = [
            txn_id for txn_id, state in transactions.items()
            if state.status == "committed"
        ]
        aborted = [
            txn_id for txn_id, state in transactions.items()
            if state.status == "aborted"
        ]

        result = RecoveryResult(
            committed_transactions=committed,
            aborted_transactions=aborted,
            redo_count=redo_count,
            undo_count=undo_count,
            recovery_time_ms=recovery_time,
        )

        logger.info(
            "Recovery completed",
            committed=len(committed),
            aborted=len(aborted),
            redo_count=redo_count,
            undo_count=undo_count,
            time_ms=recovery_time,
        )

        return result

    async def _analysis_phase(self) -> tuple[Dict[str, TransactionState], int]:
        """
        Analysis phase: Scan log to identify transactions.

        Returns transactions and the LSN to start redo from.
        """
        logger.debug("Starting analysis phase")

        records = await self.wal.backend.read_all()
        transactions: Dict[str, TransactionState] = {}
        start_lsn = 0

        for record in records:
            # Track checkpoint for redo start point
            if record.record_type == LogRecordType.CHECKPOINT:
                start_lsn = record.lsn
                # Recover active transaction list from checkpoint
                active_txns = record.data.get("active_transactions", {})
                for txn_id, info in active_txns.items():
                    if txn_id not in transactions:
                        transactions[txn_id] = TransactionState(
                            transaction_id=txn_id,
                            status="active",
                            start_lsn=info["start_lsn"],
                            last_lsn=info["last_lsn"],
                        )

            elif record.record_type == LogRecordType.BEGIN:
                transactions[record.transaction_id] = TransactionState(
                    transaction_id=record.transaction_id,
                    status="active",
                    start_lsn=record.lsn,
                    last_lsn=record.lsn,
                )

            elif record.record_type == LogRecordType.WRITE:
                if record.transaction_id in transactions:
                    transactions[record.transaction_id].writes.append(record.data)
                    transactions[record.transaction_id].last_lsn = record.lsn

            elif record.record_type == LogRecordType.COMMIT:
                if record.transaction_id in transactions:
                    transactions[record.transaction_id].status = "committed"
                    transactions[record.transaction_id].last_lsn = record.lsn

            elif record.record_type == LogRecordType.ABORT:
                if record.transaction_id in transactions:
                    transactions[record.transaction_id].status = "aborted"
                    transactions[record.transaction_id].last_lsn = record.lsn

        logger.debug(
            "Analysis phase complete",
            transactions=len(transactions),
            start_lsn=start_lsn,
        )

        return transactions, start_lsn

    async def _redo_phase(
        self,
        transactions: Dict[str, TransactionState],
        start_lsn: int,
    ) -> int:
        """
        Redo phase: Replay committed transaction writes.
        """
        logger.debug("Starting redo phase", start_lsn=start_lsn)

        records = await self.wal.backend.read_all(from_lsn=start_lsn)
        redo_count = 0

        for record in records:
            if record.record_type != LogRecordType.WRITE:
                continue

            txn_state = transactions.get(record.transaction_id)
            if not txn_state or txn_state.status != "committed":
                continue

            # Redo the write
            resource_id = record.data["resource_id"]
            new_value = record.data["new_value"]

            try:
                await self._apply_async(self.apply_write, resource_id, new_value)
                redo_count += 1
            except Exception as e:
                logger.error(
                    "Redo failed",
                    resource_id=resource_id,
                    error=str(e),
                )

        logger.debug("Redo phase complete", redo_count=redo_count)
        return redo_count

    async def _undo_phase(
        self,
        transactions: Dict[str, TransactionState],
    ) -> int:
        """
        Undo phase: Rollback uncommitted transaction writes.
        """
        logger.debug("Starting undo phase")

        undo_count = 0

        # Find transactions that need to be undone
        for txn_id, state in transactions.items():
            if state.status != "active":
                continue

            # Undo writes in reverse order
            for write in reversed(state.writes):
                resource_id = write["resource_id"]
                old_value = write.get("old_value")

                try:
                    await self._apply_async(self.apply_undo, resource_id, old_value)
                    undo_count += 1
                except Exception as e:
                    logger.error(
                        "Undo failed",
                        resource_id=resource_id,
                        error=str(e),
                    )

            # Log the abort
            await self.wal.abort_transaction(txn_id)

        logger.debug("Undo phase complete", undo_count=undo_count)
        return undo_count

    async def _apply_async(self, func: Callable, *args) -> Any:
        """Apply a function, handling both sync and async."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args)
        return func(*args)


class TransactionLogger:
    """
    High-level transaction logging interface.

    Integrates with TransactionManager to provide automatic
    logging of transaction operations.
    """

    def __init__(
        self,
        wal: WriteAheadLog,
        enabled: bool = True,
    ):
        self.wal = wal
        self.enabled = enabled
        self._transaction_lsns: Dict[str, int] = {}

    async def on_transaction_begin(self, transaction_id: str) -> None:
        """Called when a transaction begins."""
        if not self.enabled:
            return

        lsn = await self.wal.begin_transaction(transaction_id)
        self._transaction_lsns[transaction_id] = lsn

    async def on_write(
        self,
        transaction_id: str,
        resource_id: str,
        old_value: Optional[str],
        new_value: str,
    ) -> None:
        """Called when a write occurs."""
        if not self.enabled:
            return

        await self.wal.log_write(transaction_id, resource_id, old_value, new_value)

    async def on_transaction_commit(self, transaction_id: str) -> None:
        """Called when a transaction commits."""
        if not self.enabled:
            return

        await self.wal.commit_transaction(transaction_id)
        self._transaction_lsns.pop(transaction_id, None)

    async def on_transaction_abort(self, transaction_id: str) -> None:
        """Called when a transaction aborts."""
        if not self.enabled:
            return

        try:
            await self.wal.abort_transaction(transaction_id)
        except ValueError:
            pass  # Transaction may not have started logging yet
        self._transaction_lsns.pop(transaction_id, None)

    async def on_compensate(
        self,
        transaction_id: str,
        resource_id: str,
        action: str,
    ) -> None:
        """Called when a compensation action is performed."""
        if not self.enabled:
            return

        await self.wal.log_compensate(transaction_id, resource_id, action)

    async def close(self) -> None:
        """Close the logger."""
        await self.wal.close()

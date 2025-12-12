"""
Transaction Abstraction Layer for AgentMesh-STM.

This module provides the core transaction abstraction that wraps agent operations
as explicit transactions with optimistic concurrency control. Transactions support:
- Consistent snapshots at transaction start
- Read set tracking with version numbers
- Write buffering (writes go to local buffer, not shared state)
- Conflict validation at commit time
- Automatic rollback on conflict with optional retry
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    TypeVar,
)

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from agentmesh_stm.storage.mvcc import MVCCStorage
    from agentmesh_stm.conflict.detector import ConflictDetector
    from agentmesh_stm.compensation.manager import CompensationManager
    from agentmesh_stm.core.logging import TransactionLogger


class TransactionState(Enum):
    """States in the transaction lifecycle."""

    CREATED = auto()  # Transaction created but not started
    ACTIVE = auto()  # Transaction is executing
    VALIDATING = auto()  # Transaction is being validated for commit
    COMMITTED = auto()  # Transaction successfully committed
    ABORTED = auto()  # Transaction was aborted/rolled back
    RETRYING = auto()  # Transaction is retrying after conflict


@dataclass
class ReadSetEntry:
    """An entry in the transaction's read set."""

    resource_id: str
    version: int
    timestamp: datetime
    content_hash: Optional[str] = None


@dataclass
class WriteSetEntry:
    """An entry in the transaction's write set."""

    resource_id: str
    old_content: Optional[str]  # Original content for rollback
    new_content: str  # New content to be written
    operation_type: str = "write"  # write, delete, create


class ReadSet:
    """Tracks all resources read during a transaction."""

    def __init__(self):
        self._entries: Dict[str, ReadSetEntry] = {}

    def add(
        self,
        resource_id: str,
        version: int,
        timestamp: datetime,
        content_hash: Optional[str] = None,
    ) -> None:
        """Add a read operation to the read set."""
        self._entries[resource_id] = ReadSetEntry(
            resource_id=resource_id,
            version=version,
            timestamp=timestamp,
            content_hash=content_hash,
        )

    def get(self, resource_id: str) -> Optional[ReadSetEntry]:
        """Get read set entry for a resource."""
        return self._entries.get(resource_id)

    def contains(self, resource_id: str) -> bool:
        """Check if resource is in read set."""
        return resource_id in self._entries

    def get_all(self) -> List[ReadSetEntry]:
        """Get all read set entries."""
        return list(self._entries.values())

    def get_resource_ids(self) -> Set[str]:
        """Get all resource IDs in read set."""
        return set(self._entries.keys())

    def clear(self) -> None:
        """Clear the read set."""
        self._entries.clear()


class WriteSet:
    """Buffers all write operations during a transaction."""

    def __init__(self):
        self._entries: Dict[str, WriteSetEntry] = {}

    def add(
        self,
        resource_id: str,
        old_content: Optional[str],
        new_content: str,
        operation_type: str = "write",
    ) -> None:
        """Add a write operation to the write set."""
        self._entries[resource_id] = WriteSetEntry(
            resource_id=resource_id,
            old_content=old_content,
            new_content=new_content,
            operation_type=operation_type,
        )

    def get(self, resource_id: str) -> Optional[WriteSetEntry]:
        """Get write set entry for a resource."""
        return self._entries.get(resource_id)

    def contains(self, resource_id: str) -> bool:
        """Check if resource is in write set."""
        return resource_id in self._entries

    def get_all(self) -> List[WriteSetEntry]:
        """Get all write set entries."""
        return list(self._entries.values())

    def get_resource_ids(self) -> Set[str]:
        """Get all resource IDs in write set."""
        return set(self._entries.keys())

    def clear(self) -> None:
        """Clear the write set."""
        self._entries.clear()


class TransactionConfig(BaseModel):
    """Configuration for a transaction."""

    max_retries: int = Field(default=3, description="Maximum retry attempts on conflict")
    retry_delay_ms: int = Field(default=100, description="Delay between retries in milliseconds")
    timeout_seconds: Optional[float] = Field(
        default=300.0, description="Transaction timeout in seconds"
    )
    isolation_level: str = Field(
        default="snapshot", description="Isolation level: snapshot or serializable"
    )
    enable_compensation: bool = Field(
        default=True, description="Enable compensation for rollback"
    )


class Transaction:
    """
    Represents a single transaction in the STM system.

    A transaction provides:
    - Consistent snapshot of shared state at start time
    - Read tracking to detect read-write conflicts
    - Write buffering to delay commits
    - Automatic validation and conflict detection at commit
    - Compensation-based rollback for side effects
    """

    def __init__(
        self,
        transaction_id: Optional[str] = None,
        config: Optional[TransactionConfig] = None,
        storage: Optional["MVCCStorage"] = None,
        conflict_detector: Optional["ConflictDetector"] = None,
        compensation_manager: Optional["CompensationManager"] = None,
    ):
        self.id = transaction_id or str(uuid.uuid4())
        self.config = config or TransactionConfig()
        self._storage = storage
        self._conflict_detector = conflict_detector
        self._compensation_manager = compensation_manager

        self._state = TransactionState.CREATED
        self._read_set = ReadSet()
        self._write_set = WriteSet()
        self._snapshot_version: Optional[int] = None
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None
        self._retry_count = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> TransactionState:
        """Current transaction state."""
        return self._state

    @property
    def read_set(self) -> ReadSet:
        """The transaction's read set."""
        return self._read_set

    @property
    def write_set(self) -> WriteSet:
        """The transaction's write set."""
        return self._write_set

    @property
    def snapshot_version(self) -> Optional[int]:
        """The snapshot version this transaction is based on."""
        return self._snapshot_version

    @property
    def duration_seconds(self) -> Optional[float]:
        """Duration of the transaction in seconds."""
        if self._start_time is None:
            return None
        end = self._end_time or datetime.utcnow()
        return (end - self._start_time).total_seconds()

    async def begin(self) -> None:
        """Begin the transaction and capture a consistent snapshot."""
        async with self._lock:
            if self._state not in (TransactionState.CREATED, TransactionState.RETRYING):
                raise TransactionError(
                    f"Cannot begin transaction in state {self._state}"
                )

            self._state = TransactionState.ACTIVE
            self._start_time = datetime.utcnow()

            # Get current global version for snapshot
            if self._storage:
                self._snapshot_version = await self._storage.get_current_version()
            else:
                self._snapshot_version = 0

    async def read(self, resource_id: str) -> Optional[str]:
        """
        Read a resource within the transaction.

        Reads from the write set first (read-your-writes consistency),
        then from the snapshot version in storage.

        Args:
            resource_id: Identifier of the resource to read

        Returns:
            Resource content or None if not found
        """
        if self._state != TransactionState.ACTIVE:
            raise TransactionError(f"Cannot read in transaction state {self._state}")

        # Check write set first for read-your-writes consistency
        if self._write_set.contains(resource_id):
            entry = self._write_set.get(resource_id)
            return entry.new_content if entry else None

        # Read from storage at snapshot version
        if self._storage:
            resource = await self._storage.read(
                resource_id, version=self._snapshot_version
            )
            if resource:
                self._read_set.add(
                    resource_id=resource_id,
                    version=resource.version,
                    timestamp=resource.timestamp,
                    content_hash=resource.content_hash,
                )
                return resource.content
            return None
        else:
            # No storage configured, track the read but return None
            self._read_set.add(
                resource_id=resource_id,
                version=0,
                timestamp=datetime.utcnow(),
            )
            return None

    async def write(self, resource_id: str, content: str) -> None:
        """
        Write to a resource within the transaction.

        Writes are buffered in the write set and not applied to shared
        state until commit.

        Args:
            resource_id: Identifier of the resource to write
            content: New content for the resource
        """
        if self._state != TransactionState.ACTIVE:
            raise TransactionError(f"Cannot write in transaction state {self._state}")

        # Get old content for potential rollback
        old_content = None
        if self._storage:
            resource = await self._storage.read(
                resource_id, version=self._snapshot_version
            )
            if resource:
                old_content = resource.content

        # Buffer the write
        self._write_set.add(
            resource_id=resource_id,
            old_content=old_content,
            new_content=content,
            operation_type="write" if old_content else "create",
        )

    async def delete(self, resource_id: str) -> None:
        """
        Mark a resource for deletion within the transaction.

        Args:
            resource_id: Identifier of the resource to delete
        """
        if self._state != TransactionState.ACTIVE:
            raise TransactionError(f"Cannot delete in transaction state {self._state}")

        # Get old content for potential rollback
        old_content = None
        if self._storage:
            resource = await self._storage.read(
                resource_id, version=self._snapshot_version
            )
            if resource:
                old_content = resource.content

        self._write_set.add(
            resource_id=resource_id,
            old_content=old_content,
            new_content="",
            operation_type="delete",
        )

    async def validate(self) -> bool:
        """
        Validate the transaction for conflicts.

        Checks that no resources in the read set have been modified
        since the transaction started.

        Returns:
            True if validation passes, False if conflicts detected
        """
        async with self._lock:
            if self._state != TransactionState.ACTIVE:
                raise TransactionError(
                    f"Cannot validate transaction in state {self._state}"
                )

            self._state = TransactionState.VALIDATING

            # Check for read-write conflicts
            for entry in self._read_set.get_all():
                if self._storage:
                    current = await self._storage.get_current_resource_version(
                        entry.resource_id
                    )
                    if current and current > entry.version:
                        # Resource was modified after we read it
                        if self._conflict_detector:
                            # Use semantic conflict detection
                            conflict = await self._conflict_detector.check_conflict(
                                entry.resource_id, entry.version, current
                            )
                            if conflict.has_conflict:
                                self._state = TransactionState.ACTIVE
                                return False
                        else:
                            # Simple version-based conflict detection
                            self._state = TransactionState.ACTIVE
                            return False

            # Check for write-write conflicts
            for entry in self._write_set.get_all():
                if self._storage:
                    current_version = await self._storage.get_current_resource_version(
                        entry.resource_id
                    )
                    read_entry = self._read_set.get(entry.resource_id)
                    base_version = read_entry.version if read_entry else 0

                    if current_version and current_version > base_version:
                        if self._conflict_detector:
                            conflict = await self._conflict_detector.check_write_conflict(
                                entry.resource_id,
                                entry.new_content,
                                base_version,
                                current_version,
                            )
                            if conflict.has_conflict:
                                self._state = TransactionState.ACTIVE
                                return False
                        else:
                            self._state = TransactionState.ACTIVE
                            return False

            self._state = TransactionState.ACTIVE
            return True

    async def commit(self) -> bool:
        """
        Commit the transaction.

        Validates the transaction and atomically applies all buffered
        writes if validation passes.

        Returns:
            True if commit succeeded, False if aborted due to conflict
        """
        async with self._lock:
            if self._state != TransactionState.ACTIVE:
                raise TransactionError(
                    f"Cannot commit transaction in state {self._state}"
                )

            # Validate first
            self._state = TransactionState.VALIDATING
            is_valid = await self._validate_internal()

            if not is_valid:
                self._state = TransactionState.ACTIVE
                return False

            # Apply all writes atomically
            if self._storage:
                try:
                    await self._storage.atomic_write_batch(
                        self.id,
                        [
                            (entry.resource_id, entry.new_content, entry.operation_type)
                            for entry in self._write_set.get_all()
                        ],
                    )
                except Exception as e:
                    self._state = TransactionState.ACTIVE
                    raise TransactionError(f"Failed to apply writes: {e}") from e

            self._state = TransactionState.COMMITTED
            self._end_time = datetime.utcnow()
            return True

    async def _validate_internal(self) -> bool:
        """Internal validation without state management."""
        # Check read set for conflicts
        for entry in self._read_set.get_all():
            if self._storage:
                current = await self._storage.get_current_resource_version(
                    entry.resource_id
                )
                if current is not None and current > entry.version:
                    return False
        return True

    async def abort(self) -> None:
        """
        Abort the transaction and rollback any side effects.
        """
        async with self._lock:
            if self._state == TransactionState.COMMITTED:
                raise TransactionError("Cannot abort a committed transaction")

            if self._state == TransactionState.ABORTED:
                return  # Already aborted

            # Execute compensation for any side effects
            if self._compensation_manager and self.config.enable_compensation:
                await self._compensation_manager.rollback(self.id)

            self._state = TransactionState.ABORTED
            self._end_time = datetime.utcnow()

    async def retry(self) -> bool:
        """
        Prepare for retry after conflict.

        Returns:
            True if retry is allowed, False if max retries exceeded
        """
        async with self._lock:
            if self._retry_count >= self.config.max_retries:
                return False

            self._retry_count += 1
            self._read_set.clear()
            self._write_set.clear()
            self._state = TransactionState.RETRYING

            # Wait before retry
            await asyncio.sleep(self.config.retry_delay_ms / 1000.0)

            return True


class TransactionError(Exception):
    """Exception raised for transaction errors."""

    pass


class TransactionContext:
    """
    Context manager for transaction execution.

    Provides a convenient way to execute code within a transaction
    with automatic commit/rollback handling.
    """

    def __init__(self, transaction: Transaction, auto_retry: bool = True):
        self.transaction = transaction
        self.auto_retry = auto_retry
        self._committed = False

    async def __aenter__(self) -> Transaction:
        await self.transaction.begin()
        return self.transaction

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            # Exception occurred, abort transaction
            await self.transaction.abort()
            return False

        if not self._committed:
            # Try to commit
            success = await self.transaction.commit()
            if not success and self.auto_retry:
                # Conflict detected, try retry
                while await self.transaction.retry():
                    await self.transaction.begin()
                    # Note: The transaction body needs to be re-executed
                    # This context manager doesn't support automatic re-execution
                    # Use TransactionManager for full retry support
                    break

        return False


T = TypeVar("T")


class TransactionManager:
    """
    Manages transaction lifecycle and provides high-level transaction APIs.

    The TransactionManager coordinates between storage, conflict detection,
    and compensation systems to provide a complete STM implementation.
    """

    def __init__(
        self,
        storage: Optional["MVCCStorage"] = None,
        conflict_detector: Optional["ConflictDetector"] = None,
        compensation_manager: Optional["CompensationManager"] = None,
        default_config: Optional[TransactionConfig] = None,
        transaction_logger: Optional["TransactionLogger"] = None,
    ):
        self._storage = storage
        self._conflict_detector = conflict_detector
        self._compensation_manager = compensation_manager
        self._default_config = default_config or TransactionConfig()
        self._transaction_logger = transaction_logger
        self._active_transactions: Dict[str, Transaction] = {}
        self._lock = asyncio.Lock()

    async def create_transaction(
        self, config: Optional[TransactionConfig] = None
    ) -> Transaction:
        """Create a new transaction."""
        txn = Transaction(
            config=config or self._default_config,
            storage=self._storage,
            conflict_detector=self._conflict_detector,
            compensation_manager=self._compensation_manager,
        )

        async with self._lock:
            self._active_transactions[txn.id] = txn

        return txn

    async def get_transaction(self, transaction_id: str) -> Optional[Transaction]:
        """Get an active transaction by ID."""
        return self._active_transactions.get(transaction_id)

    async def execute(
        self,
        func: Callable[[Transaction], T],
        config: Optional[TransactionConfig] = None,
    ) -> T:
        """
        Execute a function within a transaction with automatic retry.

        This is the primary API for executing transactional operations.
        The function will be re-executed on conflict until it succeeds
        or max retries are exceeded.

        Args:
            func: Async function that takes a Transaction and returns a result
            config: Optional transaction configuration

        Returns:
            Result of the function execution

        Raises:
            TransactionError: If transaction cannot commit after all retries
        """
        txn = await self.create_transaction(config)
        retry_count = 0
        max_retries = txn.config.max_retries

        while retry_count <= max_retries:
            try:
                await txn.begin()

                # Log transaction begin
                if self._transaction_logger:
                    await self._transaction_logger.on_transaction_begin(txn.id)

                # Execute the user function
                if asyncio.iscoroutinefunction(func):
                    result = await func(txn)
                else:
                    result = func(txn)

                # Log writes before commit
                if self._transaction_logger:
                    for entry in txn.write_set.get_all():
                        await self._transaction_logger.on_write(
                            txn.id,
                            entry.resource_id,
                            entry.old_content,
                            entry.new_content,
                        )

                # Try to commit
                if await txn.commit():
                    # Log commit
                    if self._transaction_logger:
                        await self._transaction_logger.on_transaction_commit(txn.id)

                    async with self._lock:
                        self._active_transactions.pop(txn.id, None)
                    return result

                # Commit failed due to conflict, log abort
                if self._transaction_logger:
                    await self._transaction_logger.on_transaction_abort(txn.id)

                retry_count += 1
                if retry_count <= max_retries:
                    await asyncio.sleep(txn.config.retry_delay_ms / 1000.0)
                    txn._read_set.clear()
                    txn._write_set.clear()
                    txn._state = TransactionState.RETRYING

            except Exception as e:
                # Log abort on exception
                if self._transaction_logger:
                    await self._transaction_logger.on_transaction_abort(txn.id)

                await txn.abort()
                async with self._lock:
                    self._active_transactions.pop(txn.id, None)
                raise

        # Max retries exceeded, log abort
        if self._transaction_logger:
            await self._transaction_logger.on_transaction_abort(txn.id)

        await txn.abort()
        async with self._lock:
            self._active_transactions.pop(txn.id, None)
        raise TransactionError(
            f"Transaction failed after {max_retries} retries due to conflicts"
        )

    @asynccontextmanager
    async def transaction(self, config: Optional[TransactionConfig] = None):
        """
        Context manager for manual transaction control.

        Usage:
            async with manager.transaction() as txn:
                data = await txn.read("resource")
                await txn.write("resource", modified_data)
                # Commit happens automatically if no exception

        Args:
            config: Optional transaction configuration

        Yields:
            Transaction instance
        """
        txn = await self.create_transaction(config)

        try:
            await txn.begin()

            # Log transaction begin
            if self._transaction_logger:
                await self._transaction_logger.on_transaction_begin(txn.id)

            yield txn

            if txn.state == TransactionState.ACTIVE:
                # Log writes before commit
                if self._transaction_logger:
                    for entry in txn.write_set.get_all():
                        await self._transaction_logger.on_write(
                            txn.id,
                            entry.resource_id,
                            entry.old_content,
                            entry.new_content,
                        )

                if await txn.commit():
                    # Log commit
                    if self._transaction_logger:
                        await self._transaction_logger.on_transaction_commit(txn.id)
                else:
                    # Log abort
                    if self._transaction_logger:
                        await self._transaction_logger.on_transaction_abort(txn.id)
                    await txn.abort()
                    raise TransactionError("Transaction aborted due to conflict")

        except Exception:
            if txn.state not in (TransactionState.COMMITTED, TransactionState.ABORTED):
                # Log abort
                if self._transaction_logger:
                    await self._transaction_logger.on_transaction_abort(txn.id)
                await txn.abort()
            raise
        finally:
            async with self._lock:
                self._active_transactions.pop(txn.id, None)

    async def get_active_transaction_count(self) -> int:
        """Get the number of active transactions."""
        return len(self._active_transactions)

    async def abort_all(self) -> None:
        """Abort all active transactions."""
        async with self._lock:
            for txn in list(self._active_transactions.values()):
                try:
                    await txn.abort()
                except Exception:
                    pass
            self._active_transactions.clear()

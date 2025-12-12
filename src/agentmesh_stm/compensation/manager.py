"""
Compensation Transaction Manager for AgentMesh-STM.

This module handles rollback of side effects when transactions abort.
It maintains a log of compensable operations and executes them in
reverse order during rollback.

Supported side effects:
- File writes: Record original content, restore on rollback
- Git commits: Record commit hash, revert on rollback
- API calls: Record inverse operations, execute on rollback
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type
from uuid import uuid4

from pydantic import BaseModel, Field

from agentmesh_stm.utils.logging import get_logger

logger = get_logger(__name__)


class OperationType(Enum):
    """Types of compensable operations."""

    FILE_WRITE = auto()
    FILE_DELETE = auto()
    FILE_CREATE = auto()
    GIT_COMMIT = auto()
    GIT_BRANCH = auto()
    API_CALL = auto()
    CUSTOM = auto()


class CompensationState(Enum):
    """States of a compensation operation."""

    PENDING = auto()  # Logged but not compensated
    COMPENSATING = auto()  # Currently executing compensation
    COMPENSATED = auto()  # Successfully compensated
    FAILED = auto()  # Compensation failed


@dataclass
class CompensationLog:
    """Log entry for a compensable operation."""

    id: str
    transaction_id: str
    operation_type: OperationType
    timestamp: datetime
    state: CompensationState
    operation_data: Dict[str, Any]
    compensation_data: Dict[str, Any]
    error_message: Optional[str] = None

    @classmethod
    def create(
        cls,
        transaction_id: str,
        operation_type: OperationType,
        operation_data: Dict[str, Any],
        compensation_data: Dict[str, Any],
    ) -> "CompensationLog":
        """Create a new compensation log entry."""
        return cls(
            id=str(uuid4()),
            transaction_id=transaction_id,
            operation_type=operation_type,
            timestamp=datetime.utcnow(),
            state=CompensationState.PENDING,
            operation_data=operation_data,
            compensation_data=compensation_data,
        )


class CompensableOperation(ABC):
    """Base class for compensable operations."""

    operation_type: OperationType = OperationType.CUSTOM

    @abstractmethod
    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the operation.

        Args:
            data: Operation parameters

        Returns:
            Data needed for compensation
        """
        pass

    @abstractmethod
    async def compensate(self, compensation_data: Dict[str, Any]) -> bool:
        """
        Compensate (rollback) the operation.

        Args:
            compensation_data: Data from execute() needed for rollback

        Returns:
            True if compensation succeeded
        """
        pass

    @abstractmethod
    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate operation parameters."""
        pass


class CompensationManagerConfig(BaseModel):
    """Configuration for compensation manager."""

    log_dir: str = Field(default=".agentmesh/compensation", description="Directory for logs")
    max_retry_attempts: int = Field(default=3, description="Max compensation retries")
    retry_delay_seconds: float = Field(default=1.0, description="Delay between retries")
    persist_logs: bool = Field(default=True, description="Persist logs to disk")


class CompensationManager:
    """
    Manages compensation for transaction side effects.

    The manager maintains a log of all side effects performed during
    a transaction and can roll them back in reverse order if the
    transaction aborts.
    """

    def __init__(self, config: Optional[CompensationManagerConfig] = None):
        self._config = config or CompensationManagerConfig()
        self._logs: Dict[str, List[CompensationLog]] = {}  # transaction_id -> logs
        self._operations: Dict[OperationType, CompensableOperation] = {}
        self._lock = asyncio.Lock()

        # Register built-in operations
        self._register_builtin_operations()

    def _register_builtin_operations(self) -> None:
        """Register built-in compensable operations."""
        from agentmesh_stm.compensation.operations import (
            FileWriteCompensation,
            FileDeleteCompensation,
            FileCreateCompensation,
            GitCommitCompensation,
            APICallCompensation,
        )

        self.register_operation(FileWriteCompensation())
        self.register_operation(FileDeleteCompensation())
        self.register_operation(FileCreateCompensation())
        self.register_operation(GitCommitCompensation())
        self.register_operation(APICallCompensation())

    def register_operation(self, operation: CompensableOperation) -> None:
        """Register a compensable operation handler."""
        self._operations[operation.operation_type] = operation

    async def log_operation(
        self,
        transaction_id: str,
        operation_type: OperationType,
        operation_data: Dict[str, Any],
        compensation_data: Dict[str, Any],
    ) -> CompensationLog:
        """
        Log a compensable operation.

        Args:
            transaction_id: ID of the transaction
            operation_type: Type of operation
            operation_data: Parameters used for the operation
            compensation_data: Data needed for compensation

        Returns:
            The created CompensationLog entry
        """
        async with self._lock:
            log = CompensationLog.create(
                transaction_id=transaction_id,
                operation_type=operation_type,
                operation_data=operation_data,
                compensation_data=compensation_data,
            )

            if transaction_id not in self._logs:
                self._logs[transaction_id] = []

            self._logs[transaction_id].append(log)

            if self._config.persist_logs:
                await self._persist_log(log)

            logger.info(
                "Logged compensable operation",
                transaction_id=transaction_id,
                operation_type=operation_type.name,
                log_id=log.id,
            )

            return log

    async def execute_with_compensation(
        self,
        transaction_id: str,
        operation_type: OperationType,
        operation_data: Dict[str, Any],
    ) -> Any:
        """
        Execute an operation and log it for potential compensation.

        Args:
            transaction_id: ID of the transaction
            operation_type: Type of operation to execute
            operation_data: Parameters for the operation

        Returns:
            Result of the operation
        """
        operation = self._operations.get(operation_type)
        if not operation:
            raise ValueError(f"Unknown operation type: {operation_type}")

        if not operation.validate(operation_data):
            raise ValueError(f"Invalid operation data for {operation_type}")

        # Execute the operation
        compensation_data = await operation.execute(operation_data)

        # Log for compensation
        await self.log_operation(
            transaction_id=transaction_id,
            operation_type=operation_type,
            operation_data=operation_data,
            compensation_data=compensation_data,
        )

        return compensation_data

    async def rollback(self, transaction_id: str) -> bool:
        """
        Rollback all operations for a transaction.

        Operations are compensated in reverse order (LIFO).

        Args:
            transaction_id: ID of the transaction to rollback

        Returns:
            True if all compensations succeeded
        """
        async with self._lock:
            logs = self._logs.get(transaction_id, [])

        if not logs:
            logger.debug("No operations to rollback", transaction_id=transaction_id)
            return True

        # Reverse order for compensation
        logs_to_compensate = list(reversed(logs))
        all_succeeded = True

        for log in logs_to_compensate:
            if log.state == CompensationState.COMPENSATED:
                continue

            success = await self._compensate_single(log)
            if not success:
                all_succeeded = False

        # Clean up logs if all succeeded
        if all_succeeded:
            async with self._lock:
                self._logs.pop(transaction_id, None)
                if self._config.persist_logs:
                    await self._delete_transaction_logs(transaction_id)

        return all_succeeded

    async def _compensate_single(self, log: CompensationLog) -> bool:
        """Compensate a single operation with retries."""
        operation = self._operations.get(log.operation_type)
        if not operation:
            logger.error(
                "Unknown operation type for compensation",
                operation_type=log.operation_type.name,
                log_id=log.id,
            )
            return False

        log.state = CompensationState.COMPENSATING

        for attempt in range(self._config.max_retry_attempts):
            try:
                success = await operation.compensate(log.compensation_data)
                if success:
                    log.state = CompensationState.COMPENSATED
                    logger.info(
                        "Compensation succeeded",
                        log_id=log.id,
                        operation_type=log.operation_type.name,
                    )
                    return True
                else:
                    logger.warning(
                        "Compensation returned false",
                        log_id=log.id,
                        attempt=attempt + 1,
                    )
            except Exception as e:
                logger.error(
                    "Compensation failed",
                    log_id=log.id,
                    attempt=attempt + 1,
                    error=str(e),
                )
                log.error_message = str(e)

            if attempt < self._config.max_retry_attempts - 1:
                await asyncio.sleep(self._config.retry_delay_seconds)

        log.state = CompensationState.FAILED
        return False

    async def commit(self, transaction_id: str) -> None:
        """
        Commit a transaction, discarding compensation logs.

        Args:
            transaction_id: ID of the transaction to commit
        """
        async with self._lock:
            self._logs.pop(transaction_id, None)
            if self._config.persist_logs:
                await self._delete_transaction_logs(transaction_id)

        logger.debug("Transaction committed, compensation logs cleared", transaction_id=transaction_id)

    async def get_pending_compensations(
        self, transaction_id: str
    ) -> List[CompensationLog]:
        """Get all pending compensation logs for a transaction."""
        return [
            log
            for log in self._logs.get(transaction_id, [])
            if log.state == CompensationState.PENDING
        ]

    async def _persist_log(self, log: CompensationLog) -> None:
        """Persist a compensation log to disk."""
        log_dir = Path(self._config.log_dir) / log.transaction_id
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"{log.id}.json"
        log_data = {
            "id": log.id,
            "transaction_id": log.transaction_id,
            "operation_type": log.operation_type.name,
            "timestamp": log.timestamp.isoformat(),
            "state": log.state.name,
            "operation_data": log.operation_data,
            "compensation_data": log.compensation_data,
            "error_message": log.error_message,
        }

        async with asyncio.Lock():
            with open(log_file, "w") as f:
                json.dump(log_data, f, indent=2)

    async def _delete_transaction_logs(self, transaction_id: str) -> None:
        """Delete persisted logs for a transaction."""
        log_dir = Path(self._config.log_dir) / transaction_id
        if log_dir.exists():
            shutil.rmtree(log_dir)

    async def recover_pending(self) -> List[str]:
        """
        Recover pending compensation logs from disk.

        Returns:
            List of transaction IDs with pending compensations
        """
        if not self._config.persist_logs:
            return []

        log_base = Path(self._config.log_dir)
        if not log_base.exists():
            return []

        recovered_transactions = []

        for txn_dir in log_base.iterdir():
            if not txn_dir.is_dir():
                continue

            transaction_id = txn_dir.name
            logs = []

            for log_file in txn_dir.glob("*.json"):
                try:
                    with open(log_file) as f:
                        data = json.load(f)

                    log = CompensationLog(
                        id=data["id"],
                        transaction_id=data["transaction_id"],
                        operation_type=OperationType[data["operation_type"]],
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        state=CompensationState[data["state"]],
                        operation_data=data["operation_data"],
                        compensation_data=data["compensation_data"],
                        error_message=data.get("error_message"),
                    )
                    logs.append(log)
                except Exception as e:
                    logger.error(
                        "Failed to recover log",
                        file=str(log_file),
                        error=str(e),
                    )

            if logs:
                self._logs[transaction_id] = logs
                recovered_transactions.append(transaction_id)

        return recovered_transactions

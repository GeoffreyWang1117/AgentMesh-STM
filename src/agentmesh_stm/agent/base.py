"""
Base Agent Abstraction for AgentMesh-STM.

This module provides the base Agent class that wraps agent operations
as STM transactions, ensuring consistent state management and
automatic conflict handling.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field

from agentmesh_stm.core.transaction import (
    Transaction,
    TransactionConfig,
    TransactionError,
    TransactionManager,
    TransactionState,
)
from agentmesh_stm.utils.logging import get_logger

logger = get_logger(__name__)


class TaskStatus(Enum):
    """Status of an agent task."""

    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    RETRYING = auto()


class TaskPriority(Enum):
    """Priority levels for tasks."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AgentTask:
    """Represents a task to be executed by an agent."""

    id: str
    description: str
    input_data: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # Task IDs this depends on
    retry_count: int = 0
    max_retries: int = 3

    @classmethod
    def create(
        cls,
        description: str,
        input_data: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        dependencies: Optional[List[str]] = None,
        **metadata,
    ) -> "AgentTask":
        """Create a new agent task."""
        return cls(
            id=str(uuid4()),
            description=description,
            input_data=input_data,
            priority=priority,
            dependencies=dependencies or [],
            metadata=metadata,
        )


@dataclass
class AgentResult:
    """Result of an agent task execution."""

    task_id: str
    success: bool
    output_data: Dict[str, Any]
    error_message: Optional[str] = None
    execution_time_seconds: float = 0.0
    transaction_id: Optional[str] = None
    retry_count: int = 0
    modified_resources: List[str] = field(default_factory=list)

    @classmethod
    def success_result(
        cls,
        task_id: str,
        output_data: Dict[str, Any],
        execution_time: float,
        transaction_id: Optional[str] = None,
        modified_resources: Optional[List[str]] = None,
    ) -> "AgentResult":
        """Create a successful result."""
        return cls(
            task_id=task_id,
            success=True,
            output_data=output_data,
            execution_time_seconds=execution_time,
            transaction_id=transaction_id,
            modified_resources=modified_resources or [],
        )

    @classmethod
    def failure_result(
        cls,
        task_id: str,
        error_message: str,
        execution_time: float,
        retry_count: int = 0,
    ) -> "AgentResult":
        """Create a failure result."""
        return cls(
            task_id=task_id,
            success=False,
            output_data={},
            error_message=error_message,
            execution_time_seconds=execution_time,
            retry_count=retry_count,
        )


class AgentConfig(BaseModel):
    """Configuration for an agent."""

    name: str = Field(default="agent", description="Agent name")
    max_concurrent_tasks: int = Field(default=1, description="Max concurrent tasks")
    transaction_config: Optional[TransactionConfig] = Field(
        default=None, description="Transaction configuration"
    )
    auto_retry_on_conflict: bool = Field(
        default=True, description="Auto retry on transaction conflict"
    )
    task_timeout_seconds: float = Field(
        default=300.0, description="Task execution timeout"
    )


class Agent(ABC):
    """
    Base class for agents in the AgentMesh-STM framework.

    An Agent:
    - Executes tasks within STM transactions
    - Automatically handles transaction conflicts and retries
    - Provides hooks for task lifecycle events
    - Supports concurrent task execution with proper isolation
    """

    def __init__(
        self,
        transaction_manager: TransactionManager,
        config: Optional[AgentConfig] = None,
    ):
        self._transaction_manager = transaction_manager
        self._config = config or AgentConfig()
        self._id = str(uuid4())
        self._active_tasks: Dict[str, AgentTask] = {}
        self._completed_tasks: Dict[str, AgentResult] = {}
        self._running = False
        self._semaphore = asyncio.Semaphore(self._config.max_concurrent_tasks)
        self._lock = asyncio.Lock()

    @property
    def id(self) -> str:
        """Agent unique identifier."""
        return self._id

    @property
    def name(self) -> str:
        """Agent name."""
        return self._config.name

    @property
    def is_running(self) -> bool:
        """Whether the agent is currently running."""
        return self._running

    @abstractmethod
    async def execute_task(
        self, task: AgentTask, transaction: Transaction
    ) -> Dict[str, Any]:
        """
        Execute a task within a transaction.

        This method should be implemented by subclasses to define
        the agent's specific behavior.

        Args:
            task: The task to execute
            transaction: Active transaction for the task

        Returns:
            Output data from task execution
        """
        pass

    async def submit_task(self, task: AgentTask) -> str:
        """
        Submit a task for execution.

        Args:
            task: Task to submit

        Returns:
            Task ID
        """
        async with self._lock:
            self._active_tasks[task.id] = task

        logger.info(
            "Task submitted",
            agent=self.name,
            task_id=task.id,
            description=task.description,
        )

        # Start execution if agent is running
        if self._running:
            asyncio.create_task(self._execute_task_with_transaction(task))

        return task.id

    async def run_task(self, task: AgentTask) -> AgentResult:
        """
        Run a task synchronously and return the result.

        Args:
            task: Task to run

        Returns:
            AgentResult with execution outcome
        """
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.utcnow()

        start_time = datetime.utcnow()

        try:
            result = await asyncio.wait_for(
                self._execute_task_with_transaction(task),
                timeout=self._config.task_timeout_seconds,
            )
            return result
        except asyncio.TimeoutError:
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.utcnow()
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            return AgentResult.failure_result(
                task_id=task.id,
                error_message="Task execution timed out",
                execution_time=execution_time,
            )

    async def _execute_task_with_transaction(self, task: AgentTask) -> AgentResult:
        """Execute a task within a transaction with retry handling."""
        async with self._semaphore:
            start_time = datetime.utcnow()
            retry_count = 0

            while retry_count <= task.max_retries:
                task.status = TaskStatus.RUNNING if retry_count == 0 else TaskStatus.RETRYING
                task.retry_count = retry_count

                try:
                    # Execute within transaction
                    transaction = await self._transaction_manager.create_transaction(
                        self._config.transaction_config
                    )

                    await transaction.begin()

                    # Pre-execution hook
                    await self.on_task_started(task, transaction)

                    # Execute the task
                    output_data = await self.execute_task(task, transaction)

                    # Commit transaction
                    if await transaction.commit():
                        execution_time = (datetime.utcnow() - start_time).total_seconds()

                        task.status = TaskStatus.COMPLETED
                        task.completed_at = datetime.utcnow()

                        result = AgentResult.success_result(
                            task_id=task.id,
                            output_data=output_data,
                            execution_time=execution_time,
                            transaction_id=transaction.id,
                            modified_resources=list(transaction.write_set.get_resource_ids()),
                        )

                        # Post-execution hook
                        await self.on_task_completed(task, result)

                        async with self._lock:
                            self._completed_tasks[task.id] = result
                            self._active_tasks.pop(task.id, None)

                        return result

                    # Commit failed - conflict detected
                    logger.warning(
                        "Transaction conflict",
                        agent=self.name,
                        task_id=task.id,
                        retry=retry_count,
                    )

                    await transaction.abort()

                    if not self._config.auto_retry_on_conflict:
                        break

                    retry_count += 1
                    await asyncio.sleep(0.1 * (2**retry_count))  # Exponential backoff

                except TransactionError as e:
                    logger.error(
                        "Transaction error",
                        agent=self.name,
                        task_id=task.id,
                        error=str(e),
                    )
                    retry_count += 1

                except Exception as e:
                    logger.error(
                        "Task execution error",
                        agent=self.name,
                        task_id=task.id,
                        error=str(e),
                    )

                    task.status = TaskStatus.FAILED
                    task.completed_at = datetime.utcnow()
                    execution_time = (datetime.utcnow() - start_time).total_seconds()

                    result = AgentResult.failure_result(
                        task_id=task.id,
                        error_message=str(e),
                        execution_time=execution_time,
                        retry_count=retry_count,
                    )

                    await self.on_task_failed(task, result)

                    async with self._lock:
                        self._completed_tasks[task.id] = result
                        self._active_tasks.pop(task.id, None)

                    return result

            # Max retries exceeded
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.utcnow()

            result = AgentResult.failure_result(
                task_id=task.id,
                error_message=f"Max retries exceeded ({task.max_retries})",
                execution_time=execution_time,
                retry_count=retry_count,
            )

            await self.on_task_failed(task, result)

            async with self._lock:
                self._completed_tasks[task.id] = result
                self._active_tasks.pop(task.id, None)

            return result

    async def start(self) -> None:
        """Start the agent to process submitted tasks."""
        self._running = True
        logger.info("Agent started", agent=self.name, agent_id=self.id)

        # Process any pending tasks
        async with self._lock:
            pending = [t for t in self._active_tasks.values() if t.status == TaskStatus.PENDING]

        for task in pending:
            asyncio.create_task(self._execute_task_with_transaction(task))

    async def stop(self) -> None:
        """Stop the agent."""
        self._running = False
        logger.info("Agent stopped", agent=self.name, agent_id=self.id)

    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get the status of a task."""
        if task_id in self._active_tasks:
            return self._active_tasks[task_id].status
        if task_id in self._completed_tasks:
            return (
                TaskStatus.COMPLETED
                if self._completed_tasks[task_id].success
                else TaskStatus.FAILED
            )
        return None

    async def get_task_result(self, task_id: str) -> Optional[AgentResult]:
        """Get the result of a completed task."""
        return self._completed_tasks.get(task_id)

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        async with self._lock:
            if task_id in self._active_tasks:
                task = self._active_tasks[task_id]
                if task.status == TaskStatus.PENDING:
                    task.status = TaskStatus.CANCELLED
                    del self._active_tasks[task_id]
                    return True
        return False

    # Lifecycle hooks for subclasses
    async def on_task_started(self, task: AgentTask, transaction: Transaction) -> None:
        """Called when a task starts executing."""
        pass

    async def on_task_completed(self, task: AgentTask, result: AgentResult) -> None:
        """Called when a task completes successfully."""
        pass

    async def on_task_failed(self, task: AgentTask, result: AgentResult) -> None:
        """Called when a task fails."""
        pass


T = TypeVar("T")


class SimpleAgent(Agent):
    """
    A simple agent that executes a callable function within transactions.

    Useful for wrapping existing functions with STM semantics.
    """

    def __init__(
        self,
        transaction_manager: TransactionManager,
        handler: Callable[[AgentTask, Transaction], Dict[str, Any]],
        config: Optional[AgentConfig] = None,
    ):
        super().__init__(transaction_manager, config)
        self._handler = handler

    async def execute_task(
        self, task: AgentTask, transaction: Transaction
    ) -> Dict[str, Any]:
        """Execute the task using the configured handler."""
        if asyncio.iscoroutinefunction(self._handler):
            return await self._handler(task, transaction)
        return self._handler(task, transaction)

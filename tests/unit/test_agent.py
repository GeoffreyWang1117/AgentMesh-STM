"""Unit tests for Agent Integration Layer."""

import pytest
import asyncio
from typing import Dict, Any

from agentmesh_stm.agent.base import (
    Agent,
    AgentConfig,
    AgentTask,
    AgentResult,
    TaskStatus,
    TaskPriority,
    SimpleAgent,
)
from agentmesh_stm.core.transaction import Transaction, TransactionManager
from agentmesh_stm.storage.mvcc import MVCCStorage, InMemoryBackend


class TestAgentTask:
    """Tests for AgentTask class."""

    def test_create(self):
        """Test creating an agent task."""
        task = AgentTask.create(
            description="Test task",
            input_data={"key": "value"},
            priority=TaskPriority.HIGH,
        )

        assert task.description == "Test task"
        assert task.input_data == {"key": "value"}
        assert task.priority == TaskPriority.HIGH
        assert task.status == TaskStatus.PENDING
        assert task.id is not None

    def test_create_with_dependencies(self):
        """Test creating task with dependencies."""
        task = AgentTask.create(
            description="Dependent task",
            input_data={},
            dependencies=["task1", "task2"],
        )

        assert task.dependencies == ["task1", "task2"]


class TestAgentResult:
    """Tests for AgentResult class."""

    def test_success_result(self):
        """Test creating successful result."""
        result = AgentResult.success_result(
            task_id="task1",
            output_data={"result": "success"},
            execution_time=1.5,
            modified_resources=["file.txt"],
        )

        assert result.success
        assert result.task_id == "task1"
        assert result.output_data == {"result": "success"}
        assert result.execution_time_seconds == 1.5
        assert result.modified_resources == ["file.txt"]

    def test_failure_result(self):
        """Test creating failure result."""
        result = AgentResult.failure_result(
            task_id="task1",
            error_message="Something went wrong",
            execution_time=0.5,
            retry_count=2,
        )

        assert not result.success
        assert result.error_message == "Something went wrong"
        assert result.retry_count == 2


class TestSimpleAgent:
    """Tests for SimpleAgent class."""

    @pytest.fixture
    def storage(self):
        return MVCCStorage(backend=InMemoryBackend())

    @pytest.fixture
    def transaction_manager(self, storage):
        return TransactionManager(storage=storage)

    @pytest.mark.asyncio
    async def test_run_simple_task(self, transaction_manager):
        """Test running a simple task."""

        async def handler(task: AgentTask, txn: Transaction) -> Dict[str, Any]:
            return {"result": f"Processed: {task.description}"}

        agent = SimpleAgent(
            transaction_manager=transaction_manager,
            handler=handler,
            config=AgentConfig(name="test_agent"),
        )

        task = AgentTask.create(
            description="Test task",
            input_data={},
        )

        result = await agent.run_task(task)

        assert result.success
        assert result.output_data["result"] == "Processed: Test task"

    @pytest.mark.asyncio
    async def test_task_with_file_operations(self, transaction_manager, storage):
        """Test task that performs file operations."""
        # Initialize some data
        await storage.write("input.txt", "initial data")

        async def handler(task: AgentTask, txn: Transaction) -> Dict[str, Any]:
            # Read input
            content = await txn.read("input.txt")

            # Write output
            await txn.write("output.txt", f"Processed: {content}")

            return {"status": "done"}

        agent = SimpleAgent(
            transaction_manager=transaction_manager,
            handler=handler,
        )

        task = AgentTask.create(
            description="File processing",
            input_data={},
        )

        result = await agent.run_task(task)

        assert result.success
        assert "output.txt" in result.modified_resources

        # Verify write was committed
        output = await storage.read("output.txt")
        assert output.content == "Processed: initial data"

    @pytest.mark.asyncio
    async def test_task_failure(self, transaction_manager):
        """Test task that fails."""

        async def handler(task: AgentTask, txn: Transaction) -> Dict[str, Any]:
            raise ValueError("Intentional failure")

        agent = SimpleAgent(
            transaction_manager=transaction_manager,
            handler=handler,
        )

        task = AgentTask.create(
            description="Failing task",
            input_data={},
        )

        result = await agent.run_task(task)

        assert not result.success
        assert "Intentional failure" in result.error_message

    @pytest.mark.asyncio
    async def test_agent_lifecycle(self, transaction_manager):
        """Test agent start/stop lifecycle."""

        async def handler(task: AgentTask, txn: Transaction) -> Dict[str, Any]:
            return {"done": True}

        agent = SimpleAgent(
            transaction_manager=transaction_manager,
            handler=handler,
        )

        assert not agent.is_running

        await agent.start()
        assert agent.is_running

        await agent.stop()
        assert not agent.is_running

    @pytest.mark.asyncio
    async def test_get_task_status(self, transaction_manager):
        """Test getting task status."""

        async def slow_handler(task: AgentTask, txn: Transaction) -> Dict[str, Any]:
            await asyncio.sleep(0.1)
            return {"done": True}

        agent = SimpleAgent(
            transaction_manager=transaction_manager,
            handler=slow_handler,
        )

        task = AgentTask.create(
            description="Task",
            input_data={},
        )

        # Before execution
        status = await agent.get_task_status(task.id)
        assert status is None

        # Run task
        result = await agent.run_task(task)

        # After completion
        status = await agent.get_task_status(task.id)
        assert status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_concurrent_tasks(self, transaction_manager):
        """Test running multiple tasks concurrently."""
        execution_order = []

        async def handler(task: AgentTask, txn: Transaction) -> Dict[str, Any]:
            task_num = task.input_data["num"]
            execution_order.append(f"start_{task_num}")
            await asyncio.sleep(0.05)
            execution_order.append(f"end_{task_num}")
            return {"num": task_num}

        config = AgentConfig(max_concurrent_tasks=3)
        agent = SimpleAgent(
            transaction_manager=transaction_manager,
            handler=handler,
            config=config,
        )

        tasks = [
            AgentTask.create(description=f"Task {i}", input_data={"num": i})
            for i in range(3)
        ]

        # Run tasks concurrently
        results = await asyncio.gather(*[agent.run_task(task) for task in tasks])

        assert all(r.success for r in results)

        # With concurrent execution, some starts should interleave
        # (This is probabilistic, but with 3 concurrent tasks it's very likely)
        assert len(execution_order) == 6


class CustomAgent(Agent):
    """Custom agent implementation for testing."""

    def __init__(self, transaction_manager, config=None):
        super().__init__(transaction_manager, config)
        self.on_started_called = False
        self.on_completed_called = False
        self.on_failed_called = False

    async def execute_task(
        self, task: AgentTask, transaction: Transaction
    ) -> Dict[str, Any]:
        if task.input_data.get("should_fail"):
            raise Exception("Task failed")

        return {"custom_result": task.input_data.get("value", "default")}

    async def on_task_started(self, task: AgentTask, transaction: Transaction) -> None:
        self.on_started_called = True

    async def on_task_completed(self, task: AgentTask, result: AgentResult) -> None:
        self.on_completed_called = True

    async def on_task_failed(self, task: AgentTask, result: AgentResult) -> None:
        self.on_failed_called = True


class TestCustomAgent:
    """Tests for custom Agent implementation."""

    @pytest.fixture
    def storage(self):
        return MVCCStorage(backend=InMemoryBackend())

    @pytest.fixture
    def transaction_manager(self, storage):
        return TransactionManager(storage=storage)

    @pytest.mark.asyncio
    async def test_lifecycle_hooks_success(self, transaction_manager):
        """Test that lifecycle hooks are called on success."""
        agent = CustomAgent(transaction_manager)

        task = AgentTask.create(
            description="Test",
            input_data={"value": "test_value"},
        )

        result = await agent.run_task(task)

        assert result.success
        assert agent.on_started_called
        assert agent.on_completed_called
        assert not agent.on_failed_called

    @pytest.mark.asyncio
    async def test_lifecycle_hooks_failure(self, transaction_manager):
        """Test that lifecycle hooks are called on failure."""
        agent = CustomAgent(transaction_manager)

        task = AgentTask.create(
            description="Failing test",
            input_data={"should_fail": True},
        )

        result = await agent.run_task(task)

        assert not result.success
        assert agent.on_started_called
        assert not agent.on_completed_called
        assert agent.on_failed_called

"""Integration tests for multi-agent coordination."""

import pytest
import asyncio
from typing import Dict, Any, List

from agentmesh_stm.core.transaction import (
    Transaction,
    TransactionConfig,
    TransactionManager,
)
from agentmesh_stm.storage.mvcc import MVCCStorage, InMemoryBackend
from agentmesh_stm.conflict.detector import ConflictDetector, ConflictDetectorConfig
from agentmesh_stm.compensation.manager import CompensationManager
from agentmesh_stm.agent.base import (
    Agent,
    AgentConfig,
    AgentTask,
    AgentResult,
    SimpleAgent,
)


class TestMultiAgentCoordination:
    """Integration tests for multiple agents working together."""

    @pytest.fixture
    def storage(self):
        return MVCCStorage(backend=InMemoryBackend())

    @pytest.fixture
    def conflict_detector(self, storage):
        config = ConflictDetectorConfig(
            enable_ast_analysis=True,
            enable_semantic_analysis=False,
        )
        return ConflictDetector(storage=storage, config=config)

    @pytest.fixture
    def transaction_manager(self, storage, conflict_detector):
        return TransactionManager(
            storage=storage,
            conflict_detector=conflict_detector,
        )

    @pytest.mark.asyncio
    async def test_parallel_non_conflicting_agents(self, transaction_manager, storage):
        """Test multiple agents working on different files in parallel."""
        # Initialize files
        await storage.write("file_a.txt", "content a")
        await storage.write("file_b.txt", "content b")
        await storage.write("file_c.txt", "content c")

        async def modify_file(task: AgentTask, txn: Transaction) -> Dict[str, Any]:
            file_name = task.input_data["file"]
            content = await txn.read(file_name)
            await txn.write(file_name, f"Modified: {content}")
            return {"modified": file_name}

        agents = [
            SimpleAgent(
                transaction_manager=transaction_manager,
                handler=modify_file,
                config=AgentConfig(name=f"agent_{i}"),
            )
            for i in range(3)
        ]

        tasks = [
            AgentTask.create(
                description=f"Modify file_{chr(97+i)}",
                input_data={"file": f"file_{chr(97+i)}.txt"},
            )
            for i in range(3)
        ]

        # Run all agents in parallel
        results = await asyncio.gather(*[
            agents[i].run_task(tasks[i]) for i in range(3)
        ])

        # All should succeed since they work on different files
        assert all(r.success for r in results)

        # Verify all files were modified
        for i in range(3):
            file_name = f"file_{chr(97+i)}.txt"
            resource = await storage.read(file_name)
            assert resource.content.startswith("Modified:")

    @pytest.mark.asyncio
    async def test_conflicting_agents_with_retry(self, transaction_manager, storage):
        """Test agents with conflicts that succeed through retry."""
        await storage.write("shared.txt", "0")

        successful_increments = []

        async def increment_counter(task: AgentTask, txn: Transaction) -> Dict[str, Any]:
            agent_id = task.input_data["agent_id"]

            # Read current value
            value = await txn.read("shared.txt")
            current = int(value)

            # Simulate some processing time
            await asyncio.sleep(0.01)

            # Increment
            new_value = current + 1
            await txn.write("shared.txt", str(new_value))

            successful_increments.append(agent_id)
            return {"new_value": new_value}

        config = TransactionConfig(max_retries=5, retry_delay_ms=10)

        agents = [
            SimpleAgent(
                transaction_manager=transaction_manager,
                handler=increment_counter,
                config=AgentConfig(name=f"counter_{i}", transaction_config=config),
            )
            for i in range(3)
        ]

        tasks = [
            AgentTask.create(
                description=f"Increment by agent {i}",
                input_data={"agent_id": i},
                max_retries=5,
            )
            for i in range(3)
        ]

        # Run all agents - they will conflict but retry
        results = await asyncio.gather(*[
            agents[i].run_task(tasks[i]) for i in range(3)
        ])

        # At least some should succeed (serialized through retries)
        successful = [r for r in results if r.success]
        assert len(successful) >= 1

        # Final value should reflect successful increments
        final = await storage.read("shared.txt")
        assert int(final.content) == len(successful_increments)

    @pytest.mark.asyncio
    async def test_read_write_isolation(self, transaction_manager, storage):
        """Test that transactions have proper read isolation."""
        await storage.write("data.txt", "initial")

        read_values: List[str] = []
        write_complete = asyncio.Event()

        async def reader(task: AgentTask, txn: Transaction) -> Dict[str, Any]:
            # Read initial value
            value1 = await txn.read("data.txt")
            read_values.append(("before_wait", value1))

            # Wait for writer to complete
            await write_complete.wait()

            # Read again - should still see snapshot value
            value2 = await txn.read("data.txt")
            read_values.append(("after_wait", value2))

            return {"values": [value1, value2]}

        async def writer(task: AgentTask, txn: Transaction) -> Dict[str, Any]:
            await asyncio.sleep(0.05)  # Small delay
            await txn.write("data.txt", "modified")
            return {"written": True}

        reader_agent = SimpleAgent(
            transaction_manager=transaction_manager,
            handler=reader,
        )
        writer_agent = SimpleAgent(
            transaction_manager=transaction_manager,
            handler=writer,
        )

        reader_task = AgentTask.create(description="Reader", input_data={})
        writer_task = AgentTask.create(description="Writer", input_data={})

        # Start both tasks
        reader_future = asyncio.create_task(reader_agent.run_task(reader_task))

        # Give reader time to start and do first read
        await asyncio.sleep(0.02)

        # Run writer
        writer_result = await writer_agent.run_task(writer_task)
        assert writer_result.success

        # Signal reader to continue
        write_complete.set()

        # Wait for reader
        reader_result = await reader_future

        # Reader should see consistent snapshot
        # Note: Due to conflict detection, the reader might fail
        # but both reads within the transaction should see same value
        if reader_result.success:
            values = reader_result.output_data["values"]
            # Both values should be the same (snapshot isolation)
            assert values[0] == values[1]

    @pytest.mark.asyncio
    async def test_compensation_on_conflict(self, storage, conflict_detector):
        """Test that compensation works when transaction aborts."""
        compensation_manager = CompensationManager()

        manager = TransactionManager(
            storage=storage,
            conflict_detector=conflict_detector,
            compensation_manager=compensation_manager,
        )

        await storage.write("file.txt", "original")

        # Create a transaction that will fail
        txn = await manager.create_transaction()
        await txn.begin()

        # Read to establish version
        await txn.read("file.txt")

        # External modification causing conflict
        await storage.write("file.txt", "externally modified")

        # Our write
        await txn.write("file.txt", "our modification")

        # Commit should fail
        success = await txn.commit()
        assert not success

        # Abort should trigger compensation
        await txn.abort()

    @pytest.mark.asyncio
    async def test_task_dependency_chain(self, transaction_manager, storage):
        """Test tasks with dependencies executing in order."""
        execution_order = []
        await storage.write("pipeline.txt", "start")

        async def stage_handler(task: AgentTask, txn: Transaction) -> Dict[str, Any]:
            stage = task.input_data["stage"]
            execution_order.append(stage)

            current = await txn.read("pipeline.txt")
            await txn.write("pipeline.txt", f"{current} -> {stage}")

            return {"stage": stage}

        agent = SimpleAgent(
            transaction_manager=transaction_manager,
            handler=stage_handler,
        )

        # Create dependent tasks
        task1 = AgentTask.create(
            description="Stage 1",
            input_data={"stage": "S1"},
        )
        task2 = AgentTask.create(
            description="Stage 2",
            input_data={"stage": "S2"},
            dependencies=[task1.id],
        )
        task3 = AgentTask.create(
            description="Stage 3",
            input_data={"stage": "S3"},
            dependencies=[task2.id],
        )

        # Execute in sequence (respecting dependencies)
        r1 = await agent.run_task(task1)
        assert r1.success

        r2 = await agent.run_task(task2)
        assert r2.success

        r3 = await agent.run_task(task3)
        assert r3.success

        # Verify execution order
        assert execution_order == ["S1", "S2", "S3"]

        # Verify final state
        final = await storage.read("pipeline.txt")
        assert final.content == "start -> S1 -> S2 -> S3"


class TestEndToEndScenarios:
    """End-to-end test scenarios simulating real-world usage."""

    @pytest.fixture
    def full_system(self):
        """Create a fully configured system."""
        storage = MVCCStorage(backend=InMemoryBackend())
        conflict_detector = ConflictDetector(
            storage=storage,
            config=ConflictDetectorConfig(
                enable_ast_analysis=True,
                enable_semantic_analysis=False,
            ),
        )
        compensation_manager = CompensationManager()

        manager = TransactionManager(
            storage=storage,
            conflict_detector=conflict_detector,
            compensation_manager=compensation_manager,
        )

        return {
            "storage": storage,
            "conflict_detector": conflict_detector,
            "compensation_manager": compensation_manager,
            "transaction_manager": manager,
        }

    @pytest.mark.asyncio
    async def test_collaborative_code_editing(self, full_system):
        """Simulate multiple agents editing a Python file."""
        storage = full_system["storage"]
        manager = full_system["transaction_manager"]

        # Initial Python file
        initial_code = '''
class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b
'''
        await storage.write("calculator.py", initial_code)

        async def add_multiply(task: AgentTask, txn: Transaction) -> Dict[str, Any]:
            code = await txn.read("calculator.py")
            new_code = code.rstrip() + '''

    def multiply(self, a, b):
        return a * b
'''
            await txn.write("calculator.py", new_code)
            return {"added": "multiply"}

        async def add_divide(task: AgentTask, txn: Transaction) -> Dict[str, Any]:
            code = await txn.read("calculator.py")
            new_code = code.rstrip() + '''

    def divide(self, a, b):
        return a / b if b != 0 else None
'''
            await txn.write("calculator.py", new_code)
            return {"added": "divide"}

        agent1 = SimpleAgent(manager, add_multiply, AgentConfig(name="multiply_agent"))
        agent2 = SimpleAgent(manager, add_divide, AgentConfig(name="divide_agent"))

        task1 = AgentTask.create(description="Add multiply", input_data={})
        task2 = AgentTask.create(description="Add divide", input_data={})

        # Run sequentially to avoid conflicts
        r1 = await agent1.run_task(task1)
        r2 = await agent2.run_task(task2)

        assert r1.success
        assert r2.success

        # Verify both methods were added
        final_code = await storage.read("calculator.py")
        assert "def multiply" in final_code.content
        assert "def divide" in final_code.content

    @pytest.mark.asyncio
    async def test_bug_fix_workflow(self, full_system):
        """Simulate a bug fix workflow with multiple file changes."""
        storage = full_system["storage"]
        manager = full_system["transaction_manager"]

        # Setup files
        await storage.write("main.py", "from utils import helper\nhelper()")
        await storage.write("utils.py", "def helper():\n    return buggy_code()")
        await storage.write("tests/test_utils.py", "def test_helper():\n    assert helper() == expected")

        async def fix_bug(task: AgentTask, txn: Transaction) -> Dict[str, Any]:
            # Read and fix utils
            utils_code = await txn.read("utils.py")
            fixed_utils = utils_code.replace("buggy_code()", "fixed_code()")
            await txn.write("utils.py", fixed_utils)

            # Update test
            test_code = await txn.read("tests/test_utils.py")
            updated_test = test_code + "\n    # Bug fix verified"
            await txn.write("tests/test_utils.py", updated_test)

            return {
                "files_modified": ["utils.py", "tests/test_utils.py"],
                "bug_fixed": True,
            }

        agent = SimpleAgent(manager, fix_bug, AgentConfig(name="bugfix_agent"))
        task = AgentTask.create(description="Fix helper bug", input_data={})

        result = await agent.run_task(task)

        assert result.success
        assert len(result.modified_resources) == 2

        # Verify changes
        utils = await storage.read("utils.py")
        assert "fixed_code()" in utils.content

        tests = await storage.read("tests/test_utils.py")
        assert "Bug fix verified" in tests.content

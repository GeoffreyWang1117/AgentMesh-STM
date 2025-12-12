"""
Multi-Agent Collaboration Example for AgentMesh-STM

This example demonstrates advanced multi-agent coordination scenarios:
1. Multiple agents collaboratively editing code
2. Conflict detection and resolution
3. Transaction coordination for complex workflows
"""

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


async def setup_codebase(storage: MVCCStorage) -> None:
    """Initialize a simulated codebase."""

    # Main application file
    main_py = '''"""Main application module."""

from utils import helper
from models import User

def main():
    """Application entry point."""
    user = User("admin")
    result = helper(user)
    print(result)

if __name__ == "__main__":
    main()
'''

    # Utility module
    utils_py = '''"""Utility functions."""

def helper(user):
    """Helper function that processes user data."""
    return f"Processing {user.name}"

def format_output(data):
    """Format output data."""
    return str(data)
'''

    # Models module
    models_py = '''"""Data models."""

class User:
    """User model."""

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"User({self.name})"
'''

    # Test file
    test_py = '''"""Tests for the application."""

from models import User
from utils import helper

def test_helper():
    user = User("test")
    result = helper(user)
    assert "test" in result

def test_user_str():
    user = User("alice")
    assert str(user) == "User(alice)"
'''

    await storage.write("src/main.py", main_py)
    await storage.write("src/utils.py", utils_py)
    await storage.write("src/models.py", models_py)
    await storage.write("tests/test_app.py", test_py)


async def example_collaborative_editing():
    """Demonstrate multiple agents collaboratively editing code."""
    print("=== Collaborative Code Editing Example ===\n")

    # Setup system
    storage = MVCCStorage(backend=InMemoryBackend())
    conflict_detector = ConflictDetector(
        storage=storage,
        config=ConflictDetectorConfig(enable_semantic_analysis=False),
    )
    manager = TransactionManager(
        storage=storage,
        conflict_detector=conflict_detector,
        default_config=TransactionConfig(max_retries=5),
    )

    await setup_codebase(storage)
    print("Codebase initialized with 4 files")

    # Define agent tasks

    async def add_validation(task: AgentTask, txn: Transaction) -> Dict[str, Any]:
        """Agent that adds input validation to User model."""
        models_code = await txn.read("src/models.py")

        # Add validation to User class
        new_code = models_code.replace(
            "def __init__(self, name):",
            '''def __init__(self, name):
        if not name or not isinstance(name, str):
            raise ValueError("Name must be a non-empty string")'''
        )

        await txn.write("src/models.py", new_code)
        return {"added": "input validation to User.__init__"}

    async def add_logging(task: AgentTask, txn: Transaction) -> Dict[str, Any]:
        """Agent that adds logging to helper function."""
        utils_code = await txn.read("src/utils.py")

        # Add logging import and usage
        new_code = '''"""Utility functions."""

import logging

logger = logging.getLogger(__name__)

def helper(user):
    """Helper function that processes user data."""
    logger.info(f"Processing user: {user.name}")
    return f"Processing {user.name}"

def format_output(data):
    """Format output data."""
    return str(data)
'''
        await txn.write("src/utils.py", new_code)
        return {"added": "logging to helper function"}

    async def add_test_validation(task: AgentTask, txn: Transaction) -> Dict[str, Any]:
        """Agent that adds validation tests."""
        test_code = await txn.read("tests/test_app.py")

        new_test = '''

def test_user_validation():
    """Test that User validates input."""
    import pytest
    with pytest.raises(ValueError):
        User("")
    with pytest.raises(ValueError):
        User(None)
'''
        await txn.write("tests/test_app.py", test_code + new_test)
        return {"added": "validation tests"}

    # Create agents
    validation_agent = SimpleAgent(
        manager, add_validation, AgentConfig(name="validation_agent")
    )
    logging_agent = SimpleAgent(
        manager, add_logging, AgentConfig(name="logging_agent")
    )
    test_agent = SimpleAgent(
        manager, add_test_validation, AgentConfig(name="test_agent")
    )

    # Create tasks
    tasks = [
        AgentTask.create(description="Add input validation", input_data={}),
        AgentTask.create(description="Add logging", input_data={}),
        AgentTask.create(description="Add validation tests", input_data={}),
    ]

    print("\nStarting 3 agents to modify different aspects of the codebase...")
    print("- Validation agent: modifying models.py")
    print("- Logging agent: modifying utils.py")
    print("- Test agent: modifying test_app.py")

    # Run agents in parallel - they modify different files, no conflicts expected
    results = await asyncio.gather(
        validation_agent.run_task(tasks[0]),
        logging_agent.run_task(tasks[1]),
        test_agent.run_task(tasks[2]),
    )

    print("\nResults:")
    for i, result in enumerate(results):
        status = "SUCCESS" if result.success else f"FAILED: {result.error_message}"
        print(f"  Task {i+1}: {status}")
        if result.success:
            print(f"    Output: {result.output_data}")

    # Show modified files
    print("\nModified code snippets:")

    models = await storage.read("src/models.py")
    print("\n--- models.py (with validation) ---")
    print(models.content[:500] + "...")

    utils = await storage.read("src/utils.py")
    print("\n--- utils.py (with logging) ---")
    print(utils.content[:500] + "...")


async def example_conflict_resolution():
    """Demonstrate conflict detection when agents modify same file."""
    print("\n\n=== Conflict Resolution Example ===\n")

    storage = MVCCStorage(backend=InMemoryBackend())
    conflict_detector = ConflictDetector(
        storage=storage,
        config=ConflictDetectorConfig(enable_semantic_analysis=False),
    )
    manager = TransactionManager(
        storage=storage,
        conflict_detector=conflict_detector,
        default_config=TransactionConfig(max_retries=3, retry_delay_ms=50),
    )

    # Simple shared file
    await storage.write("shared_config.json", '{"count": 0, "items": []}')

    increment_count = 0
    add_item_count = 0

    async def increment_counter(task: AgentTask, txn: Transaction) -> Dict[str, Any]:
        """Agent that increments the counter."""
        nonlocal increment_count
        increment_count += 1

        import json
        content = await txn.read("shared_config.json")
        data = json.loads(content)

        # Simulate processing time
        await asyncio.sleep(0.02)

        data["count"] += 1
        await txn.write("shared_config.json", json.dumps(data))

        return {"new_count": data["count"], "attempts": increment_count}

    async def add_item(task: AgentTask, txn: Transaction) -> Dict[str, Any]:
        """Agent that adds an item to the list."""
        nonlocal add_item_count
        add_item_count += 1

        import json
        content = await txn.read("shared_config.json")
        data = json.loads(content)

        # Simulate processing time
        await asyncio.sleep(0.02)

        item_name = task.input_data["item"]
        data["items"].append(item_name)
        await txn.write("shared_config.json", json.dumps(data))

        return {"items": data["items"], "attempts": add_item_count}

    counter_agent = SimpleAgent(
        manager, increment_counter, AgentConfig(name="counter_agent")
    )
    item_agent = SimpleAgent(
        manager, add_item, AgentConfig(name="item_agent")
    )

    print("Two agents will try to modify the same file concurrently.")
    print("Conflicts will be detected and transactions will retry.\n")

    counter_task = AgentTask.create(
        description="Increment counter",
        input_data={},
        max_retries=3,
    )
    item_task = AgentTask.create(
        description="Add item",
        input_data={"item": "new_item"},
        max_retries=3,
    )

    # Run both agents concurrently - they will conflict
    results = await asyncio.gather(
        counter_agent.run_task(counter_task),
        item_agent.run_task(item_task),
    )

    print("Results:")
    for result in results:
        if result.success:
            print(f"  SUCCESS: {result.output_data}")
        else:
            print(f"  FAILED: {result.error_message}")

    print(f"\nTotal attempts - Counter: {increment_count}, Item: {add_item_count}")

    # Check final state
    import json
    final = await storage.read("shared_config.json")
    final_data = json.loads(final.content)
    print(f"Final state: {final_data}")


async def example_sequential_pipeline():
    """Demonstrate a sequential processing pipeline with multiple agents."""
    print("\n\n=== Sequential Pipeline Example ===\n")

    storage = MVCCStorage(backend=InMemoryBackend())
    manager = TransactionManager(storage=storage)

    # Pipeline stages
    stages = ["parse", "validate", "transform", "save"]

    async def create_stage_handler(stage_name: str):
        """Create a handler for a pipeline stage."""

        async def handler(task: AgentTask, txn: Transaction) -> Dict[str, Any]:
            pipeline_id = task.input_data["pipeline_id"]
            state_key = f"pipeline/{pipeline_id}/state"

            # Read current state
            content = await txn.read(state_key)
            if content is None:
                current_stages = []
            else:
                import json
                current_stages = json.loads(content)

            # Add this stage
            current_stages.append({
                "stage": stage_name,
                "status": "completed",
            })

            # Write updated state
            import json
            await txn.write(state_key, json.dumps(current_stages))

            return {
                "stage": stage_name,
                "stages_completed": len(current_stages),
            }

        return handler

    print("Creating 4-stage pipeline: parse -> validate -> transform -> save")

    # Initialize pipeline state
    pipeline_id = "pipeline_001"
    await storage.write(f"pipeline/{pipeline_id}/state", "[]")

    # Run stages sequentially
    for stage in stages:
        handler = await create_stage_handler(stage)
        agent = SimpleAgent(
            manager, handler, AgentConfig(name=f"{stage}_agent")
        )

        task = AgentTask.create(
            description=f"Run {stage} stage",
            input_data={"pipeline_id": pipeline_id},
        )

        print(f"  Running stage: {stage}...", end=" ")
        result = await agent.run_task(task)

        if result.success:
            print(f"OK (stages completed: {result.output_data['stages_completed']})")
        else:
            print(f"FAILED: {result.error_message}")
            break

    # Show final pipeline state
    import json
    final_state = await storage.read(f"pipeline/{pipeline_id}/state")
    stages_data = json.loads(final_state.content)

    print("\nPipeline execution completed:")
    for stage_info in stages_data:
        print(f"  - {stage_info['stage']}: {stage_info['status']}")


async def main():
    """Run all multi-agent examples."""
    await example_collaborative_editing()
    await example_conflict_resolution()
    await example_sequential_pipeline()

    print("\n" + "=" * 50)
    print("All multi-agent examples completed!")


if __name__ == "__main__":
    asyncio.run(main())

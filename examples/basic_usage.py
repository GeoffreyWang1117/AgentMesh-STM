"""
Basic Usage Example for AgentMesh-STM

This example demonstrates the core concepts of the AgentMesh-STM framework:
1. Creating transactions for atomic operations
2. Using MVCC storage for versioned data
3. Running agents with automatic conflict handling
"""

import asyncio
from typing import Dict, Any

from agentmesh_stm.core.transaction import (
    Transaction,
    TransactionConfig,
    TransactionManager,
)
from agentmesh_stm.storage.mvcc import MVCCStorage, InMemoryBackend
from agentmesh_stm.agent.base import (
    Agent,
    AgentConfig,
    AgentTask,
    SimpleAgent,
)


async def example_basic_transaction():
    """Demonstrate basic transaction usage."""
    print("=== Basic Transaction Example ===\n")

    # Create storage and transaction manager
    storage = MVCCStorage(backend=InMemoryBackend())
    manager = TransactionManager(storage=storage)

    # Write some initial data
    await storage.write("config.json", '{"version": 1}')
    print("Initial data written: config.json")

    # Use context manager for transaction
    async with manager.transaction() as txn:
        # Read data
        content = await txn.read("config.json")
        print(f"Read within transaction: {content}")

        # Modify and write
        await txn.write("config.json", '{"version": 2}')
        await txn.write("new_file.txt", "Created in transaction")

        # Both writes will be committed atomically
        print("Writes buffered, will commit on context exit")

    # Verify changes were committed
    config = await storage.read("config.json")
    new_file = await storage.read("new_file.txt")
    print(f"\nAfter commit:")
    print(f"  config.json: {config.content}")
    print(f"  new_file.txt: {new_file.content}")


async def example_conflict_handling():
    """Demonstrate conflict detection and retry."""
    print("\n=== Conflict Handling Example ===\n")

    storage = MVCCStorage(backend=InMemoryBackend())
    manager = TransactionManager(
        storage=storage,
        default_config=TransactionConfig(max_retries=3),
    )

    # Initialize a counter
    await storage.write("counter", "0")

    attempt_count = 0

    async def increment_with_conflict_simulation(txn: Transaction):
        nonlocal attempt_count
        attempt_count += 1

        # Read current value
        value = await txn.read("counter")
        current = int(value)
        print(f"  Attempt {attempt_count}: Read counter = {current}")

        # Simulate external modification on first attempt
        if attempt_count == 1:
            print("  [Simulating external modification...]")
            await storage.write("counter", "100")

        # Try to increment
        await txn.write("counter", str(current + 1))
        return {"incremented_from": current}

    # Execute with automatic retry
    result = await manager.execute(increment_with_conflict_simulation)

    final = await storage.read("counter")
    print(f"\nFinal counter value: {final.content}")
    print(f"Total attempts: {attempt_count}")


async def example_simple_agent():
    """Demonstrate using SimpleAgent for task execution."""
    print("\n=== Simple Agent Example ===\n")

    storage = MVCCStorage(backend=InMemoryBackend())
    manager = TransactionManager(storage=storage)

    # Initialize some files
    await storage.write("data/input.txt", "Hello, World!")

    # Define a task handler
    async def process_file(task: AgentTask, txn: Transaction) -> Dict[str, Any]:
        """Process input file and create output."""
        input_path = task.input_data["input"]
        output_path = task.input_data["output"]

        # Read input
        content = await txn.read(input_path)
        print(f"  Agent reading: {input_path}")

        # Process (uppercase in this example)
        processed = content.upper()

        # Write output
        await txn.write(output_path, processed)
        print(f"  Agent writing: {output_path}")

        return {
            "input_length": len(content),
            "output_length": len(processed),
        }

    # Create agent
    agent = SimpleAgent(
        transaction_manager=manager,
        handler=process_file,
        config=AgentConfig(name="file_processor"),
    )

    # Create and run task
    task = AgentTask.create(
        description="Process input file",
        input_data={
            "input": "data/input.txt",
            "output": "data/output.txt",
        },
    )

    print(f"Running task: {task.description}")
    result = await agent.run_task(task)

    print(f"\nTask completed:")
    print(f"  Success: {result.success}")
    print(f"  Output: {result.output_data}")
    print(f"  Modified files: {result.modified_resources}")

    # Verify output
    output = await storage.read("data/output.txt")
    print(f"  Output content: {output.content}")


async def example_parallel_agents():
    """Demonstrate multiple agents working in parallel."""
    print("\n=== Parallel Agents Example ===\n")

    storage = MVCCStorage(backend=InMemoryBackend())
    manager = TransactionManager(storage=storage)

    # Initialize files
    for i in range(3):
        await storage.write(f"file_{i}.txt", f"Content {i}")

    async def modify_file(task: AgentTask, txn: Transaction) -> Dict[str, Any]:
        file_path = task.input_data["file"]
        agent_name = task.input_data["agent"]

        content = await txn.read(file_path)
        modified = f"[Modified by {agent_name}] {content}"
        await txn.write(file_path, modified)

        return {"agent": agent_name, "file": file_path}

    # Create multiple agents
    agents = [
        SimpleAgent(
            transaction_manager=manager,
            handler=modify_file,
            config=AgentConfig(name=f"agent_{i}"),
        )
        for i in range(3)
    ]

    # Create tasks for different files
    tasks = [
        AgentTask.create(
            description=f"Modify file_{i}",
            input_data={"file": f"file_{i}.txt", "agent": f"agent_{i}"},
        )
        for i in range(3)
    ]

    print("Starting 3 agents in parallel...")

    # Run all agents in parallel
    results = await asyncio.gather(*[
        agents[i].run_task(tasks[i]) for i in range(3)
    ])

    print("\nResults:")
    for result in results:
        if result.success:
            print(f"  {result.output_data['agent']}: modified {result.output_data['file']}")
        else:
            print(f"  Failed: {result.error_message}")

    # Show final state
    print("\nFinal file contents:")
    for i in range(3):
        content = await storage.read(f"file_{i}.txt")
        print(f"  file_{i}.txt: {content.content}")


async def example_version_history():
    """Demonstrate MVCC version history."""
    print("\n=== Version History Example ===\n")

    storage = MVCCStorage(backend=InMemoryBackend())

    # Create multiple versions
    await storage.write("document.txt", "Version 1: Initial content")
    await storage.write("document.txt", "Version 2: Updated content")
    await storage.write("document.txt", "Version 3: Final content")

    print("Created 3 versions of document.txt")

    # Get version history
    history = await storage.get_version_history("document.txt")

    print("\nVersion history (newest first):")
    for version in history:
        print(f"  v{version.version}: {version.content[:30]}... ({version.timestamp})")

    # Read at specific version
    print("\nReading at specific versions:")
    v1 = await storage.read("document.txt", version=1)
    v2 = await storage.read("document.txt", version=2)
    latest = await storage.read("document.txt")

    print(f"  At version 1: {v1.content if v1 else 'N/A'}")
    print(f"  At version 2: {v2.content if v2 else 'N/A'}")
    print(f"  Latest: {latest.content}")


async def main():
    """Run all examples."""
    await example_basic_transaction()
    await example_conflict_handling()
    await example_simple_agent()
    await example_parallel_agents()
    await example_version_history()

    print("\n" + "=" * 50)
    print("All examples completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())

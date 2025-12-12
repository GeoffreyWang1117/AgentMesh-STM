# AgentMesh-STM

A Software Transactional Memory (STM) based Multi-Agent Coordination Framework for LLM Agents.

## Overview

AgentMesh-STM provides a robust coordination framework for multiple LLM agents working concurrently on shared resources. It addresses the unique challenges of multi-agent collaboration:

- **Long-running transactions**: LLM agent operations take seconds to minutes, far exceeding traditional millisecond transactions
- **Irreversible side effects**: File writes, Git commits, and API calls require compensation-based rollback
- **Semantic conflict detection**: Code modifications need semantic understanding, not just memory address comparison

## Key Features

- **Transaction Abstraction**: Wrap agent operations as explicit transactions with read/write tracking
- **MVCC Storage**: Multi-version concurrency control for reading any historical version
- **Semantic Conflict Detection**: Hierarchical detection at file, AST, and semantic levels
- **Compensation Management**: Rollback side effects when transactions abort
- **LLM Integration**: Native support for OpenAI and Anthropic APIs

## Installation

```bash
pip install -e .
```

Or with development dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic Transaction

```python
import asyncio
from agentmesh_stm import TransactionManager, MVCCStorage

async def main():
    storage = MVCCStorage()
    manager = TransactionManager(storage=storage)

    async with manager.transaction() as txn:
        # Read data
        content = await txn.read("config.json")

        # Modify and write
        await txn.write("config.json", modified_content)

        # Commit happens automatically on context exit

asyncio.run(main())
```

### Simple Agent

```python
from agentmesh_stm import TransactionManager, MVCCStorage, SimpleAgent, AgentTask

async def process_file(task, txn):
    content = await txn.read(task.input_data["file"])
    processed = content.upper()
    await txn.write(task.input_data["output"], processed)
    return {"status": "done"}

storage = MVCCStorage()
manager = TransactionManager(storage=storage)

agent = SimpleAgent(
    transaction_manager=manager,
    handler=process_file,
)

task = AgentTask.create(
    description="Process file",
    input_data={"file": "input.txt", "output": "output.txt"},
)

result = await agent.run_task(task)
```

### Parallel Agents with Conflict Handling

```python
from agentmesh_stm import (
    TransactionManager,
    MVCCStorage,
    ConflictDetector,
    SimpleAgent,
    AgentTask,
    TransactionConfig,
)

storage = MVCCStorage()
conflict_detector = ConflictDetector(storage=storage)
manager = TransactionManager(
    storage=storage,
    conflict_detector=conflict_detector,
    default_config=TransactionConfig(max_retries=3),
)

# Create multiple agents
agents = [
    SimpleAgent(manager, handler, name=f"agent_{i}")
    for i in range(3)
]

# Run in parallel - conflicts auto-retry
results = await asyncio.gather(*[
    agents[i].run_task(tasks[i]) for i in range(3)
])
```

## Architecture

### Core Components

1. **Transaction Layer** (`agentmesh_stm.core.transaction`)
   - `Transaction`: Wraps operations with read/write sets
   - `TransactionManager`: Coordinates transaction lifecycle
   - Optimistic concurrency with automatic retry

2. **MVCC Storage** (`agentmesh_stm.storage.mvcc`)
   - `MVCCStorage`: Versioned storage with history
   - `InMemoryBackend`: Fast in-memory storage
   - `SQLiteBackend`: Persistent storage option

3. **Conflict Detection** (`agentmesh_stm.conflict`)
   - `ConflictDetector`: Hierarchical conflict detection
   - `ASTAnalyzer`: Code structure analysis
   - `ConflictResolver`: Automatic merge strategies

4. **Compensation** (`agentmesh_stm.compensation`)
   - `CompensationManager`: Tracks and rolls back side effects
   - Built-in support for file, Git, and API operations

5. **Agent Layer** (`agentmesh_stm.agent`)
   - `Agent`: Base class for transactional agents
   - `SimpleAgent`: Callable wrapper for quick usage
   - `LLMAgent`: Full LLM integration with tools

## Conflict Detection Levels

1. **File Level**: Quick filter - different files = no conflict
2. **AST Level**: Parse code structure - different regions = likely no conflict
3. **Semantic Level**: LLM analysis for overlapping regions

## Configuration

### Transaction Config

```python
from agentmesh_stm import TransactionConfig

config = TransactionConfig(
    max_retries=3,           # Retry attempts on conflict
    retry_delay_ms=100,      # Delay between retries
    timeout_seconds=300.0,   # Transaction timeout
    isolation_level="snapshot",  # snapshot or serializable
)
```

### Conflict Detector Config

```python
from agentmesh_stm.conflict import ConflictDetectorConfig

config = ConflictDetectorConfig(
    enable_ast_analysis=True,
    enable_semantic_analysis=True,
    semantic_model="gpt-4o-mini",
    semantic_threshold=0.7,
)
```

## Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=agentmesh_stm

# Specific test file
pytest tests/unit/test_transaction.py
```

## Examples

See the `examples/` directory:

- `basic_usage.py`: Core concepts and simple usage
- `multi_agent_collaboration.py`: Advanced multi-agent scenarios

## Theory

AgentMesh-STM provides formal guarantees:

- **Conflict Serializability**: Committed transactions are equivalent to some serial order
- **Snapshot Isolation**: Transactions see consistent snapshots
- **Progress Guarantee**: No livelocks under reasonable assumptions

## Dependencies

- Python 3.10+
- tree-sitter (AST parsing)
- libcst (Python CST)
- openai / anthropic (LLM integration)
- aiosqlite / lmdb (storage backends)

## License

MIT License

## Contributing

Contributions welcome! Please read the contributing guidelines and submit pull requests.

#!/usr/bin/env python3
"""
Parallel Bug Fixing Example

This example demonstrates how multiple agents can work on fixing
different bugs in the same codebase concurrently, with STM ensuring
consistency when their changes overlap.

Scenario:
- A bug tracking system has multiple open bugs
- Multiple agents pick up different bugs to fix
- Agents may need to modify the same files
- STM handles conflicts automatically with retry
"""

import asyncio
import random
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from agentmesh_stm.core.transaction import TransactionManager, TransactionConfig, Transaction
from agentmesh_stm.storage.mvcc import MVCCStorage, InMemoryBackend
from agentmesh_stm.conflict.detector import ConflictDetector
from agentmesh_stm.agent.base import Agent, AgentTask, AgentResult, AgentConfig


# Bug definitions
@dataclass
class Bug:
    id: str
    title: str
    description: str
    affected_file: str
    fix_description: str


BUGS = [
    Bug(
        id="BUG-001",
        title="Division by zero in calculate_average",
        description="calculate_average crashes when given empty list",
        affected_file="src/math_utils.py",
        fix_description="Add check for empty list before division",
    ),
    Bug(
        id="BUG-002",
        title="Missing null check in get_user_name",
        description="get_user_name raises error when user not found",
        affected_file="src/user_service.py",
        fix_description="Add null check and return default value",
    ),
    Bug(
        id="BUG-003",
        title="Off-by-one error in pagination",
        description="Last item missing in paginated results",
        affected_file="src/pagination.py",
        fix_description="Fix range boundary condition",
    ),
    Bug(
        id="BUG-004",
        title="Race condition in counter increment",
        description="Counter loses updates under concurrent access",
        affected_file="src/counter.py",
        fix_description="Add synchronization to increment method",
    ),
    Bug(
        id="BUG-005",
        title="Memory leak in cache cleanup",
        description="Cache entries not properly removed",
        affected_file="src/cache.py",
        fix_description="Implement proper cleanup in __del__",
    ),
]


# Initial codebase files
INITIAL_FILES = {
    "src/math_utils.py": '''"""Math utility functions."""

def calculate_average(numbers):
    """Calculate average of numbers."""
    return sum(numbers) / len(numbers)

def calculate_sum(numbers):
    """Calculate sum of numbers."""
    return sum(numbers)

def find_max(numbers):
    """Find maximum value."""
    return max(numbers)
''',

    "src/user_service.py": '''"""User service for managing users."""

class UserService:
    def __init__(self):
        self.users = {}

    def add_user(self, user_id: str, name: str):
        """Add a new user."""
        self.users[user_id] = {"name": name}

    def get_user_name(self, user_id: str) -> str:
        """Get user name by ID."""
        return self.users[user_id]["name"]

    def list_users(self):
        """List all users."""
        return list(self.users.values())
''',

    "src/pagination.py": '''"""Pagination utilities."""

def paginate(items, page_size: int, page_number: int):
    """Return a page of items."""
    start = page_number * page_size
    end = start + page_size - 1  # Bug: should be start + page_size
    return items[start:end]

def get_total_pages(total_items: int, page_size: int) -> int:
    """Calculate total number of pages."""
    return (total_items + page_size - 1) // page_size
''',

    "src/counter.py": '''"""Thread-safe counter implementation."""

class Counter:
    def __init__(self, initial_value: int = 0):
        self.value = initial_value

    def increment(self):
        """Increment the counter."""
        current = self.value
        self.value = current + 1  # Bug: not atomic

    def decrement(self):
        """Decrement the counter."""
        current = self.value
        self.value = current - 1

    def get_value(self) -> int:
        """Get current value."""
        return self.value
''',

    "src/cache.py": '''"""Simple cache implementation."""

class Cache:
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.data = {}
        self.access_order = []

    def get(self, key: str):
        """Get value from cache."""
        if key in self.data:
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.data[key]
        return None

    def set(self, key: str, value):
        """Set value in cache."""
        if len(self.data) >= self.max_size:
            oldest = self.access_order[0]
            del self.data[oldest]
            # Bug: forgot to remove from access_order
        self.data[key] = value
        self.access_order.append(key)
''',
}


# Bug fixes (what each agent will apply)
BUG_FIXES = {
    "BUG-001": (
        "src/math_utils.py",
        '''"""Calculate average of numbers."""
    return sum(numbers) / len(numbers)''',
        '''"""Calculate average of numbers."""
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)''',
    ),

    "BUG-002": (
        "src/user_service.py",
        '''def get_user_name(self, user_id: str) -> str:
        """Get user name by ID."""
        return self.users[user_id]["name"]''',
        '''def get_user_name(self, user_id: str) -> Optional[str]:
        """Get user name by ID. Returns None if user not found."""
        user = self.users.get(user_id)
        return user["name"] if user else None''',
    ),

    "BUG-003": (
        "src/pagination.py",
        '''end = start + page_size - 1  # Bug: should be start + page_size''',
        '''end = start + page_size  # Fixed: correct boundary''',
    ),

    "BUG-004": (
        "src/counter.py",
        '''"""Thread-safe counter implementation."""

class Counter:
    def __init__(self, initial_value: int = 0):
        self.value = initial_value

    def increment(self):
        """Increment the counter."""
        current = self.value
        self.value = current + 1  # Bug: not atomic''',
        '''"""Thread-safe counter implementation."""
import threading

class Counter:
    def __init__(self, initial_value: int = 0):
        self.value = initial_value
        self._lock = threading.Lock()

    def increment(self):
        """Increment the counter atomically."""
        with self._lock:
            self.value += 1  # Fixed: atomic with lock''',
    ),

    "BUG-005": (
        "src/cache.py",
        '''if len(self.data) >= self.max_size:
            oldest = self.access_order[0]
            del self.data[oldest]
            # Bug: forgot to remove from access_order''',
        '''if len(self.data) >= self.max_size:
            oldest = self.access_order.pop(0)  # Fixed: properly remove from order
            del self.data[oldest]''',
    ),
}


class BugFixAgent(Agent):
    """Agent that fixes a specific bug."""

    def __init__(
        self,
        transaction_manager: TransactionManager,
        bug: Bug,
    ):
        super().__init__(
            transaction_manager,
            AgentConfig(
                name=f"BugFixer-{bug.id}",
                description=f"Fixes {bug.id}: {bug.title}",
            ),
        )
        self.bug = bug

    async def execute_task(
        self, task: AgentTask, transaction: Transaction
    ) -> Dict[str, Any]:
        """Execute the bug fix."""
        # Simulate some "thinking" time
        await asyncio.sleep(random.uniform(0.1, 0.5))

        # Read the affected file
        content = await transaction.read(self.bug.affected_file)
        if content is None:
            return {
                "bug_id": self.bug.id,
                "success": False,
                "error": f"File not found: {self.bug.affected_file}",
            }

        # Apply the fix
        fix = BUG_FIXES.get(self.bug.id)
        if not fix:
            return {
                "bug_id": self.bug.id,
                "success": False,
                "error": "No fix available",
            }

        file_path, old_code, new_code = fix

        if old_code not in content:
            return {
                "bug_id": self.bug.id,
                "success": False,
                "error": "Could not find code to fix (may already be fixed)",
            }

        # Apply the fix
        fixed_content = content.replace(old_code, new_code)

        # Write the fixed file
        await transaction.write(file_path, fixed_content)

        return {
            "bug_id": self.bug.id,
            "success": True,
            "file_modified": file_path,
            "description": self.bug.fix_description,
        }


async def run_parallel_bug_fixing():
    """Run multiple bug-fixing agents in parallel."""
    print("=" * 60)
    print("Parallel Bug Fixing Example")
    print("=" * 60)

    # Setup STM infrastructure
    storage = MVCCStorage(InMemoryBackend())
    conflict_detector = ConflictDetector(storage)
    manager = TransactionManager(
        storage=storage,
        conflict_detector=conflict_detector,
        default_config=TransactionConfig(
            max_retries=10,  # Allow more retries for concurrent fixes
            retry_delay_ms=50,
        ),
    )

    # Initialize codebase
    print("\nInitializing codebase...")
    async with manager.transaction() as txn:
        for file_path, content in INITIAL_FILES.items():
            await txn.write(file_path, content)
    print(f"Loaded {len(INITIAL_FILES)} files")

    # Create bug-fixing agents
    agents = [BugFixAgent(manager, bug) for bug in BUGS]

    print(f"\nCreated {len(agents)} bug-fixing agents")
    print("-" * 40)
    for bug in BUGS:
        print(f"  {bug.id}: {bug.title}")

    # Create tasks
    tasks = [
        AgentTask(
            task_id=f"fix-{agent.bug.id}",
            description=f"Fix {agent.bug.id}",
            input_data={},
        )
        for agent in agents
    ]

    print("\n" + "=" * 60)
    print("Starting parallel bug fixes...")
    print("=" * 60)

    # Run all agents concurrently
    start_time = asyncio.get_event_loop().time()
    results = await asyncio.gather(
        *[agent.run(task) for agent, task in zip(agents, tasks)],
        return_exceptions=True,
    )
    elapsed = asyncio.get_event_loop().time() - start_time

    # Process results
    print("\n" + "=" * 60)
    print("Bug Fix Results")
    print("=" * 60)

    successful = 0
    failed = 0

    for i, result in enumerate(results):
        bug = BUGS[i]
        if isinstance(result, Exception):
            print(f"\n[{bug.id}] EXCEPTION: {result}")
            failed += 1
        else:
            output = result.output_data or {}
            if output.get("success"):
                print(f"\n[{bug.id}] FIXED ✓")
                print(f"  Title: {bug.title}")
                print(f"  File: {output.get('file_modified')}")
                print(f"  Fix: {output.get('description')}")
                successful += 1
            else:
                print(f"\n[{bug.id}] FAILED ✗")
                print(f"  Title: {bug.title}")
                print(f"  Error: {output.get('error')}")
                failed += 1

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total bugs: {len(BUGS)}")
    print(f"Fixed: {successful}")
    print(f"Failed: {failed}")
    print(f"Time elapsed: {elapsed:.2f}s")

    # Show a sample of fixed code
    print("\n" + "=" * 60)
    print("Sample Fixed Code: src/math_utils.py")
    print("=" * 60)
    async with manager.transaction() as txn:
        math_utils = await txn.read("src/math_utils.py")
        print(math_utils)


async def run_conflict_scenario():
    """
    Run a scenario that intentionally creates conflicts
    to demonstrate STM conflict resolution.
    """
    print("\n" + "=" * 60)
    print("Conflict Resolution Scenario")
    print("=" * 60)

    # Setup
    storage = MVCCStorage(InMemoryBackend())
    conflict_detector = ConflictDetector(storage)
    manager = TransactionManager(
        storage=storage,
        conflict_detector=conflict_detector,
        default_config=TransactionConfig(max_retries=5),
    )

    # Initialize with a shared file
    shared_file = "src/shared_module.py"
    initial_content = '''"""Shared module that multiple agents will modify."""

CONFIG = {
    "timeout": 30,
    "retries": 3,
    "debug": False,
}

def process(data):
    """Process data."""
    return data
'''

    async with manager.transaction() as txn:
        await txn.write(shared_file, initial_content)

    print(f"\nInitialized shared file: {shared_file}")

    # Create two agents that will modify the same file
    async def agent_a_task(txn: Transaction):
        """Agent A modifies the timeout config."""
        content = await txn.read(shared_file)
        # Simulate work
        await asyncio.sleep(0.1)
        # Modify timeout
        modified = content.replace('"timeout": 30', '"timeout": 60')
        await txn.write(shared_file, modified)
        return "Agent A: Updated timeout to 60"

    async def agent_b_task(txn: Transaction):
        """Agent B modifies the debug config."""
        content = await txn.read(shared_file)
        # Simulate work
        await asyncio.sleep(0.1)
        # Modify debug
        modified = content.replace('"debug": False', '"debug": True')
        await txn.write(shared_file, modified)
        return "Agent B: Enabled debug mode"

    print("\nRunning two agents that modify the same file concurrently...")
    print("(One may need to retry due to conflict)")

    # Run both agents concurrently
    try:
        results = await asyncio.gather(
            manager.execute(agent_a_task),
            manager.execute(agent_b_task),
        )
        print("\nResults:")
        for result in results:
            print(f"  {result}")
    except Exception as e:
        print(f"\nOne agent failed after retries: {e}")

    # Show final state
    print("\nFinal file content:")
    async with manager.transaction() as txn:
        final_content = await txn.read(shared_file)
        print(final_content)


if __name__ == "__main__":
    asyncio.run(run_parallel_bug_fixing())
    asyncio.run(run_conflict_scenario())

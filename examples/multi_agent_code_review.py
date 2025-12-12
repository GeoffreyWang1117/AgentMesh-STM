#!/usr/bin/env python3
"""
Multi-Agent Code Review Example

This example demonstrates how multiple LLM agents can concurrently
review and improve different aspects of a codebase while maintaining
consistency through STM transactions.

Agents:
- SecurityAgent: Reviews code for security vulnerabilities
- PerformanceAgent: Optimizes code for performance
- StyleAgent: Ensures code follows style guidelines
- DocumentationAgent: Improves code documentation
"""

import asyncio
import os
from typing import Dict, Any, List

from agentmesh_stm.core.transaction import TransactionManager, TransactionConfig
from agentmesh_stm.storage.mvcc import MVCCStorage, InMemoryBackend
from agentmesh_stm.conflict.detector import ConflictDetector
from agentmesh_stm.agent.base import Agent, AgentTask, AgentResult, AgentConfig
from agentmesh_stm.agent.llm_agent import LLMAgent, LLMConfig, LLMProvider


# Sample code to review
SAMPLE_CODE = '''
def get_user_data(user_id):
    # Get user from database
    query = f"SELECT * FROM users WHERE id = {user_id}"
    result = db.execute(query)
    return result

def process_items(items):
    processed = []
    for item in items:
        for i in range(len(items)):
            if items[i] == item:
                processed.append(item * 2)
    return processed

def calculate_total(prices):
    total = 0
    for p in prices:
        total = total + p
    return total

class UserManager:
    def __init__(self):
        self.users = {}

    def add_user(self, id, name, email):
        self.users[id] = {"name": name, "email": email}

    def get_user(self, id):
        return self.users.get(id)
'''


class ReviewAgent(Agent):
    """Base class for review agents."""

    def __init__(
        self,
        transaction_manager: TransactionManager,
        agent_config: AgentConfig,
        review_type: str,
    ):
        super().__init__(transaction_manager, agent_config)
        self.review_type = review_type

    async def execute_task(self, task: AgentTask, transaction) -> Dict[str, Any]:
        """Execute the review task."""
        # Read the file
        content = await transaction.read(task.input_data["file_path"])
        if content is None:
            return {"error": "File not found"}

        # Perform review (simulated - in real use, would call LLM)
        issues = await self._review_code(content)
        fixes = await self._generate_fixes(content, issues)

        if fixes:
            # Write the fixed content
            await transaction.write(task.input_data["file_path"], fixes)

        return {
            "review_type": self.review_type,
            "issues_found": len(issues),
            "issues": issues,
            "fixes_applied": bool(fixes),
        }

    async def _review_code(self, content: str) -> List[Dict[str, Any]]:
        """Override in subclasses to implement specific review logic."""
        return []

    async def _generate_fixes(
        self, content: str, issues: List[Dict[str, Any]]
    ) -> str | None:
        """Override in subclasses to implement specific fix logic."""
        return None


class SecurityReviewAgent(ReviewAgent):
    """Agent that reviews code for security vulnerabilities."""

    def __init__(self, transaction_manager: TransactionManager):
        super().__init__(
            transaction_manager,
            AgentConfig(name="SecurityAgent", description="Reviews for security issues"),
            "security",
        )

    async def _review_code(self, content: str) -> List[Dict[str, Any]]:
        """Review code for security issues."""
        issues = []

        # Check for SQL injection
        if "f\"SELECT" in content or 'f"SELECT' in content:
            issues.append({
                "type": "SQL_INJECTION",
                "severity": "HIGH",
                "description": "Potential SQL injection vulnerability detected",
                "suggestion": "Use parameterized queries instead of string formatting",
            })

        # Check for hardcoded credentials
        if "password" in content.lower() and "=" in content:
            issues.append({
                "type": "HARDCODED_CREDENTIALS",
                "severity": "HIGH",
                "description": "Possible hardcoded credentials",
                "suggestion": "Use environment variables or secure vault",
            })

        return issues

    async def _generate_fixes(
        self, content: str, issues: List[Dict[str, Any]]
    ) -> str | None:
        """Fix security issues."""
        fixed = content

        for issue in issues:
            if issue["type"] == "SQL_INJECTION":
                # Fix SQL injection
                fixed = fixed.replace(
                    'query = f"SELECT * FROM users WHERE id = {user_id}"',
                    'query = "SELECT * FROM users WHERE id = ?"\n    result = db.execute(query, (user_id,))'
                )
                fixed = fixed.replace(
                    "result = db.execute(query)",
                    "# Fixed: Using parameterized query above"
                )

        return fixed if fixed != content else None


class PerformanceReviewAgent(ReviewAgent):
    """Agent that reviews and optimizes code for performance."""

    def __init__(self, transaction_manager: TransactionManager):
        super().__init__(
            transaction_manager,
            AgentConfig(name="PerformanceAgent", description="Optimizes performance"),
            "performance",
        )

    async def _review_code(self, content: str) -> List[Dict[str, Any]]:
        """Review code for performance issues."""
        issues = []

        # Check for O(n²) patterns
        if "for item in items:" in content and "for i in range(len(items))" in content:
            issues.append({
                "type": "QUADRATIC_COMPLEXITY",
                "severity": "MEDIUM",
                "description": "Nested loop creating O(n²) complexity",
                "suggestion": "Consider using a set or dictionary for O(1) lookups",
            })

        # Check for inefficient string concatenation
        if "total = total +" in content:
            issues.append({
                "type": "INEFFICIENT_SUM",
                "severity": "LOW",
                "description": "Using loop for summing instead of built-in sum()",
                "suggestion": "Use sum() built-in function",
            })

        return issues

    async def _generate_fixes(
        self, content: str, issues: List[Dict[str, Any]]
    ) -> str | None:
        """Fix performance issues."""
        fixed = content

        for issue in issues:
            if issue["type"] == "INEFFICIENT_SUM":
                # Fix inefficient sum
                old_func = '''def calculate_total(prices):
    total = 0
    for p in prices:
        total = total + p
    return total'''
                new_func = '''def calculate_total(prices):
    """Calculate total price efficiently."""
    return sum(prices)'''
                fixed = fixed.replace(old_func, new_func)

        return fixed if fixed != content else None


class DocumentationAgent(ReviewAgent):
    """Agent that improves code documentation."""

    def __init__(self, transaction_manager: TransactionManager):
        super().__init__(
            transaction_manager,
            AgentConfig(name="DocumentationAgent", description="Improves documentation"),
            "documentation",
        )

    async def _review_code(self, content: str) -> List[Dict[str, Any]]:
        """Review code for documentation issues."""
        issues = []

        # Check for missing docstrings
        if "def get_user_data" in content and '"""' not in content.split("def get_user_data")[1].split("\n")[1]:
            issues.append({
                "type": "MISSING_DOCSTRING",
                "severity": "LOW",
                "location": "get_user_data",
                "suggestion": "Add docstring explaining function purpose",
            })

        if "class UserManager" in content:
            class_section = content.split("class UserManager")[1]
            if '"""' not in class_section[:100]:
                issues.append({
                    "type": "MISSING_CLASS_DOCSTRING",
                    "severity": "LOW",
                    "location": "UserManager",
                    "suggestion": "Add class docstring",
                })

        return issues

    async def _generate_fixes(
        self, content: str, issues: List[Dict[str, Any]]
    ) -> str | None:
        """Add documentation."""
        fixed = content

        for issue in issues:
            if issue["type"] == "MISSING_CLASS_DOCSTRING":
                fixed = fixed.replace(
                    "class UserManager:\n    def __init__(self):",
                    '''class UserManager:
    """
    Manages user data storage and retrieval.

    Provides methods to add and retrieve user information
    using an in-memory dictionary store.
    """

    def __init__(self):'''
                )

        return fixed if fixed != content else None


async def run_multi_agent_review():
    """Run multiple review agents concurrently."""
    print("=" * 60)
    print("Multi-Agent Code Review Example")
    print("=" * 60)

    # Setup STM infrastructure
    storage = MVCCStorage(InMemoryBackend())
    conflict_detector = ConflictDetector(storage)
    manager = TransactionManager(
        storage=storage,
        conflict_detector=conflict_detector,
        default_config=TransactionConfig(max_retries=5),
    )

    # Initialize storage with sample code
    async with manager.transaction() as txn:
        await txn.write("src/user_module.py", SAMPLE_CODE)

    print("\nInitial code loaded into storage")
    print("-" * 40)

    # Create review agents
    agents = [
        SecurityReviewAgent(manager),
        PerformanceReviewAgent(manager),
        DocumentationAgent(manager),
    ]

    # Create tasks for each agent
    tasks = []
    for agent in agents:
        task = AgentTask(
            task_id=f"review-{agent.review_type}",
            description=f"{agent.review_type.capitalize()} review",
            input_data={"file_path": "src/user_module.py"},
        )
        tasks.append((agent, task))

    print(f"\nStarting {len(agents)} concurrent review agents...")
    print("-" * 40)

    # Run all agents concurrently
    async def run_agent(agent: Agent, task: AgentTask) -> AgentResult:
        return await agent.run(task)

    results = await asyncio.gather(
        *[run_agent(agent, task) for agent, task in tasks],
        return_exceptions=True,
    )

    # Process results
    print("\n" + "=" * 60)
    print("Review Results")
    print("=" * 60)

    for i, result in enumerate(results):
        agent = agents[i]
        if isinstance(result, Exception):
            print(f"\n[{agent.review_type.upper()}] Error: {result}")
        else:
            print(f"\n[{agent.review_type.upper()}]")
            print(f"  Status: {result.status.value}")
            if result.output_data:
                print(f"  Issues found: {result.output_data.get('issues_found', 0)}")
                for issue in result.output_data.get("issues", []):
                    print(f"    - {issue['type']}: {issue['description']}")
                if result.output_data.get("fixes_applied"):
                    print(f"  Fixes applied: Yes")

    # Show final code
    print("\n" + "=" * 60)
    print("Final Code After Reviews")
    print("=" * 60)

    async with manager.transaction() as txn:
        final_code = await txn.read("src/user_module.py")
        print(final_code)

    # Show transaction statistics
    print("\n" + "=" * 60)
    print("Transaction Statistics")
    print("=" * 60)
    print(f"Active transactions: {await manager.get_active_transaction_count()}")


if __name__ == "__main__":
    asyncio.run(run_multi_agent_review())

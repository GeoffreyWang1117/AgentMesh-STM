"""
LLM Agent Implementation for AgentMesh-STM.

This module provides LLM-powered agents that can execute complex
tasks using language models while maintaining STM transaction
semantics.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field

from agentmesh_stm.agent.base import (
    Agent,
    AgentConfig,
    AgentResult,
    AgentTask,
)
from agentmesh_stm.agent.tools import (
    CodeAnalysisTools,
    GitTools,
    ProjectTools,
    ShellTools,
    ToolResult,
    create_tool_definitions,
)
from agentmesh_stm.core.transaction import Transaction, TransactionManager
from agentmesh_stm.utils.logging import get_logger

logger = get_logger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""

    OPENAI = auto()
    ANTHROPIC = auto()
    LOCAL = auto()  # For local models via transformers


class LLMConfig(BaseModel):
    """Configuration for LLM integration."""

    provider: LLMProvider = Field(default=LLMProvider.OPENAI, description="LLM provider")
    model: str = Field(default="gpt-4o", description="Model to use")
    api_key: Optional[str] = Field(default=None, description="API key")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    max_tokens: int = Field(default=4096, description="Max tokens in response")
    system_prompt: Optional[str] = Field(default=None, description="System prompt")


@dataclass
class Tool:
    """Represents a tool available to the LLM agent."""

    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[..., Any]

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


class LLMAgent(Agent):
    """
    An agent powered by a Large Language Model.

    The LLM agent can:
    - Execute complex tasks using natural language understanding
    - Use tools to interact with the codebase within transactions
    - Handle multi-step reasoning and planning
    - Automatically manage file reads/writes through the transaction
    """

    def __init__(
        self,
        transaction_manager: TransactionManager,
        llm_config: Optional[LLMConfig] = None,
        agent_config: Optional[AgentConfig] = None,
        enable_advanced_tools: bool = True,
        working_directory: str = ".",
    ):
        super().__init__(transaction_manager, agent_config)
        self._llm_config = llm_config or LLMConfig()
        self._client = None
        self._tools: Dict[str, Tool] = {}
        self._working_directory = working_directory
        self._register_default_tools()
        if enable_advanced_tools:
            self._register_advanced_tools()

    def _register_default_tools(self) -> None:
        """Register default tools for file operations."""
        self.register_tool(
            Tool(
                name="read_file",
                description="Read the contents of a file",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to read",
                        }
                    },
                    "required": ["path"],
                },
                handler=self._tool_read_file,
            )
        )

        self.register_tool(
            Tool(
                name="write_file",
                description="Write content to a file",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to write",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write to the file",
                        },
                    },
                    "required": ["path", "content"],
                },
                handler=self._tool_write_file,
            )
        )

        self.register_tool(
            Tool(
                name="list_files",
                description="List files in a directory",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path to list",
                        },
                        "pattern": {
                            "type": "string",
                            "description": "Optional glob pattern to filter files",
                        },
                    },
                    "required": ["path"],
                },
                handler=self._tool_list_files,
            )
        )

        self.register_tool(
            Tool(
                name="search_code",
                description="Search for a pattern in code files",
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Search pattern (regex supported)",
                        },
                        "path": {
                            "type": "string",
                            "description": "Directory to search in",
                        },
                    },
                    "required": ["pattern"],
                },
                handler=self._tool_search_code,
            )
        )

    def register_tool(self, tool: Tool) -> None:
        """Register a tool for the agent to use."""
        self._tools[tool.name] = tool
        logger.debug("Tool registered", tool=tool.name, agent=self.name)

    def _register_advanced_tools(self) -> None:
        """Register advanced tools for code analysis, git, and project operations."""
        # Code Analysis Tools
        self.register_tool(
            Tool(
                name="analyze_dependencies",
                description="Analyze imports and dependencies in a source file",
                parameters={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to analyze",
                        },
                        "language": {
                            "type": "string",
                            "enum": ["python", "javascript", "typescript"],
                            "description": "Programming language (default: python)",
                        },
                    },
                    "required": ["file_path"],
                },
                handler=self._tool_analyze_dependencies,
            )
        )

        self.register_tool(
            Tool(
                name="find_function_definition",
                description="Find where a function is defined in the codebase",
                parameters={
                    "type": "object",
                    "properties": {
                        "function_name": {
                            "type": "string",
                            "description": "Name of the function to find",
                        },
                        "search_path": {
                            "type": "string",
                            "description": "Directory to search in",
                        },
                    },
                    "required": ["function_name"],
                },
                handler=self._tool_find_function_definition,
            )
        )

        self.register_tool(
            Tool(
                name="find_class_definition",
                description="Find where a class is defined in the codebase",
                parameters={
                    "type": "object",
                    "properties": {
                        "class_name": {
                            "type": "string",
                            "description": "Name of the class to find",
                        },
                        "search_path": {
                            "type": "string",
                            "description": "Directory to search in",
                        },
                    },
                    "required": ["class_name"],
                },
                handler=self._tool_find_class_definition,
            )
        )

        self.register_tool(
            Tool(
                name="get_function_callers",
                description="Find all places where a function is called",
                parameters={
                    "type": "object",
                    "properties": {
                        "function_name": {
                            "type": "string",
                            "description": "Name of the function",
                        },
                        "search_path": {
                            "type": "string",
                            "description": "Directory to search in",
                        },
                    },
                    "required": ["function_name"],
                },
                handler=self._tool_get_function_callers,
            )
        )

        # Git Tools
        self.register_tool(
            Tool(
                name="get_git_status",
                description="Get current Git status (staged, modified, untracked files)",
                parameters={
                    "type": "object",
                    "properties": {
                        "repo_path": {
                            "type": "string",
                            "description": "Path to the repository",
                        },
                    },
                },
                handler=self._tool_git_status,
            )
        )

        self.register_tool(
            Tool(
                name="get_git_diff",
                description="Get Git diff showing changes",
                parameters={
                    "type": "object",
                    "properties": {
                        "repo_path": {
                            "type": "string",
                            "description": "Path to the repository",
                        },
                        "staged": {
                            "type": "boolean",
                            "description": "Show staged changes only",
                        },
                        "file_path": {
                            "type": "string",
                            "description": "Specific file to diff",
                        },
                    },
                },
                handler=self._tool_git_diff,
            )
        )

        self.register_tool(
            Tool(
                name="get_git_log",
                description="Get recent Git commit history",
                parameters={
                    "type": "object",
                    "properties": {
                        "repo_path": {
                            "type": "string",
                            "description": "Path to the repository",
                        },
                        "num_commits": {
                            "type": "integer",
                            "description": "Number of commits to show (default: 10)",
                        },
                        "file_path": {
                            "type": "string",
                            "description": "Show history for specific file",
                        },
                    },
                },
                handler=self._tool_git_log,
            )
        )

        # Project Tools
        self.register_tool(
            Tool(
                name="get_project_structure",
                description="Get project directory structure as a tree",
                parameters={
                    "type": "object",
                    "properties": {
                        "root_path": {
                            "type": "string",
                            "description": "Root directory",
                        },
                        "max_depth": {
                            "type": "integer",
                            "description": "Maximum depth to traverse (default: 3)",
                        },
                    },
                },
                handler=self._tool_project_structure,
            )
        )

        self.register_tool(
            Tool(
                name="find_files",
                description="Find files matching a glob pattern",
                parameters={
                    "type": "object",
                    "properties": {
                        "root_path": {
                            "type": "string",
                            "description": "Root directory",
                        },
                        "pattern": {
                            "type": "string",
                            "description": "Glob pattern (e.g., '*.py', '**/*.ts')",
                        },
                    },
                    "required": ["pattern"],
                },
                handler=self._tool_find_files,
            )
        )

        self.register_tool(
            Tool(
                name="get_file_stats",
                description="Get statistics about a file (size, lines, etc.)",
                parameters={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file",
                        },
                    },
                    "required": ["file_path"],
                },
                handler=self._tool_file_stats,
            )
        )

        # Shell Tools
        self.register_tool(
            Tool(
                name="run_tests",
                description="Run tests using a test framework",
                parameters={
                    "type": "object",
                    "properties": {
                        "test_path": {
                            "type": "string",
                            "description": "Path to tests",
                        },
                        "test_framework": {
                            "type": "string",
                            "enum": ["pytest", "jest", "unittest"],
                            "description": "Test framework to use (default: pytest)",
                        },
                        "verbose": {
                            "type": "boolean",
                            "description": "Enable verbose output",
                        },
                    },
                },
                handler=self._tool_run_tests,
            )
        )

        logger.info(
            "Advanced tools registered",
            agent=self.name,
            tools=[
                "analyze_dependencies", "find_function_definition", "find_class_definition",
                "get_function_callers", "get_git_status", "get_git_diff", "get_git_log",
                "get_project_structure", "find_files", "get_file_stats", "run_tests"
            ],
        )

    async def _get_client(self) -> Any:
        """Get or create the LLM client."""
        if self._client:
            return self._client

        if self._llm_config.provider == LLMProvider.OPENAI:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self._llm_config.api_key)

        elif self._llm_config.provider == LLMProvider.ANTHROPIC:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=self._llm_config.api_key)

        return self._client

    async def execute_task(
        self, task: AgentTask, transaction: Transaction
    ) -> Dict[str, Any]:
        """
        Execute a task using the LLM.

        Args:
            task: Task to execute
            transaction: Active transaction

        Returns:
            Output data from task execution
        """
        # Store transaction reference for tools
        self._current_transaction = transaction

        # Build messages
        messages = self._build_messages(task)

        # Get tool definitions
        tools = self._get_tool_definitions()

        # Execute LLM loop
        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            response = await self._call_llm(messages, tools)

            # Check if we need to call tools
            tool_calls = self._extract_tool_calls(response)

            if not tool_calls:
                # No more tool calls, return final response
                final_content = self._extract_content(response)
                return {
                    "response": final_content,
                    "iterations": iteration,
                    "modified_files": list(transaction.write_set.get_resource_ids()),
                }

            # Execute tool calls
            tool_results = await self._execute_tool_calls(tool_calls, transaction)

            # Add assistant message and tool results to conversation
            messages.append(self._format_assistant_message(response))
            messages.extend(self._format_tool_results(tool_calls, tool_results))

        return {
            "response": "Max iterations reached",
            "iterations": max_iterations,
            "modified_files": list(transaction.write_set.get_resource_ids()),
        }

    def _build_messages(self, task: AgentTask) -> List[Dict[str, Any]]:
        """Build the message list for the LLM."""
        messages = []

        # System prompt
        system_prompt = self._llm_config.system_prompt or self._default_system_prompt()
        messages.append({"role": "system", "content": system_prompt})

        # Task description
        user_content = f"""Task: {task.description}

Input data:
{json.dumps(task.input_data, indent=2)}

Please complete this task. Use the available tools to read and modify files as needed.
Ensure all changes are logically consistent and maintain code quality."""

        messages.append({"role": "user", "content": user_content})

        return messages

    def _default_system_prompt(self) -> str:
        """Return the default system prompt."""
        return """You are a skilled software engineering agent working within a transactional framework.
Your changes are tracked and can be rolled back if they conflict with other agents' work.

Guidelines:
1. Read files before modifying them to understand the context
2. Make minimal, focused changes that accomplish the task
3. Preserve existing code style and conventions
4. Handle errors gracefully
5. Document significant changes

You have access to tools for file operations and code search. Use them to understand
the codebase and make necessary changes."""

    def _get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get tool definitions in provider-specific format."""
        if self._llm_config.provider == LLMProvider.OPENAI:
            return [tool.to_openai_format() for tool in self._tools.values()]
        elif self._llm_config.provider == LLMProvider.ANTHROPIC:
            return [tool.to_anthropic_format() for tool in self._tools.values()]
        return []

    async def _call_llm(
        self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]
    ) -> Any:
        """Call the LLM with messages and tools."""
        client = await self._get_client()

        if self._llm_config.provider == LLMProvider.OPENAI:
            response = await client.chat.completions.create(
                model=self._llm_config.model,
                messages=messages,
                tools=tools if tools else None,
                temperature=self._llm_config.temperature,
                max_tokens=self._llm_config.max_tokens,
            )
            return response

        elif self._llm_config.provider == LLMProvider.ANTHROPIC:
            # Convert messages for Anthropic format
            anthropic_messages = []
            system = None

            for msg in messages:
                if msg["role"] == "system":
                    system = msg["content"]
                else:
                    anthropic_messages.append(msg)

            response = await client.messages.create(
                model=self._llm_config.model,
                system=system,
                messages=anthropic_messages,
                tools=tools if tools else None,
                max_tokens=self._llm_config.max_tokens,
            )
            return response

        raise ValueError(f"Unsupported provider: {self._llm_config.provider}")

    def _extract_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Extract tool calls from LLM response."""
        tool_calls = []

        if self._llm_config.provider == LLMProvider.OPENAI:
            message = response.choices[0].message
            if message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append({
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments),
                    })

        elif self._llm_config.provider == LLMProvider.ANTHROPIC:
            for block in response.content:
                if block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "name": block.name,
                        "arguments": block.input,
                    })

        return tool_calls

    def _extract_content(self, response: Any) -> str:
        """Extract text content from LLM response."""
        if self._llm_config.provider == LLMProvider.OPENAI:
            return response.choices[0].message.content or ""

        elif self._llm_config.provider == LLMProvider.ANTHROPIC:
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return ""

        return ""

    async def _execute_tool_calls(
        self, tool_calls: List[Dict[str, Any]], transaction: Transaction
    ) -> List[Dict[str, Any]]:
        """Execute tool calls and return results."""
        results = []

        for tc in tool_calls:
            tool_name = tc["name"]
            arguments = tc["arguments"]

            tool = self._tools.get(tool_name)
            if not tool:
                results.append({
                    "id": tc["id"],
                    "error": f"Unknown tool: {tool_name}",
                })
                continue

            try:
                # Execute tool with transaction context
                if asyncio.iscoroutinefunction(tool.handler):
                    result = await tool.handler(transaction=transaction, **arguments)
                else:
                    result = tool.handler(transaction=transaction, **arguments)

                results.append({
                    "id": tc["id"],
                    "result": result,
                })

            except Exception as e:
                logger.error("Tool execution error", tool=tool_name, error=str(e))
                results.append({
                    "id": tc["id"],
                    "error": str(e),
                })

        return results

    def _format_assistant_message(self, response: Any) -> Dict[str, Any]:
        """Format assistant message for conversation history."""
        if self._llm_config.provider == LLMProvider.OPENAI:
            message = response.choices[0].message
            return {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in (message.tool_calls or [])
                ],
            }

        elif self._llm_config.provider == LLMProvider.ANTHROPIC:
            return {"role": "assistant", "content": response.content}

        return {"role": "assistant", "content": ""}

    def _format_tool_results(
        self,
        tool_calls: List[Dict[str, Any]],
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Format tool results for conversation history."""
        messages = []

        if self._llm_config.provider == LLMProvider.OPENAI:
            for tc, result in zip(tool_calls, results):
                content = (
                    json.dumps(result.get("result", result.get("error")))
                    if isinstance(result.get("result"), dict)
                    else str(result.get("result", result.get("error")))
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": content,
                })

        elif self._llm_config.provider == LLMProvider.ANTHROPIC:
            tool_results = []
            for tc, result in zip(tool_calls, results):
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": str(result.get("result", result.get("error"))),
                })
            messages.append({"role": "user", "content": tool_results})

        return messages

    # Default tool implementations
    async def _tool_read_file(self, transaction: Transaction, path: str) -> str:
        """Read a file through the transaction."""
        content = await transaction.read(path)
        if content is None:
            # Try reading directly if not in transaction storage
            import aiofiles
            import os

            if os.path.exists(path):
                async with aiofiles.open(path, "r") as f:
                    content = await f.read()
            else:
                return f"File not found: {path}"
        return content

    async def _tool_write_file(
        self, transaction: Transaction, path: str, content: str
    ) -> str:
        """Write to a file through the transaction."""
        await transaction.write(path, content)
        return f"Written to {path}"

    async def _tool_list_files(
        self, transaction: Transaction, path: str, pattern: str = "*"
    ) -> List[str]:
        """List files in a directory."""
        import glob
        import os

        full_pattern = os.path.join(path, pattern)
        files = glob.glob(full_pattern, recursive=True)
        return files

    async def _tool_search_code(
        self, transaction: Transaction, pattern: str, path: str = "."
    ) -> List[Dict[str, Any]]:
        """Search for pattern in code files."""
        import os
        import re

        results = []
        code_extensions = {".py", ".js", ".ts", ".java", ".go", ".rs", ".cpp", ".c", ".h"}

        for root, dirs, files in os.walk(path):
            # Skip hidden and common non-code directories
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in {"node_modules", "venv", "__pycache__"}]

            for file in files:
                ext = os.path.splitext(file)[1]
                if ext not in code_extensions:
                    continue

                filepath = os.path.join(root, file)
                try:
                    with open(filepath, "r") as f:
                        content = f.read()

                    for i, line in enumerate(content.split("\n"), 1):
                        if re.search(pattern, line):
                            results.append({
                                "file": filepath,
                                "line": i,
                                "content": line.strip()[:200],
                            })

                except Exception:
                    continue

        return results[:50]  # Limit results

    # Advanced tool handlers
    async def _tool_analyze_dependencies(
        self,
        transaction: Transaction,
        file_path: str,
        language: str = "python",
    ) -> Dict[str, Any]:
        """Analyze dependencies in a file."""
        result = await CodeAnalysisTools.analyze_dependencies(
            transaction, file_path, language
        )
        if result.success:
            return {"dependencies": result.output}
        return {"error": result.error}

    async def _tool_find_function_definition(
        self,
        transaction: Transaction,
        function_name: str,
        search_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Find function definition."""
        path = search_path or self._working_directory
        result = await CodeAnalysisTools.find_function_definition(
            transaction, function_name, path
        )
        if result.success:
            return {"locations": result.output}
        return {"error": result.error}

    async def _tool_find_class_definition(
        self,
        transaction: Transaction,
        class_name: str,
        search_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Find class definition."""
        path = search_path or self._working_directory
        result = await CodeAnalysisTools.find_class_definition(
            transaction, class_name, path
        )
        if result.success:
            return {"locations": result.output}
        return {"error": result.error}

    async def _tool_get_function_callers(
        self,
        transaction: Transaction,
        function_name: str,
        search_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Find function callers."""
        path = search_path or self._working_directory
        result = await CodeAnalysisTools.get_function_callers(
            transaction, function_name, path
        )
        if result.success:
            return {"callers": result.output}
        return {"error": result.error}

    async def _tool_git_status(
        self,
        transaction: Transaction,
        repo_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get git status."""
        path = repo_path or self._working_directory
        result = await GitTools.get_git_status(path)
        if result.success:
            return result.output
        return {"error": result.error}

    async def _tool_git_diff(
        self,
        transaction: Transaction,
        repo_path: Optional[str] = None,
        staged: bool = False,
        file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get git diff."""
        path = repo_path or self._working_directory
        result = await GitTools.get_git_diff(path, staged, file_path)
        if result.success:
            return {"diff": result.output}
        return {"error": result.error}

    async def _tool_git_log(
        self,
        transaction: Transaction,
        repo_path: Optional[str] = None,
        num_commits: int = 10,
        file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get git log."""
        path = repo_path or self._working_directory
        result = await GitTools.get_git_log(path, num_commits, file_path)
        if result.success:
            return {"commits": result.output}
        return {"error": result.error}

    async def _tool_project_structure(
        self,
        transaction: Transaction,
        root_path: Optional[str] = None,
        max_depth: int = 3,
    ) -> Dict[str, Any]:
        """Get project structure."""
        path = root_path or self._working_directory
        result = await ProjectTools.get_project_structure(path, max_depth)
        if result.success:
            return {"structure": result.output}
        return {"error": result.error}

    async def _tool_find_files(
        self,
        transaction: Transaction,
        pattern: str,
        root_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Find files by pattern."""
        path = root_path or self._working_directory
        result = await ProjectTools.find_files_by_pattern(path, pattern)
        if result.success:
            return {"files": result.output}
        return {"error": result.error}

    async def _tool_file_stats(
        self,
        transaction: Transaction,
        file_path: str,
    ) -> Dict[str, Any]:
        """Get file statistics."""
        result = await ProjectTools.get_file_stats(file_path)
        if result.success:
            return result.output
        return {"error": result.error}

    async def _tool_run_tests(
        self,
        transaction: Transaction,
        test_path: Optional[str] = None,
        test_framework: str = "pytest",
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Run tests."""
        path = test_path or "."
        result = await ShellTools.run_tests(path, test_framework, verbose)
        return {
            "success": result.success,
            "output": result.output,
            "error": result.error,
        }

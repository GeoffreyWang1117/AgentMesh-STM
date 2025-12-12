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
    ):
        super().__init__(transaction_manager, agent_config)
        self._llm_config = llm_config or LLMConfig()
        self._client = None
        self._tools: Dict[str, Tool] = {}
        self._register_default_tools()

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

"""Agent integration layer."""

from agentmesh_stm.agent.base import Agent, AgentTask, AgentResult, AgentConfig, SimpleAgent
from agentmesh_stm.agent.llm_agent import LLMAgent, LLMConfig, LLMProvider, Tool
from agentmesh_stm.agent.tools import (
    CodeAnalysisTools,
    GitTools,
    ProjectTools,
    ShellTools,
    ToolResult,
    create_tool_definitions,
)

__all__ = [
    # Base
    "Agent",
    "AgentTask",
    "AgentResult",
    "AgentConfig",
    "SimpleAgent",
    # LLM Agent
    "LLMAgent",
    "LLMConfig",
    "LLMProvider",
    "Tool",
    # Advanced Tools
    "CodeAnalysisTools",
    "GitTools",
    "ProjectTools",
    "ShellTools",
    "ToolResult",
    "create_tool_definitions",
]

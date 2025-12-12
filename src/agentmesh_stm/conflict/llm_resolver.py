"""
LLM-based Semantic Conflict Resolution for AgentMesh-STM.

This module provides intelligent conflict resolution using LLMs to:
- Understand the semantic intent of conflicting changes
- Merge changes that modify different logical parts of code
- Generate merged code that preserves both changes' intent
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from agentmesh_stm.utils.logging import get_logger

logger = get_logger(__name__)


class MergeStrategy(Enum):
    """Strategies for resolving conflicts."""

    AUTO = auto()         # Let LLM decide best approach
    PRESERVE_BOTH = auto()  # Try to keep both changes
    PREFER_NEWER = auto()   # Prefer the newer change
    PREFER_OLDER = auto()   # Prefer the older change
    SEMANTIC_MERGE = auto()  # Deep semantic understanding


@dataclass
class ConflictInfo:
    """Information about a conflict."""

    resource_id: str
    base_content: str       # Original content
    local_content: str      # Local (transaction) changes
    remote_content: str     # Remote (committed) changes
    conflict_type: str      # "write-write", "semantic", etc.
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MergeResult:
    """Result of a merge operation."""

    success: bool
    merged_content: Optional[str] = None
    explanation: str = ""
    conflicts_remaining: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class LLMResolverConfig:
    """Configuration for LLM-based resolver."""

    provider: str = "openai"  # "openai" or "anthropic"
    model: str = "gpt-4o"
    api_key: Optional[str] = None
    temperature: float = 0.2  # Lower for more deterministic merges
    max_tokens: int = 4096
    timeout_seconds: float = 60.0


class LLMConflictResolver:
    """
    LLM-based conflict resolver that uses semantic understanding
    to merge conflicting code changes.
    """

    def __init__(self, config: Optional[LLMResolverConfig] = None):
        self.config = config or LLMResolverConfig()
        self._client = None

    async def _get_client(self) -> Any:
        """Get or create the LLM client."""
        if self._client:
            return self._client

        if self.config.provider == "openai":
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self.config.api_key)
        elif self.config.provider == "anthropic":
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(api_key=self.config.api_key)

        return self._client

    async def resolve(
        self,
        conflict: ConflictInfo,
        strategy: MergeStrategy = MergeStrategy.AUTO,
    ) -> MergeResult:
        """
        Resolve a conflict using LLM-based semantic understanding.

        Args:
            conflict: Information about the conflict
            strategy: Merge strategy to use

        Returns:
            MergeResult with merged content or failure info
        """
        logger.info(
            "Resolving conflict with LLM",
            resource=conflict.resource_id,
            strategy=strategy.name,
        )

        try:
            # Build the prompt based on strategy
            prompt = self._build_merge_prompt(conflict, strategy)

            # Call the LLM
            response = await self._call_llm(prompt)

            # Parse the response
            result = self._parse_merge_response(response, conflict)

            logger.info(
                "Conflict resolved",
                resource=conflict.resource_id,
                success=result.success,
                confidence=result.confidence,
            )

            return result

        except Exception as e:
            logger.error(
                "Failed to resolve conflict",
                resource=conflict.resource_id,
                error=str(e),
            )
            return MergeResult(
                success=False,
                explanation=f"LLM resolution failed: {str(e)}",
            )

    def _build_merge_prompt(
        self,
        conflict: ConflictInfo,
        strategy: MergeStrategy,
    ) -> str:
        """Build the prompt for the LLM."""
        strategy_instruction = self._get_strategy_instruction(strategy)

        prompt = f"""You are a code merge expert. Your task is to merge conflicting changes to a file.

## Context
- File: {conflict.resource_id}
- Conflict Type: {conflict.conflict_type}
{self._format_context(conflict.context)}

## Strategy
{strategy_instruction}

## Base Version (Original)
```
{conflict.base_content}
```

## Local Changes (Transaction A)
```
{conflict.local_content}
```

## Remote Changes (Transaction B - Already Committed)
```
{conflict.remote_content}
```

## Instructions
1. Analyze the semantic intent of both sets of changes
2. Identify which parts of the code each transaction modified
3. Determine if the changes are compatible (modify different logical units)
4. If compatible, merge both changes while preserving correct syntax
5. If incompatible, explain why and suggest the best resolution

## Response Format
Respond with a JSON object containing:
- "success": boolean indicating if merge was possible
- "merged_content": the merged code (or null if failed)
- "explanation": detailed explanation of the merge decision
- "conflicts_remaining": list of any unresolved conflict descriptions
- "confidence": number from 0-1 indicating confidence in the merge

Example response:
```json
{{
    "success": true,
    "merged_content": "// merged code here",
    "explanation": "Both changes modify different functions...",
    "conflicts_remaining": [],
    "confidence": 0.95
}}
```

Provide your analysis and merged result:"""

        return prompt

    def _get_strategy_instruction(self, strategy: MergeStrategy) -> str:
        """Get instruction text for a strategy."""
        instructions = {
            MergeStrategy.AUTO: (
                "Analyze both changes and determine the best merge approach. "
                "Prioritize preserving the semantic intent of both changes if possible."
            ),
            MergeStrategy.PRESERVE_BOTH: (
                "Try to preserve both sets of changes. If they modify different "
                "parts of the code, combine them. Only report failure if the "
                "changes directly conflict."
            ),
            MergeStrategy.PREFER_NEWER: (
                "When changes conflict, prefer the remote (newer) changes. "
                "But still try to incorporate local changes that don't conflict."
            ),
            MergeStrategy.PREFER_OLDER: (
                "When changes conflict, prefer the local (older) changes. "
                "But still try to incorporate remote changes that don't conflict."
            ),
            MergeStrategy.SEMANTIC_MERGE: (
                "Perform a deep semantic analysis. Understand what each change "
                "is trying to accomplish and create a merge that achieves both "
                "goals, even if it requires restructuring the code."
            ),
        }
        return instructions.get(strategy, instructions[MergeStrategy.AUTO])

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format additional context information."""
        if not context:
            return ""

        lines = ["## Additional Context"]
        for key, value in context.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the prompt."""
        client = await self._get_client()

        if self.config.provider == "openai":
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a code merge expert that specializes in resolving conflicts in software code. Always respond with valid JSON.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    response_format={"type": "json_object"},
                ),
                timeout=self.config.timeout_seconds,
            )
            return response.choices[0].message.content

        elif self.config.provider == "anthropic":
            response = await asyncio.wait_for(
                client.messages.create(
                    model=self.config.model,
                    system="You are a code merge expert that specializes in resolving conflicts in software code. Always respond with valid JSON.",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.config.max_tokens,
                ),
                timeout=self.config.timeout_seconds,
            )
            return response.content[0].text

        raise ValueError(f"Unsupported provider: {self.config.provider}")

    def _parse_merge_response(
        self,
        response: str,
        conflict: ConflictInfo,
    ) -> MergeResult:
        """Parse the LLM response into a MergeResult."""
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]

            data = json.loads(json_str.strip())

            return MergeResult(
                success=data.get("success", False),
                merged_content=data.get("merged_content"),
                explanation=data.get("explanation", ""),
                conflicts_remaining=data.get("conflicts_remaining", []),
                confidence=float(data.get("confidence", 0.0)),
            )

        except json.JSONDecodeError as e:
            logger.warning(
                "Failed to parse LLM response as JSON",
                error=str(e),
                response=response[:200],
            )
            # Try to extract useful information from non-JSON response
            return MergeResult(
                success=False,
                explanation=f"Failed to parse response: {str(e)}",
            )


class BatchConflictResolver:
    """
    Resolves multiple conflicts efficiently using batching.
    """

    def __init__(
        self,
        resolver: LLMConflictResolver,
        max_concurrent: int = 5,
    ):
        self.resolver = resolver
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def resolve_all(
        self,
        conflicts: List[ConflictInfo],
        strategy: MergeStrategy = MergeStrategy.AUTO,
    ) -> Dict[str, MergeResult]:
        """
        Resolve multiple conflicts concurrently.

        Args:
            conflicts: List of conflicts to resolve
            strategy: Merge strategy to use

        Returns:
            Dictionary mapping resource_id to MergeResult
        """
        async def resolve_with_semaphore(conflict: ConflictInfo) -> Tuple[str, MergeResult]:
            async with self._semaphore:
                result = await self.resolver.resolve(conflict, strategy)
                return conflict.resource_id, result

        tasks = [resolve_with_semaphore(c) for c in conflicts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        resolved = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                resolved[conflicts[i].resource_id] = MergeResult(
                    success=False,
                    explanation=f"Resolution failed: {str(result)}",
                )
            else:
                resource_id, merge_result = result
                resolved[resource_id] = merge_result

        return resolved


class ThreeWayMerger:
    """
    Performs three-way merge using LLM for semantic understanding.
    """

    def __init__(self, resolver: LLMConflictResolver):
        self.resolver = resolver

    async def merge(
        self,
        base: str,
        local: str,
        remote: str,
        file_path: str,
        strategy: MergeStrategy = MergeStrategy.AUTO,
    ) -> MergeResult:
        """
        Perform a three-way merge.

        Args:
            base: Base version content
            local: Local changes
            remote: Remote changes
            file_path: Path to the file (for context)
            strategy: Merge strategy

        Returns:
            MergeResult with merged content
        """
        # First, check if traditional merge would work
        simple_merge = self._try_simple_merge(base, local, remote)
        if simple_merge:
            return MergeResult(
                success=True,
                merged_content=simple_merge,
                explanation="Simple line-by-line merge succeeded",
                confidence=1.0,
            )

        # Fall back to LLM-based semantic merge
        conflict = ConflictInfo(
            resource_id=file_path,
            base_content=base,
            local_content=local,
            remote_content=remote,
            conflict_type="three-way-merge",
        )

        return await self.resolver.resolve(conflict, strategy)

    def _try_simple_merge(
        self,
        base: str,
        local: str,
        remote: str,
    ) -> Optional[str]:
        """
        Try a simple line-by-line merge.

        Returns merged content if successful, None if conflicts detected.
        """
        base_lines = base.split("\n")
        local_lines = local.split("\n")
        remote_lines = remote.split("\n")

        # Find changes from base in each version
        local_changes = self._find_changes(base_lines, local_lines)
        remote_changes = self._find_changes(base_lines, remote_lines)

        # Check for overlapping changes
        for line_num in local_changes:
            if line_num in remote_changes:
                if local_changes[line_num] != remote_changes[line_num]:
                    # Conflict detected
                    return None

        # Apply non-conflicting changes
        result_lines = base_lines.copy()

        # Apply remote changes first
        for line_num, change in sorted(remote_changes.items(), reverse=True):
            self._apply_change(result_lines, line_num, change)

        # Apply local changes
        for line_num, change in sorted(local_changes.items(), reverse=True):
            if line_num not in remote_changes:
                self._apply_change(result_lines, line_num, change)

        return "\n".join(result_lines)

    def _find_changes(
        self,
        base: List[str],
        modified: List[str],
    ) -> Dict[int, Tuple[str, str]]:
        """
        Find changed lines between base and modified versions.

        Returns dict mapping line number to (action, content).
        """
        changes = {}

        # Simple diff - for production, use a proper diff algorithm
        min_len = min(len(base), len(modified))

        for i in range(min_len):
            if base[i] != modified[i]:
                changes[i] = ("modify", modified[i])

        # Handle length differences
        if len(modified) > len(base):
            for i in range(len(base), len(modified)):
                changes[i] = ("add", modified[i])
        elif len(modified) < len(base):
            for i in range(len(modified), len(base)):
                changes[i] = ("delete", "")

        return changes

    def _apply_change(
        self,
        lines: List[str],
        line_num: int,
        change: Tuple[str, str],
    ) -> None:
        """Apply a change to the lines list."""
        action, content = change

        if action == "modify" and line_num < len(lines):
            lines[line_num] = content
        elif action == "add":
            if line_num >= len(lines):
                lines.append(content)
            else:
                lines.insert(line_num, content)
        elif action == "delete" and line_num < len(lines):
            lines.pop(line_num)


class ConflictAnalyzer:
    """
    Analyzes conflicts to provide detailed information for resolution.
    """

    def __init__(self, resolver: LLMConflictResolver):
        self.resolver = resolver

    async def analyze(
        self,
        conflict: ConflictInfo,
    ) -> Dict[str, Any]:
        """
        Analyze a conflict without resolving it.

        Returns detailed analysis including:
        - Change classification
        - Overlap detection
        - Resolution recommendations
        """
        prompt = self._build_analysis_prompt(conflict)
        response = await self.resolver._call_llm(prompt)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"error": "Failed to parse analysis response"}

    def _build_analysis_prompt(self, conflict: ConflictInfo) -> str:
        """Build prompt for conflict analysis."""
        return f"""Analyze this code conflict without resolving it.

## File: {conflict.resource_id}

## Base Version
```
{conflict.base_content}
```

## Local Changes
```
{conflict.local_content}
```

## Remote Changes
```
{conflict.remote_content}
```

Provide a JSON analysis with:
- "local_changes": list of changes made in local version with descriptions
- "remote_changes": list of changes made in remote version with descriptions
- "overlap_regions": list of code regions where both versions made changes
- "semantic_compatibility": boolean indicating if changes are semantically compatible
- "resolution_difficulty": "easy", "medium", or "hard"
- "recommended_strategy": one of "preserve_both", "prefer_local", "prefer_remote", "manual"
- "reasoning": detailed explanation of the analysis

Respond with valid JSON only."""

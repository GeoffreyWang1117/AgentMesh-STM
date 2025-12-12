"""
Semantic Conflict Detection for AgentMesh-STM.

This module provides hierarchical conflict detection that goes beyond
simple version comparison to identify true semantic conflicts:

1. File-level: Different files = no conflict
2. AST-level: Same file but different code regions = likely no conflict
3. Semantic-level: Same region, use LLM to determine compatibility
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from agentmesh_stm.conflict.ast_analyzer import ASTAnalyzer, CodeRegion, CodeRegionType

if TYPE_CHECKING:
    from agentmesh_stm.storage.mvcc import MVCCStorage


class ConflictType(Enum):
    """Types of conflicts that can be detected."""

    NONE = auto()  # No conflict
    READ_WRITE = auto()  # Another transaction modified data we read
    WRITE_WRITE = auto()  # Another transaction modified data we're writing
    SEMANTIC = auto()  # Changes are semantically incompatible
    STRUCTURAL = auto()  # Changes to same code structure


class ConflictLevel(Enum):
    """Severity levels for detected conflicts."""

    NONE = auto()  # No conflict
    LOW = auto()  # Likely mergeable
    MEDIUM = auto()  # May be mergeable with review
    HIGH = auto()  # Definite conflict
    CRITICAL = auto()  # Cannot proceed


@dataclass
class ConflictResult:
    """Result of conflict detection."""

    has_conflict: bool
    conflict_type: ConflictType
    conflict_level: ConflictLevel
    resource_id: str
    description: str
    conflicting_regions: List[CodeRegion] = field(default_factory=list)
    suggested_resolution: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def no_conflict(cls, resource_id: str) -> "ConflictResult":
        """Create a no-conflict result."""
        return cls(
            has_conflict=False,
            conflict_type=ConflictType.NONE,
            conflict_level=ConflictLevel.NONE,
            resource_id=resource_id,
            description="No conflict detected",
        )

    @classmethod
    def conflict(
        cls,
        resource_id: str,
        conflict_type: ConflictType,
        conflict_level: ConflictLevel,
        description: str,
        **kwargs,
    ) -> "ConflictResult":
        """Create a conflict result."""
        return cls(
            has_conflict=True,
            conflict_type=conflict_type,
            conflict_level=conflict_level,
            resource_id=resource_id,
            description=description,
            **kwargs,
        )


class ConflictDetectorConfig(BaseModel):
    """Configuration for conflict detection."""

    enable_ast_analysis: bool = Field(
        default=True, description="Enable AST-based conflict analysis"
    )
    enable_semantic_analysis: bool = Field(
        default=True, description="Enable LLM-based semantic analysis"
    )
    semantic_model: str = Field(
        default="gpt-4o-mini", description="Model for semantic analysis"
    )
    semantic_threshold: float = Field(
        default=0.7, description="Confidence threshold for semantic conflicts"
    )
    max_semantic_tokens: int = Field(
        default=2000, description="Max tokens for semantic analysis context"
    )
    file_extensions_to_analyze: List[str] = Field(
        default=[".py", ".js", ".ts", ".java", ".go", ".rs"],
        description="File extensions for AST analysis",
    )


class ConflictDetector:
    """
    Hierarchical semantic conflict detector.

    Detection proceeds in three levels:
    1. File-level: Quick check if resources are different files
    2. AST-level: Parse code and check if changes are in different regions
    3. Semantic-level: Use LLM to determine if overlapping changes conflict
    """

    def __init__(
        self,
        storage: Optional["MVCCStorage"] = None,
        config: Optional[ConflictDetectorConfig] = None,
        llm_client: Optional[Any] = None,
    ):
        self._storage = storage
        self._config = config or ConflictDetectorConfig()
        self._llm_client = llm_client
        self._ast_analyzer = ASTAnalyzer()
        self._cache: Dict[str, ConflictResult] = {}

    async def check_conflict(
        self,
        resource_id: str,
        read_version: int,
        current_version: int,
    ) -> ConflictResult:
        """
        Check for read-write conflict on a resource.

        Args:
            resource_id: ID of the resource
            read_version: Version that was read
            current_version: Current version of the resource

        Returns:
            ConflictResult indicating if there's a conflict
        """
        if read_version >= current_version:
            return ConflictResult.no_conflict(resource_id)

        # Get the content at both versions
        if not self._storage:
            # No storage, assume conflict based on version
            return ConflictResult.conflict(
                resource_id=resource_id,
                conflict_type=ConflictType.READ_WRITE,
                conflict_level=ConflictLevel.HIGH,
                description=f"Resource modified: version {read_version} -> {current_version}",
            )

        old_resource = await self._storage.read(resource_id, version=read_version)
        new_resource = await self._storage.read(resource_id, version=current_version)

        if not old_resource or not new_resource:
            return ConflictResult.conflict(
                resource_id=resource_id,
                conflict_type=ConflictType.READ_WRITE,
                conflict_level=ConflictLevel.HIGH,
                description="Resource version not found",
            )

        # If content hash is the same, no real conflict
        if old_resource.content_hash == new_resource.content_hash:
            return ConflictResult.no_conflict(resource_id)

        # Determine if this is code that can be analyzed
        language = self._get_language(resource_id)
        if not language or not self._config.enable_ast_analysis:
            return ConflictResult.conflict(
                resource_id=resource_id,
                conflict_type=ConflictType.READ_WRITE,
                conflict_level=ConflictLevel.MEDIUM,
                description="Resource modified, unable to determine semantic conflict",
            )

        # Perform AST analysis
        return await self._analyze_code_conflict(
            resource_id,
            old_resource.content,
            new_resource.content,
            language,
        )

    async def check_write_conflict(
        self,
        resource_id: str,
        our_content: str,
        base_version: int,
        current_version: int,
    ) -> ConflictResult:
        """
        Check for write-write conflict.

        Args:
            resource_id: ID of the resource
            our_content: Content we're trying to write
            base_version: Version we based our changes on
            current_version: Current version someone else wrote

        Returns:
            ConflictResult indicating if there's a conflict
        """
        if base_version >= current_version:
            return ConflictResult.no_conflict(resource_id)

        if not self._storage:
            return ConflictResult.conflict(
                resource_id=resource_id,
                conflict_type=ConflictType.WRITE_WRITE,
                conflict_level=ConflictLevel.HIGH,
                description="Concurrent write detected",
            )

        base_resource = await self._storage.read(resource_id, version=base_version)
        current_resource = await self._storage.read(resource_id, version=current_version)

        if not base_resource:
            base_content = ""
        else:
            base_content = base_resource.content

        if not current_resource:
            return ConflictResult.no_conflict(resource_id)

        their_content = current_resource.content

        # Determine language
        language = self._get_language(resource_id)
        if not language or not self._config.enable_ast_analysis:
            return ConflictResult.conflict(
                resource_id=resource_id,
                conflict_type=ConflictType.WRITE_WRITE,
                conflict_level=ConflictLevel.HIGH,
                description="Concurrent writes to same resource",
            )

        # Analyze the three-way conflict
        return await self._analyze_three_way_conflict(
            resource_id,
            base_content,
            our_content,
            their_content,
            language,
        )

    async def _analyze_code_conflict(
        self,
        resource_id: str,
        old_content: str,
        new_content: str,
        language: str,
    ) -> ConflictResult:
        """Analyze conflict between two versions of code."""
        try:
            modified_regions, modified_lines = self._ast_analyzer.get_modified_regions(
                old_content, new_content, language
            )

            if not modified_regions:
                # Changes are outside any tracked region (e.g., whitespace)
                return ConflictResult.no_conflict(resource_id)

            return ConflictResult.conflict(
                resource_id=resource_id,
                conflict_type=ConflictType.READ_WRITE,
                conflict_level=ConflictLevel.MEDIUM,
                description=f"Modified regions: {', '.join(r.full_name for r in modified_regions[:3])}",
                conflicting_regions=modified_regions,
                metadata={"modified_lines": modified_lines},
            )

        except Exception as e:
            # AST parsing failed, fall back to version-based conflict
            return ConflictResult.conflict(
                resource_id=resource_id,
                conflict_type=ConflictType.READ_WRITE,
                conflict_level=ConflictLevel.MEDIUM,
                description=f"AST analysis failed: {e}",
            )

    async def _analyze_three_way_conflict(
        self,
        resource_id: str,
        base_content: str,
        our_content: str,
        their_content: str,
        language: str,
    ) -> ConflictResult:
        """Analyze three-way merge conflict."""
        try:
            # Get modified regions for both changes
            our_regions, our_lines = self._ast_analyzer.get_modified_regions(
                base_content, our_content, language
            )
            their_regions, their_lines = self._ast_analyzer.get_modified_regions(
                base_content, their_content, language
            )

            # Check for overlapping regions
            overlapping = []
            for our_region in our_regions:
                for their_region in their_regions:
                    if our_region.overlaps_with(their_region):
                        overlapping.append((our_region, their_region))

            if not overlapping:
                # No overlapping regions, changes can likely be merged
                return ConflictResult(
                    has_conflict=False,
                    conflict_type=ConflictType.NONE,
                    conflict_level=ConflictLevel.LOW,
                    resource_id=resource_id,
                    description="Changes are in different code regions, may be auto-mergeable",
                    metadata={
                        "our_regions": [r.full_name for r in our_regions],
                        "their_regions": [r.full_name for r in their_regions],
                    },
                )

            # Overlapping regions - need semantic analysis
            if self._config.enable_semantic_analysis and self._llm_client:
                return await self._semantic_conflict_analysis(
                    resource_id,
                    base_content,
                    our_content,
                    their_content,
                    overlapping,
                    language,
                )

            # No semantic analysis available, report as conflict
            return ConflictResult.conflict(
                resource_id=resource_id,
                conflict_type=ConflictType.WRITE_WRITE,
                conflict_level=ConflictLevel.HIGH,
                description=f"Overlapping changes in: {', '.join(r[0].full_name for r in overlapping[:3])}",
                conflicting_regions=[r[0] for r in overlapping],
            )

        except Exception as e:
            return ConflictResult.conflict(
                resource_id=resource_id,
                conflict_type=ConflictType.WRITE_WRITE,
                conflict_level=ConflictLevel.HIGH,
                description=f"Conflict analysis failed: {e}",
            )

    async def _semantic_conflict_analysis(
        self,
        resource_id: str,
        base_content: str,
        our_content: str,
        their_content: str,
        overlapping_regions: List[tuple],
        language: str,
    ) -> ConflictResult:
        """Use LLM to determine if overlapping changes are semantically compatible."""
        if not self._llm_client:
            return ConflictResult.conflict(
                resource_id=resource_id,
                conflict_type=ConflictType.SEMANTIC,
                conflict_level=ConflictLevel.MEDIUM,
                description="Semantic analysis unavailable",
            )

        # Extract relevant code sections
        region_names = [r[0].full_name for r in overlapping_regions[:3]]

        prompt = f"""Analyze the following concurrent code changes for semantic compatibility.

Base code (original):
```{language}
{self._truncate(base_content)}
```

Change A (our changes):
```{language}
{self._truncate(our_content)}
```

Change B (their changes):
```{language}
{self._truncate(their_content)}
```

Both changes modify the following regions: {', '.join(region_names)}

Determine:
1. Are these changes semantically compatible (can be merged without logical conflicts)?
2. If incompatible, what is the nature of the conflict?
3. If compatible, how might they be merged?

Respond in JSON format:
{{"compatible": true/false, "confidence": 0.0-1.0, "reason": "explanation", "merge_suggestion": "if compatible"}}
"""

        try:
            # Call LLM for analysis
            if hasattr(self._llm_client, "chat"):
                response = await self._llm_client.chat.completions.create(
                    model=self._config.semantic_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                )
                result_text = response.choices[0].message.content
            else:
                # Assume anthropic client
                response = await self._llm_client.messages.create(
                    model=self._config.semantic_model,
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}],
                )
                result_text = response.content[0].text

            # Parse response
            import json

            result = json.loads(result_text)

            if result.get("compatible", False) and result.get("confidence", 0) >= self._config.semantic_threshold:
                return ConflictResult(
                    has_conflict=False,
                    conflict_type=ConflictType.NONE,
                    conflict_level=ConflictLevel.LOW,
                    resource_id=resource_id,
                    description=result.get("reason", "Changes are semantically compatible"),
                    suggested_resolution=result.get("merge_suggestion"),
                    metadata={"confidence": result.get("confidence")},
                )
            else:
                return ConflictResult.conflict(
                    resource_id=resource_id,
                    conflict_type=ConflictType.SEMANTIC,
                    conflict_level=ConflictLevel.HIGH,
                    description=result.get("reason", "Semantic conflict detected"),
                    conflicting_regions=[r[0] for r in overlapping_regions],
                    metadata={"confidence": result.get("confidence")},
                )

        except Exception as e:
            # LLM analysis failed, fall back to structural conflict
            return ConflictResult.conflict(
                resource_id=resource_id,
                conflict_type=ConflictType.STRUCTURAL,
                conflict_level=ConflictLevel.MEDIUM,
                description=f"Semantic analysis error: {e}. Overlapping regions detected.",
                conflicting_regions=[r[0] for r in overlapping_regions],
            )

    def _get_language(self, resource_id: str) -> Optional[str]:
        """Determine language from resource ID (typically file path)."""
        for ext in self._config.file_extensions_to_analyze:
            if resource_id.endswith(ext):
                return ext
        return None

    def _truncate(self, content: str, max_lines: int = 100) -> str:
        """Truncate content for LLM context."""
        lines = content.split("\n")
        if len(lines) <= max_lines:
            return content
        return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"

    async def check_batch_conflicts(
        self,
        conflicts_to_check: List[Dict[str, Any]],
    ) -> List[ConflictResult]:
        """
        Check multiple potential conflicts in parallel.

        Args:
            conflicts_to_check: List of dicts with resource_id, read_version, current_version

        Returns:
            List of ConflictResults
        """
        tasks = []
        for check in conflicts_to_check:
            task = self.check_conflict(
                check["resource_id"],
                check["read_version"],
                check["current_version"],
            )
            tasks.append(task)

        return await asyncio.gather(*tasks)


class MergeStrategy(Enum):
    """Strategies for resolving conflicts."""

    OURS = auto()  # Keep our changes
    THEIRS = auto()  # Keep their changes
    MANUAL = auto()  # Require manual resolution
    AUTO_MERGE = auto()  # Attempt automatic merge


@dataclass
class MergeResult:
    """Result of attempting to merge conflicting changes."""

    success: bool
    merged_content: Optional[str]
    strategy_used: MergeStrategy
    conflicts_remaining: List[ConflictResult] = field(default_factory=list)


class ConflictResolver:
    """Attempts to resolve detected conflicts."""

    def __init__(self, detector: ConflictDetector):
        self._detector = detector

    async def resolve(
        self,
        resource_id: str,
        base_content: str,
        our_content: str,
        their_content: str,
        strategy: MergeStrategy = MergeStrategy.AUTO_MERGE,
    ) -> MergeResult:
        """
        Attempt to resolve a conflict.

        Args:
            resource_id: ID of the resource
            base_content: Original content
            our_content: Our changes
            their_content: Their changes
            strategy: Resolution strategy

        Returns:
            MergeResult with outcome
        """
        if strategy == MergeStrategy.OURS:
            return MergeResult(
                success=True,
                merged_content=our_content,
                strategy_used=MergeStrategy.OURS,
            )

        if strategy == MergeStrategy.THEIRS:
            return MergeResult(
                success=True,
                merged_content=their_content,
                strategy_used=MergeStrategy.THEIRS,
            )

        if strategy == MergeStrategy.AUTO_MERGE:
            return await self._auto_merge(
                resource_id, base_content, our_content, their_content
            )

        return MergeResult(
            success=False,
            merged_content=None,
            strategy_used=MergeStrategy.MANUAL,
        )

    async def _auto_merge(
        self,
        resource_id: str,
        base_content: str,
        our_content: str,
        their_content: str,
    ) -> MergeResult:
        """Attempt automatic merge using diff3-style algorithm."""
        # Simple line-based merge
        base_lines = base_content.split("\n")
        our_lines = our_content.split("\n")
        their_lines = their_content.split("\n")

        # Find lines that changed in each version
        our_changes = set()
        their_changes = set()

        for i, line in enumerate(our_lines):
            if i >= len(base_lines) or line != base_lines[i]:
                our_changes.add(i)

        for i, line in enumerate(their_lines):
            if i >= len(base_lines) or line != base_lines[i]:
                their_changes.add(i)

        # Check for overlapping changes
        overlap = our_changes & their_changes
        if overlap:
            return MergeResult(
                success=False,
                merged_content=None,
                strategy_used=MergeStrategy.MANUAL,
            )

        # Merge non-overlapping changes
        max_len = max(len(our_lines), len(their_lines), len(base_lines))
        merged = []

        for i in range(max_len):
            if i in our_changes and i < len(our_lines):
                merged.append(our_lines[i])
            elif i in their_changes and i < len(their_lines):
                merged.append(their_lines[i])
            elif i < len(base_lines):
                merged.append(base_lines[i])

        return MergeResult(
            success=True,
            merged_content="\n".join(merged),
            strategy_used=MergeStrategy.AUTO_MERGE,
        )

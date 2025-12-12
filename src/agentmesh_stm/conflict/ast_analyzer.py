"""
AST Analysis for Code Conflict Detection.

This module provides AST-based analysis for detecting code regions and
understanding the structure of code changes for conflict detection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple


class CodeRegionType(Enum):
    """Types of code regions."""

    FILE = auto()
    CLASS = auto()
    FUNCTION = auto()
    METHOD = auto()
    BLOCK = auto()
    IMPORT = auto()
    VARIABLE = auto()
    CONSTANT = auto()
    DECORATOR = auto()
    COMMENT = auto()


@dataclass
class CodeRegion:
    """Represents a region of code with semantic meaning."""

    type: CodeRegionType
    name: str
    start_line: int
    end_line: int
    start_col: int = 0
    end_col: int = 0
    parent: Optional["CodeRegion"] = None
    children: List["CodeRegion"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_name(self) -> str:
        """Get the fully qualified name including parent regions."""
        if self.parent and self.parent.type != CodeRegionType.FILE:
            return f"{self.parent.full_name}.{self.name}"
        return self.name

    def overlaps_with(self, other: "CodeRegion") -> bool:
        """Check if this region overlaps with another."""
        return not (self.end_line < other.start_line or self.start_line > other.end_line)

    def contains(self, line: int) -> bool:
        """Check if a line number is within this region."""
        return self.start_line <= line <= self.end_line

    def get_overlap_lines(self, other: "CodeRegion") -> Set[int]:
        """Get the set of overlapping line numbers."""
        if not self.overlaps_with(other):
            return set()
        start = max(self.start_line, other.start_line)
        end = min(self.end_line, other.end_line)
        return set(range(start, end + 1))


class ASTAnalyzer:
    """
    Analyzes code to extract structural information for conflict detection.

    Supports multiple languages through tree-sitter parsers.
    """

    def __init__(self):
        self._parsers: Dict[str, Any] = {}
        self._initialized = False

    def _init_parsers(self) -> None:
        """Initialize tree-sitter parsers for supported languages."""
        if self._initialized:
            return

        try:
            import tree_sitter_python
            from tree_sitter import Language, Parser

            # Python parser
            py_parser = Parser(Language(tree_sitter_python.language()))
            self._parsers["python"] = py_parser
            self._parsers[".py"] = py_parser

        except ImportError:
            pass

        try:
            import tree_sitter_javascript
            from tree_sitter import Language, Parser

            # JavaScript parser
            js_parser = Parser(Language(tree_sitter_javascript.language()))
            self._parsers["javascript"] = js_parser
            self._parsers[".js"] = js_parser
            self._parsers[".jsx"] = js_parser

        except ImportError:
            pass

        try:
            import tree_sitter_java
            from tree_sitter import Language, Parser

            # Java parser
            java_parser = Parser(Language(tree_sitter_java.language()))
            self._parsers["java"] = java_parser
            self._parsers[".java"] = java_parser

        except ImportError:
            pass

        self._initialized = True

    def get_parser(self, language: str) -> Optional[Any]:
        """Get parser for a language."""
        self._init_parsers()
        return self._parsers.get(language)

    def analyze(self, code: str, language: str = "python") -> List[CodeRegion]:
        """
        Analyze code and extract code regions.

        Args:
            code: Source code to analyze
            language: Programming language

        Returns:
            List of CodeRegion objects representing code structure
        """
        parser = self.get_parser(language)

        if parser:
            return self._analyze_with_tree_sitter(code, parser, language)
        else:
            # Fallback to regex-based analysis
            return self._analyze_with_regex(code, language)

    def _analyze_with_tree_sitter(
        self, code: str, parser: Any, language: str
    ) -> List[CodeRegion]:
        """Analyze code using tree-sitter parser."""
        tree = parser.parse(bytes(code, "utf-8"))
        root = tree.root_node

        regions = []
        file_region = CodeRegion(
            type=CodeRegionType.FILE,
            name="<file>",
            start_line=1,
            end_line=code.count("\n") + 1,
        )
        regions.append(file_region)

        self._extract_regions_recursive(root, file_region, regions, language)

        return regions

    def _extract_regions_recursive(
        self,
        node: Any,
        parent_region: CodeRegion,
        regions: List[CodeRegion],
        language: str,
    ) -> None:
        """Recursively extract code regions from AST nodes."""
        region = None

        # Python-specific node handling
        if language in ("python", ".py"):
            if node.type == "class_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    region = CodeRegion(
                        type=CodeRegionType.CLASS,
                        name=name_node.text.decode("utf-8"),
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        start_col=node.start_point[1],
                        end_col=node.end_point[1],
                        parent=parent_region,
                    )

            elif node.type == "function_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    region_type = (
                        CodeRegionType.METHOD
                        if parent_region.type == CodeRegionType.CLASS
                        else CodeRegionType.FUNCTION
                    )
                    region = CodeRegion(
                        type=region_type,
                        name=name_node.text.decode("utf-8"),
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        start_col=node.start_point[1],
                        end_col=node.end_point[1],
                        parent=parent_region,
                    )

            elif node.type == "import_statement" or node.type == "import_from_statement":
                region = CodeRegion(
                    type=CodeRegionType.IMPORT,
                    name=node.text.decode("utf-8")[:50],  # Truncate long imports
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    start_col=node.start_point[1],
                    end_col=node.end_point[1],
                    parent=parent_region,
                )

        # JavaScript-specific node handling
        elif language in ("javascript", ".js", ".jsx"):
            if node.type == "class_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    region = CodeRegion(
                        type=CodeRegionType.CLASS,
                        name=name_node.text.decode("utf-8"),
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        parent=parent_region,
                    )

            elif node.type in ("function_declaration", "arrow_function", "method_definition"):
                name_node = node.child_by_field_name("name")
                name = name_node.text.decode("utf-8") if name_node else "<anonymous>"
                region_type = (
                    CodeRegionType.METHOD
                    if parent_region.type == CodeRegionType.CLASS
                    else CodeRegionType.FUNCTION
                )
                region = CodeRegion(
                    type=region_type,
                    name=name,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    parent=parent_region,
                )

        # Java-specific node handling
        elif language in ("java", ".java"):
            if node.type == "class_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    region = CodeRegion(
                        type=CodeRegionType.CLASS,
                        name=name_node.text.decode("utf-8"),
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        parent=parent_region,
                    )

            elif node.type == "method_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    region = CodeRegion(
                        type=CodeRegionType.METHOD,
                        name=name_node.text.decode("utf-8"),
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        parent=parent_region,
                    )

        if region:
            regions.append(region)
            parent_region.children.append(region)
            parent_region = region

        # Recurse into children
        for child in node.children:
            self._extract_regions_recursive(child, parent_region, regions, language)

    def _analyze_with_regex(self, code: str, language: str) -> List[CodeRegion]:
        """Fallback regex-based analysis for unsupported languages."""
        regions = []
        lines = code.split("\n")

        file_region = CodeRegion(
            type=CodeRegionType.FILE,
            name="<file>",
            start_line=1,
            end_line=len(lines),
        )
        regions.append(file_region)

        # Python patterns
        if language in ("python", ".py"):
            class_pattern = re.compile(r"^(\s*)class\s+(\w+)")
            func_pattern = re.compile(r"^(\s*)def\s+(\w+)")

            stack: List[Tuple[int, CodeRegion]] = []

            for i, line in enumerate(lines, 1):
                class_match = class_pattern.match(line)
                func_match = func_pattern.match(line)

                if class_match:
                    indent = len(class_match.group(1))
                    name = class_match.group(2)

                    # Pop regions with >= indent
                    while stack and stack[-1][0] >= indent:
                        stack.pop()

                    parent = stack[-1][1] if stack else file_region
                    region = CodeRegion(
                        type=CodeRegionType.CLASS,
                        name=name,
                        start_line=i,
                        end_line=i,  # Will be updated
                        parent=parent,
                    )
                    regions.append(region)
                    parent.children.append(region)
                    stack.append((indent, region))

                elif func_match:
                    indent = len(func_match.group(1))
                    name = func_match.group(2)

                    while stack and stack[-1][0] >= indent:
                        stack.pop()

                    parent = stack[-1][1] if stack else file_region
                    region_type = (
                        CodeRegionType.METHOD
                        if parent.type == CodeRegionType.CLASS
                        else CodeRegionType.FUNCTION
                    )
                    region = CodeRegion(
                        type=region_type,
                        name=name,
                        start_line=i,
                        end_line=i,
                        parent=parent,
                    )
                    regions.append(region)
                    parent.children.append(region)
                    stack.append((indent, region))

            # Update end lines
            for i, region in enumerate(regions[1:], 1):
                if i < len(regions) - 1:
                    next_region = regions[i + 1]
                    if next_region.parent == region.parent or next_region.start_line > region.start_line:
                        region.end_line = next_region.start_line - 1
                else:
                    region.end_line = len(lines)

        return regions

    def get_modified_regions(
        self,
        old_code: str,
        new_code: str,
        language: str = "python",
    ) -> Tuple[List[CodeRegion], List[int]]:
        """
        Identify which code regions were modified between two versions.

        Args:
            old_code: Original code
            new_code: Modified code
            language: Programming language

        Returns:
            Tuple of (modified_regions, modified_lines)
        """
        old_lines = old_code.split("\n")
        new_lines = new_code.split("\n")

        # Find modified lines using simple diff
        modified_lines = set()

        # Simple line-by-line comparison
        max_lines = max(len(old_lines), len(new_lines))
        for i in range(max_lines):
            old_line = old_lines[i] if i < len(old_lines) else None
            new_line = new_lines[i] if i < len(new_lines) else None
            if old_line != new_line:
                modified_lines.add(i + 1)

        # Analyze new code structure
        regions = self.analyze(new_code, language)

        # Find regions containing modified lines
        modified_regions = []
        for region in regions:
            if region.type == CodeRegionType.FILE:
                continue
            for line in modified_lines:
                if region.contains(line):
                    modified_regions.append(region)
                    break

        return modified_regions, sorted(modified_lines)

    def find_region_at_line(
        self, regions: List[CodeRegion], line: int
    ) -> Optional[CodeRegion]:
        """Find the most specific region containing a line."""
        candidates = [r for r in regions if r.contains(line) and r.type != CodeRegionType.FILE]
        if not candidates:
            return None

        # Return the most specific (smallest) region
        return min(candidates, key=lambda r: r.end_line - r.start_line)

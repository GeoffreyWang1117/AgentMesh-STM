"""
Advanced Tools for LLM Agents.

This module provides additional tools for LLM agents:
- Code analysis tools
- Git operation tools
- Shell command execution
- Project structure analysis
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentmesh_stm.core.transaction import Transaction
from agentmesh_stm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ToolResult:
    """Result from a tool execution."""

    success: bool
    output: Any
    error: Optional[str] = None


class CodeAnalysisTools:
    """Tools for analyzing code."""

    @staticmethod
    async def analyze_dependencies(
        transaction: Transaction,
        file_path: str,
        language: str = "python",
    ) -> ToolResult:
        """
        Analyze dependencies/imports in a file.

        Args:
            transaction: Active transaction
            file_path: Path to the file
            language: Programming language

        Returns:
            List of dependencies/imports
        """
        try:
            content = await transaction.read(file_path)
            if content is None:
                # Try reading directly
                if os.path.exists(file_path):
                    with open(file_path) as f:
                        content = f.read()
                else:
                    return ToolResult(success=False, output=[], error="File not found")

            dependencies = []

            if language == "python":
                # Match import statements
                import_pattern = re.compile(
                    r'^(?:from\s+([\w.]+)\s+)?import\s+([\w., ]+)',
                    re.MULTILINE
                )
                for match in import_pattern.finditer(content):
                    from_module = match.group(1)
                    imports = match.group(2)
                    if from_module:
                        dependencies.append(f"from {from_module}: {imports}")
                    else:
                        dependencies.append(imports)

            elif language in ("javascript", "typescript"):
                # Match import/require statements
                import_pattern = re.compile(
                    r'(?:import\s+.*?\s+from\s+[\'"](.+?)[\'"]|'
                    r'require\s*\(\s*[\'"](.+?)[\'"]\s*\))',
                    re.MULTILINE
                )
                for match in import_pattern.finditer(content):
                    dep = match.group(1) or match.group(2)
                    if dep:
                        dependencies.append(dep)

            return ToolResult(success=True, output=dependencies)

        except Exception as e:
            return ToolResult(success=False, output=[], error=str(e))

    @staticmethod
    async def find_function_definition(
        transaction: Transaction,
        function_name: str,
        search_path: str = ".",
    ) -> ToolResult:
        """
        Find where a function is defined.

        Args:
            transaction: Active transaction
            function_name: Name of the function to find
            search_path: Directory to search in

        Returns:
            List of locations where the function is defined
        """
        try:
            locations = []
            code_extensions = {".py", ".js", ".ts", ".java", ".go"}

            for root, dirs, files in os.walk(search_path):
                # Skip hidden and common non-code directories
                dirs[:] = [
                    d for d in dirs
                    if not d.startswith(".") and d not in {"node_modules", "venv", "__pycache__"}
                ]

                for file in files:
                    ext = os.path.splitext(file)[1]
                    if ext not in code_extensions:
                        continue

                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, "r") as f:
                            content = f.read()

                        # Pattern for function definitions
                        patterns = [
                            rf"def\s+{re.escape(function_name)}\s*\(",  # Python
                            rf"function\s+{re.escape(function_name)}\s*\(",  # JS
                            rf"const\s+{re.escape(function_name)}\s*=",  # JS arrow
                            rf"func\s+{re.escape(function_name)}\s*\(",  # Go
                        ]

                        for pattern in patterns:
                            for i, line in enumerate(content.split("\n"), 1):
                                if re.search(pattern, line):
                                    locations.append({
                                        "file": filepath,
                                        "line": i,
                                        "content": line.strip()[:100],
                                    })

                    except Exception:
                        continue

            return ToolResult(success=True, output=locations)

        except Exception as e:
            return ToolResult(success=False, output=[], error=str(e))

    @staticmethod
    async def find_class_definition(
        transaction: Transaction,
        class_name: str,
        search_path: str = ".",
    ) -> ToolResult:
        """
        Find where a class is defined.

        Args:
            transaction: Active transaction
            class_name: Name of the class to find
            search_path: Directory to search in

        Returns:
            List of locations where the class is defined
        """
        try:
            locations = []
            code_extensions = {".py", ".js", ".ts", ".java"}

            for root, dirs, files in os.walk(search_path):
                dirs[:] = [
                    d for d in dirs
                    if not d.startswith(".") and d not in {"node_modules", "venv", "__pycache__"}
                ]

                for file in files:
                    ext = os.path.splitext(file)[1]
                    if ext not in code_extensions:
                        continue

                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, "r") as f:
                            content = f.read()

                        # Pattern for class definitions
                        patterns = [
                            rf"class\s+{re.escape(class_name)}\s*[:\(]",  # Python/JS
                            rf"class\s+{re.escape(class_name)}\s+",  # Java
                        ]

                        for pattern in patterns:
                            for i, line in enumerate(content.split("\n"), 1):
                                if re.search(pattern, line):
                                    locations.append({
                                        "file": filepath,
                                        "line": i,
                                        "content": line.strip()[:100],
                                    })

                    except Exception:
                        continue

            return ToolResult(success=True, output=locations)

        except Exception as e:
            return ToolResult(success=False, output=[], error=str(e))

    @staticmethod
    async def get_function_callers(
        transaction: Transaction,
        function_name: str,
        search_path: str = ".",
    ) -> ToolResult:
        """
        Find all places where a function is called.

        Args:
            transaction: Active transaction
            function_name: Name of the function
            search_path: Directory to search in

        Returns:
            List of call sites
        """
        try:
            callers = []
            code_extensions = {".py", ".js", ".ts", ".java", ".go"}

            for root, dirs, files in os.walk(search_path):
                dirs[:] = [
                    d for d in dirs
                    if not d.startswith(".") and d not in {"node_modules", "venv", "__pycache__"}
                ]

                for file in files:
                    ext = os.path.splitext(file)[1]
                    if ext not in code_extensions:
                        continue

                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, "r") as f:
                            content = f.read()

                        # Pattern for function calls (not definitions)
                        call_pattern = rf"(?<!def\s)(?<!function\s){re.escape(function_name)}\s*\("

                        for i, line in enumerate(content.split("\n"), 1):
                            if re.search(call_pattern, line):
                                callers.append({
                                    "file": filepath,
                                    "line": i,
                                    "content": line.strip()[:100],
                                })

                    except Exception:
                        continue

            return ToolResult(success=True, output=callers)

        except Exception as e:
            return ToolResult(success=False, output=[], error=str(e))


class GitTools:
    """Tools for Git operations."""

    @staticmethod
    async def get_git_status(repo_path: str = ".") -> ToolResult:
        """
        Get Git status for the repository.

        Args:
            repo_path: Path to the repository

        Returns:
            Git status information
        """
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_path,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                return ToolResult(success=False, output={}, error=result.stderr)

            files = {"staged": [], "modified": [], "untracked": []}

            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                status = line[:2]
                filepath = line[3:]

                if status[0] in "MADRC":
                    files["staged"].append(filepath)
                if status[1] == "M":
                    files["modified"].append(filepath)
                if status == "??":
                    files["untracked"].append(filepath)

            return ToolResult(success=True, output=files)

        except Exception as e:
            return ToolResult(success=False, output={}, error=str(e))

    @staticmethod
    async def get_git_diff(
        repo_path: str = ".",
        staged: bool = False,
        file_path: Optional[str] = None,
    ) -> ToolResult:
        """
        Get Git diff output.

        Args:
            repo_path: Path to the repository
            staged: Show staged changes
            file_path: Specific file to diff

        Returns:
            Diff output
        """
        try:
            cmd = ["git", "diff"]
            if staged:
                cmd.append("--staged")
            if file_path:
                cmd.append(file_path)

            result = subprocess.run(
                cmd,
                cwd=repo_path,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                return ToolResult(success=False, output="", error=result.stderr)

            return ToolResult(success=True, output=result.stdout)

        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    @staticmethod
    async def get_git_log(
        repo_path: str = ".",
        num_commits: int = 10,
        file_path: Optional[str] = None,
    ) -> ToolResult:
        """
        Get Git log.

        Args:
            repo_path: Path to the repository
            num_commits: Number of commits to show
            file_path: Show history for specific file

        Returns:
            Log entries
        """
        try:
            cmd = [
                "git", "log",
                f"-{num_commits}",
                "--pretty=format:%h|%an|%ar|%s",
            ]
            if file_path:
                cmd.extend(["--", file_path])

            result = subprocess.run(
                cmd,
                cwd=repo_path,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                return ToolResult(success=False, output=[], error=result.stderr)

            commits = []
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split("|", 3)
                if len(parts) == 4:
                    commits.append({
                        "hash": parts[0],
                        "author": parts[1],
                        "date": parts[2],
                        "message": parts[3],
                    })

            return ToolResult(success=True, output=commits)

        except Exception as e:
            return ToolResult(success=False, output=[], error=str(e))

    @staticmethod
    async def get_file_blame(
        repo_path: str,
        file_path: str,
        start_line: int = 1,
        end_line: Optional[int] = None,
    ) -> ToolResult:
        """
        Get Git blame for a file.

        Args:
            repo_path: Path to the repository
            file_path: Path to the file
            start_line: Start line number
            end_line: End line number

        Returns:
            Blame information
        """
        try:
            cmd = ["git", "blame", "-L", f"{start_line},"]
            if end_line:
                cmd[-1] = f"{start_line},{end_line}"
            cmd.append(file_path)

            result = subprocess.run(
                cmd,
                cwd=repo_path,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                return ToolResult(success=False, output=[], error=result.stderr)

            return ToolResult(success=True, output=result.stdout)

        except Exception as e:
            return ToolResult(success=False, output=[], error=str(e))


class ProjectTools:
    """Tools for project structure analysis."""

    @staticmethod
    async def get_project_structure(
        root_path: str = ".",
        max_depth: int = 3,
        include_hidden: bool = False,
    ) -> ToolResult:
        """
        Get project directory structure.

        Args:
            root_path: Root directory
            max_depth: Maximum depth to traverse
            include_hidden: Include hidden files/directories

        Returns:
            Project structure tree
        """
        try:
            structure = []

            def traverse(path: Path, depth: int, prefix: str = ""):
                if depth > max_depth:
                    return

                try:
                    entries = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
                except PermissionError:
                    return

                # Filter entries
                entries = [
                    e for e in entries
                    if include_hidden or not e.name.startswith(".")
                ]

                # Skip common non-essential directories
                skip_dirs = {"node_modules", "venv", "__pycache__", ".git", "dist", "build"}
                entries = [
                    e for e in entries
                    if not (e.is_dir() and e.name in skip_dirs)
                ]

                for i, entry in enumerate(entries):
                    is_last = i == len(entries) - 1
                    connector = "└── " if is_last else "├── "
                    extension = "    " if is_last else "│   "

                    if entry.is_dir():
                        structure.append(f"{prefix}{connector}{entry.name}/")
                        traverse(entry, depth + 1, prefix + extension)
                    else:
                        structure.append(f"{prefix}{connector}{entry.name}")

            root = Path(root_path)
            structure.append(f"{root.name}/")
            traverse(root, 1)

            return ToolResult(success=True, output="\n".join(structure))

        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    @staticmethod
    async def find_files_by_pattern(
        root_path: str = ".",
        pattern: str = "*.py",
        exclude_dirs: Optional[List[str]] = None,
    ) -> ToolResult:
        """
        Find files matching a pattern.

        Args:
            root_path: Root directory
            pattern: Glob pattern
            exclude_dirs: Directories to exclude

        Returns:
            List of matching files
        """
        try:
            if exclude_dirs is None:
                exclude_dirs = ["node_modules", "venv", "__pycache__", ".git"]

            root = Path(root_path)
            matches = []

            for path in root.rglob(pattern):
                # Check if in excluded directory
                if any(ex in path.parts for ex in exclude_dirs):
                    continue
                matches.append(str(path))

            return ToolResult(success=True, output=matches)

        except Exception as e:
            return ToolResult(success=False, output=[], error=str(e))

    @staticmethod
    async def get_file_stats(file_path: str) -> ToolResult:
        """
        Get statistics about a file.

        Args:
            file_path: Path to the file

        Returns:
            File statistics
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return ToolResult(success=False, output={}, error="File not found")

            stat = path.stat()

            with open(path, "r") as f:
                content = f.read()

            lines = content.split("\n")

            stats = {
                "path": str(path),
                "size_bytes": stat.st_size,
                "lines": len(lines),
                "non_empty_lines": len([l for l in lines if l.strip()]),
                "characters": len(content),
                "extension": path.suffix,
            }

            return ToolResult(success=True, output=stats)

        except Exception as e:
            return ToolResult(success=False, output={}, error=str(e))


class ShellTools:
    """Tools for shell command execution."""

    ALLOWED_COMMANDS = {
        "ls", "cat", "head", "tail", "wc", "grep", "find",
        "pip", "npm", "yarn", "pytest", "python", "node",
        "make", "cmake", "cargo",
    }

    @staticmethod
    async def run_command(
        command: str,
        working_dir: str = ".",
        timeout: int = 30,
        allowed_commands: Optional[set] = None,
    ) -> ToolResult:
        """
        Run a shell command safely.

        Args:
            command: Command to run
            working_dir: Working directory
            timeout: Timeout in seconds
            allowed_commands: Set of allowed command prefixes

        Returns:
            Command output
        """
        try:
            if allowed_commands is None:
                allowed_commands = ShellTools.ALLOWED_COMMANDS

            # Check if command is allowed
            cmd_parts = command.split()
            if not cmd_parts:
                return ToolResult(success=False, output="", error="Empty command")

            base_cmd = cmd_parts[0]
            if base_cmd not in allowed_commands:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Command '{base_cmd}' not in allowed list",
                )

            # Run command
            result = subprocess.run(
                command,
                shell=True,
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            return ToolResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr if result.returncode != 0 else None,
            )

        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output="", error="Command timed out")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    @staticmethod
    async def run_tests(
        test_path: str = ".",
        test_framework: str = "pytest",
        verbose: bool = False,
    ) -> ToolResult:
        """
        Run tests using specified framework.

        Args:
            test_path: Path to tests
            test_framework: Test framework to use
            verbose: Enable verbose output

        Returns:
            Test results
        """
        try:
            if test_framework == "pytest":
                cmd = ["pytest", test_path]
                if verbose:
                    cmd.append("-v")
            elif test_framework == "jest":
                cmd = ["npx", "jest", test_path]
                if verbose:
                    cmd.append("--verbose")
            elif test_framework == "unittest":
                cmd = ["python", "-m", "unittest", "discover", "-s", test_path]
                if verbose:
                    cmd.append("-v")
            else:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Unknown test framework: {test_framework}",
                )

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout for tests
            )

            return ToolResult(
                success=result.returncode == 0,
                output=result.stdout + result.stderr,
                error=None if result.returncode == 0 else "Tests failed",
            )

        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output="", error="Tests timed out")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


def create_tool_definitions() -> List[Dict[str, Any]]:
    """Create tool definitions for LLM agents."""
    return [
        # Code Analysis Tools
        {
            "name": "analyze_dependencies",
            "description": "Analyze imports/dependencies in a source file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to analyze",
                    },
                    "language": {
                        "type": "string",
                        "enum": ["python", "javascript", "typescript"],
                        "description": "Programming language",
                    },
                },
                "required": ["file_path"],
            },
            "handler": CodeAnalysisTools.analyze_dependencies,
        },
        {
            "name": "find_function_definition",
            "description": "Find where a function is defined in the codebase",
            "parameters": {
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
            "handler": CodeAnalysisTools.find_function_definition,
        },
        {
            "name": "find_class_definition",
            "description": "Find where a class is defined in the codebase",
            "parameters": {
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
            "handler": CodeAnalysisTools.find_class_definition,
        },
        {
            "name": "get_function_callers",
            "description": "Find all places where a function is called",
            "parameters": {
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
            "handler": CodeAnalysisTools.get_function_callers,
        },
        # Git Tools
        {
            "name": "get_git_status",
            "description": "Get current Git status (staged, modified, untracked files)",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_path": {
                        "type": "string",
                        "description": "Path to the repository",
                    },
                },
            },
            "handler": GitTools.get_git_status,
        },
        {
            "name": "get_git_diff",
            "description": "Get Git diff showing changes",
            "parameters": {
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
            "handler": GitTools.get_git_diff,
        },
        {
            "name": "get_git_log",
            "description": "Get recent Git commit history",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_path": {
                        "type": "string",
                        "description": "Path to the repository",
                    },
                    "num_commits": {
                        "type": "integer",
                        "description": "Number of commits to show",
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Show history for specific file",
                    },
                },
            },
            "handler": GitTools.get_git_log,
        },
        # Project Tools
        {
            "name": "get_project_structure",
            "description": "Get project directory structure as a tree",
            "parameters": {
                "type": "object",
                "properties": {
                    "root_path": {
                        "type": "string",
                        "description": "Root directory",
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum depth to traverse",
                    },
                },
            },
            "handler": ProjectTools.get_project_structure,
        },
        {
            "name": "find_files",
            "description": "Find files matching a glob pattern",
            "parameters": {
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
            "handler": ProjectTools.find_files_by_pattern,
        },
        {
            "name": "get_file_stats",
            "description": "Get statistics about a file (size, lines, etc.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file",
                    },
                },
                "required": ["file_path"],
            },
            "handler": ProjectTools.get_file_stats,
        },
        # Shell Tools
        {
            "name": "run_tests",
            "description": "Run tests using a test framework",
            "parameters": {
                "type": "object",
                "properties": {
                    "test_path": {
                        "type": "string",
                        "description": "Path to tests",
                    },
                    "test_framework": {
                        "type": "string",
                        "enum": ["pytest", "jest", "unittest"],
                        "description": "Test framework to use",
                    },
                    "verbose": {
                        "type": "boolean",
                        "description": "Enable verbose output",
                    },
                },
            },
            "handler": ShellTools.run_tests,
        },
    ]

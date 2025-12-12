"""
Dataset Loaders for AgentMesh-STM Evaluation.

This module provides loaders for benchmark datasets:
- SWE-bench: Real GitHub issues and fixes
- CodeContests: Competitive programming problems
- Defects4J: Java project defects
- BigCloneBench: Code clone pairs
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from pydantic import BaseModel, Field


@dataclass
class DatasetSample:
    """A single sample from a dataset."""

    id: str
    description: str
    input_data: Dict[str, Any]
    expected_output: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CodeFixSample(DatasetSample):
    """Sample for code fixing tasks (SWE-bench style)."""

    repo: str = ""
    issue_text: str = ""
    base_commit: str = ""
    patch: str = ""
    test_patch: str = ""
    hints_text: str = ""
    instance_id: str = ""

    @classmethod
    def from_swebench(cls, data: Dict[str, Any]) -> "CodeFixSample":
        """Create from SWE-bench format."""
        return cls(
            id=data.get("instance_id", ""),
            description=data.get("problem_statement", ""),
            input_data={
                "repo": data.get("repo", ""),
                "base_commit": data.get("base_commit", ""),
                "issue": data.get("problem_statement", ""),
            },
            expected_output={
                "patch": data.get("patch", ""),
            },
            repo=data.get("repo", ""),
            issue_text=data.get("problem_statement", ""),
            base_commit=data.get("base_commit", ""),
            patch=data.get("patch", ""),
            test_patch=data.get("test_patch", ""),
            hints_text=data.get("hints_text", ""),
            instance_id=data.get("instance_id", ""),
            metadata={
                "created_at": data.get("created_at", ""),
                "version": data.get("version", ""),
                "fail_to_pass": data.get("FAIL_TO_PASS", []),
                "pass_to_pass": data.get("PASS_TO_PASS", []),
            },
        )


@dataclass
class CodingProblemSample(DatasetSample):
    """Sample for coding problems (CodeContests style)."""

    problem_text: str = ""
    input_format: str = ""
    output_format: str = ""
    examples: List[Dict[str, str]] = field(default_factory=list)
    solutions: List[str] = field(default_factory=list)
    difficulty: str = ""
    tags: List[str] = field(default_factory=list)

    @classmethod
    def from_codecontests(cls, data: Dict[str, Any]) -> "CodingProblemSample":
        """Create from CodeContests format."""
        return cls(
            id=data.get("name", ""),
            description=data.get("description", ""),
            input_data={
                "problem": data.get("description", ""),
                "public_tests": data.get("public_tests", {}),
            },
            expected_output={
                "solutions": data.get("solutions", {}),
            },
            problem_text=data.get("description", ""),
            input_format=data.get("input_spec", ""),
            output_format=data.get("output_spec", ""),
            examples=[
                {"input": i, "output": o}
                for i, o in zip(
                    data.get("public_tests", {}).get("input", []),
                    data.get("public_tests", {}).get("output", []),
                )
            ],
            solutions=data.get("solutions", {}).get("solution", []),
            difficulty=str(data.get("difficulty", "")),
            tags=data.get("cf_tags", []),
            metadata={
                "time_limit": data.get("time_limit", {}),
                "memory_limit": data.get("memory_limit_bytes", 0),
                "source": data.get("source", 0),
            },
        )


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.expanduser("~/.agentmesh/datasets")
        os.makedirs(self.cache_dir, exist_ok=True)

    @abstractmethod
    def load(self, split: str = "test") -> List[DatasetSample]:
        """Load the dataset."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[DatasetSample]:
        """Iterate over samples."""
        pass


class SWEBenchLoader(DatasetLoader):
    """
    Loader for SWE-bench dataset.

    SWE-bench contains real GitHub issues from popular Python repositories
    along with the ground truth patches that fix them.

    Dataset: https://huggingface.co/datasets/princeton-nlp/SWE-bench
    """

    DATASET_NAME = "princeton-nlp/SWE-bench"
    LITE_DATASET_NAME = "princeton-nlp/SWE-bench_Lite"

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        use_lite: bool = True,
    ):
        super().__init__(cache_dir)
        self.use_lite = use_lite
        self._samples: List[CodeFixSample] = []
        self._loaded = False

    def load(self, split: str = "test") -> List[CodeFixSample]:
        """Load the SWE-bench dataset."""
        if self._loaded:
            return self._samples

        try:
            from datasets import load_dataset

            dataset_name = self.LITE_DATASET_NAME if self.use_lite else self.DATASET_NAME
            dataset = load_dataset(dataset_name, split=split, cache_dir=self.cache_dir)

            self._samples = [
                CodeFixSample.from_swebench(sample)
                for sample in dataset
            ]
            self._loaded = True

        except ImportError:
            raise ImportError(
                "Please install the 'datasets' package: pip install datasets"
            )
        except Exception as e:
            # Try loading from local cache
            cache_path = Path(self.cache_dir) / f"swebench_{split}.json"
            if cache_path.exists():
                with open(cache_path) as f:
                    data = json.load(f)
                self._samples = [CodeFixSample.from_swebench(s) for s in data]
                self._loaded = True
            else:
                raise RuntimeError(f"Failed to load SWE-bench: {e}")

        return self._samples

    def __len__(self) -> int:
        if not self._loaded:
            self.load()
        return len(self._samples)

    def __iter__(self) -> Iterator[CodeFixSample]:
        if not self._loaded:
            self.load()
        return iter(self._samples)

    def get_by_repo(self, repo: str) -> List[CodeFixSample]:
        """Get all samples for a specific repository."""
        if not self._loaded:
            self.load()
        return [s for s in self._samples if s.repo == repo]

    def get_repos(self) -> List[str]:
        """Get list of unique repositories."""
        if not self._loaded:
            self.load()
        return list(set(s.repo for s in self._samples))

    def filter_by_difficulty(
        self, max_patch_lines: int = 50
    ) -> List[CodeFixSample]:
        """Filter samples by approximate difficulty."""
        if not self._loaded:
            self.load()
        return [
            s for s in self._samples
            if len(s.patch.split("\n")) <= max_patch_lines
        ]


class CodeContestsLoader(DatasetLoader):
    """
    Loader for CodeContests dataset.

    CodeContests contains competitive programming problems from
    Codeforces, LeetCode, and other platforms.

    Dataset: https://huggingface.co/datasets/deepmind/code_contests
    """

    DATASET_NAME = "deepmind/code_contests"

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(cache_dir)
        self._samples: List[CodingProblemSample] = []
        self._loaded = False

    def load(self, split: str = "test") -> List[CodingProblemSample]:
        """Load the CodeContests dataset."""
        if self._loaded:
            return self._samples

        try:
            from datasets import load_dataset

            dataset = load_dataset(
                self.DATASET_NAME,
                split=split,
                cache_dir=self.cache_dir,
            )

            self._samples = [
                CodingProblemSample.from_codecontests(sample)
                for sample in dataset
            ]
            self._loaded = True

        except ImportError:
            raise ImportError(
                "Please install the 'datasets' package: pip install datasets"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load CodeContests: {e}")

        return self._samples

    def __len__(self) -> int:
        if not self._loaded:
            self.load()
        return len(self._samples)

    def __iter__(self) -> Iterator[CodingProblemSample]:
        if not self._loaded:
            self.load()
        return iter(self._samples)

    def filter_by_difficulty(
        self, min_difficulty: int = 0, max_difficulty: int = 2000
    ) -> List[CodingProblemSample]:
        """Filter by Codeforces-style difficulty rating."""
        if not self._loaded:
            self.load()
        return [
            s for s in self._samples
            if s.difficulty.isdigit()
            and min_difficulty <= int(s.difficulty) <= max_difficulty
        ]

    def filter_by_tags(self, tags: List[str]) -> List[CodingProblemSample]:
        """Filter by problem tags."""
        if not self._loaded:
            self.load()
        tag_set = set(tags)
        return [
            s for s in self._samples
            if tag_set.intersection(s.tags)
        ]


class Defects4JLoader(DatasetLoader):
    """
    Loader for Defects4J dataset.

    Defects4J contains real bugs from Java projects with
    triggering tests and correct fixes.

    Note: Requires Defects4J to be installed locally.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        defects4j_home: Optional[str] = None,
    ):
        super().__init__(cache_dir)
        self.defects4j_home = defects4j_home or os.environ.get("DEFECTS4J_HOME")
        self._samples: List[DatasetSample] = []
        self._loaded = False

    def load(self, split: str = "all") -> List[DatasetSample]:
        """Load Defects4J bug information."""
        if self._loaded:
            return self._samples

        # Try loading from local Defects4J installation
        if self.defects4j_home and os.path.exists(self.defects4j_home):
            self._samples = self._load_from_installation()
        else:
            # Try loading from cached metadata
            cache_path = Path(self.cache_dir) / "defects4j_metadata.json"
            if cache_path.exists():
                with open(cache_path) as f:
                    data = json.load(f)
                self._samples = [
                    DatasetSample(
                        id=f"{d['project']}_{d['bug_id']}",
                        description=d.get("bug_description", ""),
                        input_data=d,
                    )
                    for d in data
                ]
            else:
                # Return empty list if not available
                self._samples = []

        self._loaded = True
        return self._samples

    def _load_from_installation(self) -> List[DatasetSample]:
        """Load from Defects4J installation."""
        import subprocess

        samples = []

        # Get list of projects
        projects_result = subprocess.run(
            ["defects4j", "pids"],
            capture_output=True,
            text=True,
            cwd=self.defects4j_home,
        )

        if projects_result.returncode != 0:
            return []

        projects = projects_result.stdout.strip().split("\n")

        for project in projects:
            # Get bug IDs for each project
            bids_result = subprocess.run(
                ["defects4j", "bids", "-p", project],
                capture_output=True,
                text=True,
                cwd=self.defects4j_home,
            )

            if bids_result.returncode != 0:
                continue

            bug_ids = bids_result.stdout.strip().split("\n")

            for bug_id in bug_ids:
                samples.append(
                    DatasetSample(
                        id=f"{project}_{bug_id}",
                        description=f"Bug {bug_id} in {project}",
                        input_data={
                            "project": project,
                            "bug_id": bug_id,
                        },
                    )
                )

        return samples

    def __len__(self) -> int:
        if not self._loaded:
            self.load()
        return len(self._samples)

    def __iter__(self) -> Iterator[DatasetSample]:
        if not self._loaded:
            self.load()
        return iter(self._samples)


class BigCloneBenchLoader(DatasetLoader):
    """
    Loader for BigCloneBench dataset.

    BigCloneBench contains code clone pairs for evaluating
    semantic similarity detection.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(cache_dir)
        self._samples: List[DatasetSample] = []
        self._loaded = False

    def load(self, split: str = "test") -> List[DatasetSample]:
        """Load BigCloneBench dataset."""
        if self._loaded:
            return self._samples

        # BigCloneBench is typically loaded from local files
        bcb_path = Path(self.cache_dir) / "bigclonebench"
        if bcb_path.exists():
            self._samples = self._load_from_directory(bcb_path)
        else:
            self._samples = []

        self._loaded = True
        return self._samples

    def _load_from_directory(self, path: Path) -> List[DatasetSample]:
        """Load clone pairs from directory."""
        samples = []

        # Look for clone pair files
        for clone_file in path.glob("*.txt"):
            with open(clone_file) as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 2:
                        samples.append(
                            DatasetSample(
                                id=f"clone_{len(samples)}",
                                description="Code clone pair",
                                input_data={
                                    "code1_id": parts[0],
                                    "code2_id": parts[1],
                                    "is_clone": True,
                                },
                            )
                        )

        return samples

    def __len__(self) -> int:
        if not self._loaded:
            self.load()
        return len(self._samples)

    def __iter__(self) -> Iterator[DatasetSample]:
        if not self._loaded:
            self.load()
        return iter(self._samples)


class MultiAgentTaskGenerator:
    """
    Generates multi-agent collaboration tasks from datasets.

    Transforms single-agent tasks into multi-agent scenarios
    where agents must coordinate to solve the problem.
    """

    def __init__(self, num_agents: int = 3):
        self.num_agents = num_agents

    def generate_from_swebench(
        self, sample: CodeFixSample
    ) -> List[Dict[str, Any]]:
        """
        Generate multi-agent tasks from a SWE-bench sample.

        Splits the task into:
        1. Analysis agent: Understand the issue
        2. Implementation agent: Write the fix
        3. Testing agent: Verify the fix
        """
        tasks = [
            {
                "agent": "analyzer",
                "description": f"Analyze issue in {sample.repo}",
                "input": {
                    "issue": sample.issue_text,
                    "repo": sample.repo,
                    "hints": sample.hints_text,
                },
                "output_resource": f"analysis/{sample.id}",
            },
            {
                "agent": "implementer",
                "description": f"Implement fix for {sample.id}",
                "input": {
                    "issue": sample.issue_text,
                    "repo": sample.repo,
                    "base_commit": sample.base_commit,
                },
                "dependencies": [f"analysis/{sample.id}"],
                "output_resource": f"patches/{sample.id}",
            },
            {
                "agent": "tester",
                "description": f"Test fix for {sample.id}",
                "input": {
                    "repo": sample.repo,
                    "test_patch": sample.test_patch,
                },
                "dependencies": [f"patches/{sample.id}"],
                "output_resource": f"results/{sample.id}",
            },
        ]

        return tasks

    def generate_parallel_tasks(
        self,
        samples: List[DatasetSample],
        tasks_per_agent: int = 5,
    ) -> List[List[Dict[str, Any]]]:
        """
        Generate parallel task batches for multiple agents.

        Returns batches of tasks that can be executed concurrently
        while testing conflict handling.
        """
        batches = []

        for i in range(0, len(samples), self.num_agents):
            batch = []
            for j in range(self.num_agents):
                if i + j < len(samples):
                    sample = samples[i + j]
                    batch.append({
                        "agent": f"agent_{j}",
                        "sample_id": sample.id,
                        "description": sample.description,
                        "input": sample.input_data,
                    })
            if batch:
                batches.append(batch)

        return batches

"""
Benchmarking Framework for AgentMesh-STM.

This module provides tools for benchmarking the framework:
- Configurable benchmark scenarios
- Multi-agent concurrency testing
- Comparison with baseline approaches
- Reproducible experiments
"""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from agentmesh_stm.core.transaction import TransactionConfig, TransactionManager
from agentmesh_stm.storage.mvcc import MVCCStorage, InMemoryBackend
from agentmesh_stm.conflict.detector import ConflictDetector, ConflictDetectorConfig
from agentmesh_stm.agent.base import Agent, AgentConfig, AgentTask, SimpleAgent
from agentmesh_stm.evaluation.metrics import MetricsCollector, PerformanceReport


class BenchmarkType(Enum):
    """Types of benchmarks."""

    THROUGHPUT = auto()  # Measure transaction throughput
    LATENCY = auto()  # Measure transaction latency
    CONCURRENCY = auto()  # Test concurrent agent behavior
    CONFLICT = auto()  # Test conflict detection
    SCALABILITY = auto()  # Test scalability with increasing load


class BaselineType(Enum):
    """Baseline approaches for comparison."""

    SERIAL = auto()  # Serial execution (no concurrency)
    GLOBAL_LOCK = auto()  # Single global lock
    FINE_GRAINED_LOCK = auto()  # Per-resource locking
    MESSAGE_QUEUE = auto()  # Async message queue


class BenchmarkConfig(BaseModel):
    """Configuration for a benchmark run."""

    name: str = Field(description="Benchmark name")
    benchmark_type: BenchmarkType = Field(default=BenchmarkType.THROUGHPUT)
    num_agents: int = Field(default=4, description="Number of concurrent agents")
    num_tasks: int = Field(default=100, description="Total tasks to execute")
    num_resources: int = Field(default=10, description="Number of shared resources")
    task_duration_ms: float = Field(default=10.0, description="Simulated task duration")
    conflict_probability: float = Field(default=0.3, description="Probability of conflict")
    warmup_tasks: int = Field(default=10, description="Warmup tasks before measurement")
    timeout_seconds: float = Field(default=300.0, description="Benchmark timeout")
    transaction_config: Optional[TransactionConfig] = None
    baselines: List[BaselineType] = Field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    config: BenchmarkConfig
    start_time: datetime
    end_time: datetime
    performance_report: PerformanceReport
    baseline_results: Dict[str, "BenchmarkResult"] = field(default_factory=dict)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        """Total benchmark duration."""
        return (self.end_time - self.start_time).total_seconds()

    @property
    def throughput(self) -> float:
        """Tasks completed per second."""
        total = self.performance_report.transaction_stats.get("total", 0)
        return total / self.duration_seconds if self.duration_seconds > 0 else 0

    @property
    def success_rate(self) -> float:
        """Task success rate."""
        stats = self.performance_report.transaction_stats
        total = stats.get("total", 0)
        committed = stats.get("committed", 0)
        return committed / total if total > 0 else 0

    def speedup_vs(self, baseline_name: str) -> Optional[float]:
        """Calculate speedup compared to a baseline."""
        if baseline_name not in self.baseline_results:
            return None
        baseline = self.baseline_results[baseline_name]
        if baseline.throughput == 0:
            return None
        return self.throughput / baseline.throughput

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "config": self.config.model_dump(),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": self.duration_seconds,
            "throughput": self.throughput,
            "success_rate": self.success_rate,
            "performance": self.performance_report.to_dict(),
            "baselines": {
                name: result.to_dict()
                for name, result in self.baseline_results.items()
            },
            "errors": self.errors,
        }

    def save(self, path: str) -> None:
        """Save results to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            f"Benchmark: {self.config.name}",
            "=" * 60,
            f"Type: {self.config.benchmark_type.name}",
            f"Agents: {self.config.num_agents}",
            f"Tasks: {self.config.num_tasks}",
            f"Resources: {self.config.num_resources}",
            "",
            f"Duration: {self.duration_seconds:.2f}s",
            f"Throughput: {self.throughput:.2f} tasks/s",
            f"Success Rate: {self.success_rate:.2%}",
            "",
        ]

        if self.baseline_results:
            lines.append("Speedup vs Baselines:")
            for name in self.baseline_results:
                speedup = self.speedup_vs(name)
                if speedup:
                    lines.append(f"  {name}: {speedup:.2f}x")

        if self.errors:
            lines.append(f"\nErrors: {len(self.errors)}")

        lines.append("=" * 60)
        return "\n".join(lines)


class Benchmark(ABC):
    """Abstract base class for benchmarks."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.metrics = MetricsCollector()

    @abstractmethod
    async def setup(self) -> None:
        """Set up benchmark resources."""
        pass

    @abstractmethod
    async def run(self) -> BenchmarkResult:
        """Run the benchmark."""
        pass

    @abstractmethod
    async def teardown(self) -> None:
        """Clean up benchmark resources."""
        pass


class ThroughputBenchmark(Benchmark):
    """Benchmark measuring transaction throughput."""

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.storage: Optional[MVCCStorage] = None
        self.manager: Optional[TransactionManager] = None
        self.agents: List[SimpleAgent] = []

    async def setup(self) -> None:
        """Set up storage and agents."""
        self.storage = MVCCStorage(backend=InMemoryBackend())
        conflict_detector = ConflictDetector(storage=self.storage)
        self.manager = TransactionManager(
            storage=self.storage,
            conflict_detector=conflict_detector,
            default_config=self.config.transaction_config,
        )

        # Initialize resources
        for i in range(self.config.num_resources):
            await self.storage.write(f"resource_{i}", f"initial_content_{i}")

        # Create agents
        for i in range(self.config.num_agents):
            agent = SimpleAgent(
                transaction_manager=self.manager,
                handler=self._create_task_handler(),
                config=AgentConfig(name=f"agent_{i}"),
            )
            self.agents.append(agent)

    def _create_task_handler(self):
        """Create a task handler for the benchmark."""
        import random

        async def handler(task: AgentTask, txn):
            # Simulate work
            resource_id = f"resource_{random.randint(0, self.config.num_resources - 1)}"

            # Read
            content = await txn.read(resource_id)
            await self.metrics.record_transaction_read(txn.id)

            # Simulate processing time
            await asyncio.sleep(self.config.task_duration_ms / 1000)

            # Write with some probability
            if random.random() < self.config.conflict_probability:
                await txn.write(resource_id, f"modified_{task.id}")
                await self.metrics.record_transaction_write(txn.id)

            return {"task_id": task.id, "resource": resource_id}

        return handler

    async def run(self) -> BenchmarkResult:
        """Run the throughput benchmark."""
        start_time = datetime.utcnow()
        errors = []

        # Warmup
        warmup_tasks = [
            AgentTask.create(description=f"warmup_{i}", input_data={})
            for i in range(self.config.warmup_tasks)
        ]

        for i, task in enumerate(warmup_tasks):
            agent = self.agents[i % len(self.agents)]
            await agent.run_task(task)

        await self.metrics.reset()

        # Main benchmark
        tasks = [
            AgentTask.create(description=f"task_{i}", input_data={})
            for i in range(self.config.num_tasks)
        ]

        # Distribute tasks across agents
        async def run_agent_tasks(agent_idx: int):
            agent = self.agents[agent_idx]
            agent_tasks = [
                t for i, t in enumerate(tasks)
                if i % len(self.agents) == agent_idx
            ]
            for task in agent_tasks:
                try:
                    await self.metrics.start_transaction(task.id)
                    result = await agent.run_task(task)
                    await self.metrics.complete_transaction(task.id, result.success)
                except Exception as e:
                    errors.append(str(e))
                    await self.metrics.complete_transaction(task.id, False)

        # Run all agents concurrently
        await asyncio.gather(*[
            run_agent_tasks(i) for i in range(len(self.agents))
        ])

        end_time = datetime.utcnow()
        report = await self.metrics.generate_report()

        return BenchmarkResult(
            config=self.config,
            start_time=start_time,
            end_time=end_time,
            performance_report=report,
            errors=errors,
        )

    async def teardown(self) -> None:
        """Clean up resources."""
        self.agents.clear()
        self.storage = None
        self.manager = None


class ConcurrencyBenchmark(Benchmark):
    """Benchmark testing concurrent agent behavior."""

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.storage: Optional[MVCCStorage] = None
        self.manager: Optional[TransactionManager] = None

    async def setup(self) -> None:
        """Set up storage."""
        self.storage = MVCCStorage(backend=InMemoryBackend())
        conflict_detector = ConflictDetector(storage=self.storage)
        self.manager = TransactionManager(
            storage=self.storage,
            conflict_detector=conflict_detector,
            default_config=self.config.transaction_config,
        )

        # Initialize a shared counter
        await self.storage.write("counter", "0")

    async def run(self) -> BenchmarkResult:
        """Run the concurrency benchmark."""
        start_time = datetime.utcnow()
        errors = []
        successful_increments = []

        async def increment_counter(agent_id: int):
            """Each agent tries to increment the counter."""
            for _ in range(self.config.num_tasks // self.config.num_agents):
                try:
                    async with self.manager.transaction() as txn:
                        await self.metrics.start_transaction(txn.id)

                        value = await txn.read("counter")
                        current = int(value)

                        # Simulate processing
                        await asyncio.sleep(self.config.task_duration_ms / 1000)

                        new_value = current + 1
                        await txn.write("counter", str(new_value))

                        successful_increments.append(agent_id)
                        await self.metrics.complete_transaction(txn.id, True)

                except Exception as e:
                    errors.append(f"Agent {agent_id}: {e}")

        # Run all agents concurrently
        await asyncio.gather(*[
            increment_counter(i) for i in range(self.config.num_agents)
        ])

        # Verify final counter value
        final = await self.storage.read("counter")
        final_value = int(final.content)

        end_time = datetime.utcnow()
        report = await self.metrics.generate_report()

        result = BenchmarkResult(
            config=self.config,
            start_time=start_time,
            end_time=end_time,
            performance_report=report,
            errors=errors,
            raw_data={
                "expected_final": len(successful_increments),
                "actual_final": final_value,
                "consistency_check": final_value == len(successful_increments),
            },
        )

        return result

    async def teardown(self) -> None:
        """Clean up resources."""
        self.storage = None
        self.manager = None


class BenchmarkRunner:
    """Runs benchmarks and collects results."""

    def __init__(self):
        self.results: List[BenchmarkResult] = []

    async def run_benchmark(
        self,
        config: BenchmarkConfig,
        benchmark_class: Optional[Type[Benchmark]] = None,
    ) -> BenchmarkResult:
        """Run a single benchmark."""
        if benchmark_class is None:
            benchmark_class = self._get_benchmark_class(config.benchmark_type)

        benchmark = benchmark_class(config)

        try:
            await benchmark.setup()
            result = await benchmark.run()

            # Run baselines if configured
            for baseline_type in config.baselines:
                baseline_result = await self._run_baseline(config, baseline_type)
                result.baseline_results[baseline_type.name] = baseline_result

            self.results.append(result)
            return result

        finally:
            await benchmark.teardown()

    def _get_benchmark_class(self, benchmark_type: BenchmarkType) -> Type[Benchmark]:
        """Get the appropriate benchmark class."""
        mapping = {
            BenchmarkType.THROUGHPUT: ThroughputBenchmark,
            BenchmarkType.CONCURRENCY: ConcurrencyBenchmark,
        }
        return mapping.get(benchmark_type, ThroughputBenchmark)

    async def _run_baseline(
        self, config: BenchmarkConfig, baseline_type: BaselineType
    ) -> BenchmarkResult:
        """Run a baseline comparison."""
        if baseline_type == BaselineType.SERIAL:
            return await self._run_serial_baseline(config)
        elif baseline_type == BaselineType.GLOBAL_LOCK:
            return await self._run_global_lock_baseline(config)
        else:
            # Default to serial for unsupported baselines
            return await self._run_serial_baseline(config)

    async def _run_serial_baseline(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Run serial execution baseline."""
        # Create a config with 1 agent
        serial_config = BenchmarkConfig(
            name=f"{config.name}_serial",
            benchmark_type=config.benchmark_type,
            num_agents=1,
            num_tasks=config.num_tasks,
            num_resources=config.num_resources,
            task_duration_ms=config.task_duration_ms,
            warmup_tasks=config.warmup_tasks,
        )

        benchmark_class = self._get_benchmark_class(config.benchmark_type)
        benchmark = benchmark_class(serial_config)

        try:
            await benchmark.setup()
            return await benchmark.run()
        finally:
            await benchmark.teardown()

    async def _run_global_lock_baseline(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Run global lock baseline."""
        # For now, implement as serial since global lock effectively serializes
        return await self._run_serial_baseline(config)

    async def run_suite(
        self, configs: List[BenchmarkConfig]
    ) -> List[BenchmarkResult]:
        """Run a suite of benchmarks."""
        results = []
        for config in configs:
            result = await self.run_benchmark(config)
            results.append(result)
        return results

    def generate_comparison_report(self) -> str:
        """Generate a comparison report of all results."""
        if not self.results:
            return "No benchmark results available."

        lines = [
            "=" * 80,
            "Benchmark Comparison Report",
            "=" * 80,
            "",
            f"{'Benchmark':<30} {'Agents':>8} {'Tasks':>8} {'Throughput':>12} {'Success':>10}",
            "-" * 80,
        ]

        for result in self.results:
            lines.append(
                f"{result.config.name:<30} "
                f"{result.config.num_agents:>8} "
                f"{result.config.num_tasks:>8} "
                f"{result.throughput:>10.2f}/s "
                f"{result.success_rate:>9.1%}"
            )

        lines.append("=" * 80)
        return "\n".join(lines)

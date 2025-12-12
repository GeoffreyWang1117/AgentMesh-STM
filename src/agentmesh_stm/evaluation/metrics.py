"""
Metrics Collection and Performance Monitoring for AgentMesh-STM.

This module provides comprehensive metrics collection for:
- Transaction performance (latency, throughput, conflicts)
- Agent execution (success rate, execution time, retries)
- Conflict detection (accuracy, false positives/negatives)
- System resource usage
"""

from __future__ import annotations

import asyncio
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

from pydantic import BaseModel, Field


class MetricType(Enum):
    """Types of metrics."""

    COUNTER = auto()  # Monotonically increasing counter
    GAUGE = auto()  # Current value
    HISTOGRAM = auto()  # Distribution of values
    TIMER = auto()  # Duration measurements


@dataclass
class MetricValue:
    """A single metric measurement."""

    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class TransactionMetrics:
    """Metrics for a single transaction."""

    transaction_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    read_count: int = 0
    write_count: int = 0
    conflict_detected: bool = False
    retry_count: int = 0
    committed: bool = False
    aborted: bool = False
    rollback_duration_ms: float = 0.0

    def complete(self, committed: bool) -> None:
        """Mark transaction as complete."""
        self.end_time = datetime.utcnow()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.committed = committed
        self.aborted = not committed


@dataclass
class AgentMetrics:
    """Metrics for agent execution."""

    agent_id: str
    agent_name: str
    task_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    success: bool = False
    error_message: Optional[str] = None
    retry_count: int = 0
    files_read: int = 0
    files_written: int = 0
    llm_calls: int = 0
    llm_tokens_used: int = 0

    def complete(self, success: bool, error_message: Optional[str] = None) -> None:
        """Mark agent execution as complete."""
        self.end_time = datetime.utcnow()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.success = success
        self.error_message = error_message


@dataclass
class ConflictMetrics:
    """Metrics for conflict detection."""

    resource_id: str
    conflict_type: str
    detection_time_ms: float
    was_true_conflict: Optional[bool] = None  # For accuracy tracking
    resolution_strategy: Optional[str] = None
    resolution_success: bool = False


class MetricsCollector:
    """
    Collects and aggregates metrics for the AgentMesh-STM system.

    Provides:
    - Real-time metric collection
    - Aggregation (sum, avg, percentiles)
    - Export to various formats
    - Alerting on thresholds
    """

    def __init__(self):
        self._transactions: Dict[str, TransactionMetrics] = {}
        self._agent_executions: Dict[str, AgentMetrics] = {}
        self._conflicts: List[ConflictMetrics] = []
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._timers: Dict[str, List[float]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._start_time = datetime.utcnow()
        self._callbacks: List[Callable[[MetricValue], None]] = []

    # Transaction metrics

    async def start_transaction(self, transaction_id: str) -> TransactionMetrics:
        """Start tracking a transaction."""
        async with self._lock:
            metrics = TransactionMetrics(
                transaction_id=transaction_id,
                start_time=datetime.utcnow(),
            )
            self._transactions[transaction_id] = metrics
            self._counters["transactions_started"] += 1
            return metrics

    async def record_transaction_read(self, transaction_id: str) -> None:
        """Record a read operation in a transaction."""
        async with self._lock:
            if transaction_id in self._transactions:
                self._transactions[transaction_id].read_count += 1
                self._counters["total_reads"] += 1

    async def record_transaction_write(self, transaction_id: str) -> None:
        """Record a write operation in a transaction."""
        async with self._lock:
            if transaction_id in self._transactions:
                self._transactions[transaction_id].write_count += 1
                self._counters["total_writes"] += 1

    async def record_transaction_conflict(self, transaction_id: str) -> None:
        """Record a conflict in a transaction."""
        async with self._lock:
            if transaction_id in self._transactions:
                self._transactions[transaction_id].conflict_detected = True
                self._counters["conflicts_detected"] += 1

    async def record_transaction_retry(self, transaction_id: str) -> None:
        """Record a retry in a transaction."""
        async with self._lock:
            if transaction_id in self._transactions:
                self._transactions[transaction_id].retry_count += 1
                self._counters["transaction_retries"] += 1

    async def complete_transaction(
        self, transaction_id: str, committed: bool
    ) -> Optional[TransactionMetrics]:
        """Complete tracking a transaction."""
        async with self._lock:
            if transaction_id not in self._transactions:
                return None

            metrics = self._transactions[transaction_id]
            metrics.complete(committed)

            if committed:
                self._counters["transactions_committed"] += 1
            else:
                self._counters["transactions_aborted"] += 1

            self._histograms["transaction_duration_ms"].append(metrics.duration_ms)
            self._histograms["transaction_reads"].append(metrics.read_count)
            self._histograms["transaction_writes"].append(metrics.write_count)

            return metrics

    # Agent metrics

    async def start_agent_execution(
        self, agent_id: str, agent_name: str, task_id: str
    ) -> AgentMetrics:
        """Start tracking an agent execution."""
        async with self._lock:
            metrics = AgentMetrics(
                agent_id=agent_id,
                agent_name=agent_name,
                task_id=task_id,
                start_time=datetime.utcnow(),
            )
            self._agent_executions[task_id] = metrics
            self._counters["agent_executions_started"] += 1
            return metrics

    async def complete_agent_execution(
        self, task_id: str, success: bool, error_message: Optional[str] = None
    ) -> Optional[AgentMetrics]:
        """Complete tracking an agent execution."""
        async with self._lock:
            if task_id not in self._agent_executions:
                return None

            metrics = self._agent_executions[task_id]
            metrics.complete(success, error_message)

            if success:
                self._counters["agent_executions_succeeded"] += 1
            else:
                self._counters["agent_executions_failed"] += 1

            self._histograms["agent_execution_duration_ms"].append(metrics.duration_ms)

            return metrics

    async def record_llm_call(
        self, task_id: str, tokens_used: int = 0
    ) -> None:
        """Record an LLM API call."""
        async with self._lock:
            if task_id in self._agent_executions:
                self._agent_executions[task_id].llm_calls += 1
                self._agent_executions[task_id].llm_tokens_used += tokens_used
            self._counters["llm_calls"] += 1
            self._counters["llm_tokens_total"] += tokens_used

    # Conflict metrics

    async def record_conflict(
        self,
        resource_id: str,
        conflict_type: str,
        detection_time_ms: float,
        was_true_conflict: Optional[bool] = None,
    ) -> ConflictMetrics:
        """Record a conflict detection."""
        async with self._lock:
            metrics = ConflictMetrics(
                resource_id=resource_id,
                conflict_type=conflict_type,
                detection_time_ms=detection_time_ms,
                was_true_conflict=was_true_conflict,
            )
            self._conflicts.append(metrics)
            self._histograms["conflict_detection_time_ms"].append(detection_time_ms)
            return metrics

    # Generic metrics

    async def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a counter."""
        async with self._lock:
            self._counters[name] += value

    async def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge value."""
        async with self._lock:
            self._gauges[name] = value
            self._notify_callbacks(MetricValue(
                name=name,
                value=value,
                timestamp=datetime.utcnow(),
                metric_type=MetricType.GAUGE,
            ))

    async def record_histogram(self, name: str, value: float) -> None:
        """Record a histogram value."""
        async with self._lock:
            self._histograms[name].append(value)

    async def record_timer(self, name: str, duration_ms: float) -> None:
        """Record a timer value."""
        async with self._lock:
            self._timers[name].append(duration_ms)

    def _notify_callbacks(self, metric: MetricValue) -> None:
        """Notify registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(metric)
            except Exception:
                pass

    def register_callback(self, callback: Callable[[MetricValue], None]) -> None:
        """Register a callback for metric updates."""
        self._callbacks.append(callback)

    # Aggregation and reporting

    async def get_transaction_stats(self) -> Dict[str, Any]:
        """Get aggregated transaction statistics."""
        async with self._lock:
            completed = [
                m for m in self._transactions.values()
                if m.end_time is not None
            ]

            if not completed:
                return {
                    "total": 0,
                    "committed": 0,
                    "aborted": 0,
                    "conflict_rate": 0.0,
                }

            durations = [m.duration_ms for m in completed]
            conflicts = sum(1 for m in completed if m.conflict_detected)
            retries = sum(m.retry_count for m in completed)

            return {
                "total": len(completed),
                "committed": sum(1 for m in completed if m.committed),
                "aborted": sum(1 for m in completed if m.aborted),
                "conflict_rate": conflicts / len(completed) if completed else 0,
                "retry_total": retries,
                "avg_duration_ms": statistics.mean(durations) if durations else 0,
                "p50_duration_ms": statistics.median(durations) if durations else 0,
                "p95_duration_ms": self._percentile(durations, 95) if durations else 0,
                "p99_duration_ms": self._percentile(durations, 99) if durations else 0,
                "avg_reads": statistics.mean([m.read_count for m in completed]),
                "avg_writes": statistics.mean([m.write_count for m in completed]),
            }

    async def get_agent_stats(self) -> Dict[str, Any]:
        """Get aggregated agent statistics."""
        async with self._lock:
            completed = [
                m for m in self._agent_executions.values()
                if m.end_time is not None
            ]

            if not completed:
                return {
                    "total": 0,
                    "success_rate": 0.0,
                }

            durations = [m.duration_ms for m in completed]

            return {
                "total": len(completed),
                "succeeded": sum(1 for m in completed if m.success),
                "failed": sum(1 for m in completed if not m.success),
                "success_rate": sum(1 for m in completed if m.success) / len(completed),
                "avg_duration_ms": statistics.mean(durations) if durations else 0,
                "p50_duration_ms": statistics.median(durations) if durations else 0,
                "p95_duration_ms": self._percentile(durations, 95) if durations else 0,
                "total_llm_calls": sum(m.llm_calls for m in completed),
                "total_llm_tokens": sum(m.llm_tokens_used for m in completed),
            }

    async def get_conflict_stats(self) -> Dict[str, Any]:
        """Get aggregated conflict statistics."""
        async with self._lock:
            if not self._conflicts:
                return {
                    "total": 0,
                    "accuracy": None,
                }

            detection_times = [c.detection_time_ms for c in self._conflicts]
            labeled = [c for c in self._conflicts if c.was_true_conflict is not None]

            true_positives = sum(1 for c in labeled if c.was_true_conflict)
            false_positives = sum(1 for c in labeled if not c.was_true_conflict)

            return {
                "total": len(self._conflicts),
                "avg_detection_time_ms": statistics.mean(detection_times),
                "p95_detection_time_ms": self._percentile(detection_times, 95),
                "true_positives": true_positives,
                "false_positives": false_positives,
                "precision": true_positives / (true_positives + false_positives) if labeled else None,
            }

    async def get_counters(self) -> Dict[str, int]:
        """Get all counter values."""
        async with self._lock:
            return dict(self._counters)

    async def get_gauges(self) -> Dict[str, float]:
        """Get all gauge values."""
        async with self._lock:
            return dict(self._gauges)

    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]

    async def generate_report(self) -> "PerformanceReport":
        """Generate a comprehensive performance report."""
        return PerformanceReport(
            start_time=self._start_time,
            end_time=datetime.utcnow(),
            transaction_stats=await self.get_transaction_stats(),
            agent_stats=await self.get_agent_stats(),
            conflict_stats=await self.get_conflict_stats(),
            counters=await self.get_counters(),
            gauges=await self.get_gauges(),
        )

    async def reset(self) -> None:
        """Reset all metrics."""
        async with self._lock:
            self._transactions.clear()
            self._agent_executions.clear()
            self._conflicts.clear()
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._timers.clear()
            self._start_time = datetime.utcnow()


@dataclass
class PerformanceReport:
    """Comprehensive performance report."""

    start_time: datetime
    end_time: datetime
    transaction_stats: Dict[str, Any]
    agent_stats: Dict[str, Any]
    conflict_stats: Dict[str, Any]
    counters: Dict[str, int]
    gauges: Dict[str, float]

    @property
    def duration_seconds(self) -> float:
        """Total duration of the measurement period."""
        return (self.end_time - self.start_time).total_seconds()

    @property
    def transactions_per_second(self) -> float:
        """Transaction throughput."""
        total = self.transaction_stats.get("total", 0)
        return total / self.duration_seconds if self.duration_seconds > 0 else 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "period": {
                "start": self.start_time.isoformat(),
                "end": self.end_time.isoformat(),
                "duration_seconds": self.duration_seconds,
            },
            "throughput": {
                "transactions_per_second": self.transactions_per_second,
            },
            "transactions": self.transaction_stats,
            "agents": self.agent_stats,
            "conflicts": self.conflict_stats,
            "counters": self.counters,
            "gauges": self.gauges,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=2, default=str)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "AgentMesh-STM Performance Report",
            "=" * 60,
            f"Period: {self.start_time} to {self.end_time}",
            f"Duration: {self.duration_seconds:.2f} seconds",
            "",
            "Transaction Statistics:",
            f"  Total: {self.transaction_stats.get('total', 0)}",
            f"  Committed: {self.transaction_stats.get('committed', 0)}",
            f"  Aborted: {self.transaction_stats.get('aborted', 0)}",
            f"  Conflict Rate: {self.transaction_stats.get('conflict_rate', 0):.2%}",
            f"  Avg Duration: {self.transaction_stats.get('avg_duration_ms', 0):.2f}ms",
            f"  P95 Duration: {self.transaction_stats.get('p95_duration_ms', 0):.2f}ms",
            f"  Throughput: {self.transactions_per_second:.2f} txn/s",
            "",
            "Agent Statistics:",
            f"  Total Executions: {self.agent_stats.get('total', 0)}",
            f"  Success Rate: {self.agent_stats.get('success_rate', 0):.2%}",
            f"  Avg Duration: {self.agent_stats.get('avg_duration_ms', 0):.2f}ms",
            f"  Total LLM Calls: {self.agent_stats.get('total_llm_calls', 0)}",
            "",
            "Conflict Detection:",
            f"  Total Conflicts: {self.conflict_stats.get('total', 0)}",
            f"  Avg Detection Time: {self.conflict_stats.get('avg_detection_time_ms', 0):.2f}ms",
            "=" * 60,
        ]
        return "\n".join(lines)


class MetricsContext:
    """Context manager for timing operations."""

    def __init__(self, collector: MetricsCollector, metric_name: str):
        self._collector = collector
        self._metric_name = metric_name
        self._start_time: Optional[float] = None

    async def __aenter__(self) -> "MetricsContext":
        self._start_time = time.perf_counter()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._start_time is not None:
            duration_ms = (time.perf_counter() - self._start_time) * 1000
            await self._collector.record_timer(self._metric_name, duration_ms)

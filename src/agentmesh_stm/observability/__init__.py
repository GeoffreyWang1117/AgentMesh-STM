"""
Observability Module for AgentMesh-STM.

Provides OpenTelemetry-based distributed tracing and metrics.
"""

from __future__ import annotations

import asyncio
import functools
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from agentmesh_stm.utils.logging import get_logger

logger = get_logger(__name__)

# Type variable for generic decorators
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Tracing
# =============================================================================

@dataclass
class SpanContext:
    """Context for a trace span."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)


@dataclass
class Span:
    """A trace span representing a unit of work."""

    name: str
    context: SpanContext
    start_time: float
    end_time: Optional[float] = None
    status: str = "OK"
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to the span."""
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })

    def set_status(self, status: str, description: Optional[str] = None) -> None:
        """Set span status."""
        self.status = status
        if description:
            self.attributes["status_description"] = description


class Tracer:
    """
    Distributed tracer for AgentMesh-STM.

    Provides span creation and context propagation for distributed tracing.
    Compatible with OpenTelemetry exporters.
    """

    def __init__(self, service_name: str = "agentmesh-stm"):
        self.service_name = service_name
        self._spans: List[Span] = []
        self._active_span: Optional[Span] = None
        self._exporters: List[SpanExporter] = []
        self._lock = asyncio.Lock()
        self._span_counter = 0

    def _generate_id(self) -> str:
        """Generate a unique ID."""
        import secrets
        return secrets.token_hex(16)

    @contextmanager
    def start_span(
        self,
        name: str,
        parent: Optional[SpanContext] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """Start a new span (sync context manager)."""
        span = self._create_span(name, parent, attributes)
        previous_span = self._active_span
        self._active_span = span

        try:
            yield span
            span.set_status("OK")
        except Exception as e:
            span.set_status("ERROR", str(e))
            span.set_attribute("exception.type", type(e).__name__)
            span.set_attribute("exception.message", str(e))
            raise
        finally:
            span.end_time = time.time()
            self._active_span = previous_span
            self._spans.append(span)
            self._export_span(span)

    @asynccontextmanager
    async def start_span_async(
        self,
        name: str,
        parent: Optional[SpanContext] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """Start a new span (async context manager)."""
        span = self._create_span(name, parent, attributes)
        previous_span = self._active_span
        self._active_span = span

        try:
            yield span
            span.set_status("OK")
        except Exception as e:
            span.set_status("ERROR", str(e))
            span.set_attribute("exception.type", type(e).__name__)
            span.set_attribute("exception.message", str(e))
            raise
        finally:
            span.end_time = time.time()
            self._active_span = previous_span
            async with self._lock:
                self._spans.append(span)
            await self._export_span_async(span)

    def _create_span(
        self,
        name: str,
        parent: Optional[SpanContext] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Span:
        """Create a new span."""
        self._span_counter += 1

        if parent:
            trace_id = parent.trace_id
            parent_span_id = parent.span_id
        elif self._active_span:
            trace_id = self._active_span.context.trace_id
            parent_span_id = self._active_span.context.span_id
        else:
            trace_id = self._generate_id()
            parent_span_id = None

        context = SpanContext(
            trace_id=trace_id,
            span_id=self._generate_id(),
            parent_span_id=parent_span_id,
        )

        span = Span(
            name=name,
            context=context,
            start_time=time.time(),
            attributes={
                "service.name": self.service_name,
                **(attributes or {}),
            },
        )

        return span

    def get_active_span(self) -> Optional[Span]:
        """Get the currently active span."""
        return self._active_span

    def add_exporter(self, exporter: "SpanExporter") -> None:
        """Add a span exporter."""
        self._exporters.append(exporter)

    def _export_span(self, span: Span) -> None:
        """Export span to all exporters (sync)."""
        for exporter in self._exporters:
            try:
                exporter.export([span])
            except Exception as e:
                logger.error(f"Failed to export span: {e}")

    async def _export_span_async(self, span: Span) -> None:
        """Export span to all exporters (async)."""
        for exporter in self._exporters:
            try:
                if hasattr(exporter, "export_async"):
                    await exporter.export_async([span])
                else:
                    exporter.export([span])
            except Exception as e:
                logger.error(f"Failed to export span: {e}")


class SpanExporter:
    """Base class for span exporters."""

    def export(self, spans: List[Span]) -> None:
        """Export spans."""
        raise NotImplementedError

    async def export_async(self, spans: List[Span]) -> None:
        """Export spans asynchronously."""
        self.export(spans)


class ConsoleSpanExporter(SpanExporter):
    """Exports spans to console for debugging."""

    def export(self, spans: List[Span]) -> None:
        for span in spans:
            logger.info(
                f"TRACE: {span.name}",
                trace_id=span.context.trace_id[:8],
                span_id=span.context.span_id[:8],
                duration_ms=f"{span.duration_ms:.2f}",
                status=span.status,
            )


class InMemorySpanExporter(SpanExporter):
    """Stores spans in memory for testing."""

    def __init__(self, max_spans: int = 10000):
        self.spans: List[Span] = []
        self.max_spans = max_spans

    def export(self, spans: List[Span]) -> None:
        self.spans.extend(spans)
        if len(self.spans) > self.max_spans:
            self.spans = self.spans[-self.max_spans:]

    def clear(self) -> None:
        self.spans = []

    def get_spans_by_trace(self, trace_id: str) -> List[Span]:
        """Get all spans for a trace."""
        return [s for s in self.spans if s.context.trace_id == trace_id]


# =============================================================================
# Metrics
# =============================================================================

class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


@dataclass
class MetricPoint:
    """A single metric data point."""

    name: str
    value: float
    timestamp: float
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    description: str = ""


class Counter:
    """A monotonically increasing counter."""

    def __init__(self, name: str, description: str = "", labels: Optional[Dict[str, str]] = None):
        self.name = name
        self.description = description
        self.labels = labels or {}
        self._value = 0.0
        self._lock = asyncio.Lock()

    def inc(self, value: float = 1.0) -> None:
        """Increment the counter."""
        self._value += value

    async def inc_async(self, value: float = 1.0) -> None:
        """Increment the counter (async)."""
        async with self._lock:
            self._value += value

    @property
    def value(self) -> float:
        return self._value

    def to_point(self) -> MetricPoint:
        """Convert to a metric point."""
        return MetricPoint(
            name=self.name,
            value=self._value,
            timestamp=time.time(),
            metric_type=MetricType.COUNTER,
            labels=self.labels,
            description=self.description,
        )


class Gauge:
    """A gauge that can go up or down."""

    def __init__(self, name: str, description: str = "", labels: Optional[Dict[str, str]] = None):
        self.name = name
        self.description = description
        self.labels = labels or {}
        self._value = 0.0
        self._lock = asyncio.Lock()

    def set(self, value: float) -> None:
        """Set the gauge value."""
        self._value = value

    def inc(self, value: float = 1.0) -> None:
        """Increment the gauge."""
        self._value += value

    def dec(self, value: float = 1.0) -> None:
        """Decrement the gauge."""
        self._value -= value

    async def set_async(self, value: float) -> None:
        """Set value (async)."""
        async with self._lock:
            self._value = value

    @property
    def value(self) -> float:
        return self._value

    def to_point(self) -> MetricPoint:
        return MetricPoint(
            name=self.name,
            value=self._value,
            timestamp=time.time(),
            metric_type=MetricType.GAUGE,
            labels=self.labels,
            description=self.description,
        )


class Histogram:
    """A histogram for measuring distributions."""

    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, float("inf"))

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None,
        buckets: Optional[tuple] = None,
    ):
        self.name = name
        self.description = description
        self.labels = labels or {}
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self._values: List[float] = []
        self._sum = 0.0
        self._count = 0
        self._lock = asyncio.Lock()

    def observe(self, value: float) -> None:
        """Record an observation."""
        self._values.append(value)
        self._sum += value
        self._count += 1

    async def observe_async(self, value: float) -> None:
        """Record observation (async)."""
        async with self._lock:
            self._values.append(value)
            self._sum += value
            self._count += 1

    @contextmanager
    def time(self):
        """Context manager to time a block of code."""
        start = time.time()
        try:
            yield
        finally:
            self.observe(time.time() - start)

    @asynccontextmanager
    async def time_async(self):
        """Async context manager to time a block of code."""
        start = time.time()
        try:
            yield
        finally:
            await self.observe_async(time.time() - start)

    @property
    def count(self) -> int:
        return self._count

    @property
    def sum(self) -> float:
        return self._sum

    def get_percentile(self, percentile: float) -> float:
        """Get a percentile value."""
        if not self._values:
            return 0.0
        sorted_values = sorted(self._values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]

    def get_bucket_counts(self) -> Dict[float, int]:
        """Get counts for each bucket."""
        counts = {b: 0 for b in self.buckets}
        for value in self._values:
            for bucket in self.buckets:
                if value <= bucket:
                    counts[bucket] += 1
                    break
        return counts


class MetricsRegistry:
    """
    Registry for all metrics.

    Provides a central place to define and collect metrics.
    """

    def __init__(self, prefix: str = "agentmesh"):
        self.prefix = prefix
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._exporters: List["MetricExporter"] = []

    def counter(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None,
    ) -> Counter:
        """Get or create a counter."""
        full_name = f"{self.prefix}_{name}"
        if full_name not in self._counters:
            self._counters[full_name] = Counter(full_name, description, labels)
        return self._counters[full_name]

    def gauge(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None,
    ) -> Gauge:
        """Get or create a gauge."""
        full_name = f"{self.prefix}_{name}"
        if full_name not in self._gauges:
            self._gauges[full_name] = Gauge(full_name, description, labels)
        return self._gauges[full_name]

    def histogram(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None,
        buckets: Optional[tuple] = None,
    ) -> Histogram:
        """Get or create a histogram."""
        full_name = f"{self.prefix}_{name}"
        if full_name not in self._histograms:
            self._histograms[full_name] = Histogram(full_name, description, labels, buckets)
        return self._histograms[full_name]

    def collect(self) -> List[MetricPoint]:
        """Collect all metric points."""
        points = []
        for counter in self._counters.values():
            points.append(counter.to_point())
        for gauge in self._gauges.values():
            points.append(gauge.to_point())
        return points

    def add_exporter(self, exporter: "MetricExporter") -> None:
        """Add a metric exporter."""
        self._exporters.append(exporter)

    async def export(self) -> None:
        """Export all metrics."""
        points = self.collect()
        for exporter in self._exporters:
            try:
                await exporter.export(points)
            except Exception as e:
                logger.error(f"Failed to export metrics: {e}")

    def get_prometheus_output(self) -> str:
        """Get metrics in Prometheus format."""
        lines = []

        for counter in self._counters.values():
            if counter.description:
                lines.append(f"# HELP {counter.name} {counter.description}")
            lines.append(f"# TYPE {counter.name} counter")
            labels_str = self._format_labels(counter.labels)
            lines.append(f"{counter.name}{labels_str} {counter.value}")

        for gauge in self._gauges.values():
            if gauge.description:
                lines.append(f"# HELP {gauge.name} {gauge.description}")
            lines.append(f"# TYPE {gauge.name} gauge")
            labels_str = self._format_labels(gauge.labels)
            lines.append(f"{gauge.name}{labels_str} {gauge.value}")

        for hist in self._histograms.values():
            if hist.description:
                lines.append(f"# HELP {hist.name} {hist.description}")
            lines.append(f"# TYPE {hist.name} histogram")
            labels_str = self._format_labels(hist.labels)
            bucket_counts = hist.get_bucket_counts()
            cumulative = 0
            for bucket, count in sorted(bucket_counts.items()):
                cumulative += count
                le = "+Inf" if bucket == float("inf") else str(bucket)
                lines.append(f'{hist.name}_bucket{{le="{le}"{labels_str}}} {cumulative}')
            lines.append(f"{hist.name}_sum{labels_str} {hist.sum}")
            lines.append(f"{hist.name}_count{labels_str} {hist.count}")

        return "\n".join(lines)

    @staticmethod
    def _format_labels(labels: Dict[str, str]) -> str:
        if not labels:
            return ""
        pairs = [f'{k}="{v}"' for k, v in labels.items()]
        return "{" + ",".join(pairs) + "}"


class MetricExporter:
    """Base class for metric exporters."""

    async def export(self, points: List[MetricPoint]) -> None:
        raise NotImplementedError


class ConsoleMetricExporter(MetricExporter):
    """Exports metrics to console."""

    async def export(self, points: List[MetricPoint]) -> None:
        for point in points:
            logger.info(
                f"METRIC: {point.name}={point.value}",
                type=point.metric_type.value,
                labels=point.labels,
            )


# =============================================================================
# Instrumentation Decorators
# =============================================================================

def traced(tracer: Tracer, name: Optional[str] = None):
    """Decorator to trace a function."""

    def decorator(func: F) -> F:
        span_name = name or func.__name__

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                async with tracer.start_span_async(span_name):
                    return await func(*args, **kwargs)
            return async_wrapper  # type: ignore
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with tracer.start_span(span_name):
                    return func(*args, **kwargs)
            return sync_wrapper  # type: ignore

    return decorator


def timed(histogram: Histogram):
    """Decorator to time a function."""

    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                async with histogram.time_async():
                    return await func(*args, **kwargs)
            return async_wrapper  # type: ignore
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with histogram.time():
                    return func(*args, **kwargs)
            return sync_wrapper  # type: ignore

    return decorator


def counted(counter: Counter):
    """Decorator to count function calls."""

    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                await counter.inc_async()
                return await func(*args, **kwargs)
            return async_wrapper  # type: ignore
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                counter.inc()
                return func(*args, **kwargs)
            return sync_wrapper  # type: ignore

    return decorator


# =============================================================================
# AgentMesh-STM Specific Metrics
# =============================================================================

class AgentMeshMetrics:
    """Pre-defined metrics for AgentMesh-STM."""

    def __init__(self, registry: Optional[MetricsRegistry] = None):
        self.registry = registry or MetricsRegistry()

        # Transaction metrics
        self.txn_started = self.registry.counter(
            "transactions_started_total",
            "Total number of transactions started"
        )
        self.txn_committed = self.registry.counter(
            "transactions_committed_total",
            "Total number of transactions committed"
        )
        self.txn_aborted = self.registry.counter(
            "transactions_aborted_total",
            "Total number of transactions aborted"
        )
        self.txn_duration = self.registry.histogram(
            "transaction_duration_seconds",
            "Transaction duration in seconds"
        )
        self.active_transactions = self.registry.gauge(
            "active_transactions",
            "Number of currently active transactions"
        )

        # Conflict metrics
        self.conflicts_detected = self.registry.counter(
            "conflicts_detected_total",
            "Total number of conflicts detected"
        )
        self.conflicts_resolved = self.registry.counter(
            "conflicts_resolved_total",
            "Total number of conflicts resolved"
        )
        self.conflict_resolution_duration = self.registry.histogram(
            "conflict_resolution_duration_seconds",
            "Time to resolve conflicts"
        )

        # Storage metrics
        self.storage_reads = self.registry.counter(
            "storage_reads_total",
            "Total number of storage read operations"
        )
        self.storage_writes = self.registry.counter(
            "storage_writes_total",
            "Total number of storage write operations"
        )
        self.storage_read_duration = self.registry.histogram(
            "storage_read_duration_seconds",
            "Storage read latency"
        )
        self.storage_write_duration = self.registry.histogram(
            "storage_write_duration_seconds",
            "Storage write latency"
        )

        # Agent metrics
        self.agent_tasks_started = self.registry.counter(
            "agent_tasks_started_total",
            "Total number of agent tasks started"
        )
        self.agent_tasks_completed = self.registry.counter(
            "agent_tasks_completed_total",
            "Total number of agent tasks completed"
        )
        self.agent_tasks_failed = self.registry.counter(
            "agent_tasks_failed_total",
            "Total number of agent tasks failed"
        )
        self.agent_task_duration = self.registry.histogram(
            "agent_task_duration_seconds",
            "Agent task execution duration"
        )

        # API metrics
        self.api_requests = self.registry.counter(
            "api_requests_total",
            "Total API requests"
        )
        self.api_request_duration = self.registry.histogram(
            "api_request_duration_seconds",
            "API request latency"
        )
        self.api_errors = self.registry.counter(
            "api_errors_total",
            "Total API errors"
        )


# =============================================================================
# Global instances
# =============================================================================

_tracer: Optional[Tracer] = None
_metrics_registry: Optional[MetricsRegistry] = None
_agentmesh_metrics: Optional[AgentMeshMetrics] = None


def get_tracer() -> Tracer:
    """Get the global tracer."""
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer


def get_metrics_registry() -> MetricsRegistry:
    """Get the global metrics registry."""
    global _metrics_registry
    if _metrics_registry is None:
        _metrics_registry = MetricsRegistry()
    return _metrics_registry


def get_agentmesh_metrics() -> AgentMeshMetrics:
    """Get AgentMesh-specific metrics."""
    global _agentmesh_metrics
    if _agentmesh_metrics is None:
        _agentmesh_metrics = AgentMeshMetrics(get_metrics_registry())
    return _agentmesh_metrics


def setup_observability(
    service_name: str = "agentmesh-stm",
    enable_console_export: bool = False,
) -> tuple[Tracer, MetricsRegistry]:
    """Set up observability with common defaults."""
    global _tracer, _metrics_registry, _agentmesh_metrics

    _tracer = Tracer(service_name)
    _metrics_registry = MetricsRegistry()
    _agentmesh_metrics = AgentMeshMetrics(_metrics_registry)

    if enable_console_export:
        _tracer.add_exporter(ConsoleSpanExporter())
        _metrics_registry.add_exporter(ConsoleMetricExporter())

    return _tracer, _metrics_registry

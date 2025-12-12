"""
Performance Monitoring Dashboard for AgentMesh-STM.

Provides real-time monitoring of:
- Transaction throughput and latency
- Conflict rates and resolution statistics
- Agent performance metrics
- System resource usage
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Deque

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from agentmesh_stm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TransactionStats:
    """Statistics for a single transaction."""

    transaction_id: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "active"  # active, committed, aborted
    retries: int = 0
    read_count: int = 0
    write_count: int = 0


@dataclass
class MetricSnapshot:
    """A point-in-time snapshot of metrics."""

    timestamp: float
    transactions_total: int
    transactions_committed: int
    transactions_aborted: int
    conflicts_total: int
    conflicts_resolved: int
    avg_latency_ms: float
    throughput_tps: float
    active_agents: int


@dataclass
class SystemStats:
    """System-level statistics."""

    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    active_transactions: int = 0
    pending_writes: int = 0


class MetricsCollector:
    """
    Collects and aggregates performance metrics.
    """

    def __init__(self, window_size: int = 100, history_size: int = 3600):
        self.window_size = window_size
        self.history_size = history_size

        # Transaction tracking
        self._transactions: Dict[str, TransactionStats] = {}
        self._completed_transactions: Deque[TransactionStats] = deque(maxlen=window_size)

        # Metric history
        self._snapshots: Deque[MetricSnapshot] = deque(maxlen=history_size)

        # Counters
        self._total_transactions = 0
        self._total_commits = 0
        self._total_aborts = 0
        self._total_conflicts = 0
        self._total_resolutions = 0

        # Timing
        self._start_time = time.time()
        self._latencies: Deque[float] = deque(maxlen=window_size)

        # Agent tracking
        self._active_agents: Dict[str, float] = {}

    def transaction_started(self, transaction_id: str) -> None:
        """Record transaction start."""
        self._transactions[transaction_id] = TransactionStats(
            transaction_id=transaction_id,
            start_time=time.time(),
        )
        self._total_transactions += 1

    def transaction_committed(self, transaction_id: str) -> None:
        """Record transaction commit."""
        if transaction_id in self._transactions:
            stats = self._transactions.pop(transaction_id)
            stats.end_time = time.time()
            stats.status = "committed"
            self._completed_transactions.append(stats)
            self._total_commits += 1
            self._latencies.append((stats.end_time - stats.start_time) * 1000)

    def transaction_aborted(self, transaction_id: str) -> None:
        """Record transaction abort."""
        if transaction_id in self._transactions:
            stats = self._transactions.pop(transaction_id)
            stats.end_time = time.time()
            stats.status = "aborted"
            self._completed_transactions.append(stats)
            self._total_aborts += 1
            self._latencies.append((stats.end_time - stats.start_time) * 1000)

    def transaction_retry(self, transaction_id: str) -> None:
        """Record transaction retry."""
        if transaction_id in self._transactions:
            self._transactions[transaction_id].retries += 1

    def conflict_detected(self) -> None:
        """Record conflict detection."""
        self._total_conflicts += 1

    def conflict_resolved(self) -> None:
        """Record conflict resolution."""
        self._total_resolutions += 1

    def read_operation(self, transaction_id: str) -> None:
        """Record read operation."""
        if transaction_id in self._transactions:
            self._transactions[transaction_id].read_count += 1

    def write_operation(self, transaction_id: str) -> None:
        """Record write operation."""
        if transaction_id in self._transactions:
            self._transactions[transaction_id].write_count += 1

    def agent_active(self, agent_id: str) -> None:
        """Record agent activity."""
        self._active_agents[agent_id] = time.time()

    def agent_inactive(self, agent_id: str) -> None:
        """Record agent going inactive."""
        self._active_agents.pop(agent_id, None)

    def take_snapshot(self) -> MetricSnapshot:
        """Take a snapshot of current metrics."""
        now = time.time()
        elapsed = now - self._start_time

        # Calculate throughput
        throughput = self._total_commits / elapsed if elapsed > 0 else 0

        # Calculate average latency
        avg_latency = sum(self._latencies) / len(self._latencies) if self._latencies else 0

        # Clean up stale agents (no activity in 60 seconds)
        stale_threshold = now - 60
        self._active_agents = {
            k: v for k, v in self._active_agents.items() if v > stale_threshold
        }

        snapshot = MetricSnapshot(
            timestamp=now,
            transactions_total=self._total_transactions,
            transactions_committed=self._total_commits,
            transactions_aborted=self._total_aborts,
            conflicts_total=self._total_conflicts,
            conflicts_resolved=self._total_resolutions,
            avg_latency_ms=avg_latency,
            throughput_tps=throughput,
            active_agents=len(self._active_agents),
        )

        self._snapshots.append(snapshot)
        return snapshot

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of metrics."""
        snapshot = self.take_snapshot()
        return {
            "uptime_seconds": time.time() - self._start_time,
            "transactions": {
                "total": snapshot.transactions_total,
                "committed": snapshot.transactions_committed,
                "aborted": snapshot.transactions_aborted,
                "active": len(self._transactions),
            },
            "conflicts": {
                "total": snapshot.conflicts_total,
                "resolved": snapshot.conflicts_resolved,
                "resolution_rate": (
                    snapshot.conflicts_resolved / snapshot.conflicts_total
                    if snapshot.conflicts_total > 0
                    else 1.0
                ),
            },
            "performance": {
                "throughput_tps": snapshot.throughput_tps,
                "avg_latency_ms": snapshot.avg_latency_ms,
            },
            "agents": {
                "active": snapshot.active_agents,
            },
        }

    def get_recent_transactions(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent transaction history."""
        recent = list(self._completed_transactions)[-count:]
        return [
            {
                "id": t.transaction_id[:8],
                "status": t.status,
                "latency_ms": (t.end_time - t.start_time) * 1000 if t.end_time else 0,
                "retries": t.retries,
                "reads": t.read_count,
                "writes": t.write_count,
            }
            for t in reversed(recent)
        ]


class Dashboard:
    """
    Real-time performance monitoring dashboard.
    """

    def __init__(self, collector: MetricsCollector, refresh_rate: float = 1.0):
        if not RICH_AVAILABLE:
            raise ImportError("Rich library required for dashboard. Install with: pip install rich")

        self.collector = collector
        self.refresh_rate = refresh_rate
        self.console = Console()
        self._running = False

    def _make_layout(self) -> Layout:
        """Create the dashboard layout."""
        layout = Layout()

        layout.split(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3),
        )

        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right"),
        )

        layout["left"].split(
            Layout(name="transactions"),
            Layout(name="performance"),
        )

        layout["right"].split(
            Layout(name="conflicts"),
            Layout(name="history"),
        )

        return layout

    def _render_header(self) -> Panel:
        """Render the header panel."""
        summary = self.collector.get_summary()
        uptime = timedelta(seconds=int(summary["uptime_seconds"]))

        text = Text()
        text.append("AgentMesh-STM Dashboard", style="bold blue")
        text.append(f"  |  Uptime: {uptime}", style="dim")
        text.append(f"  |  Agents: {summary['agents']['active']}", style="green")

        return Panel(text, style="white on blue")

    def _render_transactions(self) -> Panel:
        """Render the transactions panel."""
        summary = self.collector.get_summary()
        txn = summary["transactions"]

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row("Total", str(txn["total"]))
        table.add_row("Committed", f"[green]{txn['committed']}[/green]")
        table.add_row("Aborted", f"[red]{txn['aborted']}[/red]")
        table.add_row("Active", f"[yellow]{txn['active']}[/yellow]")

        if txn["total"] > 0:
            success_rate = txn["committed"] / txn["total"] * 100
            table.add_row("Success Rate", f"{success_rate:.1f}%")

        return Panel(table, title="Transactions", border_style="green")

    def _render_performance(self) -> Panel:
        """Render the performance panel."""
        summary = self.collector.get_summary()
        perf = summary["performance"]

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row("Throughput", f"{perf['throughput_tps']:.2f} TPS")
        table.add_row("Avg Latency", f"{perf['avg_latency_ms']:.1f} ms")

        # Add sparkline-like visualization
        recent = list(self.collector._latencies)[-20:]
        if recent:
            max_lat = max(recent) if recent else 1
            sparkline = "".join(
                "▁▂▃▄▅▆▇█"[min(int(l / max_lat * 7), 7)]
                for l in recent
            )
            table.add_row("Latency Trend", f"[dim]{sparkline}[/dim]")

        return Panel(table, title="Performance", border_style="blue")

    def _render_conflicts(self) -> Panel:
        """Render the conflicts panel."""
        summary = self.collector.get_summary()
        conf = summary["conflicts"]

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row("Total Conflicts", str(conf["total"]))
        table.add_row("Resolved", f"[green]{conf['resolved']}[/green]")
        table.add_row("Resolution Rate", f"{conf['resolution_rate'] * 100:.1f}%")

        return Panel(table, title="Conflicts", border_style="yellow")

    def _render_history(self) -> Panel:
        """Render the transaction history panel."""
        recent = self.collector.get_recent_transactions(8)

        table = Table(show_header=True, box=None, padding=(0, 1))
        table.add_column("ID", style="dim")
        table.add_column("Status")
        table.add_column("Latency", justify="right")
        table.add_column("R/W", justify="right")

        for txn in recent:
            status_style = "green" if txn["status"] == "committed" else "red"
            table.add_row(
                txn["id"],
                f"[{status_style}]{txn['status']}[/{status_style}]",
                f"{txn['latency_ms']:.0f}ms",
                f"{txn['reads']}/{txn['writes']}",
            )

        return Panel(table, title="Recent Transactions", border_style="magenta")

    def _render_footer(self) -> Panel:
        """Render the footer panel."""
        text = Text()
        text.append("Press ", style="dim")
        text.append("Ctrl+C", style="bold")
        text.append(" to exit", style="dim")
        text.append("  |  ", style="dim")
        text.append(f"Refresh: {self.refresh_rate}s", style="dim")

        return Panel(text, style="dim")

    def _update_layout(self, layout: Layout) -> None:
        """Update all layout panels."""
        layout["header"].update(self._render_header())
        layout["transactions"].update(self._render_transactions())
        layout["performance"].update(self._render_performance())
        layout["conflicts"].update(self._render_conflicts())
        layout["history"].update(self._render_history())
        layout["footer"].update(self._render_footer())

    async def run(self) -> None:
        """Run the dashboard."""
        self._running = True
        layout = self._make_layout()

        with Live(layout, refresh_per_second=4, console=self.console) as live:
            while self._running:
                self._update_layout(layout)
                await asyncio.sleep(self.refresh_rate)

    def stop(self) -> None:
        """Stop the dashboard."""
        self._running = False


class SimpleDashboard:
    """
    Simple text-based dashboard for environments without Rich.
    """

    def __init__(self, collector: MetricsCollector, refresh_rate: float = 1.0):
        self.collector = collector
        self.refresh_rate = refresh_rate
        self._running = False

    async def run(self) -> None:
        """Run the simple dashboard."""
        self._running = True

        while self._running:
            # Clear screen
            print("\033[2J\033[H", end="")

            summary = self.collector.get_summary()
            uptime = timedelta(seconds=int(summary["uptime_seconds"]))

            print("=" * 60)
            print("AgentMesh-STM Performance Monitor")
            print(f"Uptime: {uptime}")
            print("=" * 60)

            print("\n[Transactions]")
            txn = summary["transactions"]
            print(f"  Total: {txn['total']}")
            print(f"  Committed: {txn['committed']}")
            print(f"  Aborted: {txn['aborted']}")
            print(f"  Active: {txn['active']}")

            print("\n[Performance]")
            perf = summary["performance"]
            print(f"  Throughput: {perf['throughput_tps']:.2f} TPS")
            print(f"  Avg Latency: {perf['avg_latency_ms']:.1f} ms")

            print("\n[Conflicts]")
            conf = summary["conflicts"]
            print(f"  Total: {conf['total']}")
            print(f"  Resolved: {conf['resolved']}")
            print(f"  Resolution Rate: {conf['resolution_rate'] * 100:.1f}%")

            print("\n[Recent Transactions]")
            recent = self.collector.get_recent_transactions(5)
            for txn in recent:
                print(f"  {txn['id']}: {txn['status']} ({txn['latency_ms']:.0f}ms)")

            print("\n" + "-" * 60)
            print("Press Ctrl+C to exit")

            await asyncio.sleep(self.refresh_rate)

    def stop(self) -> None:
        """Stop the dashboard."""
        self._running = False


def create_dashboard(
    collector: MetricsCollector,
    refresh_rate: float = 1.0,
) -> Dashboard | SimpleDashboard:
    """Create appropriate dashboard based on available libraries."""
    if RICH_AVAILABLE:
        return Dashboard(collector, refresh_rate)
    return SimpleDashboard(collector, refresh_rate)


# Global metrics collector
_global_collector: Optional[MetricsCollector] = None


def get_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def set_collector(collector: MetricsCollector) -> None:
    """Set the global metrics collector."""
    global _global_collector
    _global_collector = collector

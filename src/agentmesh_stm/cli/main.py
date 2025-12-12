"""
Command-Line Interface for AgentMesh-STM.

Provides commands for:
- Running benchmarks
- Managing distributed coordinators
- Viewing metrics and statistics
- Running example scenarios
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="agentmesh-stm",
        description="AgentMesh-STM: STM-based Multi-Agent Coordination Framework",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Benchmark command
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Run benchmarks",
    )
    bench_parser.add_argument(
        "--type",
        choices=["throughput", "concurrency", "conflict", "all"],
        default="throughput",
        help="Benchmark type to run",
    )
    bench_parser.add_argument(
        "--agents",
        type=int,
        default=4,
        help="Number of agents",
    )
    bench_parser.add_argument(
        "--tasks",
        type=int,
        default=100,
        help="Number of tasks",
    )
    bench_parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (JSON)",
    )
    bench_parser.add_argument(
        "--baseline",
        action="store_true",
        help="Include baseline comparisons",
    )

    # Coordinator command
    coord_parser = subparsers.add_parser(
        "coordinator",
        help="Manage distributed coordinator",
    )
    coord_subparsers = coord_parser.add_subparsers(dest="coord_command")

    start_parser = coord_subparsers.add_parser("start", help="Start coordinator")
    start_parser.add_argument("--host", default="localhost", help="Host to bind to")
    start_parser.add_argument("--port", type=int, default=5555, help="Port to listen on")

    join_parser = coord_subparsers.add_parser("join", help="Join existing cluster")
    join_parser.add_argument("seed", help="Seed node address (host:port)")

    status_parser = coord_subparsers.add_parser("status", help="Show cluster status")
    status_parser.add_argument("--host", default="localhost", help="Coordinator host")
    status_parser.add_argument("--port", type=int, default=5555, help="Coordinator port")

    # Metrics command
    metrics_parser = subparsers.add_parser(
        "metrics",
        help="View metrics and statistics",
    )
    metrics_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )

    # Demo command
    demo_parser = subparsers.add_parser(
        "demo",
        help="Run demonstration scenarios",
    )
    demo_parser.add_argument(
        "scenario",
        choices=["basic", "conflict", "multiagent", "all"],
        help="Demo scenario to run",
    )

    # Init command
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize AgentMesh-STM in current directory",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing configuration",
    )

    return parser


async def run_benchmark(args: argparse.Namespace) -> int:
    """Run benchmarks."""
    from agentmesh_stm.evaluation.benchmark import (
        BenchmarkConfig,
        BenchmarkRunner,
        BenchmarkType,
        BaselineType,
    )

    print(f"Running {args.type} benchmark with {args.agents} agents and {args.tasks} tasks...")

    # Map string to enum
    type_map = {
        "throughput": BenchmarkType.THROUGHPUT,
        "concurrency": BenchmarkType.CONCURRENCY,
        "conflict": BenchmarkType.CONFLICT,
    }

    runner = BenchmarkRunner()

    if args.type == "all":
        benchmark_types = [BenchmarkType.THROUGHPUT, BenchmarkType.CONCURRENCY]
    else:
        benchmark_types = [type_map.get(args.type, BenchmarkType.THROUGHPUT)]

    results = []
    for bench_type in benchmark_types:
        config = BenchmarkConfig(
            name=f"{bench_type.name.lower()}_benchmark",
            benchmark_type=bench_type,
            num_agents=args.agents,
            num_tasks=args.tasks,
            baselines=[BaselineType.SERIAL] if args.baseline else [],
        )

        result = await runner.run_benchmark(config)
        results.append(result)
        print(result.summary())

    if args.output:
        output_data = [r.to_dict() for r in results]
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")

    print("\n" + runner.generate_comparison_report())

    return 0


async def run_coordinator(args: argparse.Namespace) -> int:
    """Run coordinator commands."""
    from agentmesh_stm.distributed.coordinator import (
        DistributedCoordinator,
        CoordinatorConfig,
    )

    if args.coord_command == "start":
        config = CoordinatorConfig(
            host=args.host,
            port=args.port,
        )
        coordinator = DistributedCoordinator(config=config)

        print(f"Starting coordinator on {args.host}:{args.port}...")
        await coordinator.start()

        print("Coordinator running. Press Ctrl+C to stop.")
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping coordinator...")
            await coordinator.stop()

    elif args.coord_command == "join":
        seed_parts = args.seed.split(":")
        seed_host = seed_parts[0]
        seed_port = int(seed_parts[1]) if len(seed_parts) > 1 else 5555

        config = CoordinatorConfig()
        coordinator = DistributedCoordinator(config=config)

        await coordinator.start()
        success = await coordinator.join_cluster(seed_host, seed_port)

        if success:
            print(f"Successfully joined cluster via {args.seed}")
            print("Coordinator running. Press Ctrl+C to stop.")
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping coordinator...")
                await coordinator.stop()
        else:
            print(f"Failed to join cluster via {args.seed}")
            return 1

    elif args.coord_command == "status":
        # Connect to coordinator and get status
        print(f"Cluster status from {args.host}:{args.port}:")
        print("  (Status query not implemented in CLI)")

    return 0


async def run_demo(args: argparse.Namespace) -> int:
    """Run demonstration scenarios."""
    print(f"Running {args.scenario} demo...\n")

    if args.scenario in ("basic", "all"):
        await run_basic_demo()

    if args.scenario in ("conflict", "all"):
        await run_conflict_demo()

    if args.scenario in ("multiagent", "all"):
        await run_multiagent_demo()

    return 0


async def run_basic_demo():
    """Run basic transaction demo."""
    from agentmesh_stm.core.transaction import TransactionManager
    from agentmesh_stm.storage.mvcc import MVCCStorage, InMemoryBackend

    print("=== Basic Transaction Demo ===\n")

    storage = MVCCStorage(backend=InMemoryBackend())
    manager = TransactionManager(storage=storage)

    # Initialize data
    await storage.write("counter", "0")
    print("Initialized counter to 0")

    # Run transaction
    async with manager.transaction() as txn:
        value = await txn.read("counter")
        print(f"Read counter: {value}")

        new_value = str(int(value) + 1)
        await txn.write("counter", new_value)
        print(f"Writing counter: {new_value}")

    # Verify
    final = await storage.read("counter")
    print(f"Final counter value: {final.content}\n")


async def run_conflict_demo():
    """Run conflict detection demo."""
    from agentmesh_stm.core.transaction import TransactionManager, TransactionConfig
    from agentmesh_stm.storage.mvcc import MVCCStorage, InMemoryBackend
    from agentmesh_stm.conflict.detector import ConflictDetector

    print("=== Conflict Detection Demo ===\n")

    storage = MVCCStorage(backend=InMemoryBackend())
    conflict_detector = ConflictDetector(storage=storage)
    manager = TransactionManager(
        storage=storage,
        conflict_detector=conflict_detector,
        default_config=TransactionConfig(max_retries=3),
    )

    await storage.write("shared", "initial")

    async def worker(name: str, delay: float):
        print(f"[{name}] Starting transaction")
        try:
            async with manager.transaction() as txn:
                value = await txn.read("shared")
                print(f"[{name}] Read: {value}")

                await asyncio.sleep(delay)

                await txn.write("shared", f"modified_by_{name}")
                print(f"[{name}] Write: modified_by_{name}")

            print(f"[{name}] Committed successfully")
        except Exception as e:
            print(f"[{name}] Failed: {e}")

    # Run two workers that will conflict
    await asyncio.gather(
        worker("Worker1", 0.1),
        worker("Worker2", 0.05),
    )

    final = await storage.read("shared")
    print(f"\nFinal value: {final.content}\n")


async def run_multiagent_demo():
    """Run multi-agent collaboration demo."""
    from agentmesh_stm.core.transaction import TransactionManager
    from agentmesh_stm.storage.mvcc import MVCCStorage, InMemoryBackend
    from agentmesh_stm.agent.base import SimpleAgent, AgentConfig, AgentTask

    print("=== Multi-Agent Collaboration Demo ===\n")

    storage = MVCCStorage(backend=InMemoryBackend())
    manager = TransactionManager(storage=storage)

    # Initialize files
    for i in range(3):
        await storage.write(f"file_{i}.txt", f"Content {i}")

    async def process_file(task, txn):
        file_path = task.input_data["file"]
        content = await txn.read(file_path)
        modified = f"[Processed] {content}"
        await txn.write(file_path, modified)
        return {"processed": file_path}

    # Create agents
    agents = [
        SimpleAgent(
            manager, process_file, AgentConfig(name=f"Agent{i}")
        )
        for i in range(3)
    ]

    # Create tasks
    tasks = [
        AgentTask.create(
            description=f"Process file_{i}",
            input_data={"file": f"file_{i}.txt"},
        )
        for i in range(3)
    ]

    print("Running 3 agents in parallel...")

    # Run in parallel
    results = await asyncio.gather(*[
        agents[i].run_task(tasks[i]) for i in range(3)
    ])

    for result in results:
        status = "SUCCESS" if result.success else "FAILED"
        print(f"  {result.output_data.get('processed', 'unknown')}: {status}")

    print("\nFinal file contents:")
    for i in range(3):
        content = await storage.read(f"file_{i}.txt")
        print(f"  file_{i}.txt: {content.content}")
    print()


def run_init(args: argparse.Namespace) -> int:
    """Initialize AgentMesh-STM configuration."""
    config_dir = Path(".agentmesh")
    config_file = config_dir / "config.json"

    if config_dir.exists() and not args.force:
        print("AgentMesh-STM already initialized. Use --force to overwrite.")
        return 1

    config_dir.mkdir(exist_ok=True)

    default_config = {
        "version": "0.1.0",
        "storage": {
            "backend": "sqlite",
            "path": ".agentmesh/storage.db",
        },
        "transaction": {
            "max_retries": 3,
            "retry_delay_ms": 100,
            "timeout_seconds": 300,
        },
        "conflict_detection": {
            "enable_ast_analysis": True,
            "enable_semantic_analysis": False,
        },
        "logging": {
            "level": "INFO",
            "format": "text",
        },
    }

    with open(config_file, "w") as f:
        json.dump(default_config, f, indent=2)

    print(f"Initialized AgentMesh-STM in {config_dir}")
    print(f"Configuration saved to {config_file}")

    return 0


def run_metrics(args: argparse.Namespace) -> int:
    """Display metrics."""
    from agentmesh_stm.evaluation.metrics import MetricsCollector

    collector = MetricsCollector()

    # For now, just show empty metrics
    if args.format == "json":
        print(json.dumps({
            "counters": {},
            "gauges": {},
            "message": "No metrics collected yet",
        }, indent=2))
    else:
        print("AgentMesh-STM Metrics")
        print("=" * 40)
        print("No metrics collected yet.")
        print("\nRun benchmarks or start agents to collect metrics.")

    return 0


def cli(args: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    if parsed_args.command is None:
        parser.print_help()
        return 0

    if parsed_args.command == "benchmark":
        return asyncio.run(run_benchmark(parsed_args))
    elif parsed_args.command == "coordinator":
        return asyncio.run(run_coordinator(parsed_args))
    elif parsed_args.command == "demo":
        return asyncio.run(run_demo(parsed_args))
    elif parsed_args.command == "init":
        return run_init(parsed_args)
    elif parsed_args.command == "metrics":
        return run_metrics(parsed_args)

    return 0


def main() -> None:
    """Main entry point."""
    sys.exit(cli())


if __name__ == "__main__":
    main()

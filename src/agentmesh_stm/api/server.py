"""
REST API Server for AgentMesh-STM.

Provides HTTP endpoints for:
- Transaction management
- Resource operations
- Agent task execution
- Metrics and monitoring
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from agentmesh_stm.core.transaction import TransactionManager, TransactionConfig
from agentmesh_stm.storage.mvcc import MVCCStorage, InMemoryBackend, SQLiteBackend
from agentmesh_stm.conflict.detector import ConflictDetector
from agentmesh_stm.monitoring.dashboard import MetricsCollector
from agentmesh_stm.utils.logging import get_logger

logger = get_logger(__name__)

# Try to import aiohttp for the server
try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None


@dataclass
class APIConfig:
    """Configuration for the API server."""

    host: str = "0.0.0.0"
    port: int = 8080
    storage_backend: str = "memory"
    storage_path: Optional[str] = None
    enable_cors: bool = True
    api_key: Optional[str] = None  # Optional API key for authentication


class TransactionAPI:
    """
    REST API for transaction operations.

    Endpoints:
    - POST /api/transactions - Create a new transaction
    - GET /api/transactions/{id} - Get transaction status
    - POST /api/transactions/{id}/read - Read a resource
    - POST /api/transactions/{id}/write - Write a resource
    - POST /api/transactions/{id}/commit - Commit transaction
    - POST /api/transactions/{id}/abort - Abort transaction
    """

    def __init__(self, manager: TransactionManager, metrics: MetricsCollector):
        self.manager = manager
        self.metrics = metrics
        self._active_transactions: Dict[str, Any] = {}

    def get_routes(self) -> list:
        """Get the API routes."""
        if not AIOHTTP_AVAILABLE:
            return []

        return [
            web.post("/api/transactions", self.create_transaction),
            web.get("/api/transactions/{id}", self.get_transaction),
            web.post("/api/transactions/{id}/read", self.read_resource),
            web.post("/api/transactions/{id}/write", self.write_resource),
            web.post("/api/transactions/{id}/commit", self.commit_transaction),
            web.post("/api/transactions/{id}/abort", self.abort_transaction),
            web.delete("/api/transactions/{id}", self.abort_transaction),
        ]

    async def create_transaction(self, request: web.Request) -> web.Response:
        """Create a new transaction."""
        try:
            body = await request.json() if request.body_exists else {}
        except json.JSONDecodeError:
            body = {}

        config = TransactionConfig(
            max_retries=body.get("max_retries", 3),
            timeout_seconds=body.get("timeout_seconds", 300),
        )

        txn = await self.manager.create_transaction(config)
        await txn.begin()

        self._active_transactions[txn.id] = txn
        self.metrics.transaction_started(txn.id)

        return web.json_response({
            "transaction_id": txn.id,
            "status": txn.state.name,
            "snapshot_version": txn.snapshot_version,
        }, status=201)

    async def get_transaction(self, request: web.Request) -> web.Response:
        """Get transaction status."""
        txn_id = request.match_info["id"]
        txn = self._active_transactions.get(txn_id)

        if not txn:
            return web.json_response(
                {"error": "Transaction not found"},
                status=404,
            )

        return web.json_response({
            "transaction_id": txn.id,
            "status": txn.state.name,
            "read_count": len(txn.read_set.get_all()),
            "write_count": len(txn.write_set.get_all()),
            "duration_seconds": txn.duration_seconds,
        })

    async def read_resource(self, request: web.Request) -> web.Response:
        """Read a resource within a transaction."""
        txn_id = request.match_info["id"]
        txn = self._active_transactions.get(txn_id)

        if not txn:
            return web.json_response(
                {"error": "Transaction not found"},
                status=404,
            )

        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response(
                {"error": "Invalid JSON body"},
                status=400,
            )

        resource_id = body.get("resource_id")
        if not resource_id:
            return web.json_response(
                {"error": "resource_id is required"},
                status=400,
            )

        content = await txn.read(resource_id)
        self.metrics.read_operation(txn_id)

        return web.json_response({
            "resource_id": resource_id,
            "content": content,
            "exists": content is not None,
        })

    async def write_resource(self, request: web.Request) -> web.Response:
        """Write a resource within a transaction."""
        txn_id = request.match_info["id"]
        txn = self._active_transactions.get(txn_id)

        if not txn:
            return web.json_response(
                {"error": "Transaction not found"},
                status=404,
            )

        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response(
                {"error": "Invalid JSON body"},
                status=400,
            )

        resource_id = body.get("resource_id")
        content = body.get("content")

        if not resource_id:
            return web.json_response(
                {"error": "resource_id is required"},
                status=400,
            )

        if content is None:
            return web.json_response(
                {"error": "content is required"},
                status=400,
            )

        await txn.write(resource_id, content)
        self.metrics.write_operation(txn_id)

        return web.json_response({
            "resource_id": resource_id,
            "status": "written",
        })

    async def commit_transaction(self, request: web.Request) -> web.Response:
        """Commit a transaction."""
        txn_id = request.match_info["id"]
        txn = self._active_transactions.get(txn_id)

        if not txn:
            return web.json_response(
                {"error": "Transaction not found"},
                status=404,
            )

        success = await txn.commit()

        if success:
            self.metrics.transaction_committed(txn_id)
            del self._active_transactions[txn_id]
            return web.json_response({
                "transaction_id": txn_id,
                "status": "committed",
            })
        else:
            self.metrics.conflict_detected()
            return web.json_response({
                "transaction_id": txn_id,
                "status": "conflict",
                "error": "Transaction could not commit due to conflicts",
            }, status=409)

    async def abort_transaction(self, request: web.Request) -> web.Response:
        """Abort a transaction."""
        txn_id = request.match_info["id"]
        txn = self._active_transactions.get(txn_id)

        if not txn:
            return web.json_response(
                {"error": "Transaction not found"},
                status=404,
            )

        await txn.abort()
        self.metrics.transaction_aborted(txn_id)
        del self._active_transactions[txn_id]

        return web.json_response({
            "transaction_id": txn_id,
            "status": "aborted",
        })


class ResourceAPI:
    """
    REST API for direct resource operations (auto-commit).

    Endpoints:
    - GET /api/resources/{path} - Read a resource
    - PUT /api/resources/{path} - Write a resource
    - DELETE /api/resources/{path} - Delete a resource
    """

    def __init__(self, manager: TransactionManager, metrics: MetricsCollector):
        self.manager = manager
        self.metrics = metrics

    def get_routes(self) -> list:
        """Get the API routes."""
        if not AIOHTTP_AVAILABLE:
            return []

        return [
            web.get("/api/resources/{path:.*}", self.read_resource),
            web.put("/api/resources/{path:.*}", self.write_resource),
            web.delete("/api/resources/{path:.*}", self.delete_resource),
        ]

    async def read_resource(self, request: web.Request) -> web.Response:
        """Read a resource (single auto-commit transaction)."""
        path = request.match_info["path"]

        async def do_read(txn):
            return await txn.read(path)

        content = await self.manager.execute(do_read)

        if content is None:
            return web.json_response(
                {"error": "Resource not found"},
                status=404,
            )

        return web.json_response({
            "path": path,
            "content": content,
        })

    async def write_resource(self, request: web.Request) -> web.Response:
        """Write a resource (single auto-commit transaction)."""
        path = request.match_info["path"]

        try:
            body = await request.json()
            content = body.get("content", "")
        except json.JSONDecodeError:
            content = await request.text()

        async def do_write(txn):
            await txn.write(path, content)

        await self.manager.execute(do_write)

        return web.json_response({
            "path": path,
            "status": "written",
        })

    async def delete_resource(self, request: web.Request) -> web.Response:
        """Delete a resource (single auto-commit transaction)."""
        path = request.match_info["path"]

        async def do_delete(txn):
            await txn.delete(path)

        await self.manager.execute(do_delete)

        return web.json_response({
            "path": path,
            "status": "deleted",
        })


class MetricsAPI:
    """
    REST API for metrics and monitoring.

    Endpoints:
    - GET /api/metrics - Get current metrics
    - GET /api/health - Health check
    """

    def __init__(self, metrics: MetricsCollector):
        self.metrics = metrics

    def get_routes(self) -> list:
        """Get the API routes."""
        if not AIOHTTP_AVAILABLE:
            return []

        return [
            web.get("/api/metrics", self.get_metrics),
            web.get("/api/health", self.health_check),
            web.get("/health", self.health_check),
        ]

    async def get_metrics(self, request: web.Request) -> web.Response:
        """Get current metrics."""
        summary = self.metrics.get_summary()
        return web.json_response(summary)

    async def health_check(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({
            "status": "healthy",
            "service": "agentmesh-stm",
        })


def create_app(config: Optional[APIConfig] = None) -> web.Application:
    """Create the aiohttp application."""
    if not AIOHTTP_AVAILABLE:
        raise ImportError(
            "aiohttp is required for the API server. "
            "Install with: pip install aiohttp"
        )

    config = config or APIConfig()

    # Create storage backend
    if config.storage_backend == "sqlite" and config.storage_path:
        backend = SQLiteBackend(config.storage_path)
    else:
        backend = InMemoryBackend()

    storage = MVCCStorage(backend)
    conflict_detector = ConflictDetector(storage)
    manager = TransactionManager(
        storage=storage,
        conflict_detector=conflict_detector,
    )
    metrics = MetricsCollector()

    # Create API handlers
    transaction_api = TransactionAPI(manager, metrics)
    resource_api = ResourceAPI(manager, metrics)
    metrics_api = MetricsAPI(metrics)

    # Create application
    app = web.Application()

    # Add routes
    app.router.add_routes(transaction_api.get_routes())
    app.router.add_routes(resource_api.get_routes())
    app.router.add_routes(metrics_api.get_routes())

    # Add CORS middleware if enabled
    if config.enable_cors:
        @web.middleware
        async def cors_middleware(request, handler):
            response = await handler(request)
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            return response

        app.middlewares.append(cors_middleware)

    # Add API key authentication if configured
    if config.api_key:
        @web.middleware
        async def auth_middleware(request, handler):
            # Skip auth for health check
            if request.path in ("/health", "/api/health"):
                return await handler(request)

            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return web.json_response(
                    {"error": "Missing or invalid Authorization header"},
                    status=401,
                )

            token = auth_header[7:]
            if token != config.api_key:
                return web.json_response(
                    {"error": "Invalid API key"},
                    status=401,
                )

            return await handler(request)

        app.middlewares.append(auth_middleware)

    # Store config and manager in app
    app["config"] = config
    app["manager"] = manager
    app["metrics"] = metrics

    logger.info(
        "API application created",
        storage=config.storage_backend,
        cors=config.enable_cors,
        auth=bool(config.api_key),
    )

    return app


async def run_server(config: Optional[APIConfig] = None) -> None:
    """Run the API server."""
    if not AIOHTTP_AVAILABLE:
        raise ImportError(
            "aiohttp is required for the API server. "
            "Install with: pip install aiohttp"
        )

    config = config or APIConfig()
    app = create_app(config)

    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, config.host, config.port)
    await site.start()

    logger.info(
        f"API server started",
        host=config.host,
        port=config.port,
    )

    print(f"AgentMesh-STM API server running at http://{config.host}:{config.port}")
    print("Press Ctrl+C to stop")

    # Keep running until interrupted
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()

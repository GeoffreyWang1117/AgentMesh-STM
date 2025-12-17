"""
Lifecycle Management Module for AgentMesh-STM.

Provides graceful startup and shutdown handling.
"""

from __future__ import annotations

import asyncio
import signal
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from agentmesh_stm.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Lifecycle States
# =============================================================================

class LifecycleState(Enum):
    """Application lifecycle states."""

    CREATED = auto()
    STARTING = auto()
    RUNNING = auto()
    DRAINING = auto()  # Stopping new requests
    STOPPING = auto()
    STOPPED = auto()
    FAILED = auto()


@dataclass
class HealthStatus:
    """Health check result."""

    healthy: bool
    state: LifecycleState
    checks: Dict[str, bool] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "healthy": self.healthy,
            "state": self.state.name,
            "checks": self.checks,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# Lifecycle Hooks
# =============================================================================

StartupHook = Callable[[], Coroutine[Any, Any, None]]
ShutdownHook = Callable[[], Coroutine[Any, Any, None]]
HealthCheck = Callable[[], Coroutine[Any, Any, bool]]


@dataclass
class LifecycleHook:
    """A lifecycle hook with metadata."""

    name: str
    callback: Callable
    priority: int = 0  # Lower = run first
    timeout: float = 30.0
    critical: bool = True  # If True, failure stops lifecycle


# =============================================================================
# Lifecycle Manager
# =============================================================================

class LifecycleManager:
    """
    Manages application lifecycle with graceful startup and shutdown.

    Features:
    - Ordered startup/shutdown hooks
    - Signal handling (SIGTERM, SIGINT)
    - Graceful drain period for in-flight requests
    - Health checks
    - Timeout handling
    """

    def __init__(
        self,
        drain_timeout: float = 30.0,
        shutdown_timeout: float = 60.0,
    ):
        self.drain_timeout = drain_timeout
        self.shutdown_timeout = shutdown_timeout

        self._state = LifecycleState.CREATED
        self._startup_hooks: List[LifecycleHook] = []
        self._shutdown_hooks: List[LifecycleHook] = []
        self._health_checks: Dict[str, HealthCheck] = {}

        self._active_requests = 0
        self._requests_lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        self._shutdown_complete = asyncio.Event()

        self._signal_handlers_installed = False
        self._original_handlers: Dict[int, Any] = {}

    @property
    def state(self) -> LifecycleState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._state == LifecycleState.RUNNING

    @property
    def is_accepting_requests(self) -> bool:
        return self._state in (LifecycleState.STARTING, LifecycleState.RUNNING)

    @property
    def is_shutting_down(self) -> bool:
        return self._state in (
            LifecycleState.DRAINING,
            LifecycleState.STOPPING,
            LifecycleState.STOPPED,
        )

    # -------------------------------------------------------------------------
    # Hook Registration
    # -------------------------------------------------------------------------

    def on_startup(
        self,
        name: str,
        callback: StartupHook,
        priority: int = 0,
        timeout: float = 30.0,
        critical: bool = True,
    ) -> None:
        """Register a startup hook."""
        hook = LifecycleHook(
            name=name,
            callback=callback,
            priority=priority,
            timeout=timeout,
            critical=critical,
        )
        self._startup_hooks.append(hook)
        self._startup_hooks.sort(key=lambda h: h.priority)
        logger.debug(f"Registered startup hook: {name}")

    def on_shutdown(
        self,
        name: str,
        callback: ShutdownHook,
        priority: int = 0,
        timeout: float = 30.0,
        critical: bool = False,
    ) -> None:
        """Register a shutdown hook."""
        hook = LifecycleHook(
            name=name,
            callback=callback,
            priority=priority,
            timeout=timeout,
            critical=critical,
        )
        self._shutdown_hooks.append(hook)
        self._shutdown_hooks.sort(key=lambda h: h.priority)
        logger.debug(f"Registered shutdown hook: {name}")

    def add_health_check(self, name: str, check: HealthCheck) -> None:
        """Register a health check."""
        self._health_checks[name] = check
        logger.debug(f"Registered health check: {name}")

    # -------------------------------------------------------------------------
    # Request Tracking
    # -------------------------------------------------------------------------

    @asynccontextmanager
    async def track_request(self):
        """Context manager to track in-flight requests."""
        if not self.is_accepting_requests:
            raise RuntimeError("Not accepting new requests")

        async with self._requests_lock:
            self._active_requests += 1

        try:
            yield
        finally:
            async with self._requests_lock:
                self._active_requests -= 1

    async def get_active_requests(self) -> int:
        """Get count of active requests."""
        async with self._requests_lock:
            return self._active_requests

    # -------------------------------------------------------------------------
    # Lifecycle Operations
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """Start the application."""
        if self._state != LifecycleState.CREATED:
            raise RuntimeError(f"Cannot start from state {self._state}")

        logger.info("Starting application...")
        self._state = LifecycleState.STARTING

        try:
            # Run startup hooks
            for hook in self._startup_hooks:
                logger.info(f"Running startup hook: {hook.name}")
                try:
                    await asyncio.wait_for(
                        hook.callback(),
                        timeout=hook.timeout,
                    )
                    logger.info(f"Startup hook completed: {hook.name}")
                except asyncio.TimeoutError:
                    logger.error(f"Startup hook timed out: {hook.name}")
                    if hook.critical:
                        raise RuntimeError(f"Critical startup hook failed: {hook.name}")
                except Exception as e:
                    logger.error(f"Startup hook failed: {hook.name}", error=str(e))
                    if hook.critical:
                        raise

            self._state = LifecycleState.RUNNING
            logger.info("Application started successfully")

        except Exception as e:
            self._state = LifecycleState.FAILED
            logger.error("Application startup failed", error=str(e))
            raise

    async def stop(self, reason: str = "shutdown requested") -> None:
        """Stop the application gracefully."""
        if self._state in (LifecycleState.STOPPING, LifecycleState.STOPPED):
            logger.info("Already stopping/stopped")
            return

        logger.info(f"Stopping application: {reason}")

        # Enter draining state
        self._state = LifecycleState.DRAINING
        self._shutdown_event.set()

        # Wait for in-flight requests to complete
        await self._drain_requests()

        # Enter stopping state
        self._state = LifecycleState.STOPPING

        # Run shutdown hooks (in reverse priority order)
        shutdown_hooks = sorted(self._shutdown_hooks, key=lambda h: -h.priority)
        for hook in shutdown_hooks:
            logger.info(f"Running shutdown hook: {hook.name}")
            try:
                await asyncio.wait_for(
                    hook.callback(),
                    timeout=hook.timeout,
                )
                logger.info(f"Shutdown hook completed: {hook.name}")
            except asyncio.TimeoutError:
                logger.error(f"Shutdown hook timed out: {hook.name}")
            except Exception as e:
                logger.error(f"Shutdown hook failed: {hook.name}", error=str(e))

        self._state = LifecycleState.STOPPED
        self._shutdown_complete.set()
        logger.info("Application stopped")

    async def _drain_requests(self) -> None:
        """Wait for in-flight requests to complete."""
        logger.info(f"Draining requests (timeout: {self.drain_timeout}s)")

        start_time = asyncio.get_event_loop().time()
        deadline = start_time + self.drain_timeout

        while True:
            active = await self.get_active_requests()
            if active == 0:
                logger.info("All requests drained")
                return

            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                logger.warning(f"Drain timeout reached with {active} active requests")
                return

            logger.info(f"Waiting for {active} requests to complete...")
            await asyncio.sleep(min(1.0, remaining))

    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown to be initiated."""
        await self._shutdown_event.wait()

    async def wait_for_complete(self) -> None:
        """Wait for shutdown to complete."""
        await self._shutdown_complete.wait()

    # -------------------------------------------------------------------------
    # Signal Handling
    # -------------------------------------------------------------------------

    def install_signal_handlers(self) -> None:
        """Install signal handlers for graceful shutdown."""
        if self._signal_handlers_installed:
            return

        if sys.platform == "win32":
            # Windows doesn't support SIGTERM
            signals = [signal.SIGINT]
        else:
            signals = [signal.SIGTERM, signal.SIGINT]

        loop = asyncio.get_event_loop()

        for sig in signals:
            self._original_handlers[sig] = signal.getsignal(sig)
            loop.add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(self._handle_signal(s))
            )

        self._signal_handlers_installed = True
        logger.info("Signal handlers installed")

    def remove_signal_handlers(self) -> None:
        """Remove installed signal handlers."""
        if not self._signal_handlers_installed:
            return

        loop = asyncio.get_event_loop()

        for sig, handler in self._original_handlers.items():
            try:
                loop.remove_signal_handler(sig)
                if handler:
                    signal.signal(sig, handler)
            except Exception:
                pass

        self._signal_handlers_installed = False
        self._original_handlers.clear()
        logger.info("Signal handlers removed")

    async def _handle_signal(self, sig: signal.Signals) -> None:
        """Handle shutdown signal."""
        logger.info(f"Received signal {sig.name}")
        await self.stop(reason=f"received {sig.name}")

    # -------------------------------------------------------------------------
    # Health Checks
    # -------------------------------------------------------------------------

    async def health_check(self) -> HealthStatus:
        """Run all health checks and return status."""
        checks: Dict[str, bool] = {}
        details: Dict[str, Any] = {}

        # Run all registered health checks
        for name, check in self._health_checks.items():
            try:
                result = await asyncio.wait_for(check(), timeout=5.0)
                checks[name] = result
            except asyncio.TimeoutError:
                checks[name] = False
                details[name] = "timeout"
            except Exception as e:
                checks[name] = False
                details[name] = str(e)

        # Add built-in checks
        checks["lifecycle"] = self._state in (
            LifecycleState.RUNNING,
            LifecycleState.DRAINING,
        )
        details["active_requests"] = await self.get_active_requests()
        details["state"] = self._state.name

        healthy = all(checks.values()) and self._state == LifecycleState.RUNNING

        return HealthStatus(
            healthy=healthy,
            state=self._state,
            checks=checks,
            details=details,
        )

    async def readiness_check(self) -> bool:
        """Check if application is ready to serve traffic."""
        return self._state == LifecycleState.RUNNING

    async def liveness_check(self) -> bool:
        """Check if application is alive (not deadlocked)."""
        return self._state not in (LifecycleState.FAILED, LifecycleState.STOPPED)


# =============================================================================
# Context Manager for Full Lifecycle
# =============================================================================

@asynccontextmanager
async def managed_lifecycle(
    manager: LifecycleManager,
    install_signals: bool = True,
):
    """
    Context manager for managing application lifecycle.

    Usage:
        manager = LifecycleManager()
        manager.on_startup("db", connect_database)
        manager.on_shutdown("db", disconnect_database)

        async with managed_lifecycle(manager):
            # Application runs here
            await manager.wait_for_shutdown()
    """
    try:
        if install_signals:
            manager.install_signal_handlers()

        await manager.start()
        yield manager

    except Exception as e:
        logger.error(f"Lifecycle error: {e}")
        raise

    finally:
        if manager.state not in (LifecycleState.STOPPING, LifecycleState.STOPPED):
            await manager.stop(reason="context exit")

        if install_signals:
            manager.remove_signal_handlers()


# =============================================================================
# Component Registration Helpers
# =============================================================================

class LifecycleAware:
    """Mixin for components that need lifecycle management."""

    async def on_start(self) -> None:
        """Called when application starts."""
        pass

    async def on_stop(self) -> None:
        """Called when application stops."""
        pass

    async def health_check(self) -> bool:
        """Return True if component is healthy."""
        return True

    def register_with(self, manager: LifecycleManager, name: str) -> None:
        """Register this component with a lifecycle manager."""
        manager.on_startup(
            f"{name}.start",
            self.on_start,
        )
        manager.on_shutdown(
            f"{name}.stop",
            self.on_stop,
        )
        manager.add_health_check(name, self.health_check)


# =============================================================================
# Global Lifecycle Manager
# =============================================================================

_lifecycle_manager: Optional[LifecycleManager] = None


def get_lifecycle_manager() -> LifecycleManager:
    """Get or create the global lifecycle manager."""
    global _lifecycle_manager
    if _lifecycle_manager is None:
        _lifecycle_manager = LifecycleManager()
    return _lifecycle_manager


def set_lifecycle_manager(manager: LifecycleManager) -> None:
    """Set the global lifecycle manager."""
    global _lifecycle_manager
    _lifecycle_manager = manager

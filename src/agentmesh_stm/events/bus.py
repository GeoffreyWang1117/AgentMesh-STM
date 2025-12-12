"""
Event Bus for AgentMesh-STM.

Provides a pub/sub system for transaction lifecycle events, enabling:
- Monitoring and logging
- Custom event handlers
- Integration with external systems
- Debugging and tracing
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Union
from weakref import WeakSet

from agentmesh_stm.utils.logging import get_logger

logger = get_logger(__name__)


class EventType(Enum):
    """Types of events in the system."""

    # Transaction events
    TRANSACTION_BEGIN = auto()
    TRANSACTION_READ = auto()
    TRANSACTION_WRITE = auto()
    TRANSACTION_COMMIT_START = auto()
    TRANSACTION_COMMIT_SUCCESS = auto()
    TRANSACTION_COMMIT_FAILED = auto()
    TRANSACTION_ABORT = auto()
    TRANSACTION_RETRY = auto()

    # Conflict events
    CONFLICT_DETECTED = auto()
    CONFLICT_RESOLVED = auto()
    CONFLICT_UNRESOLVED = auto()

    # Agent events
    AGENT_TASK_START = auto()
    AGENT_TASK_COMPLETE = auto()
    AGENT_TASK_FAILED = auto()
    AGENT_TOOL_CALL = auto()

    # System events
    STORAGE_WRITE = auto()
    STORAGE_GC = auto()
    CHECKPOINT_CREATED = auto()
    RECOVERY_STARTED = auto()
    RECOVERY_COMPLETED = auto()


@dataclass
class Event:
    """Base event class."""

    event_type: EventType
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class TransactionEvent(Event):
    """Event related to a transaction."""

    transaction_id: str = ""
    resource_id: Optional[str] = None


@dataclass
class ConflictEvent(Event):
    """Event related to conflict detection/resolution."""

    transaction_id: str = ""
    resource_id: str = ""
    conflict_type: str = ""
    resolution: Optional[str] = None


@dataclass
class AgentEvent(Event):
    """Event related to an agent."""

    agent_id: str = ""
    task_id: Optional[str] = None
    tool_name: Optional[str] = None


class EventHandler(ABC):
    """Abstract base class for event handlers."""

    @abstractmethod
    async def handle(self, event: Event) -> None:
        """Handle an event."""
        pass

    def can_handle(self, event: Event) -> bool:
        """Check if this handler can handle the event."""
        return True


class EventFilter:
    """Filters events based on criteria."""

    def __init__(
        self,
        event_types: Optional[Set[EventType]] = None,
        sources: Optional[Set[str]] = None,
        transaction_ids: Optional[Set[str]] = None,
    ):
        self.event_types = event_types
        self.sources = sources
        self.transaction_ids = transaction_ids

    def matches(self, event: Event) -> bool:
        """Check if event matches the filter."""
        if self.event_types and event.event_type not in self.event_types:
            return False

        if self.sources and event.source not in self.sources:
            return False

        if self.transaction_ids:
            txn_id = getattr(event, "transaction_id", None)
            if txn_id and txn_id not in self.transaction_ids:
                return False

        return True


class CallbackHandler(EventHandler):
    """Handler that calls a callback function."""

    def __init__(
        self,
        callback: Callable[[Event], Any],
        filter: Optional[EventFilter] = None,
    ):
        self.callback = callback
        self.filter = filter

    async def handle(self, event: Event) -> None:
        if asyncio.iscoroutinefunction(self.callback):
            await self.callback(event)
        else:
            self.callback(event)

    def can_handle(self, event: Event) -> bool:
        if self.filter:
            return self.filter.matches(event)
        return True


class LoggingHandler(EventHandler):
    """Handler that logs events."""

    def __init__(self, log_level: str = "DEBUG"):
        self.log_level = log_level.upper()

    async def handle(self, event: Event) -> None:
        log_func = getattr(logger, self.log_level.lower(), logger.debug)
        log_func(
            f"Event: {event.event_type.name}",
            **{k: v for k, v in event.data.items() if not k.startswith("_")},
            source=event.source,
        )


class MetricsHandler(EventHandler):
    """Handler that updates metrics based on events."""

    def __init__(self):
        self.counters: Dict[EventType, int] = {}
        self.last_events: Dict[EventType, Event] = {}

    async def handle(self, event: Event) -> None:
        self.counters[event.event_type] = self.counters.get(event.event_type, 0) + 1
        self.last_events[event.event_type] = event

    def get_count(self, event_type: EventType) -> int:
        return self.counters.get(event_type, 0)

    def get_counts(self) -> Dict[str, int]:
        return {et.name: count for et, count in self.counters.items()}


class BufferedHandler(EventHandler):
    """Handler that buffers events for batch processing."""

    def __init__(self, max_size: int = 1000, flush_callback: Optional[Callable] = None):
        self.max_size = max_size
        self.flush_callback = flush_callback
        self.buffer: List[Event] = []
        self._lock = asyncio.Lock()

    async def handle(self, event: Event) -> None:
        async with self._lock:
            self.buffer.append(event)
            if len(self.buffer) >= self.max_size:
                await self.flush()

    async def flush(self) -> List[Event]:
        async with self._lock:
            events = self.buffer
            self.buffer = []

        if self.flush_callback and events:
            if asyncio.iscoroutinefunction(self.flush_callback):
                await self.flush_callback(events)
            else:
                self.flush_callback(events)

        return events


class EventBus:
    """
    Central event bus for publishing and subscribing to events.

    Supports:
    - Async event handling
    - Multiple handlers per event type
    - Event filtering
    - Handler priorities
    """

    def __init__(self):
        self._handlers: List[tuple[int, EventHandler]] = []  # (priority, handler)
        self._type_handlers: Dict[EventType, List[EventHandler]] = {}
        self._lock = asyncio.Lock()
        self._running = False
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._process_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the event processing loop."""
        if self._running:
            return

        self._running = True
        self._process_task = asyncio.create_task(self._process_loop())
        logger.debug("Event bus started")

    async def stop(self) -> None:
        """Stop the event processing loop."""
        if not self._running:
            return

        self._running = False
        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass

        logger.debug("Event bus stopped")

    def subscribe(
        self,
        handler: EventHandler,
        event_types: Optional[Set[EventType]] = None,
        priority: int = 0,
    ) -> None:
        """
        Subscribe a handler to events.

        Args:
            handler: Event handler
            event_types: Optional set of event types to handle (all if None)
            priority: Handler priority (higher = earlier execution)
        """
        self._handlers.append((priority, handler))
        self._handlers.sort(key=lambda x: -x[0])  # Higher priority first

        if event_types:
            for et in event_types:
                if et not in self._type_handlers:
                    self._type_handlers[et] = []
                self._type_handlers[et].append(handler)

        logger.debug(
            "Handler subscribed",
            handler=handler.__class__.__name__,
            event_types=[et.name for et in event_types] if event_types else "all",
        )

    def unsubscribe(self, handler: EventHandler) -> None:
        """Unsubscribe a handler."""
        self._handlers = [(p, h) for p, h in self._handlers if h != handler]
        for et in self._type_handlers:
            self._type_handlers[et] = [
                h for h in self._type_handlers[et] if h != handler
            ]

    def on(
        self,
        *event_types: EventType,
        priority: int = 0,
    ) -> Callable:
        """Decorator for subscribing a function as an event handler."""
        def decorator(func: Callable) -> Callable:
            handler = CallbackHandler(func)
            self.subscribe(
                handler,
                set(event_types) if event_types else None,
                priority,
            )
            return func
        return decorator

    async def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribers.

        Events are queued and processed asynchronously.
        """
        await self._queue.put(event)

    async def publish_sync(self, event: Event) -> None:
        """
        Publish an event and wait for all handlers to complete.
        """
        await self._dispatch(event)

    async def _process_loop(self) -> None:
        """Main event processing loop."""
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0,
                )
                await self._dispatch(event)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error processing event", error=str(e))

    async def _dispatch(self, event: Event) -> None:
        """Dispatch event to handlers."""
        # Get handlers for this event type
        handlers = []

        # Type-specific handlers
        if event.event_type in self._type_handlers:
            handlers.extend(self._type_handlers[event.event_type])

        # General handlers (those not subscribed to specific types)
        for priority, handler in self._handlers:
            if handler not in handlers and handler.can_handle(event):
                handlers.append(handler)

        # Execute handlers
        for handler in handlers:
            try:
                if handler.can_handle(event):
                    await handler.handle(event)
            except Exception as e:
                logger.error(
                    "Handler error",
                    handler=handler.__class__.__name__,
                    event=event.event_type.name,
                    error=str(e),
                )


class TransactionEventEmitter:
    """
    Mixin class for emitting transaction events.

    Add to TransactionManager to automatically emit events.
    """

    def __init__(self, event_bus: Optional[EventBus] = None):
        self._event_bus = event_bus or get_event_bus()

    async def emit_transaction_begin(self, transaction_id: str) -> None:
        if self._event_bus:
            await self._event_bus.publish(
                TransactionEvent(
                    event_type=EventType.TRANSACTION_BEGIN,
                    transaction_id=transaction_id,
                    source="TransactionManager",
                )
            )

    async def emit_transaction_read(
        self, transaction_id: str, resource_id: str
    ) -> None:
        if self._event_bus:
            await self._event_bus.publish(
                TransactionEvent(
                    event_type=EventType.TRANSACTION_READ,
                    transaction_id=transaction_id,
                    resource_id=resource_id,
                    source="TransactionManager",
                )
            )

    async def emit_transaction_write(
        self, transaction_id: str, resource_id: str
    ) -> None:
        if self._event_bus:
            await self._event_bus.publish(
                TransactionEvent(
                    event_type=EventType.TRANSACTION_WRITE,
                    transaction_id=transaction_id,
                    resource_id=resource_id,
                    source="TransactionManager",
                )
            )

    async def emit_transaction_commit_success(self, transaction_id: str) -> None:
        if self._event_bus:
            await self._event_bus.publish(
                TransactionEvent(
                    event_type=EventType.TRANSACTION_COMMIT_SUCCESS,
                    transaction_id=transaction_id,
                    source="TransactionManager",
                )
            )

    async def emit_transaction_commit_failed(
        self, transaction_id: str, reason: str
    ) -> None:
        if self._event_bus:
            await self._event_bus.publish(
                TransactionEvent(
                    event_type=EventType.TRANSACTION_COMMIT_FAILED,
                    transaction_id=transaction_id,
                    data={"reason": reason},
                    source="TransactionManager",
                )
            )

    async def emit_transaction_abort(
        self, transaction_id: str, reason: str = ""
    ) -> None:
        if self._event_bus:
            await self._event_bus.publish(
                TransactionEvent(
                    event_type=EventType.TRANSACTION_ABORT,
                    transaction_id=transaction_id,
                    data={"reason": reason},
                    source="TransactionManager",
                )
            )

    async def emit_conflict_detected(
        self,
        transaction_id: str,
        resource_id: str,
        conflict_type: str,
    ) -> None:
        if self._event_bus:
            await self._event_bus.publish(
                ConflictEvent(
                    event_type=EventType.CONFLICT_DETECTED,
                    transaction_id=transaction_id,
                    resource_id=resource_id,
                    conflict_type=conflict_type,
                    source="ConflictDetector",
                )
            )

    async def emit_conflict_resolved(
        self,
        transaction_id: str,
        resource_id: str,
        resolution: str,
    ) -> None:
        if self._event_bus:
            await self._event_bus.publish(
                ConflictEvent(
                    event_type=EventType.CONFLICT_RESOLVED,
                    transaction_id=transaction_id,
                    resource_id=resource_id,
                    resolution=resolution,
                    source="ConflictResolver",
                )
            )


# Global event bus instance
_global_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get the global event bus."""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus


def set_event_bus(bus: EventBus) -> None:
    """Set the global event bus."""
    global _global_event_bus
    _global_event_bus = bus

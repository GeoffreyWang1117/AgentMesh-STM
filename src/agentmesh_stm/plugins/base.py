"""
Plugin Architecture for AgentMesh-STM.

Provides extensibility through:
- Custom storage backends
- Custom conflict resolvers
- Custom agent types
- Transaction hooks
- Event handlers
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type

from agentmesh_stm.utils.logging import get_logger

logger = get_logger(__name__)


class HookType(Enum):
    """Types of hooks in the plugin system."""

    # Transaction hooks
    PRE_TRANSACTION_BEGIN = auto()
    POST_TRANSACTION_BEGIN = auto()
    PRE_TRANSACTION_READ = auto()
    POST_TRANSACTION_READ = auto()
    PRE_TRANSACTION_WRITE = auto()
    POST_TRANSACTION_WRITE = auto()
    PRE_TRANSACTION_COMMIT = auto()
    POST_TRANSACTION_COMMIT = auto()
    PRE_TRANSACTION_ABORT = auto()
    POST_TRANSACTION_ABORT = auto()

    # Conflict hooks
    PRE_CONFLICT_CHECK = auto()
    POST_CONFLICT_CHECK = auto()
    PRE_CONFLICT_RESOLVE = auto()
    POST_CONFLICT_RESOLVE = auto()

    # Agent hooks
    PRE_AGENT_TASK = auto()
    POST_AGENT_TASK = auto()
    PRE_TOOL_CALL = auto()
    POST_TOOL_CALL = auto()

    # Storage hooks
    PRE_STORAGE_WRITE = auto()
    POST_STORAGE_WRITE = auto()
    PRE_STORAGE_READ = auto()
    POST_STORAGE_READ = auto()


@dataclass
class PluginHook:
    """A hook registration."""

    hook_type: HookType
    callback: Callable
    priority: int = 0  # Higher = earlier execution

    async def execute(self, *args, **kwargs) -> Any:
        """Execute the hook callback."""
        if asyncio.iscoroutinefunction(self.callback):
            return await self.callback(*args, **kwargs)
        return self.callback(*args, **kwargs)


@dataclass
class PluginInfo:
    """Information about a plugin."""

    name: str
    version: str
    description: str = ""
    author: str = ""
    dependencies: List[str] = field(default_factory=list)
    homepage: str = ""


class Plugin(ABC):
    """
    Base class for all plugins.

    Plugins can:
    - Register hooks to intercept operations
    - Provide custom implementations
    - Extend the framework's functionality
    """

    def __init__(self):
        self._enabled = False
        self._hooks: List[PluginHook] = []

    @property
    @abstractmethod
    def info(self) -> PluginInfo:
        """Return plugin information."""
        pass

    @property
    def enabled(self) -> bool:
        """Check if the plugin is enabled."""
        return self._enabled

    async def on_load(self) -> None:
        """Called when the plugin is loaded."""
        pass

    async def on_enable(self) -> None:
        """Called when the plugin is enabled."""
        pass

    async def on_disable(self) -> None:
        """Called when the plugin is disabled."""
        pass

    async def on_unload(self) -> None:
        """Called when the plugin is unloaded."""
        pass

    def register_hook(
        self,
        hook_type: HookType,
        callback: Callable,
        priority: int = 0,
    ) -> PluginHook:
        """Register a hook."""
        hook = PluginHook(
            hook_type=hook_type,
            callback=callback,
            priority=priority,
        )
        self._hooks.append(hook)
        return hook

    def get_hooks(self, hook_type: Optional[HookType] = None) -> List[PluginHook]:
        """Get all hooks, optionally filtered by type."""
        if hook_type is None:
            return self._hooks
        return [h for h in self._hooks if h.hook_type == hook_type]


class StoragePlugin(Plugin):
    """Plugin that provides a custom storage backend."""

    @abstractmethod
    def create_backend(self, config: Dict[str, Any]) -> Any:
        """Create a storage backend instance."""
        pass


class ConflictResolverPlugin(Plugin):
    """Plugin that provides a custom conflict resolver."""

    @abstractmethod
    def create_resolver(self, config: Dict[str, Any]) -> Any:
        """Create a conflict resolver instance."""
        pass


class AgentPlugin(Plugin):
    """Plugin that provides a custom agent type."""

    @abstractmethod
    def create_agent(self, config: Dict[str, Any]) -> Any:
        """Create an agent instance."""
        pass


class ToolPlugin(Plugin):
    """Plugin that provides additional tools for agents."""

    @abstractmethod
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get tool definitions."""
        pass


class PluginManager:
    """
    Manages plugin lifecycle and execution.

    Supports:
    - Loading plugins from Python modules
    - Loading plugins from directories
    - Hook execution with priorities
    - Plugin dependencies
    """

    def __init__(self):
        self._plugins: Dict[str, Plugin] = {}
        self._hooks: Dict[HookType, List[PluginHook]] = {}
        self._plugin_classes: Dict[str, Type[Plugin]] = {}
        self._lock = asyncio.Lock()

    async def load_plugin(self, plugin: Plugin) -> None:
        """Load a plugin instance."""
        async with self._lock:
            name = plugin.info.name

            if name in self._plugins:
                raise ValueError(f"Plugin '{name}' is already loaded")

            # Check dependencies
            for dep in plugin.info.dependencies:
                if dep not in self._plugins:
                    raise ValueError(
                        f"Plugin '{name}' requires '{dep}' which is not loaded"
                    )

            await plugin.on_load()
            self._plugins[name] = plugin

            logger.info(
                "Plugin loaded",
                name=name,
                version=plugin.info.version,
            )

    async def load_plugin_class(
        self,
        plugin_class: Type[Plugin],
        config: Optional[Dict[str, Any]] = None,
    ) -> Plugin:
        """Load a plugin from a class."""
        plugin = plugin_class()
        await self.load_plugin(plugin)
        return plugin

    async def load_from_module(self, module_path: str) -> List[Plugin]:
        """Load plugins from a Python module."""
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            logger.error(
                "Failed to import plugin module",
                module=module_path,
                error=str(e),
            )
            return []

        loaded = []
        for name in dir(module):
            obj = getattr(module, name)
            if (
                isinstance(obj, type)
                and issubclass(obj, Plugin)
                and obj is not Plugin
                and not name.startswith("_")
            ):
                try:
                    plugin = obj()
                    await self.load_plugin(plugin)
                    loaded.append(plugin)
                except Exception as e:
                    logger.error(
                        "Failed to load plugin",
                        plugin=name,
                        error=str(e),
                    )

        return loaded

    async def load_from_directory(self, directory: str) -> List[Plugin]:
        """Load all plugins from a directory."""
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.warning("Plugin directory does not exist", directory=directory)
            return []

        loaded = []
        for file_path in dir_path.glob("*.py"):
            if file_path.name.startswith("_"):
                continue

            module_name = f"agentmesh_plugins.{file_path.stem}"

            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                for name in dir(module):
                    obj = getattr(module, name)
                    if (
                        isinstance(obj, type)
                        and issubclass(obj, Plugin)
                        and obj is not Plugin
                    ):
                        try:
                            plugin = obj()
                            await self.load_plugin(plugin)
                            loaded.append(plugin)
                        except Exception as e:
                            logger.error(
                                "Failed to load plugin",
                                file=str(file_path),
                                plugin=name,
                                error=str(e),
                            )

        return loaded

    async def enable_plugin(self, name: str) -> None:
        """Enable a plugin."""
        async with self._lock:
            if name not in self._plugins:
                raise ValueError(f"Plugin '{name}' is not loaded")

            plugin = self._plugins[name]
            if plugin.enabled:
                return

            await plugin.on_enable()
            plugin._enabled = True

            # Register hooks
            for hook in plugin.get_hooks():
                if hook.hook_type not in self._hooks:
                    self._hooks[hook.hook_type] = []
                self._hooks[hook.hook_type].append(hook)
                self._hooks[hook.hook_type].sort(key=lambda h: -h.priority)

            logger.info("Plugin enabled", name=name)

    async def disable_plugin(self, name: str) -> None:
        """Disable a plugin."""
        async with self._lock:
            if name not in self._plugins:
                raise ValueError(f"Plugin '{name}' is not loaded")

            plugin = self._plugins[name]
            if not plugin.enabled:
                return

            # Unregister hooks
            for hook in plugin.get_hooks():
                if hook.hook_type in self._hooks:
                    self._hooks[hook.hook_type] = [
                        h for h in self._hooks[hook.hook_type] if h != hook
                    ]

            await plugin.on_disable()
            plugin._enabled = False

            logger.info("Plugin disabled", name=name)

    async def unload_plugin(self, name: str) -> None:
        """Unload a plugin."""
        async with self._lock:
            if name not in self._plugins:
                raise ValueError(f"Plugin '{name}' is not loaded")

            plugin = self._plugins[name]

            if plugin.enabled:
                await self.disable_plugin(name)

            await plugin.on_unload()
            del self._plugins[name]

            logger.info("Plugin unloaded", name=name)

    def get_plugin(self, name: str) -> Optional[Plugin]:
        """Get a plugin by name."""
        return self._plugins.get(name)

    def get_plugins(self) -> Dict[str, Plugin]:
        """Get all loaded plugins."""
        return dict(self._plugins)

    def get_enabled_plugins(self) -> Dict[str, Plugin]:
        """Get all enabled plugins."""
        return {
            name: plugin
            for name, plugin in self._plugins.items()
            if plugin.enabled
        }

    async def execute_hook(
        self,
        hook_type: HookType,
        *args,
        **kwargs,
    ) -> List[Any]:
        """Execute all hooks of a given type."""
        if hook_type not in self._hooks:
            return []

        results = []
        for hook in self._hooks[hook_type]:
            try:
                result = await hook.execute(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(
                    "Hook execution error",
                    hook_type=hook_type.name,
                    error=str(e),
                )

        return results

    async def execute_hook_chain(
        self,
        hook_type: HookType,
        initial_value: Any,
        *args,
        **kwargs,
    ) -> Any:
        """
        Execute hooks in a chain, passing result to next hook.

        Each hook receives the result of the previous hook.
        """
        if hook_type not in self._hooks:
            return initial_value

        value = initial_value
        for hook in self._hooks[hook_type]:
            try:
                value = await hook.execute(value, *args, **kwargs)
            except Exception as e:
                logger.error(
                    "Hook chain error",
                    hook_type=hook_type.name,
                    error=str(e),
                )
                # Continue with current value

        return value

    def get_storage_plugins(self) -> List[StoragePlugin]:
        """Get all storage plugins."""
        return [
            p for p in self._plugins.values()
            if isinstance(p, StoragePlugin) and p.enabled
        ]

    def get_conflict_resolver_plugins(self) -> List[ConflictResolverPlugin]:
        """Get all conflict resolver plugins."""
        return [
            p for p in self._plugins.values()
            if isinstance(p, ConflictResolverPlugin) and p.enabled
        ]

    def get_agent_plugins(self) -> List[AgentPlugin]:
        """Get all agent plugins."""
        return [
            p for p in self._plugins.values()
            if isinstance(p, AgentPlugin) and p.enabled
        ]

    def get_tool_plugins(self) -> List[ToolPlugin]:
        """Get all tool plugins."""
        return [
            p for p in self._plugins.values()
            if isinstance(p, ToolPlugin) and p.enabled
        ]


# Global plugin manager
_global_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager."""
    global _global_plugin_manager
    if _global_plugin_manager is None:
        _global_plugin_manager = PluginManager()
    return _global_plugin_manager


# Example plugins for demonstration
class ExampleLoggingPlugin(Plugin):
    """Example plugin that logs transaction events."""

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="example-logging",
            version="1.0.0",
            description="Logs all transaction events",
            author="AgentMesh-STM",
        )

    async def on_enable(self) -> None:
        # Register hooks
        self.register_hook(
            HookType.POST_TRANSACTION_BEGIN,
            self._log_begin,
            priority=0,
        )
        self.register_hook(
            HookType.POST_TRANSACTION_COMMIT,
            self._log_commit,
            priority=0,
        )

    async def _log_begin(self, transaction_id: str, **kwargs) -> None:
        logger.info(f"[Plugin] Transaction started: {transaction_id}")

    async def _log_commit(self, transaction_id: str, **kwargs) -> None:
        logger.info(f"[Plugin] Transaction committed: {transaction_id}")


class ExampleMetricsPlugin(Plugin):
    """Example plugin that collects metrics."""

    def __init__(self):
        super().__init__()
        self.transaction_count = 0
        self.commit_count = 0
        self.abort_count = 0

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="example-metrics",
            version="1.0.0",
            description="Collects transaction metrics",
            author="AgentMesh-STM",
        )

    async def on_enable(self) -> None:
        self.register_hook(
            HookType.POST_TRANSACTION_BEGIN,
            self._count_transaction,
        )
        self.register_hook(
            HookType.POST_TRANSACTION_COMMIT,
            self._count_commit,
        )
        self.register_hook(
            HookType.POST_TRANSACTION_ABORT,
            self._count_abort,
        )

    async def _count_transaction(self, **kwargs) -> None:
        self.transaction_count += 1

    async def _count_commit(self, **kwargs) -> None:
        self.commit_count += 1

    async def _count_abort(self, **kwargs) -> None:
        self.abort_count += 1

    def get_metrics(self) -> Dict[str, int]:
        return {
            "transactions": self.transaction_count,
            "commits": self.commit_count,
            "aborts": self.abort_count,
        }

"""
Configuration System for AgentMesh-STM.

This module provides a flexible configuration system supporting:
- YAML configuration files
- Environment variable overrides
- Programmatic configuration
- Configuration validation
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator

from agentmesh_stm.utils.logging import get_logger

logger = get_logger(__name__)


class StorageConfig(BaseModel):
    """Configuration for storage backend."""

    backend: str = Field(
        default="memory",
        description="Storage backend: 'memory', 'sqlite', 'lmdb'",
    )
    path: Optional[str] = Field(
        default=None,
        description="Path for persistent storage backends",
    )
    max_versions: int = Field(
        default=100,
        description="Maximum versions to keep per resource",
    )
    gc_interval_seconds: int = Field(
        default=300,
        description="Interval between garbage collection runs",
    )
    # LMDB specific
    lmdb_map_size: int = Field(
        default=10 * 1024 * 1024 * 1024,  # 10GB
        description="LMDB map size in bytes",
    )
    # SQLite specific
    sqlite_journal_mode: str = Field(
        default="WAL",
        description="SQLite journal mode",
    )


class TransactionConfig(BaseModel):
    """Configuration for transactions."""

    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts on conflict",
    )
    retry_delay_ms: int = Field(
        default=100,
        description="Base delay between retries in milliseconds",
    )
    retry_backoff: float = Field(
        default=1.5,
        description="Exponential backoff multiplier",
    )
    timeout_seconds: Optional[float] = Field(
        default=300.0,
        description="Transaction timeout in seconds",
    )
    isolation_level: str = Field(
        default="snapshot",
        description="Isolation level: 'snapshot' or 'serializable'",
    )
    enable_compensation: bool = Field(
        default=True,
        description="Enable compensation for side effects",
    )


class ConflictConfig(BaseModel):
    """Configuration for conflict detection."""

    detection_level: str = Field(
        default="semantic",
        description="Detection level: 'version', 'line', 'ast', 'semantic'",
    )
    enable_llm_resolution: bool = Field(
        default=False,
        description="Enable LLM-based conflict resolution",
    )
    llm_provider: str = Field(
        default="openai",
        description="LLM provider for resolution",
    )
    llm_model: str = Field(
        default="gpt-4o",
        description="LLM model for resolution",
    )
    merge_strategy: str = Field(
        default="auto",
        description="Default merge strategy",
    )


class LoggingConfig(BaseModel):
    """Configuration for transaction logging."""

    enabled: bool = Field(
        default=True,
        description="Enable Write-Ahead Logging",
    )
    log_dir: str = Field(
        default="./wal",
        description="Directory for WAL files",
    )
    max_file_size_mb: int = Field(
        default=64,
        description="Maximum log file size in MB",
    )
    checkpoint_interval: int = Field(
        default=1000,
        description="Records between checkpoints",
    )
    sync_mode: str = Field(
        default="fsync",
        description="Sync mode: 'none', 'fsync', 'fdatasync'",
    )


class DistributedConfig(BaseModel):
    """Configuration for distributed coordination."""

    enabled: bool = Field(
        default=False,
        description="Enable distributed mode",
    )
    node_id: Optional[str] = Field(
        default=None,
        description="Unique node identifier",
    )
    coordinator_address: Optional[str] = Field(
        default=None,
        description="Coordinator server address",
    )
    heartbeat_interval_ms: int = Field(
        default=1000,
        description="Heartbeat interval in milliseconds",
    )
    consensus_protocol: str = Field(
        default="2pc",
        description="Consensus protocol: '2pc', 'paxos', 'optimistic'",
    )


class LLMAgentConfig(BaseModel):
    """Configuration for LLM agents."""

    provider: str = Field(
        default="openai",
        description="LLM provider: 'openai', 'anthropic'",
    )
    model: str = Field(
        default="gpt-4o",
        description="Model to use",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key (can also use env var)",
    )
    temperature: float = Field(
        default=0.7,
        description="Sampling temperature",
    )
    max_tokens: int = Field(
        default=4096,
        description="Maximum tokens in response",
    )
    max_iterations: int = Field(
        default=10,
        description="Maximum tool use iterations",
    )
    enable_advanced_tools: bool = Field(
        default=True,
        description="Enable advanced tools (git, code analysis)",
    )


class MetricsConfig(BaseModel):
    """Configuration for metrics collection."""

    enabled: bool = Field(
        default=True,
        description="Enable metrics collection",
    )
    export_format: str = Field(
        default="json",
        description="Export format: 'json', 'prometheus', 'csv'",
    )
    export_path: Optional[str] = Field(
        default=None,
        description="Path for metrics export",
    )
    collection_interval_seconds: int = Field(
        default=60,
        description="Interval between metric snapshots",
    )


class AgentMeshConfig(BaseModel):
    """Root configuration for AgentMesh-STM."""

    # Sub-configurations
    storage: StorageConfig = Field(default_factory=StorageConfig)
    transaction: TransactionConfig = Field(default_factory=TransactionConfig)
    conflict: ConflictConfig = Field(default_factory=ConflictConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    distributed: DistributedConfig = Field(default_factory=DistributedConfig)
    llm_agent: LLMAgentConfig = Field(default_factory=LLMAgentConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)

    # Global settings
    working_directory: str = Field(
        default=".",
        description="Working directory for file operations",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "AgentMeshConfig":
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        # Apply environment variable overrides
        data = cls._apply_env_overrides(data)

        return cls(**data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMeshConfig":
        """Create configuration from a dictionary."""
        data = cls._apply_env_overrides(data)
        return cls(**data)

    @classmethod
    def _apply_env_overrides(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        env_mappings = {
            "AGENTMESH_LOG_LEVEL": ("log_level",),
            "AGENTMESH_WORKING_DIR": ("working_directory",),
            # Storage
            "AGENTMESH_STORAGE_BACKEND": ("storage", "backend"),
            "AGENTMESH_STORAGE_PATH": ("storage", "path"),
            # Transaction
            "AGENTMESH_TXN_MAX_RETRIES": ("transaction", "max_retries"),
            "AGENTMESH_TXN_TIMEOUT": ("transaction", "timeout_seconds"),
            # LLM
            "OPENAI_API_KEY": ("llm_agent", "api_key"),
            "ANTHROPIC_API_KEY": ("llm_agent", "api_key"),
            "AGENTMESH_LLM_PROVIDER": ("llm_agent", "provider"),
            "AGENTMESH_LLM_MODEL": ("llm_agent", "model"),
            # Distributed
            "AGENTMESH_NODE_ID": ("distributed", "node_id"),
            "AGENTMESH_COORDINATOR": ("distributed", "coordinator_address"),
        }

        for env_var, path in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                cls._set_nested(data, path, value)

        return data

    @staticmethod
    def _set_nested(data: Dict, path: tuple, value: Any) -> None:
        """Set a nested dictionary value."""
        for key in path[:-1]:
            data = data.setdefault(key, {})
        # Convert type if needed
        if path[-1] in ("max_retries", "max_tokens", "max_iterations"):
            value = int(value)
        elif path[-1] in ("timeout_seconds", "temperature"):
            value = float(value)
        data[path[-1]] = value

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()


class ConfigManager:
    """
    Manages configuration loading and access.

    Supports:
    - Loading from YAML files
    - Environment variable overrides
    - Default configuration
    - Configuration validation
    """

    DEFAULT_CONFIG_PATHS = [
        Path("agentmesh.yaml"),
        Path("agentmesh.yml"),
        Path(".agentmesh/config.yaml"),
        Path.home() / ".agentmesh" / "config.yaml",
    ]

    def __init__(self, config: Optional[AgentMeshConfig] = None):
        self._config = config

    @property
    def config(self) -> AgentMeshConfig:
        """Get the current configuration."""
        if self._config is None:
            self._config = self.load()
        return self._config

    def load(
        self,
        path: Optional[Union[str, Path]] = None,
    ) -> AgentMeshConfig:
        """
        Load configuration from file or use defaults.

        Args:
            path: Explicit path to config file

        Returns:
            Loaded configuration
        """
        if path:
            logger.info("Loading config from specified path", path=str(path))
            self._config = AgentMeshConfig.from_yaml(path)
            return self._config

        # Try default paths
        for default_path in self.DEFAULT_CONFIG_PATHS:
            if default_path.exists():
                logger.info("Loading config from default path", path=str(default_path))
                self._config = AgentMeshConfig.from_yaml(default_path)
                return self._config

        # Use defaults
        logger.info("Using default configuration")
        self._config = AgentMeshConfig()
        return self._config

    def save(self, path: Union[str, Path]) -> None:
        """Save current configuration to file."""
        if self._config is None:
            self._config = AgentMeshConfig()
        self._config.to_yaml(path)
        logger.info("Configuration saved", path=str(path))

    def update(self, **kwargs) -> AgentMeshConfig:
        """Update configuration with new values."""
        current = self.config.to_dict()
        self._deep_update(current, kwargs)
        self._config = AgentMeshConfig.from_dict(current)
        return self._config

    @staticmethod
    def _deep_update(base: Dict, updates: Dict) -> None:
        """Deep update a dictionary."""
        for key, value in updates.items():
            if isinstance(value, dict) and key in base:
                ConfigManager._deep_update(base[key], value)
            else:
                base[key] = value


# Global config manager instance
_config_manager: Optional[ConfigManager] = None


def get_config() -> AgentMeshConfig:
    """Get the global configuration."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager.config


def load_config(path: Optional[Union[str, Path]] = None) -> AgentMeshConfig:
    """Load configuration from file."""
    global _config_manager
    _config_manager = ConfigManager()
    return _config_manager.load(path)


def save_config(path: Union[str, Path]) -> None:
    """Save current configuration to file."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    _config_manager.save(path)


# Example YAML configuration template
EXAMPLE_CONFIG = '''
# AgentMesh-STM Configuration

# Global settings
working_directory: "."
log_level: "INFO"

# Storage configuration
storage:
  backend: "memory"  # memory, sqlite, lmdb
  path: null
  max_versions: 100
  gc_interval_seconds: 300

# Transaction settings
transaction:
  max_retries: 3
  retry_delay_ms: 100
  retry_backoff: 1.5
  timeout_seconds: 300
  isolation_level: "snapshot"
  enable_compensation: true

# Conflict detection
conflict:
  detection_level: "semantic"  # version, line, ast, semantic
  enable_llm_resolution: false
  llm_provider: "openai"
  llm_model: "gpt-4o"
  merge_strategy: "auto"

# Write-Ahead Logging
logging:
  enabled: true
  log_dir: "./wal"
  max_file_size_mb: 64
  checkpoint_interval: 1000
  sync_mode: "fsync"

# Distributed coordination
distributed:
  enabled: false
  node_id: null
  coordinator_address: null
  heartbeat_interval_ms: 1000
  consensus_protocol: "2pc"

# LLM Agent settings
llm_agent:
  provider: "openai"
  model: "gpt-4o"
  api_key: null  # Use OPENAI_API_KEY env var
  temperature: 0.7
  max_tokens: 4096
  max_iterations: 10
  enable_advanced_tools: true

# Metrics collection
metrics:
  enabled: true
  export_format: "json"
  export_path: null
  collection_interval_seconds: 60
'''


def generate_example_config(path: Union[str, Path]) -> None:
    """Generate an example configuration file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(EXAMPLE_CONFIG)
    logger.info("Generated example config", path=str(path))

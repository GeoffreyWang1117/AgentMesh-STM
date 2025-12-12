"""
AgentMesh-STM: Software Transactional Memory based Multi-Agent Coordination Framework

This framework provides STM-based coordination for multiple LLM agents working on shared
resources. It handles long-running transactions, semantic conflict detection, and
compensation-based rollback for irreversible operations.

Core Components:
- Transaction: Wraps agent operations as explicit transactions
- MVCCStorage: Multi-version concurrency control for resource versioning
- ConflictDetector: Hierarchical semantic conflict detection
- CompensationManager: Handles rollback of side effects
- LLMAgent: LLM-powered agents with tool support
- WriteAheadLog: Transaction durability and recovery
- LLMConflictResolver: AI-powered conflict resolution
"""

from agentmesh_stm.core.transaction import (
    Transaction,
    TransactionContext,
    TransactionManager,
    TransactionState,
)
from agentmesh_stm.core.logging import (
    WriteAheadLog,
    TransactionLogger,
    RecoveryManager,
)
from agentmesh_stm.storage.mvcc import MVCCStorage, ResourceVersion
from agentmesh_stm.conflict.detector import ConflictDetector, ConflictResult, ConflictType
from agentmesh_stm.conflict.llm_resolver import LLMConflictResolver, MergeStrategy
from agentmesh_stm.compensation.manager import (
    CompensationManager,
    CompensableOperation,
    CompensationLog,
)
from agentmesh_stm.agent.base import Agent, AgentTask, AgentResult
from agentmesh_stm.agent.llm_agent import LLMAgent, LLMConfig
from agentmesh_stm.config import AgentMeshConfig, load_config
from agentmesh_stm.monitoring import MetricsCollector, create_dashboard

__version__ = "0.2.0"
__all__ = [
    # Transaction
    "Transaction",
    "TransactionContext",
    "TransactionManager",
    "TransactionState",
    # Storage
    "MVCCStorage",
    "ResourceVersion",
    # Conflict Detection
    "ConflictDetector",
    "ConflictResult",
    "ConflictType",
    "LLMConflictResolver",
    "MergeStrategy",
    # Compensation
    "CompensationManager",
    "CompensableOperation",
    "CompensationLog",
    # Agent
    "Agent",
    "AgentTask",
    "AgentResult",
    "LLMAgent",
    "LLMConfig",
    # Durability
    "WriteAheadLog",
    "TransactionLogger",
    "RecoveryManager",
    # Configuration
    "AgentMeshConfig",
    "load_config",
    # Monitoring
    "MetricsCollector",
    "create_dashboard",
]

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
"""

from agentmesh_stm.core.transaction import (
    Transaction,
    TransactionContext,
    TransactionManager,
    TransactionState,
)
from agentmesh_stm.storage.mvcc import MVCCStorage, ResourceVersion
from agentmesh_stm.conflict.detector import ConflictDetector, ConflictResult, ConflictType
from agentmesh_stm.compensation.manager import (
    CompensationManager,
    CompensableOperation,
    CompensationLog,
)
from agentmesh_stm.agent.base import Agent, AgentTask, AgentResult

__version__ = "0.1.0"
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
    # Compensation
    "CompensationManager",
    "CompensableOperation",
    "CompensationLog",
    # Agent
    "Agent",
    "AgentTask",
    "AgentResult",
]

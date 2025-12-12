"""
Distributed Transaction Coordinator for AgentMesh-STM.

This module provides coordination for transactions across multiple
processes or nodes, enabling distributed multi-agent collaboration.

Features:
- Multi-process transaction coordination
- Distributed conflict detection
- Two-phase commit protocol
- Failure recovery
"""

from __future__ import annotations

import asyncio
import json
import pickle
import socket
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field

from agentmesh_stm.core.transaction import (
    Transaction,
    TransactionConfig,
    TransactionManager,
    TransactionState,
)
from agentmesh_stm.storage.mvcc import MVCCStorage
from agentmesh_stm.utils.logging import get_logger

logger = get_logger(__name__)


class NodeState(Enum):
    """State of a coordinator node."""

    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()
    FAILED = auto()


class TransactionPhase(Enum):
    """Phases in distributed transaction commit."""

    PREPARE = auto()
    VOTE = auto()
    COMMIT = auto()
    ABORT = auto()
    COMPLETE = auto()


@dataclass
class NodeInfo:
    """Information about a coordinator node."""

    node_id: str
    host: str
    port: int
    state: NodeState = NodeState.STOPPED
    last_heartbeat: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def address(self) -> str:
        """Get the node address."""
        return f"{self.host}:{self.port}"

    def is_alive(self, timeout_seconds: float = 30.0) -> bool:
        """Check if node is considered alive."""
        if self.state not in (NodeState.RUNNING, NodeState.STARTING):
            return False
        if self.last_heartbeat is None:
            return False
        elapsed = (datetime.utcnow() - self.last_heartbeat).total_seconds()
        return elapsed < timeout_seconds


@dataclass
class DistributedTransaction:
    """A transaction distributed across multiple nodes."""

    transaction_id: str
    coordinator_id: str
    participants: List[str]  # Node IDs
    phase: TransactionPhase = TransactionPhase.PREPARE
    votes: Dict[str, bool] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.utcnow)
    timeout: timedelta = field(default_factory=lambda: timedelta(seconds=60))
    read_set: Dict[str, int] = field(default_factory=dict)  # resource_id -> version
    write_set: Dict[str, str] = field(default_factory=dict)  # resource_id -> content

    @property
    def is_timed_out(self) -> bool:
        """Check if transaction has timed out."""
        return datetime.utcnow() - self.start_time > self.timeout

    @property
    def all_voted(self) -> bool:
        """Check if all participants have voted."""
        return set(self.votes.keys()) == set(self.participants)

    @property
    def commit_decision(self) -> bool:
        """Determine commit decision based on votes."""
        if not self.all_voted:
            return False
        return all(self.votes.values())


class Message:
    """Base class for coordinator messages."""

    def __init__(self, msg_type: str, sender_id: str, **data):
        self.msg_type = msg_type
        self.sender_id = sender_id
        self.timestamp = datetime.utcnow()
        self.data = data

    def serialize(self) -> bytes:
        """Serialize message to bytes."""
        return pickle.dumps({
            "msg_type": self.msg_type,
            "sender_id": self.sender_id,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
        })

    @classmethod
    def deserialize(cls, data: bytes) -> "Message":
        """Deserialize message from bytes."""
        parsed = pickle.loads(data)
        msg = cls(parsed["msg_type"], parsed["sender_id"], **parsed["data"])
        msg.timestamp = datetime.fromisoformat(parsed["timestamp"])
        return msg


class CoordinatorConfig(BaseModel):
    """Configuration for the distributed coordinator."""

    node_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    host: str = Field(default="localhost")
    port: int = Field(default=5555)
    heartbeat_interval: float = Field(default=5.0)
    transaction_timeout: float = Field(default=60.0)
    max_retries: int = Field(default=3)
    enable_persistence: bool = Field(default=True)
    log_dir: str = Field(default=".agentmesh/distributed")


class DistributedCoordinator:
    """
    Coordinates distributed transactions across multiple nodes.

    The coordinator implements:
    - Node discovery and membership
    - Distributed transaction management
    - Two-phase commit protocol
    - Failure detection and recovery
    """

    def __init__(
        self,
        config: Optional[CoordinatorConfig] = None,
        storage: Optional[MVCCStorage] = None,
        transaction_manager: Optional[TransactionManager] = None,
    ):
        self._config = config or CoordinatorConfig()
        self._storage = storage
        self._transaction_manager = transaction_manager

        self._node_info = NodeInfo(
            node_id=self._config.node_id,
            host=self._config.host,
            port=self._config.port,
        )

        self._peers: Dict[str, NodeInfo] = {}
        self._active_transactions: Dict[str, DistributedTransaction] = {}
        self._lock = asyncio.Lock()

        self._server: Optional[asyncio.Server] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._running = False

        self._message_handlers: Dict[str, Callable] = {
            "prepare": self._handle_prepare,
            "vote": self._handle_vote,
            "commit": self._handle_commit,
            "abort": self._handle_abort,
            "heartbeat": self._handle_heartbeat,
            "join": self._handle_join,
            "leave": self._handle_leave,
        }

    @property
    def node_id(self) -> str:
        """Get this node's ID."""
        return self._config.node_id

    @property
    def is_running(self) -> bool:
        """Check if coordinator is running."""
        return self._running

    async def start(self) -> None:
        """Start the coordinator."""
        if self._running:
            return

        logger.info("Starting distributed coordinator", node_id=self.node_id)

        self._node_info.state = NodeState.STARTING

        # Start server
        self._server = await asyncio.start_server(
            self._handle_connection,
            self._config.host,
            self._config.port,
        )

        # Start heartbeat
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        self._node_info.state = NodeState.RUNNING
        self._node_info.last_heartbeat = datetime.utcnow()
        self._running = True

        logger.info(
            "Coordinator started",
            node_id=self.node_id,
            address=self._node_info.address,
        )

    async def stop(self) -> None:
        """Stop the coordinator."""
        if not self._running:
            return

        logger.info("Stopping distributed coordinator", node_id=self.node_id)

        self._node_info.state = NodeState.STOPPING
        self._running = False

        # Notify peers
        await self._broadcast(Message("leave", self.node_id))

        # Cancel heartbeat
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Close server
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        self._node_info.state = NodeState.STOPPED
        logger.info("Coordinator stopped", node_id=self.node_id)

    async def join_cluster(self, seed_host: str, seed_port: int) -> bool:
        """Join an existing cluster via a seed node."""
        logger.info(
            "Joining cluster",
            seed=f"{seed_host}:{seed_port}",
            node_id=self.node_id,
        )

        try:
            # Connect to seed node
            reader, writer = await asyncio.open_connection(seed_host, seed_port)

            # Send join message
            msg = Message(
                "join",
                self.node_id,
                host=self._config.host,
                port=self._config.port,
            )
            await self._send_message(writer, msg)

            # Wait for response
            response = await self._receive_message(reader)

            writer.close()
            await writer.wait_closed()

            if response and response.data.get("accepted"):
                # Add known peers
                for peer_data in response.data.get("peers", []):
                    peer = NodeInfo(
                        node_id=peer_data["node_id"],
                        host=peer_data["host"],
                        port=peer_data["port"],
                        state=NodeState.RUNNING,
                        last_heartbeat=datetime.utcnow(),
                    )
                    self._peers[peer.node_id] = peer

                logger.info("Joined cluster", peers=len(self._peers))
                return True

            return False

        except Exception as e:
            logger.error("Failed to join cluster", error=str(e))
            return False

    async def begin_distributed_transaction(
        self,
        participants: Optional[List[str]] = None,
    ) -> DistributedTransaction:
        """Begin a new distributed transaction."""
        if participants is None:
            participants = list(self._peers.keys())

        dtxn = DistributedTransaction(
            transaction_id=str(uuid4()),
            coordinator_id=self.node_id,
            participants=participants,
            timeout=timedelta(seconds=self._config.transaction_timeout),
        )

        async with self._lock:
            self._active_transactions[dtxn.transaction_id] = dtxn

        logger.debug(
            "Started distributed transaction",
            txn_id=dtxn.transaction_id,
            participants=len(participants),
        )

        return dtxn

    async def prepare(
        self,
        dtxn: DistributedTransaction,
        read_set: Dict[str, int],
        write_set: Dict[str, str],
    ) -> bool:
        """
        Prepare phase of two-phase commit.

        Sends prepare message to all participants and collects votes.
        """
        dtxn.read_set = read_set
        dtxn.write_set = write_set
        dtxn.phase = TransactionPhase.PREPARE

        # Send prepare to all participants
        prepare_msg = Message(
            "prepare",
            self.node_id,
            txn_id=dtxn.transaction_id,
            read_set=read_set,
            write_set=write_set,
        )

        await self._broadcast_to(prepare_msg, dtxn.participants)

        # Wait for votes
        dtxn.phase = TransactionPhase.VOTE

        # Wait with timeout
        deadline = datetime.utcnow() + dtxn.timeout
        while not dtxn.all_voted and datetime.utcnow() < deadline:
            await asyncio.sleep(0.1)

        return dtxn.commit_decision

    async def commit(self, dtxn: DistributedTransaction) -> bool:
        """
        Commit phase of two-phase commit.

        Sends commit message to all participants.
        """
        if not dtxn.commit_decision:
            return await self.abort(dtxn)

        dtxn.phase = TransactionPhase.COMMIT

        # Apply local changes
        if self._storage and dtxn.write_set:
            await self._storage.atomic_write_batch(
                dtxn.transaction_id,
                [(k, v, "write") for k, v in dtxn.write_set.items()],
            )

        # Send commit to all participants
        commit_msg = Message(
            "commit",
            self.node_id,
            txn_id=dtxn.transaction_id,
        )

        await self._broadcast_to(commit_msg, dtxn.participants)

        dtxn.phase = TransactionPhase.COMPLETE

        async with self._lock:
            self._active_transactions.pop(dtxn.transaction_id, None)

        logger.debug("Committed distributed transaction", txn_id=dtxn.transaction_id)
        return True

    async def abort(self, dtxn: DistributedTransaction) -> bool:
        """Abort a distributed transaction."""
        dtxn.phase = TransactionPhase.ABORT

        # Send abort to all participants
        abort_msg = Message(
            "abort",
            self.node_id,
            txn_id=dtxn.transaction_id,
        )

        await self._broadcast_to(abort_msg, dtxn.participants)

        dtxn.phase = TransactionPhase.COMPLETE

        async with self._lock:
            self._active_transactions.pop(dtxn.transaction_id, None)

        logger.debug("Aborted distributed transaction", txn_id=dtxn.transaction_id)
        return True

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle incoming connection."""
        try:
            while self._running:
                msg = await self._receive_message(reader)
                if not msg:
                    break

                handler = self._message_handlers.get(msg.msg_type)
                if handler:
                    response = await handler(msg)
                    if response:
                        await self._send_message(writer, response)
                else:
                    logger.warning("Unknown message type", msg_type=msg.msg_type)

        except Exception as e:
            logger.error("Connection error", error=str(e))
        finally:
            writer.close()
            await writer.wait_closed()

    async def _handle_prepare(self, msg: Message) -> Optional[Message]:
        """Handle prepare message."""
        txn_id = msg.data["txn_id"]
        read_set = msg.data["read_set"]
        write_set = msg.data["write_set"]

        # Check for conflicts
        can_commit = await self._check_conflicts(read_set)

        # Send vote
        vote_msg = Message(
            "vote",
            self.node_id,
            txn_id=txn_id,
            vote=can_commit,
        )

        # Send vote back to coordinator
        await self._send_to_node(msg.sender_id, vote_msg)

        return None

    async def _handle_vote(self, msg: Message) -> Optional[Message]:
        """Handle vote message."""
        txn_id = msg.data["txn_id"]
        vote = msg.data["vote"]
        voter_id = msg.sender_id

        async with self._lock:
            if txn_id in self._active_transactions:
                self._active_transactions[txn_id].votes[voter_id] = vote

        return None

    async def _handle_commit(self, msg: Message) -> Optional[Message]:
        """Handle commit message."""
        txn_id = msg.data["txn_id"]

        # Apply changes locally
        async with self._lock:
            if txn_id in self._active_transactions:
                dtxn = self._active_transactions[txn_id]
                if self._storage and dtxn.write_set:
                    await self._storage.atomic_write_batch(
                        txn_id,
                        [(k, v, "write") for k, v in dtxn.write_set.items()],
                    )
                self._active_transactions.pop(txn_id, None)

        return None

    async def _handle_abort(self, msg: Message) -> Optional[Message]:
        """Handle abort message."""
        txn_id = msg.data["txn_id"]

        async with self._lock:
            self._active_transactions.pop(txn_id, None)

        return None

    async def _handle_heartbeat(self, msg: Message) -> Optional[Message]:
        """Handle heartbeat message."""
        sender_id = msg.sender_id

        async with self._lock:
            if sender_id in self._peers:
                self._peers[sender_id].last_heartbeat = datetime.utcnow()
                self._peers[sender_id].state = NodeState.RUNNING

        return None

    async def _handle_join(self, msg: Message) -> Optional[Message]:
        """Handle join message."""
        new_node = NodeInfo(
            node_id=msg.sender_id,
            host=msg.data["host"],
            port=msg.data["port"],
            state=NodeState.RUNNING,
            last_heartbeat=datetime.utcnow(),
        )

        async with self._lock:
            self._peers[new_node.node_id] = new_node

        # Send response with peer list
        peers_data = [
            {"node_id": p.node_id, "host": p.host, "port": p.port}
            for p in self._peers.values()
        ]
        peers_data.append({
            "node_id": self.node_id,
            "host": self._config.host,
            "port": self._config.port,
        })

        return Message(
            "join_response",
            self.node_id,
            accepted=True,
            peers=peers_data,
        )

    async def _handle_leave(self, msg: Message) -> Optional[Message]:
        """Handle leave message."""
        async with self._lock:
            self._peers.pop(msg.sender_id, None)

        return None

    async def _check_conflicts(self, read_set: Dict[str, int]) -> bool:
        """Check for conflicts with local state."""
        if not self._storage:
            return True

        for resource_id, read_version in read_set.items():
            current_version = await self._storage.get_current_resource_version(resource_id)
            if current_version is not None and current_version > read_version:
                return False

        return True

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats."""
        while self._running:
            try:
                await asyncio.sleep(self._config.heartbeat_interval)

                self._node_info.last_heartbeat = datetime.utcnow()

                # Send heartbeat to all peers
                await self._broadcast(Message("heartbeat", self.node_id))

                # Check for failed peers
                async with self._lock:
                    failed_peers = [
                        peer_id for peer_id, peer in self._peers.items()
                        if not peer.is_alive()
                    ]
                    for peer_id in failed_peers:
                        self._peers[peer_id].state = NodeState.FAILED
                        logger.warning("Peer failed", peer_id=peer_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Heartbeat error", error=str(e))

    async def _broadcast(self, msg: Message) -> None:
        """Broadcast message to all peers."""
        await self._broadcast_to(msg, list(self._peers.keys()))

    async def _broadcast_to(self, msg: Message, node_ids: List[str]) -> None:
        """Broadcast message to specific nodes."""
        tasks = []
        for node_id in node_ids:
            if node_id != self.node_id and node_id in self._peers:
                tasks.append(self._send_to_node(node_id, msg))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_to_node(self, node_id: str, msg: Message) -> bool:
        """Send message to a specific node."""
        if node_id not in self._peers:
            return False

        peer = self._peers[node_id]

        try:
            reader, writer = await asyncio.open_connection(peer.host, peer.port)
            await self._send_message(writer, msg)
            writer.close()
            await writer.wait_closed()
            return True
        except Exception as e:
            logger.error("Failed to send message", peer=node_id, error=str(e))
            return False

    async def _send_message(
        self, writer: asyncio.StreamWriter, msg: Message
    ) -> None:
        """Send a message over a stream."""
        data = msg.serialize()
        length = struct.pack(">I", len(data))
        writer.write(length + data)
        await writer.drain()

    async def _receive_message(
        self, reader: asyncio.StreamReader
    ) -> Optional[Message]:
        """Receive a message from a stream."""
        try:
            length_data = await reader.readexactly(4)
            length = struct.unpack(">I", length_data)[0]
            data = await reader.readexactly(length)
            return Message.deserialize(data)
        except asyncio.IncompleteReadError:
            return None
        except Exception as e:
            logger.error("Failed to receive message", error=str(e))
            return None

    def get_cluster_status(self) -> Dict[str, Any]:
        """Get cluster status."""
        return {
            "node_id": self.node_id,
            "state": self._node_info.state.name,
            "address": self._node_info.address,
            "peers": [
                {
                    "node_id": p.node_id,
                    "address": p.address,
                    "state": p.state.name,
                    "alive": p.is_alive(),
                }
                for p in self._peers.values()
            ],
            "active_transactions": len(self._active_transactions),
        }

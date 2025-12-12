"""
Consensus Protocols for Distributed AgentMesh-STM.

This module implements consensus protocols for distributed
transaction coordination:
- Two-Phase Commit (2PC)
- Optimistic Replication
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

from agentmesh_stm.utils.logging import get_logger

logger = get_logger(__name__)


class VoteResult(Enum):
    """Result of a participant vote."""

    COMMIT = auto()
    ABORT = auto()
    TIMEOUT = auto()


@dataclass
class PrepareRequest:
    """Request to prepare a transaction."""

    transaction_id: str
    coordinator_id: str
    read_set: Dict[str, int]
    write_set: Dict[str, str]
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class VoteResponse:
    """Response to a prepare request."""

    transaction_id: str
    participant_id: str
    vote: VoteResult
    reason: Optional[str] = None


@dataclass
class DecisionNotification:
    """Notification of coordinator decision."""

    transaction_id: str
    decision: VoteResult
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ConsensusProtocol(ABC):
    """Abstract base class for consensus protocols."""

    @abstractmethod
    async def prepare(
        self,
        transaction_id: str,
        participants: List[str],
        read_set: Dict[str, int],
        write_set: Dict[str, str],
    ) -> Dict[str, VoteResponse]:
        """Run the prepare phase."""
        pass

    @abstractmethod
    async def decide(
        self,
        transaction_id: str,
        participants: List[str],
        votes: Dict[str, VoteResponse],
    ) -> VoteResult:
        """Make and broadcast decision."""
        pass

    @abstractmethod
    async def recover(self, transaction_id: str) -> Optional[VoteResult]:
        """Recover transaction state after failure."""
        pass


class TwoPhaseCommit(ConsensusProtocol):
    """
    Two-Phase Commit (2PC) Protocol.

    Phase 1 (Prepare):
    - Coordinator sends prepare request to all participants
    - Participants validate and vote COMMIT or ABORT
    - Participants wait for decision

    Phase 2 (Decide):
    - If all vote COMMIT, coordinator decides COMMIT
    - Otherwise, coordinator decides ABORT
    - Coordinator sends decision to all participants
    """

    def __init__(
        self,
        send_to_participant: Callable[[str, Any], asyncio.Future],
        timeout_seconds: float = 30.0,
    ):
        """
        Initialize 2PC protocol.

        Args:
            send_to_participant: Async function to send messages to participants
            timeout_seconds: Timeout for waiting for votes
        """
        self._send = send_to_participant
        self._timeout = timeout_seconds
        self._pending_transactions: Dict[str, Dict[str, VoteResponse]] = {}
        self._decisions: Dict[str, VoteResult] = {}

    async def prepare(
        self,
        transaction_id: str,
        participants: List[str],
        read_set: Dict[str, int],
        write_set: Dict[str, str],
    ) -> Dict[str, VoteResponse]:
        """
        Run prepare phase.

        Sends prepare requests to all participants and collects votes.
        """
        self._pending_transactions[transaction_id] = {}

        request = PrepareRequest(
            transaction_id=transaction_id,
            coordinator_id="coordinator",
            read_set=read_set,
            write_set=write_set,
        )

        # Send prepare to all participants concurrently
        tasks = []
        for participant_id in participants:
            task = asyncio.create_task(
                self._request_vote(participant_id, request)
            )
            tasks.append((participant_id, task))

        # Collect votes with timeout
        votes = {}
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*[t for _, t in tasks], return_exceptions=True),
                timeout=self._timeout,
            )

            for (participant_id, _), result in zip(tasks, results):
                if isinstance(result, VoteResponse):
                    votes[participant_id] = result
                elif isinstance(result, Exception):
                    votes[participant_id] = VoteResponse(
                        transaction_id=transaction_id,
                        participant_id=participant_id,
                        vote=VoteResult.ABORT,
                        reason=str(result),
                    )

        except asyncio.TimeoutError:
            logger.warning("Prepare phase timed out", txn_id=transaction_id)
            # Mark missing votes as timeout
            for participant_id in participants:
                if participant_id not in votes:
                    votes[participant_id] = VoteResponse(
                        transaction_id=transaction_id,
                        participant_id=participant_id,
                        vote=VoteResult.TIMEOUT,
                        reason="Timeout waiting for vote",
                    )

        self._pending_transactions[transaction_id] = votes
        return votes

    async def decide(
        self,
        transaction_id: str,
        participants: List[str],
        votes: Dict[str, VoteResponse],
    ) -> VoteResult:
        """
        Make and broadcast decision.

        Decision is COMMIT only if all participants voted COMMIT.
        """
        # Determine decision
        all_commit = all(
            v.vote == VoteResult.COMMIT
            for v in votes.values()
        )
        decision = VoteResult.COMMIT if all_commit else VoteResult.ABORT

        # Store decision for recovery
        self._decisions[transaction_id] = decision

        # Broadcast decision
        notification = DecisionNotification(
            transaction_id=transaction_id,
            decision=decision,
        )

        tasks = [
            self._send_decision(participant_id, notification)
            for participant_id in participants
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Clean up
        self._pending_transactions.pop(transaction_id, None)

        logger.info(
            "2PC decision",
            txn_id=transaction_id,
            decision=decision.name,
        )

        return decision

    async def recover(self, transaction_id: str) -> Optional[VoteResult]:
        """Recover transaction decision."""
        return self._decisions.get(transaction_id)

    async def _request_vote(
        self, participant_id: str, request: PrepareRequest
    ) -> VoteResponse:
        """Request vote from a participant."""
        try:
            response = await self._send(participant_id, {
                "type": "prepare",
                "transaction_id": request.transaction_id,
                "read_set": request.read_set,
                "write_set": request.write_set,
            })

            return VoteResponse(
                transaction_id=request.transaction_id,
                participant_id=participant_id,
                vote=VoteResult.COMMIT if response.get("vote") else VoteResult.ABORT,
                reason=response.get("reason"),
            )

        except Exception as e:
            return VoteResponse(
                transaction_id=request.transaction_id,
                participant_id=participant_id,
                vote=VoteResult.ABORT,
                reason=str(e),
            )

    async def _send_decision(
        self, participant_id: str, notification: DecisionNotification
    ) -> None:
        """Send decision to a participant."""
        try:
            await self._send(participant_id, {
                "type": "decision",
                "transaction_id": notification.transaction_id,
                "decision": notification.decision.name,
            })
        except Exception as e:
            logger.error(
                "Failed to send decision",
                participant=participant_id,
                error=str(e),
            )


class OptimisticReplication(ConsensusProtocol):
    """
    Optimistic Replication Protocol.

    Allows tentative commits that can be rolled back:
    1. Participants tentatively commit locally
    2. Coordinator validates globally
    3. If conflict detected, affected transactions rollback
    4. Eventually consistent state

    Better for high-latency, partition-tolerant scenarios.
    """

    def __init__(
        self,
        send_to_participant: Callable[[str, Any], asyncio.Future],
        conflict_detector: Callable[[Dict[str, int]], bool],
    ):
        """
        Initialize optimistic replication.

        Args:
            send_to_participant: Function to send messages
            conflict_detector: Function to detect conflicts
        """
        self._send = send_to_participant
        self._detect_conflict = conflict_detector
        self._tentative_commits: Dict[str, Dict[str, Any]] = {}
        self._confirmed: Set[str] = set()
        self._rolled_back: Set[str] = set()

    async def prepare(
        self,
        transaction_id: str,
        participants: List[str],
        read_set: Dict[str, int],
        write_set: Dict[str, str],
    ) -> Dict[str, VoteResponse]:
        """
        Tentatively commit on all participants.

        In optimistic replication, we always vote COMMIT
        and handle conflicts later.
        """
        # Store tentative commit
        self._tentative_commits[transaction_id] = {
            "participants": participants,
            "read_set": read_set,
            "write_set": write_set,
            "timestamp": datetime.utcnow(),
        }

        # Send tentative commit to all participants
        tasks = []
        for participant_id in participants:
            task = asyncio.create_task(
                self._tentative_commit(participant_id, transaction_id, write_set)
            )
            tasks.append((participant_id, task))

        # All tentatively commit
        votes = {}
        results = await asyncio.gather(*[t for _, t in tasks], return_exceptions=True)

        for (participant_id, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                votes[participant_id] = VoteResponse(
                    transaction_id=transaction_id,
                    participant_id=participant_id,
                    vote=VoteResult.ABORT,
                    reason=str(result),
                )
            else:
                votes[participant_id] = VoteResponse(
                    transaction_id=transaction_id,
                    participant_id=participant_id,
                    vote=VoteResult.COMMIT,
                )

        return votes

    async def decide(
        self,
        transaction_id: str,
        participants: List[str],
        votes: Dict[str, VoteResponse],
    ) -> VoteResult:
        """
        Validate and confirm or rollback.

        Check for conflicts with other tentative commits.
        """
        if transaction_id not in self._tentative_commits:
            return VoteResult.ABORT

        commit_info = self._tentative_commits[transaction_id]
        read_set = commit_info["read_set"]

        # Check for conflicts
        has_conflict = self._detect_conflict(read_set)

        if has_conflict:
            # Rollback on all participants
            await self._rollback(transaction_id, participants)
            self._rolled_back.add(transaction_id)
            return VoteResult.ABORT

        # Confirm on all participants
        await self._confirm(transaction_id, participants)
        self._confirmed.add(transaction_id)

        # Clean up
        self._tentative_commits.pop(transaction_id, None)

        return VoteResult.COMMIT

    async def recover(self, transaction_id: str) -> Optional[VoteResult]:
        """Recover transaction state."""
        if transaction_id in self._confirmed:
            return VoteResult.COMMIT
        if transaction_id in self._rolled_back:
            return VoteResult.ABORT
        if transaction_id in self._tentative_commits:
            # Still pending - need to re-decide
            return None
        return None

    async def _tentative_commit(
        self,
        participant_id: str,
        transaction_id: str,
        write_set: Dict[str, str],
    ) -> bool:
        """Send tentative commit to participant."""
        try:
            response = await self._send(participant_id, {
                "type": "tentative_commit",
                "transaction_id": transaction_id,
                "write_set": write_set,
            })
            return response.get("success", False)
        except Exception:
            return False

    async def _confirm(
        self,
        transaction_id: str,
        participants: List[str],
    ) -> None:
        """Confirm tentative commit on all participants."""
        tasks = [
            self._send(participant_id, {
                "type": "confirm",
                "transaction_id": transaction_id,
            })
            for participant_id in participants
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _rollback(
        self,
        transaction_id: str,
        participants: List[str],
    ) -> None:
        """Rollback tentative commit on all participants."""
        tasks = [
            self._send(participant_id, {
                "type": "rollback",
                "transaction_id": transaction_id,
            })
            for participant_id in participants
        ]
        await asyncio.gather(*tasks, return_exceptions=True)


class PaxosLike:
    """
    Simplified Paxos-like consensus for leader election.

    Used for coordinator election when the primary fails.
    """

    def __init__(
        self,
        node_id: str,
        send_fn: Callable[[str, Any], asyncio.Future],
    ):
        self._node_id = node_id
        self._send = send_fn
        self._current_leader: Optional[str] = None
        self._proposal_number = 0
        self._promised_number = 0
        self._accepted_value: Optional[str] = None

    async def propose_leader(
        self, candidate: str, voters: List[str]
    ) -> Optional[str]:
        """
        Propose a new leader.

        Returns elected leader or None if election failed.
        """
        self._proposal_number += 1
        proposal_num = (self._proposal_number, self._node_id)

        # Phase 1: Prepare
        promises = await self._prepare_phase(proposal_num, voters)

        if len(promises) <= len(voters) // 2:
            return None  # Failed to get majority

        # Check if any promise included an accepted value
        accepted_values = [
            p["accepted_value"]
            for p in promises
            if p.get("accepted_value")
        ]

        if accepted_values:
            # Use highest accepted value
            candidate = max(
                accepted_values,
                key=lambda x: x.get("proposal_num", (0, "")),
            )["value"]

        # Phase 2: Accept
        accepts = await self._accept_phase(proposal_num, candidate, voters)

        if len(accepts) > len(voters) // 2:
            self._current_leader = candidate
            return candidate

        return None

    async def _prepare_phase(
        self,
        proposal_num: tuple,
        voters: List[str],
    ) -> List[Dict[str, Any]]:
        """Run prepare phase."""
        tasks = [
            self._send(voter, {
                "type": "prepare",
                "proposal_num": proposal_num,
            })
            for voter in voters
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        promises = []
        for result in results:
            if isinstance(result, dict) and result.get("promised"):
                promises.append(result)

        return promises

    async def _accept_phase(
        self,
        proposal_num: tuple,
        value: str,
        voters: List[str],
    ) -> List[Dict[str, Any]]:
        """Run accept phase."""
        tasks = [
            self._send(voter, {
                "type": "accept",
                "proposal_num": proposal_num,
                "value": value,
            })
            for voter in voters
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        accepts = []
        for result in results:
            if isinstance(result, dict) and result.get("accepted"):
                accepts.append(result)

        return accepts

    def handle_prepare(self, proposal_num: tuple) -> Dict[str, Any]:
        """Handle prepare request."""
        if proposal_num > (self._promised_number, ""):
            self._promised_number = proposal_num[0]
            return {
                "promised": True,
                "accepted_value": {
                    "proposal_num": (self._promised_number, self._node_id),
                    "value": self._accepted_value,
                } if self._accepted_value else None,
            }
        return {"promised": False}

    def handle_accept(
        self, proposal_num: tuple, value: str
    ) -> Dict[str, Any]:
        """Handle accept request."""
        if proposal_num >= (self._promised_number, ""):
            self._promised_number = proposal_num[0]
            self._accepted_value = value
            return {"accepted": True}
        return {"accepted": False}

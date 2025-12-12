"""
Multi-Version Concurrency Control (MVCC) Storage for AgentMesh-STM.

This module provides versioned storage for resources with support for:
- Reading any historical version
- Copy-on-write for storage efficiency
- Atomic batch writes
- Version garbage collection
- Pluggable storage backends (in-memory, SQLite, LMDB)
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from agentmesh_stm.utils.hashing import content_hash


class StorageBackendType(Enum):
    """Available storage backend types."""

    MEMORY = auto()
    SQLITE = auto()
    LMDB = auto()


@dataclass
class ResourceVersion:
    """Represents a specific version of a resource."""

    resource_id: str
    version: int
    content: str
    content_hash: str
    timestamp: datetime
    transaction_id: Optional[str] = None
    is_deleted: bool = False

    @classmethod
    def create(
        cls,
        resource_id: str,
        version: int,
        content: str,
        transaction_id: Optional[str] = None,
        is_deleted: bool = False,
    ) -> "ResourceVersion":
        """Create a new resource version."""
        return cls(
            resource_id=resource_id,
            version=version,
            content=content,
            content_hash=content_hash(content),
            timestamp=datetime.utcnow(),
            transaction_id=transaction_id,
            is_deleted=is_deleted,
        )


@dataclass
class VersionedResource:
    """A resource with its version history."""

    resource_id: str
    current_version: int
    versions: Dict[int, ResourceVersion] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def add_version(self, version: ResourceVersion) -> None:
        """Add a new version to the history."""
        self.versions[version.version] = version
        if version.version > self.current_version:
            self.current_version = version.version
            self.updated_at = version.timestamp

    def get_version(self, version: int) -> Optional[ResourceVersion]:
        """Get a specific version."""
        return self.versions.get(version)

    def get_latest(self) -> Optional[ResourceVersion]:
        """Get the latest version."""
        return self.versions.get(self.current_version)

    def get_version_at_or_before(self, max_version: int) -> Optional[ResourceVersion]:
        """Get the latest version at or before the given version number."""
        available = [v for v in self.versions.keys() if v <= max_version]
        if not available:
            return None
        return self.versions.get(max(available))


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    async def get(self, resource_id: str) -> Optional[VersionedResource]:
        """Get a versioned resource by ID."""
        pass

    @abstractmethod
    async def put(self, resource: VersionedResource) -> None:
        """Store or update a versioned resource."""
        pass

    @abstractmethod
    async def delete(self, resource_id: str) -> bool:
        """Delete a resource and all its versions."""
        pass

    @abstractmethod
    async def list_resources(self) -> List[str]:
        """List all resource IDs."""
        pass

    @abstractmethod
    async def get_global_version(self) -> int:
        """Get the current global version number."""
        pass

    @abstractmethod
    async def increment_global_version(self) -> int:
        """Atomically increment and return the new global version."""
        pass


class InMemoryBackend(StorageBackend):
    """In-memory storage backend for testing and development."""

    def __init__(self):
        self._resources: Dict[str, VersionedResource] = {}
        self._global_version: int = 0
        self._lock = asyncio.Lock()

    async def get(self, resource_id: str) -> Optional[VersionedResource]:
        return self._resources.get(resource_id)

    async def put(self, resource: VersionedResource) -> None:
        self._resources[resource.resource_id] = resource

    async def delete(self, resource_id: str) -> bool:
        if resource_id in self._resources:
            del self._resources[resource_id]
            return True
        return False

    async def list_resources(self) -> List[str]:
        return list(self._resources.keys())

    async def get_global_version(self) -> int:
        return self._global_version

    async def increment_global_version(self) -> int:
        async with self._lock:
            self._global_version += 1
            return self._global_version


class SQLiteBackend(StorageBackend):
    """SQLite storage backend for persistence."""

    def __init__(self, db_path: str = ":memory:"):
        self._db_path = db_path
        self._connection = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the database schema."""
        import aiosqlite

        self._connection = await aiosqlite.connect(self._db_path)

        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS resources (
                resource_id TEXT PRIMARY KEY,
                current_version INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS resource_versions (
                resource_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                transaction_id TEXT,
                is_deleted INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (resource_id, version),
                FOREIGN KEY (resource_id) REFERENCES resources(resource_id)
            )
        """)

        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS global_state (
                key TEXT PRIMARY KEY,
                value INTEGER NOT NULL
            )
        """)

        # Initialize global version if not exists
        await self._connection.execute("""
            INSERT OR IGNORE INTO global_state (key, value) VALUES ('global_version', 0)
        """)

        await self._connection.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None

    async def get(self, resource_id: str) -> Optional[VersionedResource]:
        if not self._connection:
            await self.initialize()

        cursor = await self._connection.execute(
            "SELECT current_version, created_at, updated_at FROM resources WHERE resource_id = ?",
            (resource_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None

        resource = VersionedResource(
            resource_id=resource_id,
            current_version=row[0],
            created_at=datetime.fromisoformat(row[1]),
            updated_at=datetime.fromisoformat(row[2]),
        )

        # Load versions
        cursor = await self._connection.execute(
            """SELECT version, content, content_hash, timestamp, transaction_id, is_deleted
               FROM resource_versions WHERE resource_id = ?""",
            (resource_id,),
        )

        async for version_row in cursor:
            version = ResourceVersion(
                resource_id=resource_id,
                version=version_row[0],
                content=version_row[1],
                content_hash=version_row[2],
                timestamp=datetime.fromisoformat(version_row[3]),
                transaction_id=version_row[4],
                is_deleted=bool(version_row[5]),
            )
            resource.versions[version.version] = version

        return resource

    async def put(self, resource: VersionedResource) -> None:
        if not self._connection:
            await self.initialize()

        async with self._lock:
            # Upsert resource
            await self._connection.execute(
                """INSERT OR REPLACE INTO resources (resource_id, current_version, created_at, updated_at)
                   VALUES (?, ?, ?, ?)""",
                (
                    resource.resource_id,
                    resource.current_version,
                    resource.created_at.isoformat(),
                    resource.updated_at.isoformat(),
                ),
            )

            # Insert new versions
            for version in resource.versions.values():
                await self._connection.execute(
                    """INSERT OR REPLACE INTO resource_versions
                       (resource_id, version, content, content_hash, timestamp, transaction_id, is_deleted)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        version.resource_id,
                        version.version,
                        version.content,
                        version.content_hash,
                        version.timestamp.isoformat(),
                        version.transaction_id,
                        int(version.is_deleted),
                    ),
                )

            await self._connection.commit()

    async def delete(self, resource_id: str) -> bool:
        if not self._connection:
            await self.initialize()

        async with self._lock:
            cursor = await self._connection.execute(
                "DELETE FROM resource_versions WHERE resource_id = ?", (resource_id,)
            )
            await self._connection.execute(
                "DELETE FROM resources WHERE resource_id = ?", (resource_id,)
            )
            await self._connection.commit()
            return cursor.rowcount > 0

    async def list_resources(self) -> List[str]:
        if not self._connection:
            await self.initialize()

        cursor = await self._connection.execute("SELECT resource_id FROM resources")
        rows = await cursor.fetchall()
        return [row[0] for row in rows]

    async def get_global_version(self) -> int:
        if not self._connection:
            await self.initialize()

        cursor = await self._connection.execute(
            "SELECT value FROM global_state WHERE key = 'global_version'"
        )
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def increment_global_version(self) -> int:
        if not self._connection:
            await self.initialize()

        async with self._lock:
            cursor = await self._connection.execute(
                "UPDATE global_state SET value = value + 1 WHERE key = 'global_version' RETURNING value"
            )
            row = await cursor.fetchone()
            await self._connection.commit()
            return row[0]


class MVCCStorage:
    """
    Multi-Version Concurrency Control Storage.

    Provides versioned storage for resources with support for reading
    historical versions, atomic batch writes, and garbage collection.
    """

    def __init__(
        self,
        backend: Optional[StorageBackend] = None,
        gc_threshold: int = 100,
        max_versions_per_resource: int = 50,
    ):
        """
        Initialize MVCC storage.

        Args:
            backend: Storage backend to use (defaults to in-memory)
            gc_threshold: Number of writes before triggering GC
            max_versions_per_resource: Maximum versions to keep per resource
        """
        self._backend = backend or InMemoryBackend()
        self._gc_threshold = gc_threshold
        self._max_versions = max_versions_per_resource
        self._write_count = 0
        self._lock = asyncio.Lock()
        self._active_snapshots: Set[int] = set()

    async def get_current_version(self) -> int:
        """Get the current global version number."""
        return await self._backend.get_global_version()

    async def get_current_resource_version(self, resource_id: str) -> Optional[int]:
        """Get the current version number of a specific resource."""
        resource = await self._backend.get(resource_id)
        return resource.current_version if resource else None

    async def read(
        self, resource_id: str, version: Optional[int] = None
    ) -> Optional[ResourceVersion]:
        """
        Read a resource at a specific version.

        Args:
            resource_id: ID of the resource to read
            version: Version to read (defaults to latest)

        Returns:
            ResourceVersion if found, None otherwise
        """
        resource = await self._backend.get(resource_id)
        if not resource:
            return None

        if version is None:
            return resource.get_latest()

        return resource.get_version_at_or_before(version)

    async def write(
        self,
        resource_id: str,
        content: str,
        transaction_id: Optional[str] = None,
    ) -> ResourceVersion:
        """
        Write a new version of a resource.

        Args:
            resource_id: ID of the resource to write
            content: New content
            transaction_id: ID of the transaction making the write

        Returns:
            The newly created ResourceVersion
        """
        async with self._lock:
            # Get or create resource
            resource = await self._backend.get(resource_id)
            if not resource:
                resource = VersionedResource(
                    resource_id=resource_id,
                    current_version=0,
                )

            # Increment global version
            new_version = await self._backend.increment_global_version()

            # Create new version
            version = ResourceVersion.create(
                resource_id=resource_id,
                version=new_version,
                content=content,
                transaction_id=transaction_id,
            )

            resource.add_version(version)
            await self._backend.put(resource)

            # Trigger GC if needed
            self._write_count += 1
            if self._write_count >= self._gc_threshold:
                asyncio.create_task(self._garbage_collect())
                self._write_count = 0

            return version

    async def delete_resource(
        self,
        resource_id: str,
        transaction_id: Optional[str] = None,
    ) -> Optional[ResourceVersion]:
        """
        Mark a resource as deleted (soft delete).

        Args:
            resource_id: ID of the resource to delete
            transaction_id: ID of the transaction making the deletion

        Returns:
            The deletion marker version, or None if resource didn't exist
        """
        async with self._lock:
            resource = await self._backend.get(resource_id)
            if not resource:
                return None

            new_version = await self._backend.increment_global_version()

            version = ResourceVersion.create(
                resource_id=resource_id,
                version=new_version,
                content="",
                transaction_id=transaction_id,
                is_deleted=True,
            )

            resource.add_version(version)
            await self._backend.put(resource)

            return version

    async def atomic_write_batch(
        self,
        transaction_id: str,
        writes: List[Tuple[str, str, str]],  # (resource_id, content, operation_type)
    ) -> int:
        """
        Atomically write multiple resources.

        Args:
            transaction_id: ID of the transaction
            writes: List of (resource_id, content, operation_type) tuples

        Returns:
            The new global version after all writes
        """
        async with self._lock:
            new_version = await self._backend.get_global_version()

            for resource_id, content, operation_type in writes:
                new_version = await self._backend.increment_global_version()

                resource = await self._backend.get(resource_id)
                if not resource:
                    resource = VersionedResource(
                        resource_id=resource_id,
                        current_version=0,
                    )

                is_deleted = operation_type == "delete"
                version = ResourceVersion.create(
                    resource_id=resource_id,
                    version=new_version,
                    content=content,
                    transaction_id=transaction_id,
                    is_deleted=is_deleted,
                )

                resource.add_version(version)
                await self._backend.put(resource)

            return new_version

    async def register_snapshot(self, version: int) -> None:
        """Register an active snapshot to prevent GC of needed versions."""
        self._active_snapshots.add(version)

    async def release_snapshot(self, version: int) -> None:
        """Release a snapshot, allowing GC of old versions."""
        self._active_snapshots.discard(version)

    async def get_version_history(
        self, resource_id: str, limit: int = 10
    ) -> List[ResourceVersion]:
        """
        Get version history for a resource.

        Args:
            resource_id: ID of the resource
            limit: Maximum number of versions to return

        Returns:
            List of versions, newest first
        """
        resource = await self._backend.get(resource_id)
        if not resource:
            return []

        versions = sorted(resource.versions.values(), key=lambda v: v.version, reverse=True)
        return versions[:limit]

    async def _garbage_collect(self) -> int:
        """
        Run garbage collection to remove old versions.

        Keeps versions that are:
        - Referenced by active snapshots
        - Within max_versions_per_resource limit

        Returns:
            Number of versions removed
        """
        removed = 0
        min_snapshot = min(self._active_snapshots) if self._active_snapshots else float("inf")

        resource_ids = await self._backend.list_resources()

        for resource_id in resource_ids:
            resource = await self._backend.get(resource_id)
            if not resource:
                continue

            # Sort versions oldest first
            versions = sorted(resource.versions.keys())

            # Calculate how many to remove
            excess = len(versions) - self._max_versions
            if excess <= 0:
                continue

            # Remove old versions that aren't needed by snapshots
            for version in versions[:excess]:
                if version < min_snapshot:
                    del resource.versions[version]
                    removed += 1

            await self._backend.put(resource)

        return removed

    async def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        resource_ids = await self._backend.list_resources()
        total_versions = 0
        total_size = 0

        for resource_id in resource_ids:
            resource = await self._backend.get(resource_id)
            if resource:
                total_versions += len(resource.versions)
                for v in resource.versions.values():
                    total_size += len(v.content)

        return {
            "resource_count": len(resource_ids),
            "total_versions": total_versions,
            "total_size_bytes": total_size,
            "global_version": await self._backend.get_global_version(),
            "active_snapshots": len(self._active_snapshots),
        }


class LMDBBackend(StorageBackend):
    """
    LMDB storage backend for high-performance persistence.

    LMDB (Lightning Memory-Mapped Database) provides:
    - Memory-mapped I/O for fast reads
    - ACID transactions
    - Multi-reader, single-writer concurrency
    - Zero-copy reads
    """

    def __init__(
        self,
        path: str,
        map_size: int = 10 * 1024 * 1024 * 1024,  # 10GB default
        max_dbs: int = 10,
    ):
        """
        Initialize LMDB backend.

        Args:
            path: Directory path for the database
            map_size: Maximum size of the database in bytes
            max_dbs: Maximum number of named databases
        """
        self._path = path
        self._map_size = map_size
        self._max_dbs = max_dbs
        self._env = None
        self._resources_db = None
        self._versions_db = None
        self._meta_db = None
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the LMDB environment and databases."""
        if self._initialized:
            return

        import lmdb
        import os

        # Create directory if needed
        os.makedirs(self._path, exist_ok=True)

        # Open environment
        self._env = lmdb.open(
            self._path,
            map_size=self._map_size,
            max_dbs=self._max_dbs,
            writemap=True,
            metasync=False,
            sync=True,
        )

        # Open named databases
        self._resources_db = self._env.open_db(b"resources")
        self._versions_db = self._env.open_db(b"versions")
        self._meta_db = self._env.open_db(b"meta")

        # Initialize global version if not exists
        with self._env.begin(write=True, db=self._meta_db) as txn:
            if txn.get(b"global_version") is None:
                txn.put(b"global_version", b"0")

        self._initialized = True

    async def close(self) -> None:
        """Close the LMDB environment."""
        if self._env:
            self._env.close()
            self._env = None
            self._initialized = False

    def _serialize_resource(self, resource: VersionedResource) -> bytes:
        """Serialize resource metadata to bytes."""
        import json
        data = {
            "resource_id": resource.resource_id,
            "current_version": resource.current_version,
            "created_at": resource.created_at.isoformat(),
            "updated_at": resource.updated_at.isoformat(),
            "version_keys": list(resource.versions.keys()),
        }
        return json.dumps(data).encode("utf-8")

    def _deserialize_resource(self, data: bytes) -> VersionedResource:
        """Deserialize resource metadata from bytes."""
        import json
        parsed = json.loads(data.decode("utf-8"))
        return VersionedResource(
            resource_id=parsed["resource_id"],
            current_version=parsed["current_version"],
            created_at=datetime.fromisoformat(parsed["created_at"]),
            updated_at=datetime.fromisoformat(parsed["updated_at"]),
        )

    def _serialize_version(self, version: ResourceVersion) -> bytes:
        """Serialize a version to bytes."""
        import json
        data = {
            "resource_id": version.resource_id,
            "version": version.version,
            "content": version.content,
            "content_hash": version.content_hash,
            "timestamp": version.timestamp.isoformat(),
            "transaction_id": version.transaction_id,
            "is_deleted": version.is_deleted,
        }
        return json.dumps(data).encode("utf-8")

    def _deserialize_version(self, data: bytes) -> ResourceVersion:
        """Deserialize a version from bytes."""
        import json
        parsed = json.loads(data.decode("utf-8"))
        return ResourceVersion(
            resource_id=parsed["resource_id"],
            version=parsed["version"],
            content=parsed["content"],
            content_hash=parsed["content_hash"],
            timestamp=datetime.fromisoformat(parsed["timestamp"]),
            transaction_id=parsed.get("transaction_id"),
            is_deleted=parsed.get("is_deleted", False),
        )

    def _version_key(self, resource_id: str, version: int) -> bytes:
        """Generate key for a version entry."""
        return f"{resource_id}:{version}".encode("utf-8")

    async def get(self, resource_id: str) -> Optional[VersionedResource]:
        """Get a versioned resource by ID."""
        if not self._initialized:
            await self.initialize()

        resource_key = resource_id.encode("utf-8")

        with self._env.begin(db=self._resources_db) as txn:
            data = txn.get(resource_key)
            if not data:
                return None

            resource = self._deserialize_resource(data)

        # Load versions
        with self._env.begin(db=self._versions_db) as txn:
            cursor = txn.cursor()
            prefix = f"{resource_id}:".encode("utf-8")

            if cursor.set_range(prefix):
                while True:
                    key, value = cursor.item()
                    if not key.startswith(prefix):
                        break

                    version = self._deserialize_version(value)
                    resource.versions[version.version] = version

                    if not cursor.next():
                        break

        return resource

    async def put(self, resource: VersionedResource) -> None:
        """Store or update a versioned resource."""
        if not self._initialized:
            await self.initialize()

        async with self._lock:
            resource_key = resource.resource_id.encode("utf-8")

            with self._env.begin(write=True) as txn:
                # Store resource metadata
                txn.put(
                    resource_key,
                    self._serialize_resource(resource),
                    db=self._resources_db,
                )

                # Store versions
                for version in resource.versions.values():
                    version_key = self._version_key(resource.resource_id, version.version)
                    txn.put(
                        version_key,
                        self._serialize_version(version),
                        db=self._versions_db,
                    )

    async def delete(self, resource_id: str) -> bool:
        """Delete a resource and all its versions."""
        if not self._initialized:
            await self.initialize()

        async with self._lock:
            resource_key = resource_id.encode("utf-8")
            deleted = False

            with self._env.begin(write=True) as txn:
                # Delete resource metadata
                if txn.delete(resource_key, db=self._resources_db):
                    deleted = True

                # Delete all versions
                cursor = txn.cursor(db=self._versions_db)
                prefix = f"{resource_id}:".encode("utf-8")

                if cursor.set_range(prefix):
                    while True:
                        key, _ = cursor.item()
                        if not key.startswith(prefix):
                            break
                        cursor.delete()
                        if not cursor.next():
                            break

            return deleted

    async def list_resources(self) -> List[str]:
        """List all resource IDs."""
        if not self._initialized:
            await self.initialize()

        resources = []
        with self._env.begin(db=self._resources_db) as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                resources.append(key.decode("utf-8"))

        return resources

    async def get_global_version(self) -> int:
        """Get the current global version number."""
        if not self._initialized:
            await self.initialize()

        with self._env.begin(db=self._meta_db) as txn:
            data = txn.get(b"global_version")
            return int(data.decode("utf-8")) if data else 0

    async def increment_global_version(self) -> int:
        """Atomically increment and return the new global version."""
        if not self._initialized:
            await self.initialize()

        async with self._lock:
            with self._env.begin(write=True, db=self._meta_db) as txn:
                data = txn.get(b"global_version")
                current = int(data.decode("utf-8")) if data else 0
                new_version = current + 1
                txn.put(b"global_version", str(new_version).encode("utf-8"))
                return new_version

    async def compact(self) -> None:
        """Compact the database to reclaim space."""
        if not self._initialized:
            await self.initialize()

        # LMDB doesn't need explicit compaction, but we can copy to a new file
        # This is a no-op for now
        pass

    def get_db_stats(self) -> Dict[str, Any]:
        """Get LMDB database statistics."""
        if not self._env:
            return {}

        stats = self._env.stat()
        info = self._env.info()

        return {
            "page_size": stats["psize"],
            "tree_depth": stats["depth"],
            "branch_pages": stats["branch_pages"],
            "leaf_pages": stats["leaf_pages"],
            "overflow_pages": stats["overflow_pages"],
            "entries": stats["entries"],
            "map_size": info["map_size"],
            "last_pgno": info["last_pgno"],
            "last_txnid": info["last_txnid"],
            "max_readers": info["max_readers"],
            "num_readers": info["num_readers"],
        }

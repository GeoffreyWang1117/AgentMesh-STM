"""
Backup and Restore Module for AgentMesh-STM.

Provides data backup, restore, and export/import functionality.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set

from agentmesh_stm.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Backup Metadata
# =============================================================================

class BackupFormat(Enum):
    """Backup file formats."""
    JSON = "json"
    JSON_GZ = "json.gz"
    TAR_GZ = "tar.gz"


class BackupType(Enum):
    """Types of backups."""
    FULL = auto()
    INCREMENTAL = auto()
    SNAPSHOT = auto()


@dataclass
class BackupMetadata:
    """Metadata for a backup."""

    backup_id: str
    backup_type: BackupType
    created_at: datetime
    version: str = "1.0"
    format: BackupFormat = BackupFormat.JSON_GZ
    resource_count: int = 0
    transaction_count: int = 0
    size_bytes: int = 0
    checksum: str = ""
    description: str = ""
    parent_backup_id: Optional[str] = None  # For incremental backups
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backup_id": self.backup_id,
            "backup_type": self.backup_type.name,
            "created_at": self.created_at.isoformat(),
            "version": self.version,
            "format": self.format.value,
            "resource_count": self.resource_count,
            "transaction_count": self.transaction_count,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "description": self.description,
            "parent_backup_id": self.parent_backup_id,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BackupMetadata":
        return cls(
            backup_id=data["backup_id"],
            backup_type=BackupType[data["backup_type"]],
            created_at=datetime.fromisoformat(data["created_at"]),
            version=data.get("version", "1.0"),
            format=BackupFormat(data.get("format", "json.gz")),
            resource_count=data.get("resource_count", 0),
            transaction_count=data.get("transaction_count", 0),
            size_bytes=data.get("size_bytes", 0),
            checksum=data.get("checksum", ""),
            description=data.get("description", ""),
            parent_backup_id=data.get("parent_backup_id"),
            tags=data.get("tags", []),
        )


@dataclass
class RestoreResult:
    """Result of a restore operation."""

    success: bool
    resources_restored: int = 0
    transactions_restored: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# Backup Manager
# =============================================================================

class BackupManager:
    """
    Manages backup and restore operations.

    Supports:
    - Full backups
    - Incremental backups
    - Point-in-time snapshots
    - Multiple storage backends
    """

    def __init__(
        self,
        storage: Any,  # MVCCStorage
        backup_dir: str = "./backups",
        max_backups: int = 10,
        compress: bool = True,
    ):
        self.storage = storage
        self.backup_dir = Path(backup_dir)
        self.max_backups = max_backups
        self.compress = compress

        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def create_backup(
        self,
        backup_type: BackupType = BackupType.FULL,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> BackupMetadata:
        """Create a new backup."""
        async with self._lock:
            backup_id = self._generate_backup_id()

            logger.info(f"Creating {backup_type.name} backup: {backup_id}")

            # Collect data to backup
            data = await self._collect_backup_data(backup_type)

            # Create backup file
            backup_path = self._get_backup_path(backup_id)
            checksum = await self._write_backup(backup_path, data)

            # Create metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                backup_type=backup_type,
                created_at=datetime.utcnow(),
                format=BackupFormat.JSON_GZ if self.compress else BackupFormat.JSON,
                resource_count=len(data.get("resources", [])),
                transaction_count=len(data.get("transactions", [])),
                size_bytes=backup_path.stat().st_size,
                checksum=checksum,
                description=description,
                tags=tags or [],
            )

            # Save metadata
            await self._save_metadata(metadata)

            # Cleanup old backups
            await self._cleanup_old_backups()

            logger.info(
                f"Backup created: {backup_id}",
                resources=metadata.resource_count,
                size=metadata.size_bytes,
            )

            return metadata

    async def restore_backup(
        self,
        backup_id: str,
        verify_checksum: bool = True,
        overwrite: bool = False,
    ) -> RestoreResult:
        """Restore from a backup."""
        async with self._lock:
            logger.info(f"Restoring backup: {backup_id}")

            result = RestoreResult(success=False)

            # Load metadata
            metadata = await self._load_metadata(backup_id)
            if not metadata:
                result.errors.append(f"Backup not found: {backup_id}")
                return result

            # Load backup data
            backup_path = self._get_backup_path(backup_id)
            if not backup_path.exists():
                result.errors.append(f"Backup file not found: {backup_path}")
                return result

            data = await self._read_backup(backup_path)

            # Verify checksum
            if verify_checksum:
                actual_checksum = self._compute_checksum(data)
                if actual_checksum != metadata.checksum:
                    result.errors.append("Checksum verification failed")
                    return result

            # Restore resources
            resources_restored = 0
            for resource_data in data.get("resources", []):
                try:
                    await self._restore_resource(resource_data, overwrite)
                    resources_restored += 1
                except Exception as e:
                    result.warnings.append(
                        f"Failed to restore resource {resource_data.get('id')}: {e}"
                    )

            result.resources_restored = resources_restored
            result.success = len(result.errors) == 0

            logger.info(
                f"Restore completed: {backup_id}",
                resources=resources_restored,
                errors=len(result.errors),
            )

            return result

    async def list_backups(
        self,
        backup_type: Optional[BackupType] = None,
        tags: Optional[List[str]] = None,
    ) -> List[BackupMetadata]:
        """List available backups."""
        backups = []

        metadata_dir = self.backup_dir / "metadata"
        if not metadata_dir.exists():
            return backups

        for metadata_file in metadata_dir.glob("*.json"):
            try:
                with open(metadata_file) as f:
                    data = json.load(f)
                    metadata = BackupMetadata.from_dict(data)

                    # Filter by type
                    if backup_type and metadata.backup_type != backup_type:
                        continue

                    # Filter by tags
                    if tags and not all(t in metadata.tags for t in tags):
                        continue

                    backups.append(metadata)

            except Exception as e:
                logger.warning(f"Failed to read metadata: {metadata_file}: {e}")

        # Sort by creation time (newest first)
        backups.sort(key=lambda b: b.created_at, reverse=True)

        return backups

    async def delete_backup(self, backup_id: str) -> bool:
        """Delete a backup."""
        async with self._lock:
            logger.info(f"Deleting backup: {backup_id}")

            backup_path = self._get_backup_path(backup_id)
            metadata_path = self.backup_dir / "metadata" / f"{backup_id}.json"

            deleted = False

            if backup_path.exists():
                backup_path.unlink()
                deleted = True

            if metadata_path.exists():
                metadata_path.unlink()
                deleted = True

            return deleted

    async def verify_backup(self, backup_id: str) -> Dict[str, Any]:
        """Verify backup integrity."""
        result = {
            "backup_id": backup_id,
            "valid": False,
            "errors": [],
        }

        metadata = await self._load_metadata(backup_id)
        if not metadata:
            result["errors"].append("Metadata not found")
            return result

        backup_path = self._get_backup_path(backup_id)
        if not backup_path.exists():
            result["errors"].append("Backup file not found")
            return result

        try:
            data = await self._read_backup(backup_path)
            actual_checksum = self._compute_checksum(data)

            if actual_checksum != metadata.checksum:
                result["errors"].append("Checksum mismatch")
            else:
                result["valid"] = True
                result["resources"] = len(data.get("resources", []))
                result["size"] = backup_path.stat().st_size

        except Exception as e:
            result["errors"].append(f"Read error: {e}")

        return result

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    def _generate_backup_id(self) -> str:
        """Generate a unique backup ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        import secrets
        random_suffix = secrets.token_hex(4)
        return f"backup_{timestamp}_{random_suffix}"

    def _get_backup_path(self, backup_id: str) -> Path:
        """Get path for a backup file."""
        ext = ".json.gz" if self.compress else ".json"
        return self.backup_dir / f"{backup_id}{ext}"

    async def _collect_backup_data(self, backup_type: BackupType) -> Dict[str, Any]:
        """Collect data for backup."""
        data = {
            "version": "1.0",
            "backup_type": backup_type.name,
            "created_at": datetime.utcnow().isoformat(),
            "resources": [],
            "transactions": [],
        }

        # Collect resources from storage
        if hasattr(self.storage, "list_resources"):
            resource_ids = await self.storage.list_resources()
            for resource_id in resource_ids:
                try:
                    content = await self.storage.read(resource_id)
                    versions = []
                    if hasattr(self.storage, "get_history"):
                        versions = await self.storage.get_history(resource_id)

                    data["resources"].append({
                        "id": resource_id,
                        "content": content,
                        "versions": versions,
                    })
                except Exception as e:
                    logger.warning(f"Failed to backup resource {resource_id}: {e}")

        return data

    async def _write_backup(self, path: Path, data: Dict[str, Any]) -> str:
        """Write backup data to file and return checksum."""
        json_data = json.dumps(data, indent=2, default=str)
        checksum = self._compute_checksum(data)

        if self.compress:
            with gzip.open(path, "wt", encoding="utf-8") as f:
                f.write(json_data)
        else:
            with open(path, "w") as f:
                f.write(json_data)

        return checksum

    async def _read_backup(self, path: Path) -> Dict[str, Any]:
        """Read backup data from file."""
        if str(path).endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        else:
            with open(path) as f:
                return json.load(f)

    def _compute_checksum(self, data: Dict[str, Any]) -> str:
        """Compute checksum of backup data."""
        json_data = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_data.encode()).hexdigest()

    async def _save_metadata(self, metadata: BackupMetadata) -> None:
        """Save backup metadata."""
        metadata_dir = self.backup_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = metadata_dir / f"{metadata.backup_id}.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

    async def _load_metadata(self, backup_id: str) -> Optional[BackupMetadata]:
        """Load backup metadata."""
        metadata_path = self.backup_dir / "metadata" / f"{backup_id}.json"

        if not metadata_path.exists():
            return None

        with open(metadata_path) as f:
            data = json.load(f)
            return BackupMetadata.from_dict(data)

    async def _restore_resource(
        self,
        resource_data: Dict[str, Any],
        overwrite: bool,
    ) -> None:
        """Restore a single resource."""
        resource_id = resource_data["id"]
        content = resource_data["content"]

        # Check if resource exists
        if hasattr(self.storage, "exists"):
            exists = await self.storage.exists(resource_id)
            if exists and not overwrite:
                raise ValueError(f"Resource exists and overwrite=False: {resource_id}")

        # Write resource
        if hasattr(self.storage, "write"):
            await self.storage.write(resource_id, content)

    async def _cleanup_old_backups(self) -> None:
        """Remove old backups exceeding max_backups limit."""
        backups = await self.list_backups()

        if len(backups) <= self.max_backups:
            return

        # Delete oldest backups
        for backup in backups[self.max_backups:]:
            logger.info(f"Cleaning up old backup: {backup.backup_id}")
            await self.delete_backup(backup.backup_id)


# =============================================================================
# Export/Import
# =============================================================================

class DataExporter:
    """
    Exports data in various formats.

    Supports JSON, CSV, and custom formats.
    """

    def __init__(self, storage: Any):
        self.storage = storage

    async def export_json(
        self,
        output_path: str,
        resource_ids: Optional[List[str]] = None,
        compress: bool = True,
    ) -> int:
        """Export resources to JSON."""
        data = await self._collect_resources(resource_ids)

        if compress:
            with gzip.open(output_path, "wt", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        else:
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

        return len(data.get("resources", []))

    async def export_csv(
        self,
        output_path: str,
        resource_ids: Optional[List[str]] = None,
    ) -> int:
        """Export resources to CSV."""
        import csv

        data = await self._collect_resources(resource_ids)
        resources = data.get("resources", [])

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "content", "version"])

            for resource in resources:
                writer.writerow([
                    resource.get("id", ""),
                    resource.get("content", ""),
                    resource.get("version", ""),
                ])

        return len(resources)

    async def _collect_resources(
        self,
        resource_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Collect resources for export."""
        data = {
            "exported_at": datetime.utcnow().isoformat(),
            "resources": [],
        }

        if resource_ids is None and hasattr(self.storage, "list_resources"):
            resource_ids = await self.storage.list_resources()

        if resource_ids:
            for resource_id in resource_ids:
                try:
                    content = await self.storage.read(resource_id)
                    data["resources"].append({
                        "id": resource_id,
                        "content": content,
                    })
                except Exception as e:
                    logger.warning(f"Failed to export {resource_id}: {e}")

        return data


class DataImporter:
    """
    Imports data from various formats.
    """

    def __init__(self, storage: Any):
        self.storage = storage

    async def import_json(
        self,
        input_path: str,
        overwrite: bool = False,
    ) -> int:
        """Import resources from JSON."""
        path = Path(input_path)

        if str(path).endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(path) as f:
                data = json.load(f)

        imported = 0
        for resource in data.get("resources", []):
            try:
                resource_id = resource["id"]
                content = resource["content"]

                if hasattr(self.storage, "exists"):
                    exists = await self.storage.exists(resource_id)
                    if exists and not overwrite:
                        continue

                await self.storage.write(resource_id, content)
                imported += 1

            except Exception as e:
                logger.warning(f"Failed to import resource: {e}")

        return imported


# =============================================================================
# Scheduled Backups
# =============================================================================

class ScheduledBackup:
    """
    Runs backups on a schedule.
    """

    def __init__(
        self,
        backup_manager: BackupManager,
        interval_hours: float = 24.0,
    ):
        self.backup_manager = backup_manager
        self.interval_hours = interval_hours
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start scheduled backups."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._backup_loop())
        logger.info(f"Scheduled backups started (interval: {self.interval_hours}h)")

    async def stop(self) -> None:
        """Stop scheduled backups."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Scheduled backups stopped")

    async def _backup_loop(self) -> None:
        """Background backup loop."""
        while self._running:
            try:
                await asyncio.sleep(self.interval_hours * 3600)

                if not self._running:
                    break

                logger.info("Running scheduled backup")
                await self.backup_manager.create_backup(
                    backup_type=BackupType.FULL,
                    description="Scheduled backup",
                    tags=["scheduled"],
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduled backup failed: {e}")


# =============================================================================
# CLI Integration
# =============================================================================

async def backup_cli(args) -> int:
    """CLI entry point for backup operations."""
    from agentmesh_stm.storage import InMemoryMVCCStorage

    storage = InMemoryMVCCStorage()
    manager = BackupManager(storage, backup_dir=args.get("backup_dir", "./backups"))

    command = args.get("command")

    if command == "create":
        metadata = await manager.create_backup(
            description=args.get("description", ""),
            tags=args.get("tags", []),
        )
        print(f"Backup created: {metadata.backup_id}")
        return 0

    elif command == "list":
        backups = await manager.list_backups()
        for backup in backups:
            print(f"{backup.backup_id} | {backup.created_at} | {backup.resource_count} resources")
        return 0

    elif command == "restore":
        result = await manager.restore_backup(args["backup_id"])
        if result.success:
            print(f"Restored {result.resources_restored} resources")
            return 0
        else:
            print(f"Restore failed: {result.errors}")
            return 1

    elif command == "verify":
        result = await manager.verify_backup(args["backup_id"])
        print(json.dumps(result, indent=2))
        return 0 if result["valid"] else 1

    elif command == "delete":
        deleted = await manager.delete_backup(args["backup_id"])
        print("Deleted" if deleted else "Not found")
        return 0 if deleted else 1

    return 1

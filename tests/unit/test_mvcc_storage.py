"""Unit tests for MVCC Storage."""

import pytest
import asyncio
from datetime import datetime

from agentmesh_stm.storage.mvcc import (
    MVCCStorage,
    InMemoryBackend,
    SQLiteBackend,
    ResourceVersion,
    VersionedResource,
)


class TestResourceVersion:
    """Tests for ResourceVersion class."""

    def test_create(self):
        """Test creating a resource version."""
        version = ResourceVersion.create(
            resource_id="test",
            version=1,
            content="test content",
            transaction_id="txn123",
        )

        assert version.resource_id == "test"
        assert version.version == 1
        assert version.content == "test content"
        assert version.content_hash is not None
        assert version.transaction_id == "txn123"
        assert not version.is_deleted

    def test_content_hash_consistency(self):
        """Test that same content produces same hash."""
        v1 = ResourceVersion.create("r1", 1, "content")
        v2 = ResourceVersion.create("r2", 2, "content")

        assert v1.content_hash == v2.content_hash

    def test_deletion_marker(self):
        """Test deletion marker version."""
        version = ResourceVersion.create(
            resource_id="test",
            version=1,
            content="",
            is_deleted=True,
        )

        assert version.is_deleted


class TestVersionedResource:
    """Tests for VersionedResource class."""

    def test_add_version(self):
        """Test adding versions."""
        resource = VersionedResource(resource_id="test", current_version=0)

        v1 = ResourceVersion.create("test", 1, "content v1")
        resource.add_version(v1)

        assert resource.current_version == 1
        assert len(resource.versions) == 1

    def test_get_version(self):
        """Test retrieving specific versions."""
        resource = VersionedResource(resource_id="test", current_version=0)

        v1 = ResourceVersion.create("test", 1, "v1")
        v2 = ResourceVersion.create("test", 2, "v2")
        resource.add_version(v1)
        resource.add_version(v2)

        assert resource.get_version(1).content == "v1"
        assert resource.get_version(2).content == "v2"
        assert resource.get_version(3) is None

    def test_get_latest(self):
        """Test getting latest version."""
        resource = VersionedResource(resource_id="test", current_version=0)

        v1 = ResourceVersion.create("test", 1, "v1")
        v2 = ResourceVersion.create("test", 2, "v2")
        resource.add_version(v1)
        resource.add_version(v2)

        latest = resource.get_latest()
        assert latest.version == 2
        assert latest.content == "v2"

    def test_get_version_at_or_before(self):
        """Test snapshot reading."""
        resource = VersionedResource(resource_id="test", current_version=0)

        v1 = ResourceVersion.create("test", 1, "v1")
        v3 = ResourceVersion.create("test", 3, "v3")
        v5 = ResourceVersion.create("test", 5, "v5")
        resource.add_version(v1)
        resource.add_version(v3)
        resource.add_version(v5)

        # Exact version
        assert resource.get_version_at_or_before(3).content == "v3"

        # Before next version
        assert resource.get_version_at_or_before(4).content == "v3"

        # Latest
        assert resource.get_version_at_or_before(10).content == "v5"

        # Before any version
        assert resource.get_version_at_or_before(0) is None


class TestInMemoryBackend:
    """Tests for InMemoryBackend."""

    @pytest.fixture
    def backend(self):
        return InMemoryBackend()

    @pytest.mark.asyncio
    async def test_put_and_get(self, backend):
        """Test storing and retrieving resources."""
        resource = VersionedResource(resource_id="test", current_version=1)
        v1 = ResourceVersion.create("test", 1, "content")
        resource.add_version(v1)

        await backend.put(resource)
        retrieved = await backend.get("test")

        assert retrieved is not None
        assert retrieved.resource_id == "test"
        assert retrieved.current_version == 1

    @pytest.mark.asyncio
    async def test_delete(self, backend):
        """Test deleting resources."""
        resource = VersionedResource(resource_id="test", current_version=1)
        await backend.put(resource)

        result = await backend.delete("test")
        assert result

        retrieved = await backend.get("test")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_list_resources(self, backend):
        """Test listing resources."""
        r1 = VersionedResource(resource_id="r1", current_version=1)
        r2 = VersionedResource(resource_id="r2", current_version=1)
        await backend.put(r1)
        await backend.put(r2)

        resources = await backend.list_resources()
        assert set(resources) == {"r1", "r2"}

    @pytest.mark.asyncio
    async def test_global_version(self, backend):
        """Test global version management."""
        initial = await backend.get_global_version()
        assert initial == 0

        v1 = await backend.increment_global_version()
        assert v1 == 1

        v2 = await backend.increment_global_version()
        assert v2 == 2

        current = await backend.get_global_version()
        assert current == 2


class TestMVCCStorage:
    """Tests for MVCCStorage."""

    @pytest.fixture
    def storage(self):
        return MVCCStorage(backend=InMemoryBackend())

    @pytest.mark.asyncio
    async def test_write_and_read(self, storage):
        """Test writing and reading resources."""
        version = await storage.write("file.txt", "hello world")

        assert version.content == "hello world"
        assert version.version > 0

        read_version = await storage.read("file.txt")
        assert read_version.content == "hello world"

    @pytest.mark.asyncio
    async def test_read_at_version(self, storage):
        """Test reading at specific version."""
        v1 = await storage.write("file.txt", "version 1")
        v2 = await storage.write("file.txt", "version 2")

        # Read latest
        latest = await storage.read("file.txt")
        assert latest.content == "version 2"

        # Read at v1's version
        at_v1 = await storage.read("file.txt", version=v1.version)
        assert at_v1.content == "version 1"

    @pytest.mark.asyncio
    async def test_atomic_write_batch(self, storage):
        """Test atomic batch writes."""
        writes = [
            ("file1.txt", "content1", "write"),
            ("file2.txt", "content2", "write"),
            ("file3.txt", "content3", "create"),
        ]

        final_version = await storage.atomic_write_batch("txn1", writes)

        assert final_version > 0

        f1 = await storage.read("file1.txt")
        f2 = await storage.read("file2.txt")
        f3 = await storage.read("file3.txt")

        assert f1.content == "content1"
        assert f2.content == "content2"
        assert f3.content == "content3"

    @pytest.mark.asyncio
    async def test_delete_resource(self, storage):
        """Test soft delete."""
        await storage.write("file.txt", "content")

        delete_version = await storage.delete_resource("file.txt")

        assert delete_version is not None
        assert delete_version.is_deleted

    @pytest.mark.asyncio
    async def test_version_history(self, storage):
        """Test retrieving version history."""
        await storage.write("file.txt", "v1")
        await storage.write("file.txt", "v2")
        await storage.write("file.txt", "v3")

        history = await storage.get_version_history("file.txt", limit=10)

        assert len(history) == 3
        assert history[0].content == "v3"  # Newest first
        assert history[2].content == "v1"

    @pytest.mark.asyncio
    async def test_snapshot_isolation(self, storage):
        """Test snapshot isolation."""
        await storage.write("file.txt", "initial")

        # Get snapshot version
        snapshot = await storage.get_current_version()

        # Write new version
        await storage.write("file.txt", "modified")

        # Read at snapshot should see old value
        at_snapshot = await storage.read("file.txt", version=snapshot)
        assert at_snapshot.content == "initial"

        # Read latest should see new value
        latest = await storage.read("file.txt")
        assert latest.content == "modified"

    @pytest.mark.asyncio
    async def test_get_stats(self, storage):
        """Test storage statistics."""
        await storage.write("f1", "content1")
        await storage.write("f2", "content2")
        await storage.write("f1", "content1 updated")

        stats = await storage.get_stats()

        assert stats["resource_count"] == 2
        assert stats["total_versions"] == 3
        assert stats["global_version"] == 3

    @pytest.mark.asyncio
    async def test_concurrent_writes(self, storage):
        """Test concurrent write handling."""

        async def writer(storage, file_id, count):
            for i in range(count):
                await storage.write(f"file_{file_id}", f"content_{i}")

        # Run concurrent writers
        await asyncio.gather(
            writer(storage, "a", 5),
            writer(storage, "b", 5),
            writer(storage, "c", 5),
        )

        stats = await storage.get_stats()
        assert stats["resource_count"] == 3
        assert stats["total_versions"] == 15


class TestSQLiteBackend:
    """Tests for SQLiteBackend."""

    @pytest.fixture
    async def backend(self):
        backend = SQLiteBackend(":memory:")
        await backend.initialize()
        yield backend
        await backend.close()

    @pytest.mark.asyncio
    async def test_persistence(self, backend):
        """Test data persistence."""
        resource = VersionedResource(resource_id="test", current_version=1)
        v1 = ResourceVersion.create("test", 1, "persistent content")
        resource.add_version(v1)

        await backend.put(resource)
        retrieved = await backend.get("test")

        assert retrieved is not None
        assert retrieved.get_version(1).content == "persistent content"

    @pytest.mark.asyncio
    async def test_global_version_persistence(self, backend):
        """Test global version persistence."""
        await backend.increment_global_version()
        await backend.increment_global_version()

        version = await backend.get_global_version()
        assert version == 2

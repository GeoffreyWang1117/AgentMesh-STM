"""Unit tests for Semantic Conflict Detection."""

import pytest

from agentmesh_stm.conflict.detector import (
    ConflictDetector,
    ConflictDetectorConfig,
    ConflictResult,
    ConflictType,
    ConflictLevel,
    ConflictResolver,
    MergeStrategy,
)
from agentmesh_stm.conflict.ast_analyzer import (
    ASTAnalyzer,
    CodeRegion,
    CodeRegionType,
)
from agentmesh_stm.storage.mvcc import MVCCStorage, InMemoryBackend


class TestCodeRegion:
    """Tests for CodeRegion class."""

    def test_overlaps_with(self):
        """Test region overlap detection."""
        r1 = CodeRegion(
            type=CodeRegionType.FUNCTION,
            name="func1",
            start_line=1,
            end_line=10,
        )
        r2 = CodeRegion(
            type=CodeRegionType.FUNCTION,
            name="func2",
            start_line=5,
            end_line=15,
        )
        r3 = CodeRegion(
            type=CodeRegionType.FUNCTION,
            name="func3",
            start_line=20,
            end_line=30,
        )

        assert r1.overlaps_with(r2)
        assert r2.overlaps_with(r1)
        assert not r1.overlaps_with(r3)

    def test_contains(self):
        """Test line containment check."""
        region = CodeRegion(
            type=CodeRegionType.FUNCTION,
            name="func",
            start_line=5,
            end_line=15,
        )

        assert region.contains(5)
        assert region.contains(10)
        assert region.contains(15)
        assert not region.contains(4)
        assert not region.contains(16)

    def test_get_overlap_lines(self):
        """Test getting overlapping lines."""
        r1 = CodeRegion(
            type=CodeRegionType.FUNCTION,
            name="f1",
            start_line=1,
            end_line=10,
        )
        r2 = CodeRegion(
            type=CodeRegionType.FUNCTION,
            name="f2",
            start_line=8,
            end_line=15,
        )

        overlap = r1.get_overlap_lines(r2)
        assert overlap == {8, 9, 10}

    def test_full_name(self):
        """Test fully qualified name generation."""
        class_region = CodeRegion(
            type=CodeRegionType.CLASS,
            name="MyClass",
            start_line=1,
            end_line=50,
        )
        method_region = CodeRegion(
            type=CodeRegionType.METHOD,
            name="my_method",
            start_line=10,
            end_line=20,
            parent=class_region,
        )

        assert method_region.full_name == "MyClass.my_method"


class TestASTAnalyzer:
    """Tests for ASTAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        return ASTAnalyzer()

    def test_analyze_python_functions(self, analyzer):
        """Test analyzing Python function definitions."""
        code = '''
def hello():
    print("Hello")

def world():
    print("World")
'''
        regions = analyzer.analyze(code, "python")

        # Find function regions
        functions = [r for r in regions if r.type == CodeRegionType.FUNCTION]

        assert len(functions) >= 2
        func_names = {f.name for f in functions}
        assert "hello" in func_names
        assert "world" in func_names

    def test_analyze_python_classes(self, analyzer):
        """Test analyzing Python class definitions."""
        code = '''
class MyClass:
    def __init__(self):
        pass

    def method1(self):
        pass

class OtherClass:
    pass
'''
        regions = analyzer.analyze(code, "python")

        classes = [r for r in regions if r.type == CodeRegionType.CLASS]
        methods = [r for r in regions if r.type == CodeRegionType.METHOD]

        assert len(classes) >= 2
        class_names = {c.name for c in classes}
        assert "MyClass" in class_names
        assert "OtherClass" in class_names

        # Methods should have class as parent
        init_method = next((m for m in methods if m.name == "__init__"), None)
        assert init_method is not None
        assert init_method.parent is not None
        assert init_method.parent.name == "MyClass"

    def test_get_modified_regions(self, analyzer):
        """Test identifying modified regions."""
        old_code = '''
def unchanged():
    pass

def modified():
    old_content()

def also_unchanged():
    pass
'''
        new_code = '''
def unchanged():
    pass

def modified():
    new_content()

def also_unchanged():
    pass
'''
        modified_regions, modified_lines = analyzer.get_modified_regions(
            old_code, new_code, "python"
        )

        # Should identify the modified function
        assert len(modified_regions) > 0
        modified_names = {r.name for r in modified_regions}
        assert "modified" in modified_names
        assert "unchanged" not in modified_names

    def test_find_region_at_line(self, analyzer):
        """Test finding the most specific region at a line."""
        code = '''
class MyClass:
    def method(self):
        pass
'''
        regions = analyzer.analyze(code, "python")

        # Line inside method (line 3 approximately)
        method_regions = [r for r in regions if r.type == CodeRegionType.METHOD]
        if method_regions:
            method = method_regions[0]
            found = analyzer.find_region_at_line(regions, method.start_line)
            assert found is not None
            assert found.type == CodeRegionType.METHOD


class TestConflictResult:
    """Tests for ConflictResult class."""

    def test_no_conflict(self):
        """Test creating no-conflict result."""
        result = ConflictResult.no_conflict("resource1")

        assert not result.has_conflict
        assert result.conflict_type == ConflictType.NONE
        assert result.conflict_level == ConflictLevel.NONE

    def test_conflict(self):
        """Test creating conflict result."""
        result = ConflictResult.conflict(
            resource_id="resource1",
            conflict_type=ConflictType.WRITE_WRITE,
            conflict_level=ConflictLevel.HIGH,
            description="Concurrent writes",
        )

        assert result.has_conflict
        assert result.conflict_type == ConflictType.WRITE_WRITE
        assert result.conflict_level == ConflictLevel.HIGH


class TestConflictDetector:
    """Tests for ConflictDetector class."""

    @pytest.fixture
    def storage(self):
        return MVCCStorage(backend=InMemoryBackend())

    @pytest.fixture
    def detector(self, storage):
        config = ConflictDetectorConfig(
            enable_ast_analysis=True,
            enable_semantic_analysis=False,  # Disable LLM for tests
        )
        return ConflictDetector(storage=storage, config=config)

    @pytest.mark.asyncio
    async def test_no_conflict_same_version(self, detector, storage):
        """Test no conflict when versions match."""
        await storage.write("file.py", "content")
        version = await storage.get_current_version()

        result = await detector.check_conflict("file.py", version, version)

        assert not result.has_conflict

    @pytest.mark.asyncio
    async def test_conflict_version_mismatch(self, detector, storage):
        """Test conflict detection on version mismatch."""
        v1 = await storage.write("file.py", "content v1")
        v2 = await storage.write("file.py", "content v2")

        result = await detector.check_conflict("file.py", v1.version, v2.version)

        assert result.has_conflict
        assert result.conflict_type == ConflictType.READ_WRITE

    @pytest.mark.asyncio
    async def test_no_conflict_same_content(self, detector, storage):
        """Test no conflict when content hash matches."""
        v1 = await storage.write("file.py", "same content")
        # Write same content again
        v2 = await storage.write("file.py", "same content")

        result = await detector.check_conflict("file.py", v1.version, v2.version)

        # Should not be a conflict since content is identical
        assert not result.has_conflict

    @pytest.mark.asyncio
    async def test_write_conflict_detection(self, detector, storage):
        """Test write-write conflict detection."""
        base = await storage.write("file.py", "base content")

        # Simulate two concurrent modifications
        their_version = await storage.write("file.py", "their content")

        result = await detector.check_write_conflict(
            "file.py",
            our_content="our content",
            base_version=base.version,
            current_version=their_version.version,
        )

        assert result.has_conflict
        assert result.conflict_type == ConflictType.WRITE_WRITE


class TestConflictResolver:
    """Tests for ConflictResolver class."""

    @pytest.fixture
    def storage(self):
        return MVCCStorage(backend=InMemoryBackend())

    @pytest.fixture
    def resolver(self, storage):
        detector = ConflictDetector(storage=storage)
        return ConflictResolver(detector)

    @pytest.mark.asyncio
    async def test_resolve_ours(self, resolver):
        """Test resolving with 'ours' strategy."""
        result = await resolver.resolve(
            "file.py",
            base_content="base",
            our_content="our changes",
            their_content="their changes",
            strategy=MergeStrategy.OURS,
        )

        assert result.success
        assert result.merged_content == "our changes"
        assert result.strategy_used == MergeStrategy.OURS

    @pytest.mark.asyncio
    async def test_resolve_theirs(self, resolver):
        """Test resolving with 'theirs' strategy."""
        result = await resolver.resolve(
            "file.py",
            base_content="base",
            our_content="our changes",
            their_content="their changes",
            strategy=MergeStrategy.THEIRS,
        )

        assert result.success
        assert result.merged_content == "their changes"
        assert result.strategy_used == MergeStrategy.THEIRS

    @pytest.mark.asyncio
    async def test_auto_merge_non_overlapping(self, resolver):
        """Test auto-merge with non-overlapping changes."""
        base = "line1\nline2\nline3\nline4"
        ours = "line1_modified\nline2\nline3\nline4"  # Changed line 1
        theirs = "line1\nline2\nline3\nline4_modified"  # Changed line 4

        result = await resolver.resolve(
            "file.py",
            base_content=base,
            our_content=ours,
            their_content=theirs,
            strategy=MergeStrategy.AUTO_MERGE,
        )

        assert result.success
        assert "line1_modified" in result.merged_content
        assert "line4_modified" in result.merged_content

    @pytest.mark.asyncio
    async def test_auto_merge_overlapping_fails(self, resolver):
        """Test auto-merge fails with overlapping changes."""
        base = "line1\nline2\nline3"
        ours = "line1\nline2_ours\nline3"  # Changed line 2
        theirs = "line1\nline2_theirs\nline3"  # Also changed line 2

        result = await resolver.resolve(
            "file.py",
            base_content=base,
            our_content=ours,
            their_content=theirs,
            strategy=MergeStrategy.AUTO_MERGE,
        )

        assert not result.success
        assert result.strategy_used == MergeStrategy.MANUAL

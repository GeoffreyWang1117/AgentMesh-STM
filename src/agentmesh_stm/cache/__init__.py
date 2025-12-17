"""
Caching Module for AgentMesh-STM.

Provides caching layers for improved read performance.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

from agentmesh_stm.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


# =============================================================================
# Cache Entry
# =============================================================================

@dataclass
class CacheEntry(Generic[V]):
    """A cache entry with metadata."""

    value: V
    created_at: float
    last_accessed: float
    ttl: Optional[float] = None
    access_count: int = 0
    size: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return time.time() > self.created_at + self.ttl

    def touch(self) -> None:
        """Update last accessed time and increment access count."""
        self.last_accessed = time.time()
        self.access_count += 1


class CacheStats:
    """Cache statistics."""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "hit_rate": f"{self.hit_rate:.2%}",
        }


# =============================================================================
# Cache Interface
# =============================================================================

class Cache(ABC, Generic[K, V]):
    """Abstract base class for caches."""

    @abstractmethod
    async def get(self, key: K) -> Optional[V]:
        """Get a value from the cache."""
        pass

    @abstractmethod
    async def set(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        """Set a value in the cache."""
        pass

    @abstractmethod
    async def delete(self, key: K) -> bool:
        """Delete a value from the cache."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all entries from the cache."""
        pass

    @abstractmethod
    async def exists(self, key: K) -> bool:
        """Check if a key exists in the cache."""
        pass

    @abstractmethod
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        pass


# =============================================================================
# LRU Cache
# =============================================================================

class LRUCache(Cache[K, V]):
    """
    Least Recently Used (LRU) cache implementation.

    Thread-safe with async support.
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[float] = None,
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[K, CacheEntry[V]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._stats = CacheStats()

    async def get(self, key: K) -> Optional[V]:
        """Get value from cache, updating access order."""
        async with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return None

            entry = self._cache[key]

            # Check expiration
            if entry.is_expired:
                del self._cache[key]
                self._stats.expirations += 1
                self._stats.misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            self._stats.hits += 1

            return entry.value

    async def set(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        """Set value in cache with optional TTL."""
        async with self._lock:
            ttl = ttl if ttl is not None else self.default_ttl

            # Remove old entry if exists
            if key in self._cache:
                del self._cache[key]

            # Evict oldest entries if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._stats.evictions += 1

            # Add new entry
            self._cache[key] = CacheEntry(
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                ttl=ttl,
            )

    async def delete(self, key: K) -> bool:
        """Delete entry from cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self) -> None:
        """Clear all entries."""
        async with self._lock:
            self._cache.clear()

    async def exists(self, key: K) -> bool:
        """Check if key exists and is not expired."""
        async with self._lock:
            if key not in self._cache:
                return False
            entry = self._cache[key]
            if entry.is_expired:
                del self._cache[key]
                self._stats.expirations += 1
                return False
            return True

    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    async def get_size(self) -> int:
        """Get current cache size."""
        async with self._lock:
            return len(self._cache)

    async def cleanup_expired(self) -> int:
        """Remove expired entries and return count removed."""
        async with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]
            for key in expired_keys:
                del self._cache[key]
                self._stats.expirations += 1
            return len(expired_keys)


# =============================================================================
# TTL Cache
# =============================================================================

class TTLCache(Cache[K, V]):
    """
    Cache with time-to-live expiration.

    All entries have a TTL and are automatically expired.
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: float = 300.0,  # 5 minutes
        cleanup_interval: float = 60.0,  # 1 minute
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        self._cache: Dict[K, CacheEntry[V]] = {}
        self._lock = asyncio.Lock()
        self._stats = CacheStats()
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start background cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """Stop background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

    async def _cleanup_loop(self) -> None:
        """Background task to clean up expired entries."""
        while True:
            await asyncio.sleep(self.cleanup_interval)
            await self.cleanup_expired()

    async def get(self, key: K) -> Optional[V]:
        async with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return None

            entry = self._cache[key]

            if entry.is_expired:
                del self._cache[key]
                self._stats.expirations += 1
                self._stats.misses += 1
                return None

            entry.touch()
            self._stats.hits += 1
            return entry.value

    async def set(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        async with self._lock:
            ttl = ttl if ttl is not None else self.default_ttl

            # Evict if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                # Remove oldest entry by creation time
                oldest_key = min(
                    self._cache.keys(),
                    key=lambda k: self._cache[k].created_at
                )
                del self._cache[oldest_key]
                self._stats.evictions += 1

            self._cache[key] = CacheEntry(
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                ttl=ttl,
            )

    async def delete(self, key: K) -> bool:
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self) -> None:
        async with self._lock:
            self._cache.clear()

    async def exists(self, key: K) -> bool:
        async with self._lock:
            if key not in self._cache:
                return False
            entry = self._cache[key]
            if entry.is_expired:
                del self._cache[key]
                self._stats.expirations += 1
                return False
            return True

    def stats(self) -> CacheStats:
        return self._stats

    async def cleanup_expired(self) -> int:
        async with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]
            for key in expired_keys:
                del self._cache[key]
                self._stats.expirations += 1
            return len(expired_keys)


# =============================================================================
# Multi-Level Cache
# =============================================================================

class MultiLevelCache(Cache[K, V]):
    """
    Multi-level cache with L1 (fast, small) and L2 (slower, larger).

    L1 is typically an in-memory LRU cache.
    L2 could be a distributed cache or persistent storage.
    """

    def __init__(
        self,
        l1_cache: Cache[K, V],
        l2_cache: Cache[K, V],
        write_through: bool = True,
    ):
        self.l1 = l1_cache
        self.l2 = l2_cache
        self.write_through = write_through
        self._stats = CacheStats()

    async def get(self, key: K) -> Optional[V]:
        """Get from L1 first, then L2."""
        # Try L1
        value = await self.l1.get(key)
        if value is not None:
            self._stats.hits += 1
            return value

        # Try L2
        value = await self.l2.get(key)
        if value is not None:
            # Promote to L1
            await self.l1.set(key, value)
            self._stats.hits += 1
            return value

        self._stats.misses += 1
        return None

    async def set(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        """Set in both L1 and optionally L2."""
        await self.l1.set(key, value, ttl)
        if self.write_through:
            await self.l2.set(key, value, ttl)

    async def delete(self, key: K) -> bool:
        """Delete from both levels."""
        l1_result = await self.l1.delete(key)
        l2_result = await self.l2.delete(key)
        return l1_result or l2_result

    async def clear(self) -> None:
        """Clear both levels."""
        await self.l1.clear()
        await self.l2.clear()

    async def exists(self, key: K) -> bool:
        """Check if exists in either level."""
        return await self.l1.exists(key) or await self.l2.exists(key)

    def stats(self) -> CacheStats:
        return self._stats


# =============================================================================
# Cached Storage Wrapper
# =============================================================================

class CachedStorage:
    """
    Wrapper that adds caching to any storage backend.

    Provides transparent read-through caching.
    """

    def __init__(
        self,
        storage: Any,  # Should be MVCCStorage or similar
        cache: Optional[Cache[str, Any]] = None,
        cache_reads: bool = True,
        cache_versions: bool = True,
        invalidate_on_write: bool = True,
    ):
        self.storage = storage
        self.cache = cache or LRUCache(max_size=10000, default_ttl=300.0)
        self.cache_reads = cache_reads
        self.cache_versions = cache_versions
        self.invalidate_on_write = invalidate_on_write

    def _make_key(self, resource_id: str, version: Optional[int] = None) -> str:
        """Generate cache key."""
        if version is not None:
            return f"v:{resource_id}:{version}"
        return f"latest:{resource_id}"

    async def read(
        self,
        resource_id: str,
        version: Optional[int] = None,
    ) -> Optional[Any]:
        """Read with caching."""
        if not self.cache_reads:
            return await self.storage.read(resource_id, version)

        cache_key = self._make_key(resource_id, version)

        # Try cache first
        cached = await self.cache.get(cache_key)
        if cached is not None:
            return cached

        # Read from storage
        value = await self.storage.read(resource_id, version)

        # Cache the result
        if value is not None:
            await self.cache.set(cache_key, value)

        return value

    async def write(
        self,
        resource_id: str,
        content: Any,
        **kwargs,
    ) -> int:
        """Write with cache invalidation."""
        # Write to storage
        version = await self.storage.write(resource_id, content, **kwargs)

        # Invalidate cache
        if self.invalidate_on_write:
            await self.cache.delete(self._make_key(resource_id))

        # Cache the new version
        if self.cache_versions:
            await self.cache.set(
                self._make_key(resource_id, version),
                content
            )

        return version

    async def invalidate(self, resource_id: str) -> None:
        """Invalidate all cached versions of a resource."""
        await self.cache.delete(self._make_key(resource_id))

    async def clear_cache(self) -> None:
        """Clear all cached data."""
        await self.cache.clear()

    def get_cache_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.cache.stats()


# =============================================================================
# Query Result Cache
# =============================================================================

class QueryCache:
    """
    Cache for query results.

    Caches complex query results with automatic invalidation.
    """

    def __init__(
        self,
        cache: Optional[Cache[str, Any]] = None,
        default_ttl: float = 60.0,
    ):
        self.cache = cache or TTLCache(max_size=1000, default_ttl=default_ttl)
        self.default_ttl = default_ttl
        self._invalidation_keys: Dict[str, List[str]] = {}  # tag -> cache_keys

    def _compute_key(self, query: str, params: Dict[str, Any]) -> str:
        """Compute cache key for a query."""
        key_data = f"{query}:{sorted(params.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()

    async def get_or_compute(
        self,
        query: str,
        params: Dict[str, Any],
        compute_fn: Callable[[], Any],
        ttl: Optional[float] = None,
        tags: Optional[List[str]] = None,
    ) -> Any:
        """Get from cache or compute and cache result."""
        cache_key = self._compute_key(query, params)

        # Try cache
        result = await self.cache.get(cache_key)
        if result is not None:
            return result

        # Compute result
        if asyncio.iscoroutinefunction(compute_fn):
            result = await compute_fn()
        else:
            result = compute_fn()

        # Cache result
        await self.cache.set(cache_key, result, ttl or self.default_ttl)

        # Track tags for invalidation
        if tags:
            for tag in tags:
                if tag not in self._invalidation_keys:
                    self._invalidation_keys[tag] = []
                self._invalidation_keys[tag].append(cache_key)

        return result

    async def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all entries with a given tag."""
        if tag not in self._invalidation_keys:
            return 0

        keys = self._invalidation_keys.pop(tag)
        count = 0
        for key in keys:
            if await self.cache.delete(key):
                count += 1

        return count


# =============================================================================
# Decorator for caching
# =============================================================================

def cached(
    cache: Cache,
    key_fn: Optional[Callable[..., str]] = None,
    ttl: Optional[float] = None,
):
    """Decorator to cache function results."""

    def decorator(func: Callable) -> Callable:
        @asyncio.coroutine
        async def async_wrapper(*args, **kwargs):
            if key_fn:
                cache_key = key_fn(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{args}:{kwargs}"

            result = await cache.get(cache_key)
            if result is not None:
                return result

            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            await cache.set(cache_key, result, ttl)
            return result

        def sync_wrapper(*args, **kwargs):
            return asyncio.get_event_loop().run_until_complete(
                async_wrapper(*args, **kwargs)
            )

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# =============================================================================
# Global cache instance
# =============================================================================

_global_cache: Optional[LRUCache] = None


def get_global_cache() -> LRUCache:
    """Get or create the global cache."""
    global _global_cache
    if _global_cache is None:
        _global_cache = LRUCache(max_size=10000, default_ttl=300.0)
    return _global_cache


def set_global_cache(cache: LRUCache) -> None:
    """Set the global cache instance."""
    global _global_cache
    _global_cache = cache

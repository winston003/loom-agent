"""Performance optimization components for audio adapter.

Implements Phase 6 production optimizations (T089-T092):
- Connection pooling for external services
- LRU caching for frequently accessed data
- Concurrency control and rate limiting
- Resource cleanup and task management

Reference: specs/002-xiaozhi-voice-adapter/phase6/
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional, Callable, TypeVar, Generic
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from loom.core.structured_logger import get_logger

logger = get_logger("audio.performance")

T = TypeVar("T")


# ============================
# T090: LRU Cache with TTL
# ============================


@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with value and expiration."""
    value: T
    expires_at: float
    hit_count: int = 0


class TTLCache(Generic[T]):
    """Thread-safe LRU cache with TTL support.
    
    Features:
    - LRU eviction when max_size exceeded
    - TTL-based expiration
    - Hit/miss statistics
    - Async-safe
    
    Args:
        max_size: Maximum number of entries (default: 1000)
        ttl_seconds: Time-to-live in seconds (default: 300 = 5 min)
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

    async def get(self, key: str) -> Optional[T]:
        """Get value from cache if not expired."""
        async with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._misses += 1
                return None
            
            # Check expiration
            if time.time() > entry.expires_at:
                del self._cache[key]
                self._misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.hit_count += 1
            self._hits += 1
            
            return entry.value

    async def set(self, key: str, value: T) -> None:
        """Set value in cache with TTL."""
        async with self._lock:
            expires_at = time.time() + self.ttl_seconds
            
            if key in self._cache:
                # Update existing
                self._cache[key].value = value
                self._cache[key].expires_at = expires_at
                self._cache.move_to_end(key)
            else:
                # Add new entry
                if len(self._cache) >= self.max_size:
                    # Evict LRU (first item)
                    evicted_key = next(iter(self._cache))
                    del self._cache[evicted_key]
                    logger.debug("Cache eviction", key=evicted_key)
                
                self._cache[key] = CacheEntry(
                    value=value,
                    expires_at=expires_at,
                    hit_count=0
                )

    async def delete(self, key: str) -> bool:
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
            self._hits = 0
            self._misses = 0

    async def cleanup_expired(self) -> int:
        """Remove expired entries, return count removed."""
        async with self._lock:
            now = time.time()
            expired_keys = [
                key for key, entry in self._cache.items()
                if now > entry.expires_at
            ]
            
            for key in expired_keys:
                del self._cache[key]
            
            if expired_keys:
                logger.debug("Cleaned up expired cache entries", count=len(expired_keys))
            
            return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "ttl_seconds": self.ttl_seconds,
        }


# ============================
# T091: Concurrency Control
# ============================


class ConcurrencyLimiter:
    """Limit concurrent operations with semaphore.
    
    Supports both global and per-key (e.g., per-device) limits.
    
    Args:
        global_limit: Maximum concurrent operations globally
        per_key_limit: Maximum concurrent operations per key
    """

    def __init__(self, global_limit: int = 100, per_key_limit: int = 5):
        self.global_limit = global_limit
        self.per_key_limit = per_key_limit
        
        self._global_semaphore = asyncio.Semaphore(global_limit)
        self._key_semaphores: Dict[str, asyncio.Semaphore] = {}
        self._lock = asyncio.Lock()
        
        # Metrics
        self._total_acquired = 0
        self._total_rejected = 0
        self._current_global = 0
        self._current_per_key: Dict[str, int] = {}

    async def acquire(self, key: Optional[str] = None) -> bool:
        """Acquire concurrency slot.
        
        Args:
            key: Optional key for per-key limiting (e.g., device_id)
        
        Returns:
            True if acquired, False if rejected
        """
        # Global limit check
        if not self._global_semaphore.locked():
            await self._global_semaphore.acquire()
        else:
            self._total_rejected += 1
            logger.warning("Global concurrency limit reached", limit=self.global_limit)
            return False
        
        # Per-key limit check
        if key:
            async with self._lock:
                if key not in self._key_semaphores:
                    self._key_semaphores[key] = asyncio.Semaphore(self.per_key_limit)
                    self._current_per_key[key] = 0
            
            semaphore = self._key_semaphores[key]
            if not semaphore.locked():
                await semaphore.acquire()
                async with self._lock:
                    self._current_per_key[key] += 1
            else:
                # Release global semaphore since per-key failed
                self._global_semaphore.release()
                self._total_rejected += 1
                logger.warning(
                    "Per-key concurrency limit reached",
                    key=key,
                    limit=self.per_key_limit
                )
                return False
        
        self._total_acquired += 1
        async with self._lock:
            self._current_global += 1
        
        return True

    async def release(self, key: Optional[str] = None) -> None:
        """Release concurrency slot."""
        if key and key in self._key_semaphores:
            self._key_semaphores[key].release()
            async with self._lock:
                self._current_per_key[key] = max(0, self._current_per_key[key] - 1)
        
        self._global_semaphore.release()
        async with self._lock:
            self._current_global = max(0, self._current_global - 1)

    async def __aenter__(self):
        """Context manager entry."""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.release()

    def get_stats(self) -> Dict[str, Any]:
        """Get limiter statistics."""
        return {
            "global_limit": self.global_limit,
            "per_key_limit": self.per_key_limit,
            "current_global": self._current_global,
            "total_acquired": self._total_acquired,
            "total_rejected": self._total_rejected,
            "rejection_rate": (
                self._total_rejected / (self._total_acquired + self._total_rejected)
                if (self._total_acquired + self._total_rejected) > 0
                else 0.0
            ),
            "active_keys": len([k for k, v in self._current_per_key.items() if v > 0]),
        }


# ============================
# T089: Connection Pool
# ============================


@dataclass
class PooledConnection(Generic[T]):
    """Pooled connection wrapper."""
    connection: T
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    is_healthy: bool = True


class ConnectionPool(Generic[T]):
    """Async connection pool for external services.
    
    Features:
    - Lazy connection creation
    - Connection reuse
    - Health checking
    - Max lifetime enforcement
    - Automatic cleanup
    
    Args:
        factory: Async function to create new connections
        min_size: Minimum pool size (default: 2)
        max_size: Maximum pool size (default: 10)
        max_lifetime: Max connection lifetime in seconds (default: 3600 = 1h)
        health_check: Optional async health check function
    """

    def __init__(
        self,
        factory: Callable[[], Any],
        min_size: int = 2,
        max_size: int = 10,
        max_lifetime: float = 3600,
        health_check: Optional[Callable[[T], bool]] = None,
    ):
        self.factory = factory
        self.min_size = min_size
        self.max_size = max_size
        self.max_lifetime = max_lifetime
        self.health_check = health_check
        
        self._pool: list[PooledConnection[T]] = []
        self._lock = asyncio.Lock()
        self._initialized = False
        
        # Metrics
        self._total_created = 0
        self._total_destroyed = 0
        self._total_checkouts = 0

    async def initialize(self) -> None:
        """Initialize pool with min_size connections."""
        if self._initialized:
            return
        
        async with self._lock:
            for _ in range(self.min_size):
                conn = await self._create_connection()
                self._pool.append(conn)
            
            self._initialized = True
            logger.info(
                "Connection pool initialized",
                min_size=self.min_size,
                max_size=self.max_size
            )

    async def _create_connection(self) -> PooledConnection[T]:
        """Create new pooled connection."""
        connection = await self.factory()
        self._total_created += 1
        
        return PooledConnection(
            connection=connection,
            created_at=time.time(),
            last_used=time.time(),
            use_count=0,
            is_healthy=True,
        )

    async def acquire(self) -> T:
        """Acquire connection from pool."""
        if not self._initialized:
            await self.initialize()
        
        async with self._lock:
            # Find healthy connection
            now = time.time()
            for pooled in self._pool:
                # Check if still healthy and not expired
                if (pooled.is_healthy and
                    (now - pooled.created_at) < self.max_lifetime):
                    
                    # Optional health check
                    if self.health_check:
                        try:
                            pooled.is_healthy = await self.health_check(pooled.connection)
                            if not pooled.is_healthy:
                                continue
                        except Exception as e:
                            logger.warning("Health check failed", error=str(e))
                            pooled.is_healthy = False
                            continue
                    
                    # Use this connection
                    pooled.last_used = now
                    pooled.use_count += 1
                    self._total_checkouts += 1
                    self._pool.remove(pooled)
                    
                    return pooled.connection
            
            # No healthy connection available, create new if under limit
            if len(self._pool) < self.max_size:
                pooled = await self._create_connection()
                pooled.use_count = 1
                self._total_checkouts += 1
                return pooled.connection
            
            # Pool exhausted
            logger.warning("Connection pool exhausted", size=len(self._pool))
            raise RuntimeError("Connection pool exhausted")

    async def release(self, connection: T) -> None:
        """Return connection to pool."""
        async with self._lock:
            # Find matching pooled connection or create wrapper
            pooled = PooledConnection(
                connection=connection,
                last_used=time.time(),
                is_healthy=True,
            )
            
            # Add back to pool if under max_size
            if len(self._pool) < self.max_size:
                self._pool.append(pooled)
            else:
                # Pool full, destroy connection
                await self._destroy_connection(connection)

    async def _destroy_connection(self, connection: T) -> None:
        """Destroy connection (cleanup)."""
        try:
            if hasattr(connection, "close"):
                await connection.close()
            elif hasattr(connection, "aclose"):
                await connection.aclose()
        except Exception as e:
            logger.warning("Error closing connection", error=str(e))
        finally:
            self._total_destroyed += 1

    async def cleanup(self) -> int:
        """Remove expired/unhealthy connections."""
        async with self._lock:
            now = time.time()
            expired = []
            
            for pooled in self._pool:
                if (not pooled.is_healthy or
                    (now - pooled.created_at) >= self.max_lifetime):
                    expired.append(pooled)
            
            for pooled in expired:
                self._pool.remove(pooled)
                await self._destroy_connection(pooled.connection)
            
            if expired:
                logger.debug("Cleaned up expired connections", count=len(expired))
            
            return len(expired)

    async def close_all(self) -> None:
        """Close all connections in pool."""
        async with self._lock:
            for pooled in self._pool:
                await self._destroy_connection(pooled.connection)
            
            self._pool.clear()
            self._initialized = False
            logger.info("Connection pool closed")

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            "size": len(self._pool),
            "min_size": self.min_size,
            "max_size": self.max_size,
            "total_created": self._total_created,
            "total_destroyed": self._total_destroyed,
            "total_checkouts": self._total_checkouts,
            "max_lifetime": self.max_lifetime,
        }


# ============================
# T092: Resource Cleanup Manager
# ============================


class ResourceCleanupManager:
    """Periodic cleanup of expired resources.
    
    Manages background tasks for:
    - Cache cleanup
    - Connection pool cleanup
    - Session timeout cleanup
    
    Args:
        cleanup_interval: Seconds between cleanup runs (default: 60)
    """

    def __init__(self, cleanup_interval: float = 60):
        self.cleanup_interval = cleanup_interval
        self._tasks: Dict[str, asyncio.Task] = {}
        self._running = False
        self._cleanup_handlers: Dict[str, Callable[[], Any]] = {}

    def register_handler(self, name: str, handler: Callable[[], Any]) -> None:
        """Register cleanup handler.
        
        Args:
            name: Handler identifier
            handler: Async function to call during cleanup
        """
        self._cleanup_handlers[name] = handler
        logger.debug("Registered cleanup handler", name=name)

    async def start(self) -> None:
        """Start background cleanup tasks."""
        if self._running:
            return
        
        self._running = True
        
        for name, handler in self._cleanup_handlers.items():
            task = asyncio.create_task(self._cleanup_loop(name, handler))
            self._tasks[name] = task
        
        logger.info(
            "Resource cleanup manager started",
            handlers=len(self._cleanup_handlers),
            interval=self.cleanup_interval
        )

    async def stop(self) -> None:
        """Stop all cleanup tasks."""
        self._running = False
        
        for name, task in self._tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._tasks.clear()
        logger.info("Resource cleanup manager stopped")

    async def _cleanup_loop(self, name: str, handler: Callable[[], Any]) -> None:
        """Background cleanup loop."""
        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                start = time.time()
                result = await handler()
                elapsed = (time.time() - start) * 1000
                
                logger.debug(
                    "Cleanup completed",
                    handler=name,
                    result=result,
                    elapsed_ms=elapsed
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Cleanup error",
                    handler=name,
                    error=str(e),
                    exc_info=True
                )

    def get_stats(self) -> Dict[str, Any]:
        """Get cleanup manager statistics."""
        return {
            "running": self._running,
            "interval": self.cleanup_interval,
            "handlers": len(self._cleanup_handlers),
            "active_tasks": len([t for t in self._tasks.values() if not t.done()]),
        }

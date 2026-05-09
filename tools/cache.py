"""
TTL Cache module for Agent Tools.

Provides an in-memory cache with time-to-live eviction to avoid
redundant API calls and rate limiting.
"""

import json
import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class CacheEntry:
    """A single cache entry with its stored value and creation timestamp."""

    value: str
    timestamp: float


class TTLCache:
    """In-memory cache with time-to-live eviction.

    Stores string results keyed by function name + parameters.
    Thread-safe via threading.Lock on all read/write operations.
    """

    def __init__(self, default_ttl: int = 300, max_age: int = 900):
        """Initialize the cache.

        Args:
            default_ttl: Time-to-live in seconds (default 300 = 5 minutes).
                         Entries older than this are considered expired on get().
            max_age: Maximum entry age before forced eviction (default 900 = 15 minutes).
                     Entries older than this are removed during eviction sweeps.
        """
        self.default_ttl = default_ttl
        self.max_age = max_age
        self._store: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()

    def make_key(self, func_name: str, **kwargs) -> str:
        """Generate a deterministic cache key from function name and parameters.

        Uses sorted JSON serialization to ensure the same parameters always
        produce the same key regardless of argument order.

        Args:
            func_name: The name of the tool function.
            **kwargs: The parameters passed to the function.

        Returns:
            A string key in the format "func_name:{sorted_params_json}".
        """
        sorted_params_json = json.dumps(kwargs, sort_keys=True)
        return f"{func_name}:{sorted_params_json}"

    def get(self, key: str) -> Optional[str]:
        """Return cached value if it exists and is within TTL, else None.

        Always triggers _evict_stale() to clean up old entries.

        Args:
            key: The cache key to look up.

        Returns:
            The cached string value if valid, or None if expired/missing.
        """
        with self._lock:
            self._evict_stale()
            entry = self._store.get(key)
            if entry is None:
                return None
            if time.time() - entry.timestamp > self.default_ttl:
                return None
            return entry.value

    def set(self, key: str, value: str) -> None:
        """Store a value in the cache with the current timestamp.

        Args:
            key: The cache key.
            value: The string value to cache.
        """
        with self._lock:
            self._store[key] = CacheEntry(value=value, timestamp=time.time())

    def _evict_stale(self) -> None:
        """Remove entries older than max_age (15 minutes by default).

        This method is called internally and assumes the lock is already held.
        """
        current_time = time.time()
        stale_keys = [
            key
            for key, entry in self._store.items()
            if current_time - entry.timestamp > self.max_age
        ]
        for key in stale_keys:
            del self._store[key]

    def clear(self) -> None:
        """Remove all entries from the cache. Useful for testing."""
        with self._lock:
            self._store.clear()

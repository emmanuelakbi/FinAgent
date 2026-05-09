"""
Tests for the TTL Cache module.

Covers:
- Cache round-trip within TTL (Property 2)
- Cache expiry after TTL (Property 3)
- Cache key determinism and uniqueness (Property 4)
- Cache eviction of stale entries (Property 5)
"""

from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st

from tools.cache import TTLCache


class TestCacheExpiryAfterTTL:
    """Feature: agent-tools, Property 3: Cache expiry after TTL

    **Validates: Requirements 3.2**

    For any cache key and value, if the entry's timestamp is older than
    5 minutes (300 seconds), a subsequent get() call SHALL return None,
    indicating a cache miss.
    """

    @given(
        key=st.text(min_size=1, max_size=100),
        value=st.text(min_size=1, max_size=200),
        extra_seconds=st.floats(min_value=0.01, max_value=600.0),
    )
    @settings(max_examples=100)
    def test_get_returns_none_after_ttl_expires(self, key, value, extra_seconds):
        """After advancing time past the 5-minute TTL, get() returns None."""
        cache = TTLCache(default_ttl=300)

        base_time = 1000000.0

        # Store the entry at base_time
        with patch("time.time", return_value=base_time):
            cache.set(key, value)

        # Retrieve after TTL has expired (300 + extra_seconds past base_time)
        expired_time = base_time + 300 + extra_seconds
        with patch("time.time", return_value=expired_time):
            result = cache.get(key)

        assert result is None, (
            f"Expected None after TTL expiry, got {result!r}. "
            f"Key={key!r}, elapsed={300 + extra_seconds}s"
        )

from hypothesis import given, settings
from hypothesis import strategies as st

from tools.cache import TTLCache


class TestCacheRoundTripWithinTTL:
    """Feature: agent-tools, Property 2: Cache round-trip within TTL

    For any cache key string and for any non-empty value string, storing the value
    in the cache and immediately retrieving it (within the 5-minute TTL window)
    SHALL return the exact same value string.

    **Validates: Requirements 3.1**
    """

    @given(
        key=st.text(min_size=1),
        value=st.text(min_size=1),
    )
    @settings(max_examples=100)
    def test_set_then_get_returns_exact_value(self, key: str, value: str) -> None:
        """Storing a value and immediately retrieving it returns the exact same value."""
        cache = TTLCache()
        cache.set(key, value)
        result = cache.get(key)
        assert result == value

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from tools.cache import TTLCache


# --- Property 4: Cache key determinism and uniqueness ---
# Feature: agent-tools, Property 4: Cache key determinism and uniqueness


# Strategy for generating param values (string, int, or float)
param_values = st.one_of(
    st.text(min_size=0, max_size=50),
    st.integers(min_value=-1000, max_value=1000),
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
)

# Strategy for generating params dicts with string keys and mixed values
params_strategy = st.dictionaries(
    keys=st.text(min_size=1, max_size=20),
    values=param_values,
    min_size=0,
    max_size=5,
)


class TestCacheKeyDeterminismAndUniqueness:
    """
    Property 4: Cache key determinism and uniqueness

    For any two calls with the same function name and identical parameter values
    (regardless of argument order), make_key SHALL produce the same key string.
    For any two calls with different function names OR different parameter values,
    make_key SHALL produce different key strings.

    **Validates: Requirements 3.3**
    """

    @settings(max_examples=100)
    @given(
        func_name=st.text(min_size=1, max_size=30),
        params=params_strategy,
    )
    def test_determinism_same_inputs_produce_same_key(self, func_name, params):
        """Same func_name and params always produce the same cache key."""
        cache = TTLCache()
        key1 = cache.make_key(func_name, **params)
        key2 = cache.make_key(func_name, **params)
        assert key1 == key2, (
            f"make_key is not deterministic: "
            f"got '{key1}' and '{key2}' for same inputs"
        )

    @settings(max_examples=100)
    @given(
        func_name1=st.text(min_size=1, max_size=30),
        func_name2=st.text(min_size=1, max_size=30),
        params1=params_strategy,
        params2=params_strategy,
    )
    def test_uniqueness_different_inputs_produce_different_keys(
        self, func_name1, func_name2, params1, params2
    ):
        """Different func_name OR different params produce different cache keys."""
        # Ensure at least one of func_name or params differs
        assume(func_name1 != func_name2 or params1 != params2)

        cache = TTLCache()
        key1 = cache.make_key(func_name1, **params1)
        key2 = cache.make_key(func_name2, **params2)
        assert key1 != key2, (
            f"make_key produced same key '{key1}' for different inputs: "
            f"({func_name1!r}, {params1!r}) vs ({func_name2!r}, {params2!r})"
        )

import time

from hypothesis import given, settings
from hypothesis import strategies as st

from tools.cache import CacheEntry, TTLCache


class TestCacheEvictionOfStaleEntries:
    """Feature: agent-tools, Property 5: Cache eviction of stale entries

    For any set of cache entries where one or more entries have timestamps
    older than 15 minutes, after any cache access (get or set), all entries
    older than 15 minutes SHALL be removed from the cache.

    **Validates: Requirements 3.5**
    """

    @given(
        num_stale=st.integers(min_value=1, max_value=10),
        num_fresh=st.integers(min_value=0, max_value=10),
        stale_age=st.floats(min_value=901.0, max_value=7200.0),
    )
    @settings(max_examples=100)
    def test_stale_entries_evicted_on_get(self, num_stale, num_fresh, stale_age):
        """Stale entries (>15 min old) are removed when get() is called.

        **Validates: Requirements 3.5**
        """
        cache = TTLCache(default_ttl=300, max_age=900)
        current_time = time.time()

        # Insert stale entries directly into _store
        for i in range(num_stale):
            key = f"stale_key_{i}"
            cache._store[key] = CacheEntry(
                value=f"stale_value_{i}",
                timestamp=current_time - stale_age,
            )

        # Insert fresh entries directly into _store
        for i in range(num_fresh):
            key = f"fresh_key_{i}"
            cache._store[key] = CacheEntry(
                value=f"fresh_value_{i}",
                timestamp=current_time,
            )

        # Trigger eviction via get()
        cache.get("any_key")

        # Verify all stale entries are removed
        for i in range(num_stale):
            assert f"stale_key_{i}" not in cache._store

        # Verify fresh entries are still present
        for i in range(num_fresh):
            assert f"fresh_key_{i}" in cache._store

    @given(
        num_stale=st.integers(min_value=1, max_value=10),
        num_fresh=st.integers(min_value=0, max_value=10),
        stale_age=st.floats(min_value=901.0, max_value=7200.0),
        trigger_key=st.text(min_size=1, max_size=20),
        trigger_value=st.text(min_size=1, max_size=50),
    )
    @settings(max_examples=100)
    def test_stale_entries_evicted_on_set_then_get(
        self, num_stale, num_fresh, stale_age, trigger_key, trigger_value
    ):
        """Stale entries (>15 min old) are removed when a subsequent get() is
        triggered after set(). The set() itself does not evict, but the next
        get() call will.

        **Validates: Requirements 3.5**
        """
        cache = TTLCache(default_ttl=300, max_age=900)
        current_time = time.time()

        # Insert stale entries directly into _store
        for i in range(num_stale):
            key = f"stale_key_{i}"
            cache._store[key] = CacheEntry(
                value=f"stale_value_{i}",
                timestamp=current_time - stale_age,
            )

        # Insert fresh entries directly into _store
        for i in range(num_fresh):
            key = f"fresh_key_{i}"
            cache._store[key] = CacheEntry(
                value=f"fresh_value_{i}",
                timestamp=current_time,
            )

        # Trigger a set (does not evict by itself)
        cache.set(trigger_key, trigger_value)

        # Now trigger eviction via get()
        cache.get("lookup_key")

        # Verify all stale entries are removed
        for i in range(num_stale):
            assert f"stale_key_{i}" not in cache._store

        # Verify fresh entries are still present
        for i in range(num_fresh):
            assert f"fresh_key_{i}" in cache._store

        # Verify the newly set entry is still present (it was just created)
        assert trigger_key in cache._store

    @given(
        num_entries=st.integers(min_value=1, max_value=15),
        stale_age=st.floats(min_value=901.0, max_value=7200.0),
    )
    @settings(max_examples=100)
    def test_all_stale_entries_evicted_leaves_empty_store(
        self, num_entries, stale_age
    ):
        """When all entries are stale, the store should be empty after eviction.

        **Validates: Requirements 3.5**
        """
        cache = TTLCache(default_ttl=300, max_age=900)
        current_time = time.time()

        # Insert only stale entries
        for i in range(num_entries):
            key = f"key_{i}"
            cache._store[key] = CacheEntry(
                value=f"value_{i}",
                timestamp=current_time - stale_age,
            )

        # Trigger eviction via get()
        cache.get("trigger")

        # All entries should be evicted
        assert len(cache._store) == 0

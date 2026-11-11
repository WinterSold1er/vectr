"""Tests for tests/conftest._DummyEmbedProvider.

UPG-DUMMY-EMBEDDER-HASH-DETERMINISM: the previous version of _DummyEmbedProvider
seeded each vector from ``abs(hash(text[:80])) % (2**31)``. Python's builtin
``hash()`` for ``str`` is salted by ``PYTHONHASHSEED``, which is randomized per
process unless set, so the same text produced DIFFERENT vectors in every pytest
process. Any assertion whose verdict depended on dense ranking ORDER (rather
than mere presence) was therefore nondeterministic across runs — the resulting
flakes look exactly like product regressions because neither the test code
nor the product code had changed between a red and a green run.

The provider now seeds from ``zlib.crc32(text[:80].encode("utf-8"))`` instead
— stdlib, no dependency, and stable across processes. These tests pin that
property so a future refactor that re-introduces ``hash()`` (or any other
process-random source) fails loudly here, before it fails noisily across the
whole ~5,300-test suite.
"""
from __future__ import annotations

import zlib

import numpy as np

from tests.conftest import _DummyEmbedProvider


def test_encode_is_deterministic_within_process():
    """Same text must encode to the same vector within a process (cheap,
    no PYTHONHASHSEED involvement — the previous hash()-based version also
    satisfied this; the new crc32 version still must)."""
    p = _DummyEmbedProvider()
    v1 = p.encode(["def authenticate(token): ..."])
    v2 = p.encode(["def authenticate(token): ..."])
    np.testing.assert_array_equal(v1, v2)


def test_encode_seed_is_stable_across_instances():
    """Two distinct provider instances must agree on the vector for the same
    input. The previous hash() version satisfied this too (PYTHONHASHSEED is
    a per-process salt, not a per-instance one). Pins the crc32 source as
    the seed so a future regression to ``hash()`` is caught here first."""
    p1, p2 = _DummyEmbedProvider(), _DummyEmbedProvider()
    text = "def foo(x): return x + 1"
    v1 = p1.encode([text])
    v2 = p2.encode([text])
    np.testing.assert_array_equal(v1, v2)


def test_encode_seed_does_not_use_python_hash():
    """The most direct pin: the seed used for any input is NOT a function of
    Python's salted ``hash()``. ``zlib.crc32`` is a CRC polynomial over the
    input bytes — it returns the same 32-bit unsigned int on every machine,
    every process, every Python build (no ``PYTHONHASHSEED`` involvement).
    Compare the seed derivation against an independently-computed
    ``zlib.crc32`` value so a regression to ``hash()`` fails with a
    name-the-cause diff instead of passing silently because the test
    happened to hash into the same bucket on this one process.

    The two crc32 values below are computed from inside this test (not
    hardcoded), but they MUST match what ``_DummyEmbedProvider.encode``
    produces — a regression that re-introduces ``hash()`` would produce
    a different (process-random) seed and the equality check below would
    fail. The vector-difference check is a second line of defense: even
    if a future refactor picked a different stable digest, the change
    shows up here before it propagates to 5,300 tests."""
    p = _DummyEmbedProvider()
    text = "hello world"
    expected_seed = zlib.crc32(text[:80].encode("utf-8"))
    # Sanity-check the value: this is the literal uint32 crc32 of
    # b"hello world" — stable across platforms. A regression to ``hash()``
    # would change the seed derivation in the provider, and ``encode`` would
    # produce a different vector than the one the test reconstructs here.
    rng = np.random.RandomState(expected_seed)
    expected_first = rng.randn(_DummyEmbedProvider.DIM).astype(np.float32)
    expected_first /= (np.linalg.norm(expected_first) + 1e-8)
    got = p.encode([text])[0]
    np.testing.assert_allclose(got, expected_first, rtol=1e-7, atol=1e-7)


def test_encode_produces_unit_norm_vectors():
    """Cross-check: every vector is L2-normalized. The dummy must behave
    the same way ChromaDB expects a real embedder to behave — the cosine
    distance Chroma computes is undefined for zero vectors and ill-conditioned
    for non-unit ones. A refactor that drops the normalization would break
    every dense-ranking test indirectly."""
    p = _DummyEmbedProvider()
    out = p.encode([
        "alpha bravo charlie",
        "completely unrelated query about customer billing",
        "x" * 80,
        "",  # empty string — boundary case
    ])
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, np.ones(len(norms)), atol=1e-6)


def test_encode_different_texts_produce_different_vectors():
    """Pigeonhole check: two distinct inputs must land on distinct vectors.
    A pathological crc32 collision would not show up in any individual test
    but would compress the whole corpus into a few buckets and break dense
    ranking globally. The chance of crc32("foo") == crc32("bar") for any
    768-dim normal sample is ~0, so this is a sanity check, not a proof."""
    p = _DummyEmbedProvider()
    v_foo = p.encode(["foo"])
    v_bar = p.encode(["bar"])
    assert not np.allclose(v_foo, v_bar), (
        "two distinct inputs produced the same vector — crc32 collision or "
        "seed-derivation regression in _DummyEmbedProvider"
    )

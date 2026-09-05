"""Naive Python neighbor sampler matching the native graph_sampler contract.

Consumes CSR ``offsets`` / ``neighbors`` views (typically from
``graph_sampler.BipartiteCSR``). It does not construct graphs, apply splits,
or ingest MovieLens files.

The sampling contract matches ``sagerec::BipartiteCSR::sample_neighbors``
(proposed ADR-005 implementation defaults; ADR-005 remains a proposal):

- Uniform without replacement
- ``k == 0`` or degree ``0`` -> empty sequence
- ``k >= degree`` -> full stored neighborhood in CSR order
- ``0 < k < degree`` -> ``k`` unique neighbors in Fisher-Yates prefix order
- Explicit per-call seed; the same ``(graph, node_id, k, seed)`` is reproducible
- No shared mutable RNG across calls
- Invalid ``node_id``, ``k < 0``, or ``seed < 0`` raise ``ValueError``

Partial-neighborhood draws use a C++ ``std::mt19937_64``-compatible engine
and the same unbiased ``uniform_below`` rejection sampler as
``cpp/src/bipartite_csr.cpp``.
"""

from __future__ import annotations

from collections.abc import Sequence

_UINT64_MASK = (1 << 64) - 1
_UINT64_MAX = _UINT64_MASK

# std::mt19937_64 parameters from C++ <random>.
_MT_W = 64
_MT_N = 312
_MT_M = 156
_MT_R = 31
_MT_A = 0xB5026F5AA96619E9
_MT_U = 29
_MT_D = 0x5555555555555555
_MT_S = 17
_MT_B = 0x71D67FFFEDA60000
_MT_T = 37
_MT_C = 0xFFF7EEE000000000
_MT_L = 43
_MT_F = 6364136223846793005
_MT_LOWER = (1 << _MT_R) - 1
_MT_UPPER = _UINT64_MASK ^ _MT_LOWER


class _Mt19937_64:
    """C++ ``std::mt19937_64`` engine seeded with a single uint64 value."""

    def __init__(self, seed: int) -> None:
        self._mt = [0] * _MT_N
        self._index = _MT_N
        self._seed(seed & _UINT64_MASK)

    def _seed(self, value: int) -> None:
        self._mt[0] = value
        for i in range(1, _MT_N):
            prev = self._mt[i - 1]
            mixed = prev ^ (prev >> (_MT_W - 2))
            self._mt[i] = (_MT_F * mixed + i) & _UINT64_MASK
        self._index = _MT_N

    def __call__(self) -> int:
        if self._index >= _MT_N:
            self._twist()
        y = self._mt[self._index]
        self._index += 1
        y ^= (y >> _MT_U) & _MT_D
        y ^= (y << _MT_S) & _MT_B
        y ^= (y << _MT_T) & _MT_C
        y ^= y >> _MT_L
        return y & _UINT64_MASK

    def _twist(self) -> None:
        mt = self._mt
        for i in range(_MT_N - _MT_M):
            y = (mt[i] & _MT_UPPER) | (mt[i + 1] & _MT_LOWER)
            mt[i] = mt[i + _MT_M] ^ (y >> 1) ^ (_MT_A if y & 1 else 0)
        for i in range(_MT_N - _MT_M, _MT_N - 1):
            y = (mt[i] & _MT_UPPER) | (mt[i + 1] & _MT_LOWER)
            mt[i] = mt[i + (_MT_M - _MT_N)] ^ (y >> 1) ^ (_MT_A if y & 1 else 0)
        y = (mt[_MT_N - 1] & _MT_UPPER) | (mt[0] & _MT_LOWER)
        mt[_MT_N - 1] = mt[_MT_M - 1] ^ (y >> 1) ^ (_MT_A if y & 1 else 0)
        self._index = 0


def _uniform_below(rng: _Mt19937_64, n: int) -> int:
    """Unbiased integer in ``[0, n)`` matching the native rejection sampler."""
    limit = (_UINT64_MAX // n) * n
    sample = rng()
    while sample >= limit:
        sample = rng()
    return sample % n


def _require_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Invalid {name} {value!r}; expected an integer")
    return value


def sample_neighbors(
    offsets: Sequence[int],
    neighbors: Sequence[int],
    node_id: int,
    k: int,
    seed: int,
) -> list[int]:
    """Sample neighbors from a CSR-like view using the native contract.

    ``offsets`` length must be ``num_nodes + 1``. ``neighbors[offsets[u]:offsets[u+1]]``
    is the stored neighborhood of global node ``u``.
    """
    node_id = _require_int("node_id", node_id)
    k = _require_int("k", k)
    seed = _require_int("seed", seed)

    if len(offsets) < 1:
        raise ValueError("Invalid offsets; expected length num_nodes + 1 (>= 1)")

    num_nodes = len(offsets) - 1
    if node_id < 0 or node_id >= num_nodes:
        raise ValueError(
            f"Invalid node_id {node_id}; expected range [0, {num_nodes})"
        )
    if k < 0:
        raise ValueError(f"Invalid sample size k={k}; expected k >= 0")
    if seed < 0:
        raise ValueError(f"Invalid seed {seed}; expected a non-negative integer")

    begin = int(offsets[node_id])
    end = int(offsets[node_id + 1])
    if begin < 0 or end < begin or end > len(neighbors):
        raise ValueError(
            f"Invalid CSR slice for node_id {node_id}: "
            f"offsets=[{begin}, {end}) neighbors_len={len(neighbors)}"
        )

    degree = end - begin
    if k == 0 or degree == 0:
        return []

    pool = [int(neighbors[index]) for index in range(begin, end)]
    if k >= degree:
        return pool

    rng = _Mt19937_64(seed)
    for i in range(k):
        remaining = degree - i
        swap_with = i + _uniform_below(rng, remaining)
        pool[i], pool[swap_with] = pool[swap_with], pool[i]
    return pool[:k]


def sample_neighbors_from_graph(
    graph: object,
    node_id: int,
    k: int,
    seed: int,
) -> list[int]:
    """Sample using ``offsets()`` / ``neighbors()`` from a BipartiteCSR-like object."""
    offsets = graph.offsets()  # type: ignore[attr-defined]
    neighbors = graph.neighbors()  # type: ignore[attr-defined]
    return sample_neighbors(offsets, neighbors, node_id, k, seed)

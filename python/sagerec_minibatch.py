"""Native-backed mini-batch neighborhood expansion (Phase 4 harness).

Builds a train-only ``graph_sampler.BipartiteCSR`` from local user-movie
pairs and expands seeded multi-hop neighborhoods by calling native
``BipartiteCSR.sample_neighbors`` (ADR-005: uniform without replacement).

This module does not implement GraphSAGE layers, PyTorch Geometric tensors,
model weights, or ranking metrics. Callers must pass training positives
only; held-out edges must never enter the sampling graph.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

_NONNEG_INT64_MASK = (1 << 63) - 1
_HOP_MIX = 0x9E3779B97F4A7C15
_INDEX_MIX = 0xBF58476D1CE4E5B9


def _require_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Invalid {name} {value!r}; expected an integer")
    return value


def _require_nonneg_int(name: str, value: object) -> int:
    parsed = _require_int(name, value)
    if parsed < 0:
        raise ValueError(f"Invalid {name} {parsed}; expected a non-negative integer")
    return parsed


def _compiled_extension_path(module: ModuleType) -> Path:
    path_text = getattr(module, "__file__", None)
    if not path_text:
        raise ImportError(
            "graph_sampler has no __file__; expected the compiled pybind11 "
            "extension (graph_sampler*.so). Build the Release CMake target and "
            "put the build directory on PYTHONPATH. Mini-batch sampling must "
            "not fall back to a pure-Python stub."
        )
    return Path(path_text)


def _assert_compiled_graph_sampler(module: ModuleType) -> None:
    path = _compiled_extension_path(module)
    name = path.name
    compiled = name.startswith("graph_sampler") and (
        name.endswith(".so") or name.endswith(".pyd") or ".so." in name
    )
    if not compiled:
        raise ImportError(
            f"graph_sampler loaded from {str(path)!r}; expected compiled "
            "graph_sampler*.so (put the CMake build directory on PYTHONPATH). "
            "Mini-batch neighborhood expansion must call the native extension."
        )
    if not hasattr(module, "BipartiteCSR"):
        raise ImportError(
            "graph_sampler is missing BipartiteCSR; rebuild the pybind11 "
            "extension and put the build directory on PYTHONPATH."
        )
    sample = getattr(module.BipartiteCSR, "sample_neighbors", None)
    if sample is None or not callable(sample):
        raise ImportError(
            "graph_sampler.BipartiteCSR.sample_neighbors is missing; rebuild "
            "the native extension. Mini-batch sampling cannot proceed."
        )


def load_graph_sampler() -> ModuleType:
    """Import the compiled ``graph_sampler`` extension or fail actionably."""
    try:
        import graph_sampler as module
    except ImportError as exc:
        raise ImportError(
            "Compiled graph_sampler extension is required for mini-batch "
            "neighborhood sampling. Configure and build the Release CMake "
            "target, then put the build directory on PYTHONPATH "
            "(for example PYTHONPATH=build:python)."
        ) from exc
    _assert_compiled_graph_sampler(module)
    return module


def train_csr_from_pairs(
    num_users: int,
    num_movies: int,
    train_pairs: Sequence[tuple[int, int]],
) -> Any:
    """Build a train-only native CSR from local ``(user_id, movie_id)`` pairs.

    Callers must pass training positives only. The native constructor stores
    each unique pair in both directions.
    """
    num_users = _require_nonneg_int("num_users", num_users)
    num_movies = _require_nonneg_int("num_movies", num_movies)
    pairs = [(int(user_id), int(movie_id)) for user_id, movie_id in train_pairs]
    module = load_graph_sampler()
    return module.BipartiteCSR(num_users, num_movies, pairs)


def derived_sample_seed(base_seed: int, hop: int, source_index: int) -> int:
    """Non-negative seed for one native ``sample_neighbors`` call.

    Mixing is independent of ``PYTHONHASHSEED``. Direct
    ``NativeMinibatchSampler.sample_neighbors`` passes ``seed`` through
    unchanged; only multi-hop expansion derives per-call seeds.
    """
    base_seed = _require_nonneg_int("seed", base_seed)
    hop = _require_nonneg_int("hop", hop)
    source_index = _require_nonneg_int("source_index", source_index)
    mixed = base_seed & _NONNEG_INT64_MASK
    mixed = (mixed + (hop + 1) * _HOP_MIX) & _NONNEG_INT64_MASK
    mixed = (mixed + (source_index + 1) * _INDEX_MIX) & _NONNEG_INT64_MASK
    return mixed


@dataclass(frozen=True)
class NeighborhoodBatch:
    """Seeded multi-hop samples produced by native ``sample_neighbors``.

    ``sources[h][i]`` is the global node expanded at hop ``h``, index ``i``.
    ``hops[h][i]`` is that node's ADR-005 neighbor sample. Hop 0 sources are
    ``seed_nodes``. Hop ``h+1`` sources are the concatenation of hop ``h``
    neighbor lists (frontier expansion).
    """

    seed_nodes: tuple[int, ...]
    fanouts: tuple[int, ...]
    seed: int
    sources: tuple[tuple[int, ...], ...]
    hops: tuple[tuple[tuple[int, ...], ...], ...]


class NativeMinibatchSampler:
    """Train-only neighborhood expander that always calls ``graph_sampler``."""

    def __init__(self, graph: object) -> None:
        module = load_graph_sampler()
        if not isinstance(graph, module.BipartiteCSR):
            raise TypeError(
                "Expected graph_sampler.BipartiteCSR, got "
                f"{type(graph).__name__}. Mini-batch neighborhood expansion "
                "must use the compiled native graph; do not pass a Python "
                "stand-in or reference-sampler wrapper."
            )
        self._graph = graph

    @classmethod
    def from_train_pairs(
        cls,
        num_users: int,
        num_movies: int,
        train_pairs: Sequence[tuple[int, int]],
    ) -> NativeMinibatchSampler:
        """Construct from local training-positive pairs only."""
        return cls(train_csr_from_pairs(num_users, num_movies, train_pairs))

    @classmethod
    def from_graph(cls, graph: object) -> NativeMinibatchSampler:
        """Wrap an existing compiled ``BipartiteCSR`` (offsets/neighbors native)."""
        return cls(graph)

    @property
    def graph(self) -> Any:
        return self._graph

    @property
    def num_users(self) -> int:
        return int(self._graph.num_users)

    @property
    def num_movies(self) -> int:
        return int(self._graph.num_movies)

    @property
    def num_nodes(self) -> int:
        return int(self._graph.num_nodes)

    def offsets(self) -> list[int]:
        return list(self._graph.offsets())

    def neighbors(self) -> list[int]:
        return list(self._graph.neighbors())

    def degree(self, node_id: int) -> int:
        return int(self._graph.degree(_require_int("node_id", node_id)))

    def sample_neighbors(self, node_id: int, k: int, seed: int) -> list[int]:
        """ADR-005 one-hop sample via native ``graph_sampler``.

        The ``(graph, node_id, k, seed)`` tuple is passed through unchanged.
        """
        node_id = _require_int("node_id", node_id)
        k = _require_int("k", k)
        seed = _require_int("seed", seed)
        return list(self._graph.sample_neighbors(node_id, k, seed))

    def sample_multihop(
        self,
        seed_nodes: Sequence[int],
        fanouts: Sequence[int],
        seed: int,
    ) -> NeighborhoodBatch:
        """Expand ``fanouts`` hops from ``seed_nodes`` using native sampling.

        Each native call uses ``derived_sample_seed(seed, hop, source_index)``.
        Empty ``seed_nodes``, isolated nodes, and ``k = 0`` yield empty
        neighbor lists (ADR-005). ``k >= degree`` returns the stored
        neighborhood in CSR order.
        """
        seed = _require_nonneg_int("seed", seed)
        seeds = tuple(_require_int("seed node", node) for node in seed_nodes)
        ks = tuple(_require_int("fanout", k) for k in fanouts)
        for k in ks:
            if k < 0:
                raise ValueError(f"Invalid fanout k={k}; expected k >= 0")

        hop_sources: list[tuple[int, ...]] = []
        hop_neighbors: list[tuple[tuple[int, ...], ...]] = []
        frontier = seeds
        for hop, k in enumerate(ks):
            hop_sources.append(frontier)
            sampled_lists: list[tuple[int, ...]] = []
            next_frontier: list[int] = []
            for index, node_id in enumerate(frontier):
                call_seed = derived_sample_seed(seed, hop, index)
                sampled = tuple(
                    self._graph.sample_neighbors(node_id, k, call_seed)
                )
                sampled_lists.append(sampled)
                next_frontier.extend(sampled)
            hop_neighbors.append(tuple(sampled_lists))
            frontier = tuple(next_frontier)

        return NeighborhoodBatch(
            seed_nodes=seeds,
            fanouts=ks,
            seed=seed,
            sources=tuple(hop_sources),
            hops=tuple(hop_neighbors),
        )

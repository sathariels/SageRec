"""Type stub for the pybind11 graph_sampler extension.

The compiled module is produced by the CMake target `graph_sampler`.
Users occupy [0, num_users); movies occupy [num_users, num_nodes).
Construction takes local (user_id, movie_id) pairs, not global movie IDs.
"""

from collections.abc import Sequence

class GraphError(ValueError):
    """Invalid graph construction or sample request."""

class BipartiteCSR:
    def __init__(
        self,
        num_users: int,
        num_movies: int,
        interactions: Sequence[tuple[int, int]],
    ) -> None: ...
    @property
    def num_users(self) -> int: ...
    @property
    def num_movies(self) -> int: ...
    @property
    def num_nodes(self) -> int: ...
    def degree(self, node_id: int) -> int: ...
    def sample_neighbors(self, node_id: int, k: int, seed: int) -> list[int]: ...
    def offsets(self) -> list[int]: ...
    def neighbors(self) -> list[int]: ...

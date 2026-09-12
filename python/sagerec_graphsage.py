"""GraphSAGE training on the native mini-batch harness (Phase 4 / Phase 5).

Neighborhood expansion always goes through ``sagerec_minibatch`` /
``graph_sampler.BipartiteCSR.sample_neighbors`` (ADR-005). This module does
not import or call a PyG ``NeighborLoader`` / neighbor sampler.

Layers are GraphSAGE-style mean aggregation implemented in PyTorch. PyTorch
Geometric remains the intended production training stack for later SAGEConv
/ tensor conversion; this slice is a native-backed PyTorch trainer, not a
NumPy stand-in. MovieLens 100K quality numbers come from
``movielens_100k_config`` + the 100K train/eval script, never from the tiny
synthetic protocol smoke.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

import sagerec_minibatch as minibatch
import sagerec_negatives as negatives
from sagerec_prep import SplitResult

_TORCH_IMPORT_ERROR = (
    "PyTorch is required for GraphSAGE training. Install the CPU wheel, for "
    "example: python3 -m pip install torch==2.6.0 "
    "--index-url https://download.pytorch.org/whl/cpu "
    "(see python/requirements-train.txt). Native C++ stays free of PyTorch."
)


def require_torch() -> Any:
    """Import torch or fail with an actionable install hint."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(_TORCH_IMPORT_ERROR) from exc
    return torch


torch = require_torch()


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an int, got {type(value).__name__}")
    return value


def _require_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a real number, got {type(value).__name__}")
    return float(value)


def _require_fanouts(value: Any) -> tuple[int, ...]:
    if isinstance(value, bool) or not isinstance(value, Sequence):
        raise ValueError(f"fanouts must be a sequence of ints, got {type(value).__name__}")
    fanouts = tuple(_require_int(item, f"fanouts[{index}]") for index, item in enumerate(value))
    if not fanouts:
        raise ValueError("fanouts must contain at least one hop")
    for index, k in enumerate(fanouts):
        if k < 0:
            raise ValueError(f"fanouts[{index}] must be >= 0, got {k}")
    return fanouts


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for a CPU-only training smoke."""
    seed = _require_int(seed, "seed")
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass(frozen=True)
class GraphSAGEConfig:
    """Explicit hyperparameters for native-backed GraphSAGE training."""

    embedding_dim: int = 8
    hidden_dim: int = 8
    n_layers: int = 2
    fanouts: tuple[int, ...] = (4, 4)
    n_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 0.05
    n_negatives: int = 2
    l2: float = 1e-4
    seed: int = 0
    init_std: float = 0.1
    optimizer: str = "sgd"

    def __post_init__(self) -> None:
        embedding_dim = _require_int(self.embedding_dim, "embedding_dim")
        hidden_dim = _require_int(self.hidden_dim, "hidden_dim")
        n_layers = _require_int(self.n_layers, "n_layers")
        n_epochs = _require_int(self.n_epochs, "n_epochs")
        batch_size = _require_int(self.batch_size, "batch_size")
        n_negatives = _require_int(self.n_negatives, "n_negatives")
        seed = _require_int(self.seed, "seed")
        learning_rate = _require_float(self.learning_rate, "learning_rate")
        l2 = _require_float(self.l2, "l2")
        init_std = _require_float(self.init_std, "init_std")
        fanouts = _require_fanouts(self.fanouts)
        if embedding_dim < 1:
            raise ValueError(f"embedding_dim must be >= 1, got {embedding_dim}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        if n_layers != len(fanouts):
            raise ValueError(
                "n_layers must match len(fanouts), "
                f"got n_layers={n_layers} and fanouts={fanouts}"
            )
        if n_epochs < 1:
            raise ValueError(f"n_epochs must be >= 1, got {n_epochs}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if n_negatives < 1:
            raise ValueError(f"n_negatives must be >= 1, got {n_negatives}")
        if seed < 0:
            raise ValueError(f"seed must be >= 0, got {seed}")
        if learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
        if l2 < 0.0:
            raise ValueError(f"l2 must be >= 0, got {l2}")
        if init_std <= 0.0:
            raise ValueError(f"init_std must be > 0, got {init_std}")
        if not isinstance(self.optimizer, str):
            raise ValueError(
                f"optimizer must be a string, got {type(self.optimizer).__name__}"
            )
        optimizer = self.optimizer.lower()
        if optimizer not in {"sgd", "adam"}:
            raise ValueError(
                f"optimizer must be 'sgd' or 'adam', got {self.optimizer!r}"
            )
        object.__setattr__(self, "fanouts", fanouts)
        object.__setattr__(self, "optimizer", optimizer)

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready hyperparameter block (fanouts as a list)."""
        return {
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "fanouts": list(self.fanouts),
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "n_negatives": self.n_negatives,
            "l2": self.l2,
            "seed": self.seed,
            "init_std": self.init_std,
            "optimizer": self.optimizer,
        }


def movielens_100k_config(seed: int = 7) -> GraphSAGEConfig:
    """Modest CPU-friendly MovieLens 100K GraphSAGE hyperparameters.

    The default seed is **7** so the run is apples-to-apples with
    ``results/mf_movielens_100k.json``. This uses **3 Adam epochs** (MF used
    1 SGD epoch); fairness is the shared ADR-003 ranking protocol, not
    identical wall-clock or optimizer.
    """
    seed = _require_int(seed, "seed")
    return GraphSAGEConfig(
        embedding_dim=16,
        hidden_dim=16,
        n_layers=2,
        fanouts=(8, 8),
        n_epochs=3,
        batch_size=256,
        learning_rate=0.01,
        n_negatives=2,
        l2=1e-4,
        seed=seed,
        init_std=0.1,
        optimizer="adam",
    )


def graphsage_training_seed_notes(seed: int) -> dict[str, int | str]:
    """Document per-step seeds derived from the experiment seed."""
    seed = _require_int(seed, "seed")
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    return {
        "experiment_seed": seed,
        "python_numpy_torch": seed,
        "pair_shuffle_and_negatives_rng": seed,
        "minibatch_neighborhood_seed": (
            "sagerec_minibatch.derived_sample_seed(seed, epoch, step)"
        ),
        "multihop_native_call_seed": (
            "sagerec_minibatch.derived_sample_seed(batch_seed, hop, source_index)"
        ),
        "eval_all_node_embedding_seed": seed,
    }


def _mean_neighbor_hidden(
    hidden: torch.Tensor,
    neighbor_lists: Sequence[Sequence[int]],
    index: Mapping[int, int],
) -> torch.Tensor:
    """Mean-pool neighbor rows; isolated sources stay the zero vector."""
    n_sources = len(neighbor_lists)
    dim = hidden.size(-1)
    if n_sources == 0:
        return hidden.new_zeros((0, dim))
    rows: list[int] = []
    cols: list[int] = []
    for row, neighbors in enumerate(neighbor_lists):
        for node in neighbors:
            rows.append(row)
            cols.append(index[int(node)])
    out = hidden.new_zeros((n_sources, dim))
    if not rows:
        return out
    row_idx = torch.tensor(rows, dtype=torch.long, device=hidden.device)
    col_idx = torch.tensor(cols, dtype=torch.long, device=hidden.device)
    out.index_add_(0, row_idx, hidden[col_idx])
    counts = torch.zeros(n_sources, dtype=hidden.dtype, device=hidden.device)
    counts.index_add_(
        0,
        row_idx,
        torch.ones(len(rows), dtype=hidden.dtype, device=hidden.device),
    )
    return out / counts.clamp(min=1.0).unsqueeze(-1)


class SAGEMeanLayer(torch.nn.Module):
    """One GraphSAGE mean layer: W · concat(self, mean(neighbors)).

    Intermediate layers use ReLU. The last layer stays linear so ranking
    scores are not forced through a non-negative squash.
    """

    def __init__(self, in_dim: int, out_dim: int, *, activate: bool) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_dim * 2, out_dim)
        self.activate = activate

    def forward(self, self_h: torch.Tensor, neigh_h: torch.Tensor) -> torch.Tensor:
        hidden = self.linear(torch.cat([self_h, neigh_h], dim=-1))
        if self.activate:
            return torch.relu(hidden)
        return hidden


class GraphSAGERecommender(torch.nn.Module):
    """Bipartite GraphSAGE encoder that scores user–item pairs.

    Implements ``PairScorer.score_pairs``. Neighborhoods come from
    ``NativeMinibatchSampler`` only.
    """

    def __init__(
        self,
        sampler: minibatch.NativeMinibatchSampler,
        config: GraphSAGEConfig,
    ) -> None:
        super().__init__()
        if not isinstance(sampler, minibatch.NativeMinibatchSampler):
            raise TypeError(
                "GraphSAGE training requires sagerec_minibatch.NativeMinibatchSampler, "
                f"got {type(sampler).__name__}. Do not substitute a PyG sampler."
            )
        self.sampler = sampler
        self.config = config
        self.embedding = torch.nn.Embedding(sampler.num_nodes, config.embedding_dim)
        torch.nn.init.normal_(self.embedding.weight, mean=0.0, std=config.init_std)
        dims = [config.embedding_dim]
        dims.extend([config.hidden_dim] * config.n_layers)
        self.layers = torch.nn.ModuleList(
            SAGEMeanLayer(
                dims[index],
                dims[index + 1],
                activate=index < config.n_layers - 1,
            )
            for index in range(config.n_layers)
        )
        self._eval_user: torch.Tensor | None = None
        self._eval_item: torch.Tensor | None = None

    @property
    def num_users(self) -> int:
        return self.sampler.num_users

    @property
    def num_items(self) -> int:
        return self.sampler.num_movies

    def item_global_id(self, item_id: int) -> int:
        return self.num_users + item_id

    def expand_neighborhoods(
        self,
        seed_nodes: Sequence[int],
        seed: int,
    ) -> minibatch.NeighborhoodBatch:
        """Single training/eval expansion entry; always native ``sample_neighbors``."""
        return self.sampler.sample_multihop(seed_nodes, self.config.fanouts, seed)

    def encode_nodes(self, seed_nodes: Sequence[int], sample_seed: int) -> torch.Tensor:
        """Encode global node IDs with GraphSAGE mean over native samples."""
        nodes = [int(node) for node in seed_nodes]
        if not nodes:
            out_dim = self.config.hidden_dim
            return torch.empty((0, out_dim), dtype=torch.float32)

        batch = self.expand_neighborhoods(nodes, sample_seed)
        involved: set[int] = set(nodes)
        for sources in batch.sources:
            involved.update(int(node) for node in sources)
        for hop in batch.hops:
            for neighbors in hop:
                involved.update(int(node) for node in neighbors)
        ordered = sorted(involved)
        index = {node: position for position, node in enumerate(ordered)}
        hidden = self.embedding(
            torch.tensor(ordered, dtype=torch.long, device=self.embedding.weight.device)
        )

        for layer_index, hop in enumerate(reversed(range(len(batch.fanouts)))):
            sources = batch.sources[hop]
            neighbor_lists = batch.hops[hop]
            if not sources:
                continue
            source_positions = [index[int(node)] for node in sources]
            self_h = hidden[source_positions]
            neigh_h = _mean_neighbor_hidden(hidden, neighbor_lists, index)
            updated = self.layers[layer_index](self_h, neigh_h)
            if updated.size(-1) != hidden.size(-1):
                next_hidden = hidden.new_zeros(hidden.size(0), updated.size(-1))
            else:
                next_hidden = hidden.clone()
            # Last occurrence wins if a node appears twice in this frontier.
            last_row: dict[int, int] = {}
            for row, position in enumerate(source_positions):
                last_row[position] = row
            write_positions = list(last_row.keys())
            write_rows = [last_row[position] for position in write_positions]
            next_hidden[write_positions] = updated[write_rows]
            hidden = next_hidden

        seed_positions = [index[node] for node in nodes]
        return hidden[seed_positions]

    def score_encoded(
        self,
        user_ids: Sequence[int],
        item_ids: Sequence[int],
        sample_seed: int,
    ) -> torch.Tensor:
        users = [int(user_id) for user_id in user_ids]
        items = [int(item_id) for item_id in item_ids]
        if len(users) != len(items):
            raise ValueError(
                "user_ids and item_ids must have the same length, "
                f"got {len(users)} and {len(items)}"
            )
        if not users:
            return torch.empty(0, dtype=torch.float32)
        globals_ids = users + [self.item_global_id(item_id) for item_id in items]
        encoded = self.encode_nodes(globals_ids, sample_seed)
        user_h = encoded[: len(users)]
        item_h = encoded[len(users) :]
        return (user_h * item_h).sum(dim=-1)

    def score_pairs(
        self,
        user_ids: Sequence[int],
        item_ids: Sequence[int],
    ) -> np.ndarray:
        if len(user_ids) != len(item_ids):
            raise ValueError(
                "user_ids and item_ids must have the same length, "
                f"got {len(user_ids)} and {len(item_ids)}"
            )
        users = np.asarray(user_ids, dtype=np.int64)
        items = np.asarray(item_ids, dtype=np.int64)
        if users.size == 0:
            return np.empty(0, dtype=np.float64)
        if np.any(users < 0) or np.any(users >= self.num_users):
            raise ValueError(
                f"user_ids must lie in [0, {self.num_users}), got min={int(users.min())} "
                f"max={int(users.max())}"
            )
        if np.any(items < 0) or np.any(items >= self.num_items):
            raise ValueError(
                f"item_ids must lie in [0, {self.num_items}), got min={int(items.min())} "
                f"max={int(items.max())}"
            )
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                if self._eval_user is None or self._eval_item is None:
                    self.materialize_eval_embeddings()
                assert self._eval_user is not None
                assert self._eval_item is not None
                user_h = self._eval_user[torch.as_tensor(users, dtype=torch.long)]
                item_h = self._eval_item[torch.as_tensor(items, dtype=torch.long)]
                scores = (user_h * item_h).sum(dim=-1)
        finally:
            if was_training:
                self.train()
        return scores.detach().cpu().numpy().astype(np.float64, copy=False)

    def clear_eval_embeddings(self) -> None:
        """Drop cached ranking embeddings so the next ``score_pairs`` re-encodes."""
        self._eval_user = None
        self._eval_item = None

    def materialize_eval_embeddings(self) -> None:
        """Encode every graph node once via native multi-hop sampling.

        Ranking then uses cached inner products. Training still samples
        independently per mini-batch through ``score_encoded``. The all-node
        encode uses ``config.seed`` (see ``graphsage_training_seed_notes``).
        """
        node_ids = list(range(self.sampler.num_nodes))
        encoded = self.encode_nodes(node_ids, self.config.seed)
        if encoded.size(0) != self.sampler.num_nodes:
            raise RuntimeError(
                "eval encode must return one row per graph node, "
                f"got {encoded.size(0)} for {self.sampler.num_nodes} nodes"
            )
        self._eval_user = encoded[: self.num_users].detach()
        self._eval_item = encoded[self.num_users :].detach()


def _unique_train_pairs(
    train_pairs: Sequence[tuple[int, int]],
    num_users: int,
    num_items: int,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for index, pair in enumerate(train_pairs):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(f"train_pairs[{index}] must be a (user_id, item_id) tuple")
        user_id = _require_int(pair[0], f"train_pairs[{index}].user_id")
        item_id = _require_int(pair[1], f"train_pairs[{index}].item_id")
        if user_id < 0 or user_id >= num_users:
            raise ValueError(
                f"train_pairs[{index}].user_id {user_id} is outside [0, {num_users})"
            )
        if item_id < 0 or item_id >= num_items:
            raise ValueError(
                f"train_pairs[{index}].item_id {item_id} is outside [0, {num_items})"
            )
        key = (user_id, item_id)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    if not pairs:
        raise ValueError("train_pairs is empty; GraphSAGE needs at least one training positive")
    return pairs


def fit_graphsage(
    train_pairs: Sequence[tuple[int, int]],
    *,
    num_users: int,
    num_items: int,
    config: GraphSAGEConfig | None = None,
    forbidden_pairs: Sequence[tuple[int, int]] | None = None,
    progress: Callable[[int, int, float], None] | None = None,
) -> GraphSAGERecommender:
    """Train GraphSAGE on ``train_pairs`` only, sampling via ``graph_sampler``.

    ``forbidden_pairs`` defaults to ``train_pairs``. Held-out positives must
    not appear in ``train_pairs`` and never enter the train-only CSR.
    """
    config = GraphSAGEConfig() if config is None else config
    num_users = _require_int(num_users, "num_users")
    num_items = _require_int(num_items, "num_items")
    if num_users < 1 or num_items < 1:
        raise ValueError(
            f"num_users and num_items must be >= 1, got {num_users}, {num_items}"
        )
    pairs = _unique_train_pairs(train_pairs, num_users, num_items)
    extra_forbidden = list(forbidden_pairs) if forbidden_pairs is not None else []
    forbidden_by_user = negatives.items_by_user(list(pairs) + extra_forbidden)
    for user_id, item_id in pairs:
        forbidden_by_user.setdefault(user_id, set()).add(item_id)
    forbidden_scope = {
        (user_id, item_id)
        for user_id, items in forbidden_by_user.items()
        for item_id in items
    }

    seed_everything(config.seed)
    sampler = minibatch.NativeMinibatchSampler.from_train_pairs(
        num_users, num_items, pairs
    )
    model = GraphSAGERecommender(sampler, config)
    model.train()
    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.l2,
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.l2,
        )
    rng = np.random.default_rng(config.seed)

    for epoch in range(config.n_epochs):
        order = rng.permutation(len(pairs))
        for step, start in enumerate(range(0, len(pairs), config.batch_size)):
            batch_index = [int(index) for index in order[start : start + config.batch_size]]
            pos_users = [pairs[index][0] for index in batch_index]
            pos_items = [pairs[index][1] for index in batch_index]
            neg_users: list[int] = []
            neg_items: list[int] = []
            for user_id in pos_users:
                forbidden_items = forbidden_by_user.get(user_id, set())
                for _ in range(config.n_negatives):
                    negative_item = negatives.sample_negative_item(
                        user_id,
                        num_items=num_items,
                        forbidden_items=forbidden_items,
                        rng=rng,
                    )
                    negatives.assert_no_positive_overlap(
                        [(user_id, negative_item)], forbidden_scope
                    )
                    neg_users.append(user_id)
                    neg_items.append(negative_item)

            sample_seed = minibatch.derived_sample_seed(config.seed, epoch, step)
            pos_scores = model.score_encoded(pos_users, pos_items, sample_seed)
            neg_scores = model.score_encoded(neg_users, neg_items, sample_seed)
            labels_pos = torch.ones_like(pos_scores)
            labels_neg = torch.zeros_like(neg_scores)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                torch.cat([pos_scores, neg_scores]),
                torch.cat([labels_pos, labels_neg]),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if progress is not None:
                progress(epoch, step, float(loss.detach().cpu()))

    model.eval()
    return model


def train_graphsage(
    split: SplitResult,
    config: GraphSAGEConfig | None = None,
    progress: Callable[[int, int, float], None] | None = None,
) -> GraphSAGERecommender:
    """Fit GraphSAGE on ADR-003 training positives only."""
    train_pairs = split.train_positive_pairs()
    held_out = split.positive_pair_set("validation") | split.positive_pair_set("test")
    overlap = set(train_pairs) & held_out
    if overlap:
        user_id, item_id = next(iter(overlap))
        raise ValueError(
            "held-out positive leaked into GraphSAGE train pairs: "
            f"(user_id={user_id}, item_id={item_id})"
        )
    return fit_graphsage(
        train_pairs,
        num_users=split.num_users,
        num_items=split.num_movies,
        config=config,
        progress=progress,
    )


def graphsage_result_payload(
    *,
    manifest: Mapping[str, Any],
    config: GraphSAGEConfig,
    report: Any,
    git_commit: str | None,
    generated_at: str,
    python_version: str,
    numpy_version: str,
    torch_version: str,
    platform_info: Mapping[str, str],
) -> dict[str, Any]:
    """Machine-readable MovieLens 100K GraphSAGE provenance (mirrors MF JSON)."""
    required_manifest = (
        "dataset_edition",
        "split_policy_id",
        "split_policy_version",
        "min_interactions_for_eval",
        "cold_start_policy",
        "counts",
    )
    missing = [key for key in required_manifest if key not in manifest]
    if missing:
        raise ValueError(f"manifest is missing required keys: {missing}")
    counts = manifest["counts"]
    return {
        "model": "graphsage",
        "dataset_edition": manifest["dataset_edition"],
        "source_url": manifest.get("source_url"),
        "checksum": manifest.get("checksum"),
        "license": manifest.get("license"),
        "split_policy_id": manifest["split_policy_id"],
        "split_policy_version": manifest["split_policy_version"],
        "min_interactions_for_eval": manifest["min_interactions_for_eval"],
        "cold_start_policy": manifest["cold_start_policy"],
        "seed": config.seed,
        "git_commit": git_commit,
        "generated_at": generated_at,
        "python": python_version,
        "numpy": numpy_version,
        "torch": torch_version,
        "platform": dict(platform_info),
        "hyperparams": config.as_dict(),
        "training_seeds": graphsage_training_seed_notes(config.seed),
        "sampler": (
            "graph_sampler.BipartiteCSR.sample_neighbors via "
            "sagerec_minibatch.NativeMinibatchSampler (ADR-005, no PyG NeighborLoader)"
        ),
        "metrics": {
            "k": report.k,
            "split": report.split,
            "recall_at_k": report.recall_at_k,
            "ndcg_at_k": report.ndcg_at_k,
            "n_evaluated_users": report.n_users,
        },
        "eligibility": {
            "users": counts["users"],
            "movies": counts["movies"],
            "interactions": counts["interactions"],
            "eligible_users": counts["eligible_users"],
            "cold_start_users": counts["cold_start_users"],
            "evaluated_users": report.n_users,
        },
        "note": (
            "Single-seed MovieLens 100K GraphSAGE run on the ADR-003 split. "
            "Neighborhoods come from native sagerec_minibatch / graph_sampler; "
            "not a PyG NeighborLoader run and not a multi-seed leaderboard. "
            f"Uses {config.n_epochs} epoch(s); fairness versus MF is the shared "
            "eval protocol, not identical wall-clock."
        ),
    }

#pragma once

#include "sagerec/error.hpp"
#include "sagerec/types.hpp"

#include <utility>
#include <vector>

namespace sagerec {

/// Train-only bipartite CSR over disjoint user and movie ID ranges.
///
/// Node-ID layout (ADR convention):
/// - users:  [0, num_users)
/// - movies: [num_users, num_users + num_movies)
///
/// Construction accepts local IDs: user_id in [0, num_users) and movie_id in
/// [0, num_movies). Each interaction becomes two directed adjacency entries
/// (user_global -> movie_global and movie_global -> user_global). Duplicate
/// interactions are dropped (proposed default). Neighborhoods are stored in
/// ascending global-ID order.
///
/// The graph does not parse MovieLens files and does not apply splits. Callers
/// must pass training-positive interactions only. MovieLens 100K text is parsed
/// by `parse_movielens_100k`; feed its `local_pairs()` here after restricting
/// to training positives.
///
/// Ownership: the instance owns offsets and neighbor buffers. Accessors return
/// references valid for the lifetime of the graph. sample_neighbors returns a
/// new vector.
///
/// Exceptions: GraphError on invalid dimensions, IDs, or sample size.
/// Construction is O(E log D) because each neighborhood is sorted. Sampling a
/// hop is O(degree) when copying the neighborhood, then O(k) for a partial
/// shuffle when 0 < k < degree.
class BipartiteCSR {
 public:
  /// Build a CSR from typed user-movie interactions.
  ///
  /// @param num_users number of user nodes; must be >= 0
  /// @param num_movies number of movie nodes; must be >= 0
  /// @param interactions local (user_id, movie_id) pairs
  static BipartiteCSR from_interactions(
      NodeId num_users,
      NodeId num_movies,
      const std::vector<std::pair<NodeId, NodeId>>& interactions);

  NodeId num_users() const noexcept { return num_users_; }
  NodeId num_movies() const noexcept { return num_movies_; }
  NodeId num_nodes() const noexcept { return num_users_ + num_movies_; }

  /// Number of stored directed adjacency entries (two per unique interaction).
  Offset num_directed_edges() const noexcept {
    return static_cast<Offset>(neighbors_.size());
  }

  Degree degree(NodeId node_id) const;

  /// offsets().size() == num_nodes() + 1. Lifetime: this object.
  const std::vector<Offset>& offsets() const noexcept { return offsets_; }

  /// Concatenated neighborhoods. Lifetime: this object.
  const std::vector<NodeId>& neighbors() const noexcept { return neighbors_; }

  /// Uniform sample without replacement (proposed ADR-005 implementation contract).
  ///
  /// - k == 0 or degree == 0: empty result
  /// - k >= degree: full neighborhood in CSR order
  /// - 0 < k < degree: k unique neighbors in Fisher-Yates prefix order
  /// - node_id outside [0, num_nodes) or k < 0: GraphError
  ///
  /// Thread-safe: const and uses a per-call RNG seeded by `seed`.
  std::vector<NodeId> sample_neighbors(NodeId node_id, Degree k, Seed seed) const;

 private:
  BipartiteCSR() = default;

  void validate_node(NodeId node_id) const;

  NodeId num_users_ = 0;
  NodeId num_movies_ = 0;
  std::vector<Offset> offsets_;
  std::vector<NodeId> neighbors_;
};

}  // namespace sagerec

#pragma once

#include "sagerec/types.hpp"

#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

namespace sagerec {

/// One normalized MovieLens 100K interaction after source-ID remapping.
///
/// `user_id` and `movie_id` are contiguous local IDs suitable for
/// `BipartiteCSR::from_interactions`. Rating and timestamp are the original
/// u.data values. This type does not assign a split (ADR-003 is still proposed).
struct MovieLens100kInteraction {
  NodeId user_id = 0;
  NodeId movie_id = 0;
  std::int32_t rating = 0;
  std::int64_t timestamp = 0;
};

/// Deterministic MovieLens 100K parse result.
///
/// Source user and movie IDs are remapped independently: each local ID is the
/// rank of that source ID among the sorted unique source IDs of its type.
/// `user_source_ids()[local]` and `movie_source_ids()[local]` recover the
/// original positive source IDs.
///
/// Interactions keep input order (blank lines omitted). The parser does not
/// build a CSR and does not apply a split; callers must pass training positives
/// only into `BipartiteCSR`.
///
/// Ownership: the instance owns interactions and mapping vectors. Accessors
/// return references valid for the lifetime of this object. `local_pairs()`
/// returns a new vector.
///
/// Exceptions: GraphError on empty input, malformed rows, non-integer fields,
/// nonpositive source IDs, integer overflow, or duplicate (source user, source
/// movie) pairs. Parsing is O(N log U + N log M) for N rows and U/M unique IDs.
class MovieLens100kRatings {
 public:
  NodeId num_users() const noexcept { return static_cast<NodeId>(user_source_ids_.size()); }
  NodeId num_movies() const noexcept { return static_cast<NodeId>(movie_source_ids_.size()); }

  /// Normalized interactions in input order. Lifetime: this object.
  const std::vector<MovieLens100kInteraction>& interactions() const noexcept {
    return interactions_;
  }

  /// Local user ID -> original source user ID. Lifetime: this object.
  const std::vector<std::int32_t>& user_source_ids() const noexcept { return user_source_ids_; }

  /// Local movie ID -> original source movie ID. Lifetime: this object.
  const std::vector<std::int32_t>& movie_source_ids() const noexcept {
    return movie_source_ids_;
  }

  /// Local (user_id, movie_id) pairs for `BipartiteCSR::from_interactions`.
  /// The graph still requires the caller to supply training positives only.
  std::vector<std::pair<NodeId, NodeId>> local_pairs() const;

 private:
  MovieLens100kRatings() = default;

  friend MovieLens100kRatings parse_movielens_100k(std::string_view text);

  std::vector<MovieLens100kInteraction> interactions_;
  std::vector<std::int32_t> user_source_ids_;
  std::vector<std::int32_t> movie_source_ids_;
};

/// Parse tab-separated MovieLens 100K `u.data` text.
///
/// Each non-blank line must be exactly four tab-separated fields:
/// `source_user_id`, `source_movie_id`, `rating`, `timestamp`. Source IDs must
/// be positive integers. Duplicate source user-movie pairs fail. Whitespace-only
/// lines are skipped. CRLF line endings are accepted.
///
/// Tests and callers pass in-memory text. This function does not read files,
/// download data, or parse MovieLens 1M (`::` delimited) ratings.
MovieLens100kRatings parse_movielens_100k(std::string_view text);

}  // namespace sagerec

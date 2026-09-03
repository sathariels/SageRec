#pragma once

#include <cstdint>

namespace sagerec {

/// Zero-based global graph node identifier.
/// Users occupy [0, num_users). Movies occupy [num_users, num_nodes).
using NodeId = std::int32_t;

/// CSR prefix-sum entry. offsets[0] == 0 and offsets.back() == neighbors.size().
using Offset = std::int64_t;

/// Neighborhood size and sample request size. Must be non-negative at API boundaries.
using Degree = std::int32_t;

/// Explicit sampler seed. The same (graph, node_id, k, seed) tuple is reproducible.
using Seed = std::uint64_t;

}  // namespace sagerec

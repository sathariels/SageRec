#include "sagerec/bipartite_csr.hpp"

#include <algorithm>
#include <limits>
#include <random>
#include <sstream>
#include <string>

namespace sagerec {
namespace {

std::string format_range_error(const char* name, NodeId value, NodeId begin, NodeId end) {
  std::ostringstream out;
  out << "Invalid " << name << ' ' << value << "; expected range [" << begin << ", " << end
      << ')';
  return out.str();
}

std::uint64_t uniform_below(std::mt19937_64& rng, std::uint64_t n) {
  const std::uint64_t limit = (std::numeric_limits<std::uint64_t>::max() / n) * n;
  std::uint64_t sample = rng();
  while (sample >= limit) {
    sample = rng();
  }
  return sample % n;
}

}  // namespace

BipartiteCSR BipartiteCSR::from_interactions(
    NodeId num_users,
    NodeId num_movies,
    const std::vector<std::pair<NodeId, NodeId>>& interactions) {
  if (num_users < 0 || num_movies < 0) {
    std::ostringstream out;
    out << "num_users and num_movies must be >= 0; got num_users=" << num_users
        << ", num_movies=" << num_movies;
    throw GraphError(out.str());
  }

  const std::int64_t total =
      static_cast<std::int64_t>(num_users) + static_cast<std::int64_t>(num_movies);
  if (total > std::numeric_limits<NodeId>::max()) {
    throw GraphError("num_users + num_movies exceeds NodeId range");
  }

  const NodeId node_count = static_cast<NodeId>(total);
  std::vector<std::vector<NodeId>> adjacency(static_cast<std::size_t>(node_count));

  for (const auto& [user_id, movie_id] : interactions) {
    if (user_id < 0 || user_id >= num_users) {
      throw GraphError(format_range_error("user_id", user_id, 0, num_users));
    }
    if (movie_id < 0 || movie_id >= num_movies) {
      throw GraphError(format_range_error("movie_id", movie_id, 0, num_movies));
    }
    const NodeId movie_global = num_users + movie_id;
    adjacency[static_cast<std::size_t>(user_id)].push_back(movie_global);
    adjacency[static_cast<std::size_t>(movie_global)].push_back(user_id);
  }

  BipartiteCSR graph;
  graph.num_users_ = num_users;
  graph.num_movies_ = num_movies;
  graph.offsets_.assign(static_cast<std::size_t>(node_count) + 1, 0);
  graph.neighbors_.clear();

  for (NodeId node = 0; node < node_count; ++node) {
    auto& neighborhood = adjacency[static_cast<std::size_t>(node)];
    std::sort(neighborhood.begin(), neighborhood.end());
    neighborhood.erase(std::unique(neighborhood.begin(), neighborhood.end()), neighborhood.end());
    graph.neighbors_.insert(graph.neighbors_.end(), neighborhood.begin(), neighborhood.end());
    graph.offsets_[static_cast<std::size_t>(node) + 1] =
        static_cast<Offset>(graph.neighbors_.size());
  }

  return graph;
}

Degree BipartiteCSR::degree(NodeId node_id) const {
  validate_node(node_id);
  const Offset begin = offsets_[static_cast<std::size_t>(node_id)];
  const Offset end = offsets_[static_cast<std::size_t>(node_id) + 1];
  return static_cast<Degree>(end - begin);
}

std::vector<NodeId> BipartiteCSR::sample_neighbors(NodeId node_id, Degree k, Seed seed) const {
  validate_node(node_id);
  if (k < 0) {
    std::ostringstream out;
    out << "Invalid sample size k=" << k << "; expected k >= 0";
    throw GraphError(out.str());
  }

  const Offset begin = offsets_[static_cast<std::size_t>(node_id)];
  const Offset end = offsets_[static_cast<std::size_t>(node_id) + 1];
  const Degree node_degree = static_cast<Degree>(end - begin);
  if (k == 0 || node_degree == 0) {
    return {};
  }

  const auto first = neighbors_.begin() + static_cast<std::ptrdiff_t>(begin);
  const auto last = neighbors_.begin() + static_cast<std::ptrdiff_t>(end);
  if (k >= node_degree) {
    return std::vector<NodeId>(first, last);
  }

  std::vector<NodeId> pool(first, last);
  std::mt19937_64 rng(seed);
  for (Degree i = 0; i < k; ++i) {
    const std::uint64_t remaining = static_cast<std::uint64_t>(node_degree - i);
    const Degree swap_with =
        i + static_cast<Degree>(uniform_below(rng, remaining));
    std::swap(pool[static_cast<std::size_t>(i)], pool[static_cast<std::size_t>(swap_with)]);
  }
  pool.resize(static_cast<std::size_t>(k));
  return pool;
}

void BipartiteCSR::validate_node(NodeId node_id) const {
  const NodeId node_count = num_nodes();
  if (node_id < 0 || node_id >= node_count) {
    throw GraphError(format_range_error("node_id", node_id, 0, node_count));
  }
}

}  // namespace sagerec

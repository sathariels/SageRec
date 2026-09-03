#include "sagerec/bipartite_csr.hpp"

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

class TestFailure : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

#define REQUIRE(cond)                                                            \
  do {                                                                           \
    if (!(cond)) {                                                               \
      throw TestFailure(std::string(#cond " failed at ") + __FILE__ + ":" +     \
                        std::to_string(__LINE__));                               \
    }                                                                            \
  } while (0)

#define REQUIRE_THROWS_MESSAGE(expr, snippet)                                    \
  do {                                                                           \
    bool threw = false;                                                          \
    try {                                                                        \
      expr;                                                                      \
    } catch (const sagerec::GraphError& ex) {                                    \
      threw = true;                                                              \
      const std::string message = ex.what();                                     \
      if (message.find(snippet) == std::string::npos) {                          \
        throw TestFailure("GraphError '" + message + "' did not contain '" +     \
                          std::string(snippet) + "' at " + __FILE__ + ":" +      \
                          std::to_string(__LINE__));                             \
      }                                                                          \
    }                                                                            \
    if (!threw) {                                                                \
      throw TestFailure(std::string("expected GraphError from ") + #expr +       \
                        " at " + __FILE__ + ":" + std::to_string(__LINE__));     \
    }                                                                            \
  } while (0)

sagerec::BipartiteCSR fixture_graph() {
  // users {0,1}, movies {0,1,2} -> global movies {2,3,4}
  // interactions: (0,0), (0,1), (1,1)
  return sagerec::BipartiteCSR::from_interactions(2, 3, {{0, 0}, {0, 1}, {1, 1}});
}

void test_construction_exact_csr() {
  const auto graph = sagerec::BipartiteCSR::from_interactions(
      2, 3, {{1, 1}, {0, 1}, {0, 0}});

  REQUIRE(graph.num_users() == 2);
  REQUIRE(graph.num_movies() == 3);
  REQUIRE(graph.num_nodes() == 5);
  REQUIRE(graph.num_directed_edges() == 6);

  const std::vector<sagerec::Offset> expected_offsets{0, 2, 3, 4, 6, 6};
  const std::vector<sagerec::NodeId> expected_neighbors{2, 3, 3, 0, 0, 1};
  REQUIRE(graph.offsets() == expected_offsets);
  REQUIRE(graph.neighbors() == expected_neighbors);
  REQUIRE(graph.offsets().front() == 0);
  REQUIRE(graph.offsets().back() == static_cast<sagerec::Offset>(graph.neighbors().size()));
  REQUIRE(graph.degree(0) == 2);
  REQUIRE(graph.degree(1) == 1);
  REQUIRE(graph.degree(2) == 1);
  REQUIRE(graph.degree(3) == 2);
  REQUIRE(graph.degree(4) == 0);

  for (sagerec::NodeId user = 0; user < graph.num_users(); ++user) {
    const auto begin = static_cast<std::size_t>(graph.offsets()[static_cast<std::size_t>(user)]);
    const auto end = static_cast<std::size_t>(graph.offsets()[static_cast<std::size_t>(user) + 1]);
    for (std::size_t i = begin; i < end; ++i) {
      REQUIRE(graph.neighbors()[i] >= graph.num_users());
      REQUIRE(graph.neighbors()[i] < graph.num_nodes());
    }
  }
  for (sagerec::NodeId movie = graph.num_users(); movie < graph.num_nodes(); ++movie) {
    const auto begin = static_cast<std::size_t>(graph.offsets()[static_cast<std::size_t>(movie)]);
    const auto end = static_cast<std::size_t>(graph.offsets()[static_cast<std::size_t>(movie) + 1]);
    for (std::size_t i = begin; i < end; ++i) {
      REQUIRE(graph.neighbors()[i] >= 0);
      REQUIRE(graph.neighbors()[i] < graph.num_users());
    }
  }
}

void test_duplicate_edges_deduped() {
  const auto graph = sagerec::BipartiteCSR::from_interactions(
      1, 1, {{0, 0}, {0, 0}, {0, 0}});
  REQUIRE(graph.degree(0) == 1);
  REQUIRE(graph.degree(1) == 1);
  REQUIRE(graph.num_directed_edges() == 2);
  REQUIRE(graph.neighbors() == std::vector<sagerec::NodeId>({1, 0}));
}

void test_isolated_and_empty_graphs() {
  const auto empty = sagerec::BipartiteCSR::from_interactions(0, 0, {});
  REQUIRE(empty.num_nodes() == 0);
  REQUIRE(empty.offsets() == std::vector<sagerec::Offset>({0}));
  REQUIRE(empty.neighbors().empty());
  REQUIRE_THROWS_MESSAGE(empty.sample_neighbors(0, 1, 1), "node_id");

  const auto isolated = sagerec::BipartiteCSR::from_interactions(2, 1, {});
  REQUIRE(isolated.num_nodes() == 3);
  REQUIRE(isolated.degree(0) == 0);
  REQUIRE(isolated.degree(1) == 0);
  REQUIRE(isolated.degree(2) == 0);
  REQUIRE(isolated.sample_neighbors(1, 4, 9).empty());

  const auto mixed = sagerec::BipartiteCSR::from_interactions(2, 1, {{0, 0}});
  REQUIRE(mixed.degree(1) == 0);
  REQUIRE(mixed.sample_neighbors(1, 8, 3).empty());
  REQUIRE(mixed.sample_neighbors(0, 0, 3).empty());
}

void test_k_zero_and_full_neighborhood() {
  const auto graph = fixture_graph();
  REQUIRE(graph.sample_neighbors(0, 0, 11).empty());

  const auto full_eq = graph.sample_neighbors(0, 2, 99);
  const auto full_gt = graph.sample_neighbors(0, 100, 0);
  const std::vector<sagerec::NodeId> expected{2, 3};
  REQUIRE(full_eq == expected);
  REQUIRE(full_gt == expected);
  REQUIRE(graph.sample_neighbors(4, 5, 1).empty());
}

void test_sample_subset_unique_and_bounded() {
  const auto graph = sagerec::BipartiteCSR::from_interactions(
      1, 4, {{0, 0}, {0, 1}, {0, 2}, {0, 3}});
  REQUIRE(graph.degree(0) == 4);

  const auto sample = graph.sample_neighbors(0, 2, 123);
  REQUIRE(sample.size() == 2);
  REQUIRE(sample[0] != sample[1]);
  const std::set<sagerec::NodeId> neighborhood{1, 2, 3, 4};
  REQUIRE(neighborhood.count(sample[0]) == 1);
  REQUIRE(neighborhood.count(sample[1]) == 1);
}

void test_invalid_ids() {
  REQUIRE_THROWS_MESSAGE(
      sagerec::BipartiteCSR::from_interactions(-1, 1, {}), "num_users");
  REQUIRE_THROWS_MESSAGE(
      sagerec::BipartiteCSR::from_interactions(1, -2, {}), "num_movies");
  REQUIRE_THROWS_MESSAGE(
      sagerec::BipartiteCSR::from_interactions(1, 1, {{1, 0}}), "user_id");
  REQUIRE_THROWS_MESSAGE(
      sagerec::BipartiteCSR::from_interactions(1, 1, {{0, 1}}), "movie_id");
  REQUIRE_THROWS_MESSAGE(
      sagerec::BipartiteCSR::from_interactions(1, 1, {{-1, 0}}), "user_id");

  const auto graph = fixture_graph();
  REQUIRE_THROWS_MESSAGE(graph.degree(5), "node_id");
  REQUIRE_THROWS_MESSAGE(graph.sample_neighbors(-1, 1, 0), "node_id");
  REQUIRE_THROWS_MESSAGE(graph.sample_neighbors(5, 1, 0), "node_id");
  REQUIRE_THROWS_MESSAGE(graph.sample_neighbors(0, -3, 0), "sample size");
}

void test_seed_reproducibility() {
  const auto graph = sagerec::BipartiteCSR::from_interactions(
      1, 5, {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}});
  const auto a = graph.sample_neighbors(0, 3, 42);
  const auto b = graph.sample_neighbors(0, 3, 42);
  const auto c = graph.sample_neighbors(0, 3, 43);
  REQUIRE(a == b);
  REQUIRE(a.size() == 3);
  REQUIRE(c.size() == 3);
  REQUIRE(std::set<sagerec::NodeId>(a.begin(), a.end()).size() == 3);

  std::set<sagerec::NodeId> seen;
  for (sagerec::Seed seed = 0; seed < 200; ++seed) {
    const auto one = graph.sample_neighbors(0, 1, seed);
    REQUIRE(one.size() == 1);
    seen.insert(one[0]);
  }
  REQUIRE(seen.size() >= 2);
}

void test_bidirectional_train_edges() {
  const auto graph = fixture_graph();
  const auto user0 = graph.sample_neighbors(0, 2, 0);
  const auto movie1 = graph.sample_neighbors(3, 2, 0);
  REQUIRE(std::find(user0.begin(), user0.end(), 3) != user0.end());
  REQUIRE(std::find(movie1.begin(), movie1.end(), 0) != movie1.end());
  REQUIRE(std::find(movie1.begin(), movie1.end(), 1) != movie1.end());
}

const std::map<std::string, std::function<void()>> kTests{
    {"construction", test_construction_exact_csr},
    {"duplicate_edges", test_duplicate_edges_deduped},
    {"isolated_nodes", test_isolated_and_empty_graphs},
    {"k_zero_and_full", test_k_zero_and_full_neighborhood},
    {"sample_subset", test_sample_subset_unique_and_bounded},
    {"invalid_ids", test_invalid_ids},
    {"seed_reproducibility", test_seed_reproducibility},
    {"bidirectional", test_bidirectional_train_edges},
};

int run_one(const std::string& name, const std::function<void()>& fn) {
  try {
    fn();
    std::cout << "PASS " << name << '\n';
    return 0;
  } catch (const TestFailure& ex) {
    std::cerr << "FAIL " << name << ": " << ex.what() << '\n';
    return 1;
  } catch (const std::exception& ex) {
    std::cerr << "FAIL " << name << ": unexpected exception: " << ex.what() << '\n';
    return 1;
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1) {
    const std::string name = argv[1];
    const auto it = kTests.find(name);
    if (it == kTests.end()) {
      std::cerr << "Unknown test '" << name << "'\n";
      return 2;
    }
    return run_one(it->first, it->second);
  }

  int failures = 0;
  for (const auto& [name, fn] : kTests) {
    failures += run_one(name, fn);
  }
  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
    return 1;
  }
  std::cout << kTests.size() << " test(s) passed\n";
  return 0;
}

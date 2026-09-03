#include "sagerec/bipartite_csr.hpp"
#include "sagerec/error.hpp"
#include "sagerec/movielens_100k.hpp"

#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <map>
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

constexpr const char* kTinyUdata =
    "10\t20\t5\t100\n"
    "7\t20\t4\t200\n"
    "10\t8\t1\t50\n";

void test_parse_valid_mappings_and_order() {
  const auto parsed = sagerec::parse_movielens_100k(kTinyUdata);

  REQUIRE(parsed.num_users() == 2);
  REQUIRE(parsed.num_movies() == 2);
  REQUIRE(parsed.user_source_ids() == std::vector<std::int32_t>({7, 10}));
  REQUIRE(parsed.movie_source_ids() == std::vector<std::int32_t>({8, 20}));
  REQUIRE(parsed.interactions().size() == 3);

  REQUIRE(parsed.interactions()[0].user_id == 1);
  REQUIRE(parsed.interactions()[0].movie_id == 1);
  REQUIRE(parsed.interactions()[0].rating == 5);
  REQUIRE(parsed.interactions()[0].timestamp == 100);

  REQUIRE(parsed.interactions()[1].user_id == 0);
  REQUIRE(parsed.interactions()[1].movie_id == 1);
  REQUIRE(parsed.interactions()[1].rating == 4);
  REQUIRE(parsed.interactions()[1].timestamp == 200);

  REQUIRE(parsed.interactions()[2].user_id == 1);
  REQUIRE(parsed.interactions()[2].movie_id == 0);
  REQUIRE(parsed.interactions()[2].rating == 1);
  REQUIRE(parsed.interactions()[2].timestamp == 50);

  const auto pairs = parsed.local_pairs();
  REQUIRE(pairs.size() == 3);
  REQUIRE((pairs[0] == std::pair<sagerec::NodeId, sagerec::NodeId>(1, 1)));
  REQUIRE((pairs[1] == std::pair<sagerec::NodeId, sagerec::NodeId>(0, 1)));
  REQUIRE((pairs[2] == std::pair<sagerec::NodeId, sagerec::NodeId>(1, 0)));
}

void test_crlf_and_blank_lines() {
  const auto parsed = sagerec::parse_movielens_100k("1\t2\t3\t4\r\n\r\n5\t6\t7\t8\r\n");
  REQUIRE(parsed.num_users() == 2);
  REQUIRE(parsed.user_source_ids() == std::vector<std::int32_t>({1, 5}));
  REQUIRE(parsed.movie_source_ids() == std::vector<std::int32_t>({2, 6}));
  REQUIRE(parsed.interactions().size() == 2);
  REQUIRE(parsed.interactions()[1].rating == 7);
  REQUIRE(parsed.interactions()[1].timestamp == 8);
}

void test_normalized_pairs_build_csr() {
  const auto parsed = sagerec::parse_movielens_100k(kTinyUdata);
  const auto graph = sagerec::BipartiteCSR::from_interactions(
      parsed.num_users(), parsed.num_movies(), parsed.local_pairs());

  REQUIRE(graph.num_users() == 2);
  REQUIRE(graph.num_movies() == 2);
  REQUIRE(graph.num_nodes() == 4);
  REQUIRE(graph.num_directed_edges() == 6);

  const std::vector<sagerec::Offset> expected_offsets{0, 1, 3, 4, 6};
  const std::vector<sagerec::NodeId> expected_neighbors{3, 2, 3, 1, 0, 1};
  REQUIRE(graph.offsets() == expected_offsets);
  REQUIRE(graph.neighbors() == expected_neighbors);
}

void test_empty_input() {
  REQUIRE_THROWS_MESSAGE(sagerec::parse_movielens_100k(""), "empty");
  REQUIRE_THROWS_MESSAGE(sagerec::parse_movielens_100k("\n"), "empty");
  REQUIRE_THROWS_MESSAGE(sagerec::parse_movielens_100k("  \n\t\n"), "empty");
}

void test_malformed_rows_and_fields() {
  REQUIRE_THROWS_MESSAGE(
      sagerec::parse_movielens_100k("1\t2\t3\n"), "expected 4 tab-separated fields");
  REQUIRE_THROWS_MESSAGE(
      sagerec::parse_movielens_100k("1\t2\t3\t4\t5\n"), "got 5");
  REQUIRE_THROWS_MESSAGE(
      sagerec::parse_movielens_100k("1 2 3 4\n"), "expected 4 tab-separated fields");
  REQUIRE_THROWS_MESSAGE(
      sagerec::parse_movielens_100k("1\t2\t3\t4\t\n"), "got 5");
}

void test_bad_field_types() {
  REQUIRE_THROWS_MESSAGE(sagerec::parse_movielens_100k("x\t2\t3\t4\n"), "user_id");
  REQUIRE_THROWS_MESSAGE(sagerec::parse_movielens_100k("x\t2\t3\t4\n"), "not an integer");
  REQUIRE_THROWS_MESSAGE(sagerec::parse_movielens_100k("1\t2.5\t3\t4\n"), "movie_id");
  REQUIRE_THROWS_MESSAGE(sagerec::parse_movielens_100k("1\t2\t3.0\t4\n"), "rating");
  REQUIRE_THROWS_MESSAGE(sagerec::parse_movielens_100k("1\t2\t3\t12x\n"), "timestamp");
  REQUIRE_THROWS_MESSAGE(sagerec::parse_movielens_100k("1\t\t3\t4\n"), "movie_id");
}

void test_nonpositive_source_ids() {
  REQUIRE_THROWS_MESSAGE(
      sagerec::parse_movielens_100k("0\t2\t3\t4\n"), "not a positive source ID");
  REQUIRE_THROWS_MESSAGE(sagerec::parse_movielens_100k("0\t2\t3\t4\n"), "user_id 0");
  REQUIRE_THROWS_MESSAGE(
      sagerec::parse_movielens_100k("-1\t2\t3\t4\n"), "not a positive source ID");
  REQUIRE_THROWS_MESSAGE(
      sagerec::parse_movielens_100k("1\t-8\t3\t4\n"), "movie_id -8");
}

void test_duplicate_source_pairs() {
  REQUIRE_THROWS_MESSAGE(
      sagerec::parse_movielens_100k("1\t2\t3\t4\n1\t2\t5\t6\n"),
      "Duplicate MovieLens 100K interaction at row 2");
  REQUIRE_THROWS_MESSAGE(
      sagerec::parse_movielens_100k("1\t2\t3\t4\n1\t2\t5\t6\n"),
      "already seen at row 1");
  REQUIRE_THROWS_MESSAGE(
      sagerec::parse_movielens_100k("9\t8\t1\t1\n\n9\t8\t2\t2\n"), "at row 3");
}

const std::map<std::string, std::function<void()>> kTests{
    {"parse_valid", test_parse_valid_mappings_and_order},
    {"crlf_blank_lines", test_crlf_and_blank_lines},
    {"normalized_csr", test_normalized_pairs_build_csr},
    {"empty_input", test_empty_input},
    {"malformed_rows", test_malformed_rows_and_fields},
    {"bad_types", test_bad_field_types},
    {"nonpositive_ids", test_nonpositive_source_ids},
    {"duplicates", test_duplicate_source_pairs},
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

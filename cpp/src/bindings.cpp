#include "sagerec/bipartite_csr.hpp"
#include "sagerec/movielens_100k.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(graph_sampler, m) {
  m.doc() =
      "SageRec bipartite CSR graph, seeded neighbor sampler, and in-memory "
      "MovieLens 100K u.data parser. "
      "Users occupy [0, num_users); movies occupy [num_users, num_nodes). "
      "Construction takes local (user_id, movie_id) pairs. "
      "parse_movielens_100k accepts tab-separated u.data text only; it does "
      "not read files, download data, apply a split, or build a CSR. "
      "sample_neighbors(node_id, k, seed) uses uniform sampling without "
      "replacement (proposed ADR-005 implementation contract).";

  py::register_exception<sagerec::GraphError>(m, "GraphError", PyExc_ValueError);

  py::class_<sagerec::MovieLens100kInteraction>(m, "MovieLens100kInteraction")
      .def_readonly("user_id", &sagerec::MovieLens100kInteraction::user_id)
      .def_readonly("movie_id", &sagerec::MovieLens100kInteraction::movie_id)
      .def_readonly("rating", &sagerec::MovieLens100kInteraction::rating)
      .def_readonly("timestamp", &sagerec::MovieLens100kInteraction::timestamp);

  py::class_<sagerec::MovieLens100kRatings>(m, "MovieLens100kRatings")
      .def_property_readonly(
          "num_users",
          &sagerec::MovieLens100kRatings::num_users)
      .def_property_readonly(
          "num_movies",
          &sagerec::MovieLens100kRatings::num_movies)
      .def_property_readonly(
          "interactions",
          [](const sagerec::MovieLens100kRatings& parsed) {
            return parsed.interactions();
          },
          "Normalized interactions in input order (copied).")
      .def_property_readonly(
          "user_source_ids",
          [](const sagerec::MovieLens100kRatings& parsed) {
            return parsed.user_source_ids();
          },
          "Local user ID to original source user ID (copied).")
      .def_property_readonly(
          "movie_source_ids",
          [](const sagerec::MovieLens100kRatings& parsed) {
            return parsed.movie_source_ids();
          },
          "Local movie ID to original source movie ID (copied).")
      .def(
          "local_pairs",
          [](const sagerec::MovieLens100kRatings& parsed) {
            py::gil_scoped_release release;
            return parsed.local_pairs();
          },
          "Local (user_id, movie_id) pairs for BipartiteCSR. "
          "Callers must still pass training positives only.");

  m.def(
      "parse_movielens_100k",
      [](std::string text) {
        // Copy Python str/bytes into owned storage before releasing the GIL.
        // Do not pass a view into Python memory across the boundary.
        py::gil_scoped_release release;
        return sagerec::parse_movielens_100k(text);
      },
      py::arg("text"),
      "Parse in-memory tab-separated MovieLens 100K u.data text. "
      "Does not read files, download data, apply a split, or build a CSR.");

  py::class_<sagerec::BipartiteCSR>(m, "BipartiteCSR")
      .def(
          py::init([](sagerec::NodeId num_users,
                      sagerec::NodeId num_movies,
                      const std::vector<std::pair<sagerec::NodeId, sagerec::NodeId>>&
                          interactions) {
            py::gil_scoped_release release;
            return sagerec::BipartiteCSR::from_interactions(
                num_users, num_movies, interactions);
          }),
          py::arg("num_users"),
          py::arg("num_movies"),
          py::arg("interactions"),
          "Construct a train-only bipartite CSR from local user-movie pairs.")
      .def_property_readonly(
          "num_users",
          &sagerec::BipartiteCSR::num_users)
      .def_property_readonly(
          "num_movies",
          &sagerec::BipartiteCSR::num_movies)
      .def_property_readonly(
          "num_nodes",
          &sagerec::BipartiteCSR::num_nodes)
      .def(
          "degree",
          &sagerec::BipartiteCSR::degree,
          py::arg("node_id"),
          "Return the stored neighborhood size of a global node ID.")
      .def(
          "sample_neighbors",
          [](const sagerec::BipartiteCSR& graph,
             sagerec::NodeId node_id,
             sagerec::Degree k,
             std::int64_t seed) {
            if (seed < 0) {
              throw sagerec::GraphError(
                  "Invalid seed " + std::to_string(seed) +
                  "; expected a non-negative integer");
            }
            py::gil_scoped_release release;
            return graph.sample_neighbors(
                node_id, k, static_cast<sagerec::Seed>(seed));
          },
          py::arg("node_id"),
          py::arg("k"),
          py::arg("seed"),
          "Sample up to k neighbors of node_id using an explicit seed.")
      .def(
          "offsets",
          [](const sagerec::BipartiteCSR& graph) { return graph.offsets(); },
          "Copy of CSR offsets. Length is num_nodes + 1.")
      .def(
          "neighbors",
          [](const sagerec::BipartiteCSR& graph) { return graph.neighbors(); },
          "Copy of concatenated CSR neighbors.");
}

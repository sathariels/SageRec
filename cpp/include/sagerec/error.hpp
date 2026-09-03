#pragma once

#include <stdexcept>
#include <string>

namespace sagerec {

/// Actionable failure at a public graph or sampler boundary.
///
/// Messages name the invalid value and the expected constraint so callers can
/// correct the input. Python bindings map this type to graph_sampler.GraphError.
class GraphError : public std::invalid_argument {
 public:
  explicit GraphError(const std::string& message)
      : std::invalid_argument(message) {}
};

}  // namespace sagerec

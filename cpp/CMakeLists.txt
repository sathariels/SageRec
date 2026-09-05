cmake_minimum_required(VERSION 3.16)
project(sagerec_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

function(sagerec_set_warnings target)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
  target_compile_definitions(${target} PRIVATE
    $<$<CONFIG:Release>:_GLIBCXX_ASSERTIONS>)
endfunction()

add_library(sagerec_csr STATIC
  src/bipartite_csr.cpp
  src/movielens_100k.cpp
)
target_include_directories(sagerec_csr PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
sagerec_set_warnings(sagerec_csr)

enable_testing()

add_executable(test_bipartite_csr
  tests/test_bipartite_csr.cpp
)
target_link_libraries(test_bipartite_csr PRIVATE sagerec_csr)
sagerec_set_warnings(test_bipartite_csr)

add_executable(test_movielens_100k
  tests/test_movielens_100k.cpp
)
target_link_libraries(test_movielens_100k PRIVATE sagerec_csr)
sagerec_set_warnings(test_movielens_100k)

set(_native_cases
  construction
  duplicate_edges
  isolated_nodes
  k_zero_and_full
  sample_subset
  invalid_ids
  seed_reproducibility
  bidirectional
)
foreach(_case IN LISTS _native_cases)
  add_test(NAME ${_case} COMMAND test_bipartite_csr ${_case})
endforeach()

set(_ingestion_cases
  parse_valid
  crlf_blank_lines
  normalized_csr
  empty_input
  malformed_rows
  bad_types
  nonpositive_ids
  duplicates
)
foreach(_case IN LISTS _ingestion_cases)
  add_test(NAME ${_case} COMMAND test_movielens_100k ${_case})
endforeach()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE _pybind11_cmakedir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _pybind11_rc
  )
  if(_pybind11_rc EQUAL 0 AND NOT _pybind11_cmakedir STREQUAL "")
    set(pybind11_DIR "${_pybind11_cmakedir}")
    find_package(pybind11 CONFIG REQUIRED)
  endif()
endif()
if(NOT pybind11_FOUND)
  message(FATAL_ERROR
    "pybind11 is required to build graph_sampler. "
    "On Debian/Ubuntu: sudo apt-get install -y pybind11-dev python3-pybind11. "
    "Alternatively: python3 -m pip install pybind11")
endif()

pybind11_add_module(graph_sampler src/bindings.cpp)
target_link_libraries(graph_sampler PRIVATE sagerec_csr)
sagerec_set_warnings(graph_sampler)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/../python/graph_sampler.pyi
  ${CMAKE_CURRENT_BINARY_DIR}/graph_sampler.pyi
  COPYONLY
)

set(_python_src_dir ${CMAKE_CURRENT_SOURCE_DIR}/../python)
set(_python_tests_dir ${_python_src_dir}/tests)
add_test(
  NAME python_binding_smoke
  COMMAND ${Python3_EXECUTABLE} -m unittest discover -s ${_python_tests_dir} -v
)
set_tests_properties(python_binding_smoke PROPERTIES
  ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:graph_sampler>:${_python_src_dir}"
)

cmake_minimum_required(VERSION 3.18)
project(surveillance_frontier LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED COMPONENTS CXX)

# Domain-agnostic evaluation engine: window scan, per-length ranking, candidate selection, trial methods.
add_library(sf_engine
  engine/src/series.cpp
  engine/src/scan.cpp
  engine/src/rank.cpp
  engine/src/select.cpp
  engine/src/detector.cpp
  engine/src/scenario.cpp
  engine/src/evaluate.cpp)
target_include_directories(sf_engine PUBLIC engine/include)
target_link_libraries(sf_engine PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(sf_engine PRIVATE -Wall -Wextra -O3 -march=native)

add_executable(sf_test engine/tests/test_engine.cpp)
target_link_libraries(sf_test PRIVATE sf_engine)

add_executable(sf_detect engine/apps/sf_detect.cpp)
target_link_libraries(sf_detect PRIVATE sf_engine)

add_executable(sf_bench engine/apps/sf_bench.cpp)
target_link_libraries(sf_bench PRIVATE sf_engine)
target_compile_options(sf_bench PRIVATE -O3 -march=native)

enable_testing()
add_test(NAME engine_gate COMMAND sf_test)

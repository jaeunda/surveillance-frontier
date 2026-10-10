cmake_minimum_required(VERSION 3.18)
project(surveillance_frontier LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Phase 2 GPU back end (engine/cuda). Off by default; the CPU-only build does not depend on CUDA.
option(SF_CUDA "Build the CUDA back end" OFF)

find_package(OpenMP REQUIRED COMPONENTS CXX)

# Domain-agnostic evaluation engine: window scan, per-length ranking, candidate selection, trial methods.
add_library(sf_engine
  engine/src/series.cpp
  engine/src/scan.cpp
  engine/src/rank.cpp
  engine/src/select.cpp
  engine/src/detector.cpp
  engine/src/scenario.cpp
  engine/src/evaluate.cpp
  engine/src/interval.cpp
  engine/src/sha256.cpp
  engine/src/check.cpp
  engine/src/batch_cpu.cpp)
target_include_directories(sf_engine PUBLIC engine/include)
target_link_libraries(sf_engine PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(sf_engine PRIVATE -Wall -Wextra -O3 -march=native)

# The GPU back end's device code run on the host (exactness tests without a GPU). No FMA contraction, no fast math:
# every operation of the device path is a single IEEE operation, as on the device with --fmad=false.
add_library(sf_gpu_emul engine/cuda/gpu_backend_emul.cpp)
target_link_libraries(sf_gpu_emul PUBLIC sf_engine)
target_compile_options(sf_gpu_emul PRIVATE -Wall -Wextra -O3 -ffp-contract=off)

add_library(sf_backends engine/src/backends.cpp)
target_link_libraries(sf_backends PUBLIC sf_engine sf_gpu_emul)
target_compile_options(sf_backends PRIVATE -Wall -Wextra -O3)

if(SF_CUDA)
  enable_language(CUDA)
  # SASS only (no PTX, so no JIT at start-up): sm_75 optional T4 check, sm_89 L40S (measured), sm_120 RTX 5060 Ti
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
    set(CMAKE_CUDA_ARCHITECTURES 75-real 89-real 120-real)
  endif()
  set(CMAKE_CUDA_STANDARD 17)
  add_library(sf_gpu_cuda engine/cuda/gpu_backend_cuda.cu)
  target_link_libraries(sf_gpu_cuda PUBLIC sf_engine)
  # SPEC §4: no --use_fast_math; no FMA contraction on the device feature path; host code keeps the engine flags
  # (-fopenmp: the G-hyb host-side ordering and merge loop is parallel over trials)
  target_compile_options(sf_gpu_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false -O3 -Xcompiler=-O3,-fopenmp>)
  target_link_libraries(sf_backends PUBLIC sf_gpu_cuda)
  target_compile_definitions(sf_backends PRIVATE SF_HAVE_CUDA=1)
endif()

add_executable(sf_test engine/tests/test_engine.cpp)
target_link_libraries(sf_test PRIVATE sf_engine)

add_executable(sf_detect engine/apps/sf_detect.cpp)
target_link_libraries(sf_detect PRIVATE sf_engine)

add_executable(sf_bench engine/apps/sf_bench.cpp)
target_link_libraries(sf_bench PRIVATE sf_engine)
target_compile_options(sf_bench PRIVATE -O3 -march=native)

# Phase 2: task runner and correctness gate
add_executable(sf_run engine/apps/sf_run.cpp)
target_link_libraries(sf_run PRIVATE sf_backends)
target_compile_options(sf_run PRIVATE -Wall -Wextra -O3 -march=native)

add_executable(sf_gate engine/apps/sf_gate.cpp)
target_link_libraries(sf_gate PRIVATE sf_backends)
target_compile_options(sf_gate PRIVATE -Wall -Wextra -O3 -march=native)

add_executable(sf_test2 engine/tests/test_phase2.cpp)
target_link_libraries(sf_test2 PRIVATE sf_backends)
target_compile_options(sf_test2 PRIVATE -Wall -Wextra -O2)

enable_testing()
add_test(NAME engine_gate COMMAND sf_test)
add_test(NAME phase2_units COMMAND sf_test2)

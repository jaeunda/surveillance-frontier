// Device runtime used by the GPU back end: memory, copies, flat kernel launches, timing. Two implementations of the
// same functions: CUDA (gpu_backend_cuda.cu) and the host emulation (gpu_backend_emul.cpp, SF_GPU_EMUL), which runs
// each kernel body as a parallel host loop. Included once per back end inside its own namespace (SF_GPU_NS).
#pragma once

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#if !defined(SF_GPU_EMUL)
#include <cuda_runtime.h>
#endif

namespace sf {
namespace SF_GPU_NS {

#if defined(SF_GPU_EMUL)

inline void init(int) {}
inline std::string device_name() { return "host emulation"; }
inline void mem_info(size_t& free_b, size_t& total_b) {
  free_b = total_b = size_t{8} << 30;  // nominal: the emulation never sizes batches from memory
}
inline void* dev_malloc(size_t bytes) {
  void* p = std::malloc(bytes ? bytes : 1);
  if (!p) throw std::runtime_error("emulated device allocation failed: " + std::to_string(bytes) + " bytes");
  return p;
}
inline void dev_free(void* p) { std::free(p); }
inline void h2d(void* d, const void* h, size_t bytes) { if (bytes) std::memcpy(d, h, bytes); }
inline void d2h(void* h, const void* d, size_t bytes) { if (bytes) std::memcpy(h, d, bytes); }
inline void dev_memset(void* d, int v, size_t bytes) { if (bytes) std::memset(d, v, bytes); }
inline void dev_sync() {}

inline int& launch_threads() {
  static int t = 1;
  return t;
}
template <typename A, void (*F)(int64_t, const A&)>
void launch(int64_t total, const A& a) {
#pragma omp parallel for num_threads(launch_threads()) schedule(dynamic, 16) if (launch_threads() > 1)
  for (int64_t i = 0; i < total; ++i) F(i, a);
}

// Phase timer: host clock (the emulation is synchronous)
struct PhaseTimer {
  std::chrono::steady_clock::time_point t;
  void mark() { t = std::chrono::steady_clock::now(); }
  double ms_since(const PhaseTimer& o) const { return std::chrono::duration<double, std::milli>(t - o.t).count(); }
  void wait() {}
};

#else

inline void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string("CUDA: ") + what + ": " + cudaGetErrorString(e));
}
inline void init(int device) {
  check(cudaSetDevice(device), "cudaSetDevice");
  check(cudaFree(nullptr), "context creation");
}
inline std::string device_name() {
  int d = 0;
  cudaGetDevice(&d);
  cudaDeviceProp p{};
  cudaGetDeviceProperties(&p, d);
  return std::string(p.name) + " sm_" + std::to_string(p.major) + std::to_string(p.minor) + " " +
         std::to_string(p.totalGlobalMem >> 20) + " MiB";
}
inline void mem_info(size_t& free_b, size_t& total_b) { check(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo"); }
inline void* dev_malloc(size_t bytes) {
  void* p = nullptr;
  check(cudaMalloc(&p, bytes ? bytes : 1), ("cudaMalloc " + std::to_string(bytes) + " bytes").c_str());
  return p;
}
inline void dev_free(void* p) { cudaFree(p); }
inline void h2d(void* d, const void* h, size_t bytes) {
  if (bytes) check(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice), "h2d");
}
inline void d2h(void* h, const void* d, size_t bytes) {
  if (bytes) check(cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost), "d2h");
}
inline void dev_memset(void* d, int v, size_t bytes) {
  if (bytes) check(cudaMemset(d, v, bytes), "memset");
}
inline void dev_sync() { check(cudaDeviceSynchronize(), "synchronize"); }

inline int& launch_threads() {
  static int t = 1;
  return t;
}
template <typename A, void (*F)(int64_t, const A&)>
__global__ void flat_kernel(int64_t total, A a) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < total) F(i, a);
}
template <typename A, void (*F)(int64_t, const A&)>
void launch(int64_t total, const A& a) {
  if (total <= 0) return;
  constexpr int kBlock = 128;
  flat_kernel<A, F><<<static_cast<unsigned>((total + kBlock - 1) / kBlock), kBlock>>>(total, a);
  check(cudaGetLastError(), "kernel launch");
}

// Phase timer: CUDA events on the default stream (no host synchronisation until wait())
struct PhaseTimer {
  cudaEvent_t e = nullptr;
  PhaseTimer() { check(cudaEventCreate(&e), "cudaEventCreate"); }
  ~PhaseTimer() { if (e) cudaEventDestroy(e); }
  PhaseTimer(const PhaseTimer&) = delete;
  PhaseTimer& operator=(const PhaseTimer&) = delete;
  void mark() { check(cudaEventRecord(e), "cudaEventRecord"); }
  double ms_since(const PhaseTimer& o) const {
    float ms = 0;
    check(cudaEventElapsedTime(&ms, o.e, e), "cudaEventElapsedTime");
    return ms;
  }
  void wait() { check(cudaEventSynchronize(e), "cudaEventSynchronize"); }
};

#endif

}  // namespace SF_GPU_NS
}  // namespace sf

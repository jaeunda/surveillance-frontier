// The GPU back end on a CUDA device (CMake option SF_CUDA=ON). Device code: gpu_kernels.hpp; host side:
// gpu_backend.inl. Built for sm_89 (L40S, measured), sm_120 (desktop RTX 5060 Ti) and sm_75 (optional T4 check) as
// SASS only, without --use_fast_math and with --fmad=false (SPEC §4).
#define SF_GPU_NS gpu_cuda
#include "gpu_backend.inl"

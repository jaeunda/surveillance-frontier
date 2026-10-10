// The GPU back end's device code run on the host ("emul"): every kernel body of gpu_kernels.hpp executes as a
// parallel host loop with the same arithmetic. Used for exactness tests without a GPU; never for timings.
#define SF_GPU_EMUL 1
#define SF_GPU_NS gpu_emul
#include "gpu_backend.inl"

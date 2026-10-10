// SF_HD marks functions compiled for both the host and, under nvcc, the device, so CPU and GPU share one source.
#pragma once

#if defined(__CUDACC__)
#define SF_HD __host__ __device__ inline
#else
#define SF_HD inline
#endif

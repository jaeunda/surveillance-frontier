// Back-end factory. The CUDA back end exists only in builds with SF_CUDA=ON (SF_HAVE_CUDA).
#include <stdexcept>

#include "sf/batch.hpp"

namespace sf {

std::unique_ptr<BatchEvaluator> make_cpu_backend(const BaseState& base, const BackendConfig& cfg);
namespace gpu_emul {
std::unique_ptr<BatchEvaluator> make_backend(const BaseState& base, const BackendConfig& cfg);
}
#if defined(SF_HAVE_CUDA)
namespace gpu_cuda {
std::unique_ptr<BatchEvaluator> make_backend(const BaseState& base, const BackendConfig& cfg);
void device_init(int device);
}
#endif

void device_init(const std::string& name, const BackendConfig& cfg) {
  if (name == "cpu") {
    int started = 0;
#pragma omp parallel num_threads(cfg.threads) reduction(+ : started)
    started += 1;
    if (started != cfg.threads) throw std::runtime_error("thread pool started with fewer threads than requested");
  }
#if defined(SF_HAVE_CUDA)
  if (name == "cuda") gpu_cuda::device_init(cfg.device);
#endif
}

bool backend_available(const std::string& name) {
#if defined(SF_HAVE_CUDA)
  if (name == "cuda") return true;
#endif
  return name == "cpu" || name == "emul";
}

std::unique_ptr<BatchEvaluator> make_backend(const std::string& name, const BaseState& base, const BackendConfig& cfg) {
  if (name == "cpu") return make_cpu_backend(base, cfg);
  if (name == "emul") return gpu_emul::make_backend(base, cfg);
#if defined(SF_HAVE_CUDA)
  if (name == "cuda") return gpu_cuda::make_backend(base, cfg);
#endif
  throw std::invalid_argument("back end not available in this build: " + name);
}

}  // namespace sf

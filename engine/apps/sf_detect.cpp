// Run the detector once on a series file and print the top episodes as CSV.
//
//   sf_detect <series.bin> [--scan direct|optimized] [--threads P] [--top K]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "sf/detector.hpp"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <series.bin> [--scan direct|optimized] [--threads P] [--top K]\n", argv[0]);
    return 2;
  }
  sf::DetectorConfig cfg;
  int threads = 1;
  for (int i = 2; i + 1 < argc; i += 2) {
    const std::string a = argv[i], v = argv[i + 1];
    if (a == "--scan")
      cfg.scan = v == "direct" ? sf::ScanKind::Direct : sf::ScanKind::Optimized;
    else if (a == "--threads")
      threads = std::atoi(v.c_str());
    else if (a == "--top")
      cfg.select.top_k = std::atoi(v.c_str());
    else {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      return 2;
    }
  }
  const sf::Series s = sf::load_series(argv[1]);
  sf::Workspace ws;
  sf::StageTimes t;
  const auto top = sf::detect(sf::view(s), cfg, threads, ws, &t);
  std::printf("k,start,window,S,start_ts_ns\n");
  for (const auto& e : top)
    std::printf("%d,%lld,%d,%.15g,%lld\n", e.k, static_cast<long long>(e.start), e.window, e.S,
                static_cast<long long>(s.ts_ns[e.start]));
  std::fprintf(stderr, "n=%lld scan=%.1f ms rank=%.1f ms select=%.1f ms\n", static_cast<long long>(s.size()),
               t.scan_ms, t.rank_ms, t.select_ms);
  return 0;
}

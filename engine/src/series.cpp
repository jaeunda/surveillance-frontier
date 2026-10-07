#include "sf/series.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace sf {

namespace {
constexpr char kMagic[8] = {'S', 'F', 'S', 'E', 'R', '0', '0', '1'};

template <typename T>
void read_array(std::ifstream& f, std::vector<T>& v, uint64_t n) {
  v.resize(n);
  f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(T)));
}

template <typename T>
void write_array(std::ofstream& f, const std::vector<T>& v) {
  f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
}
}  // namespace

Series Series::head(int64_t n) const {
  if (n <= 0 || n >= size()) return *this;
  Series out;
  out.ts_ns.assign(ts_ns.begin(), ts_ns.begin() + n);
  out.low.assign(low.begin(), low.begin() + n);
  out.high.assign(high.begin(), high.begin() + n);
  out.volume.assign(volume.begin(), volume.begin() + n);
  return out;
}

Series load_series(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  char magic[8];
  uint64_t n = 0;
  f.read(magic, 8);
  f.read(reinterpret_cast<char*>(&n), sizeof(n));
  if (!f || std::memcmp(magic, kMagic, 8) != 0) throw std::runtime_error("not a series file: " + path);
  Series s;
  read_array(f, s.ts_ns, n);
  read_array(f, s.low, n);
  read_array(f, s.high, n);
  read_array(f, s.volume, n);
  if (!f) throw std::runtime_error("truncated series file: " + path);
  for (uint64_t i = 0; i < n; ++i) {
    if (!(s.low[i] > 0.0f) || !(s.high[i] >= s.low[i]) || !(s.volume[i] >= 0.0f))
      throw std::runtime_error("invalid bar " + std::to_string(i) + " in " + path);
  }
  return s;
}

void save_series(const std::string& path, const Series& s) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + path);
  const uint64_t n = static_cast<uint64_t>(s.size());
  f.write(kMagic, 8);
  f.write(reinterpret_cast<const char*>(&n), sizeof(n));
  write_array(f, s.ts_ns);
  write_array(f, s.low);
  write_array(f, s.high);
  write_array(f, s.volume);
}

}  // namespace sf

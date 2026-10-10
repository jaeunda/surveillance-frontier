// SHA-256 (FIPS 180-4) for input verification and output identity hashes.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace sf {

class Sha256 {
 public:
  Sha256();
  void update(const void* data, size_t len);
  std::string hex();  // finalises; the object must not be updated afterwards

 private:
  void block(const uint8_t* p);
  uint32_t h_[8];
  uint8_t buf_[64];
  size_t used_ = 0;
  uint64_t bytes_ = 0;
};

std::string sha256_hex(const void* data, size_t len);
std::string sha256_file(const std::string& path);  // throws if the file cannot be read

}  // namespace sf

// Trial stream sf-stream-v1 (Phase 2 SPEC §3): a trial's start depends only on (key, task, series, cell, j,
// replicate, purpose), never on device, thread count, batch, or schedule.
//
// Philox4x32-10 (Salmon et al. 2011, Random123) keyed by the seed; the counter encodes the trial. Each attempt yields
// two 64-bit candidates, mapped to [0, M) by Lemire's multiply-and-reject method. The header compiles as host C++ and
// as CUDA device code (SF_HD), so both back ends draw the same starts from the same source.
#pragma once

#include <cstdint>

#include "sf/hd.hpp"

namespace sf {

constexpr uint64_t kStreamSeed = 20261010;
constexpr int kStreamMaxAttempts = 256;

// Codes from protocol.json ("stream").
enum StreamTask : uint32_t { kTaskT1 = 1, kTaskT2 = 2, kTaskTRefEst = 3, kTaskTDiag = 4, kTaskGate = 5 };
enum StreamPurpose : uint32_t {
  kPurposeMeasured = 0,
  kPurposeWarmup = 1,
  kPurposeSpot = 2,
  kPurposeOracle = 3,
  kPurposeCalibration = 4,
  kPurposeGate = 5
};

struct Philox4x32 {
  uint32_t v[4];
};

SF_HD void philox_mulhilo(uint32_t a, uint32_t b, uint32_t& hi, uint32_t& lo) {
  const uint64_t p = static_cast<uint64_t>(a) * b;
  hi = static_cast<uint32_t>(p >> 32);
  lo = static_cast<uint32_t>(p);
}

// Philox4x32 with 10 rounds (Random123 constants).
SF_HD Philox4x32 philox4x32_10(Philox4x32 ctr, uint32_t k0, uint32_t k1) {
  for (int r = 0; r < 10; ++r) {
    if (r > 0) {
      k0 += 0x9E3779B9u;
      k1 += 0xBB67AE85u;
    }
    uint32_t hi0, lo0, hi1, lo1;
    philox_mulhilo(0xD2511F53u, ctr.v[0], hi0, lo0);
    philox_mulhilo(0xCD9E8D57u, ctr.v[2], hi1, lo1);
    ctr = Philox4x32{{hi1 ^ ctr.v[1] ^ k0, lo1, hi0 ^ ctr.v[3] ^ k1, lo0}};
  }
  return ctr;
}

// (m * M) as a 128-bit product: high and low 64 bits.
SF_HD void mul_64x64(uint64_t a, uint64_t b, uint64_t& hi, uint64_t& lo) {
#if defined(__CUDA_ARCH__)
  lo = a * b;
  hi = __umul64hi(a, b);
#else
  const unsigned __int128 p = static_cast<unsigned __int128>(a) * b;
  hi = static_cast<uint64_t>(p >> 64);
  lo = static_cast<uint64_t>(p);
#endif
}

struct StreamKey {
  uint32_t task, series, cell, replicate, purpose;
};

SF_HD Philox4x32 stream_counter(const StreamKey& key, uint64_t j, uint32_t attempt) {
  return Philox4x32{{static_cast<uint32_t>(j), static_cast<uint32_t>(j >> 32),
                     (key.task & 0xFFu) << 24 | (key.series & 0xFu) << 20 | (key.cell & 0xFFFFFu),
                     (key.purpose & 0xFu) << 28 | (key.replicate & 0xFFFFFu) << 8 | (attempt & 0xFFu)}};
}

// Uniform draw in [0, M) for trial j of the key; returns false after kStreamMaxAttempts attempts (the runner aborts).
// `rejections` (optional) counts rejected candidates.
SF_HD bool stream_draw(const StreamKey& key, uint64_t j, uint64_t M, uint64_t& out, int* rejections = nullptr) {
  const uint32_t k0 = static_cast<uint32_t>(kStreamSeed), k1 = static_cast<uint32_t>(kStreamSeed >> 32);
  const uint64_t t = (0 - M) % M;  // (2^64 - M) mod M
  for (uint32_t attempt = 0; attempt < static_cast<uint32_t>(kStreamMaxAttempts); ++attempt) {
    const Philox4x32 x = philox4x32_10(stream_counter(key, j, attempt), k0, k1);
    const uint64_t cand[2] = {x.v[0] | static_cast<uint64_t>(x.v[1]) << 32, x.v[2] | static_cast<uint64_t>(x.v[3]) << 32};
    for (int h = 0; h < 2; ++h) {
      uint64_t hi, lo;
      mul_64x64(cand[h], M, hi, lo);
      if (lo < M && lo < t) {
        if (rejections) ++*rejections;
        continue;
      }
      out = hi;
      return true;
    }
  }
  return false;
}

}  // namespace sf

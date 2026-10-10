#include "sf/check.hpp"

#include <cstring>

namespace sf {

const char* check_stage_name(CheckStage s) {
  switch (s) {
    case CheckStage::None: return "none";
    case CheckStage::Features: return "features";
    case CheckStage::TailCounts: return "tail counts";
    case CheckStage::Candidates: return "ordered candidates";
    case CheckStage::Episodes: return "episodes";
    case CheckStage::Bits: return "hit/overlap";
  }
  return "?";
}

bool same_float_bits(float a, float b) { return std::memcmp(&a, &b, sizeof a) == 0; }
bool same_double_bits(double a, double b) { return std::memcmp(&a, &b, sizeof a) == 0; }

bool same_episode(const Episode& a, const Episode& b) {
  return a.k == b.k && a.start == b.start && a.window == b.window && a.count == b.count && same_double_bits(a.S, b.S);
}

bool same_episode_list(const std::vector<Episode>& a, const std::vector<Episode>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (!same_episode(a[i], b[i])) return false;
  return true;
}

namespace {
CheckResult fail(CheckStage s, const std::string& d) { return {s, d}; }
}  // namespace

CheckResult compare_traces(const TrialTrace& ref, const TrialTrace& got, bool stages) {
  if (stages) {
    if (!ref.has_stages || !got.has_stages) return fail(CheckStage::Features, "trace without stages");
    if (ref.rows.size() != got.rows.size()) return fail(CheckStage::Features, "row count");
    for (size_t k = 0; k < ref.rows.size(); ++k) {
      const auto &a = ref.rows[k], &b = got.rows[k];
      if (a.c0 != b.c0 || a.c1 != b.c1 || a.range.size() != b.range.size() || a.volume.size() != b.volume.size())
        return fail(CheckStage::Features, "row " + std::to_string(k) + ": changed range");
      for (size_t j = 0; j < a.range.size(); ++j)
        if (!same_float_bits(a.range[j], b.range[j]) || !same_float_bits(a.volume[j], b.volume[j]))
          return fail(CheckStage::Features, "row " + std::to_string(k) + " start " + std::to_string(a.c0 + j));
    }
    for (size_t k = 0; k < ref.rows.size(); ++k) {
      const auto &a = ref.rows[k], &b = got.rows[k];
      if (a.both_start != b.both_start || a.both_count != b.both_count)
        return fail(CheckStage::TailCounts, "row " + std::to_string(k) + ": both-set (" +
                                                std::to_string(a.both_start.size()) + " vs " +
                                                std::to_string(b.both_start.size()) + " entries)");
    }
    if (!same_episode_list(ref.ordered, got.ordered))
      return fail(CheckStage::Candidates, std::to_string(ref.ordered.size()) + " vs " +
                                              std::to_string(got.ordered.size()) + " candidates");
  }
  if (ref.episodes.size() != got.episodes.size()) return fail(CheckStage::Episodes, "policy count");
  for (size_t p = 0; p < ref.episodes.size(); ++p)
    if (!same_episode_list(ref.episodes[p], got.episodes[p]))
      return fail(CheckStage::Episodes, "policy " + std::to_string(p));
  return {};
}

CheckResult compare_bits(const uint64_t* ref_hit, const uint64_t* ref_overlap, const uint64_t* got_hit,
                         const uint64_t* got_overlap, size_t policies) {
  for (size_t p = 0; p < policies; ++p) {
    const uint64_t m = uint64_t{1} << (p % 64);
    const size_t w = p / 64;
    if ((ref_hit[w] & m) != (got_hit[w] & m) || (ref_overlap[w] & m) != (got_overlap[w] & m))
      return fail(CheckStage::Bits, "policy " + std::to_string(p));
  }
  return {};
}

}  // namespace sf

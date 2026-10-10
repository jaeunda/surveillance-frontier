// Exactness checker shared by the gate, spot verification, and the oracle (Phase 2 SPEC §5.1).
//
// Compares two results of one trial with no tolerance: per stage, features of the changed starts by bit pattern,
// the both-set (starts and counts), the ordered candidates, then per policy the episodes (k, start, window, count,
// and S by bit pattern) and the hit and overlap bits. Reports the first stage that differs.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "sf/evaluate.hpp"

namespace sf {

enum class CheckStage { None, Features, TailCounts, Candidates, Episodes, Bits };
const char* check_stage_name(CheckStage s);

struct CheckResult {
  CheckStage first = CheckStage::None;
  std::string detail;
  bool ok() const { return first == CheckStage::None; }
};

bool same_float_bits(float a, float b);
bool same_double_bits(double a, double b);
bool same_episode(const Episode& a, const Episode& b);  // every field, S by bit pattern
bool same_episode_list(const std::vector<Episode>& a, const std::vector<Episode>& b);

// stages: also compare features, both-set and candidates (both traces must have them).
CheckResult compare_traces(const TrialTrace& ref, const TrialTrace& got, bool stages);

// Hit and overlap bits of all policies of one trial (word arrays of policy_words(P) words each).
CheckResult compare_bits(const uint64_t* ref_hit, const uint64_t* ref_overlap, const uint64_t* got_hit,
                         const uint64_t* got_overlap, size_t policies);

}  // namespace sf

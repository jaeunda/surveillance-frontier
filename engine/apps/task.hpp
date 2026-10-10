// Phase 2 task files (SPEC §2): what sf_run and sf_gate read. Generated from protocol.json by
// experiments/phase2-gpu/make_tasks.py; the runner never derives task content on its own.
//
//   {"task": "T2-policy-map", "task_code": 2,
//    "series": {"name": "1s-week", "path": "data/...bin", "sha256": "...", "code": 1, "prefix": 0},
//    "window_ladder": [2, 3, ...], "policies": [{"K": 8, "s_min": 1.5, "episode_gap": 30, "top_k": 20}, ...],
//    "max_length": 256,
//    "cells": [{"L": 2, "q": 1.0, "kind": "both", "stream_cell": 0}, ...],
//    "trials": {"order": "cell-major", "per_cell": 11726} | {"order": "enumerate"} | {"order": "round-robin"},
//    "stream": {"purpose": "measured", "replicate": 0},
//    "alpha": 1.53e-05, "half_width": 0.02,                      fixed-n tasks
//    "checkpoints": [256, ...], "alpha_per_interval": ..., "estimate_if_n0": 0.5}   deadline tasks
#pragma once

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "sf/batch.hpp"
#include "sf/json.hpp"

namespace sf {

struct TaskCell {
  int32_t length;
  double strength;
  EventKind kind;
  uint32_t stream_cell;
};

struct Task {
  std::string name, path_json;
  uint32_t task_code = 0;
  std::string series_name, series_path, series_sha256;
  uint32_t series_code = 0;
  int64_t prefix = 0;
  std::vector<int32_t> ladder;
  std::vector<Policy> policies;
  int32_t max_length = 0;
  std::vector<TaskCell> cells;
  std::string order;  // cell-major | enumerate | round-robin
  int64_t per_cell = 0;
  std::string purpose = "measured";
  uint32_t replicate = 0;
  double alpha = 0.05, half_width = 0;
  std::vector<int64_t> checkpoints;
  double alpha_per_interval = 0, estimate_if_n0 = 0.5;
};

inline std::string read_text(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

inline EventKind parse_kind(const std::string& k) {
  if (k == "both") return EventKind::Both;
  if (k == "price") return EventKind::Price;
  if (k == "volume") return EventKind::Volume;
  throw std::invalid_argument("unknown event kind " + k);
}
inline const char* kind_name(EventKind k) {
  return k == EventKind::Both ? "both" : k == EventKind::Price ? "price" : "volume";
}

inline uint32_t purpose_code(const std::string& p) {
  if (p == "measured") return kPurposeMeasured;
  if (p == "warmup") return kPurposeWarmup;
  if (p == "spot") return kPurposeSpot;
  if (p == "oracle") return kPurposeOracle;
  if (p == "calibration") return kPurposeCalibration;
  if (p == "gate") return kPurposeGate;
  throw std::invalid_argument("unknown stream purpose " + p);
}

inline Task parse_task(const Json& j, const std::string& path = "") {
  Task t;
  t.path_json = path;
  t.name = j["task"].str();
  t.task_code = static_cast<uint32_t>(j["task_code"].int64());
  const Json& s = j["series"];
  t.series_name = s["name"].str();
  t.series_path = s["path"].str();
  t.series_sha256 = s.str_or("sha256", "");
  t.series_code = static_cast<uint32_t>(s["code"].int64());
  t.prefix = s.int_or("prefix", 0);
  for (const auto& w : j["window_ladder"].items()) t.ladder.push_back(static_cast<int32_t>(w.int64()));
  for (const auto& p : j["policies"].items())
    t.policies.push_back({static_cast<int>(p["K"].int64()),
                          {p["s_min"].num(), static_cast<int>(p.int_or("top_k", 20)), p["episode_gap"].int64()}});
  t.max_length = static_cast<int32_t>(j["max_length"].int64());
  for (const auto& c : j["cells"].items())
    t.cells.push_back({static_cast<int32_t>(c["L"].int64()), c["q"].num(), parse_kind(c.str_or("kind", "both")),
                       static_cast<uint32_t>(c["stream_cell"].int64())});
  const Json& tr = j["trials"];
  t.order = tr["order"].str();
  t.per_cell = tr.int_or("per_cell", 0);
  if (j.has("stream")) {
    t.purpose = j["stream"].str_or("purpose", "measured");
    t.replicate = static_cast<uint32_t>(j["stream"].int_or("replicate", 0));
  }
  t.alpha = j.num_or("alpha", 0.05);
  t.half_width = j.num_or("half_width", 0);
  if (j.has("checkpoints"))
    for (const auto& c : j["checkpoints"].items()) t.checkpoints.push_back(c.int64());
  t.alpha_per_interval = j.num_or("alpha_per_interval", 0);
  t.estimate_if_n0 = j.num_or("estimate_if_n0", 0.5);
  if (t.cells.empty() || t.cells.size() > static_cast<size_t>(kMaxCells)) throw std::invalid_argument("cell count");
  if (t.order != "cell-major" && t.order != "enumerate" && t.order != "round-robin")
    throw std::invalid_argument("unknown trial order " + t.order);
  if (t.order == "cell-major" && t.per_cell <= 0) throw std::invalid_argument("cell-major task without per_cell");
  for (const auto& c : t.cells)
    if (c.length < 1 || c.length > t.max_length) throw std::invalid_argument("cell length outside max_length");
  return t;
}

inline Task load_task(const std::string& path) { return parse_task(Json::parse(read_text(path)), path); }

// Trial space of the task on a series of n bars, for a stream purpose and replicate.
inline TrialSpace make_space(const Task& t, int64_t n, uint32_t purpose, uint32_t replicate) {
  TrialSpace sp{};
  sp.cells = static_cast<int32_t>(t.cells.size());
  sp.key = StreamKey{t.task_code, t.series_code, 0, replicate, purpose};
  sp.offset[0] = 0;
  for (int32_t c = 0; c < sp.cells; ++c) {
    sp.stream_cell[c] = t.cells[c].stream_cell;
    sp.valid[c] = n - t.cells[c].length + 1;
    if (sp.valid[c] < 1) throw std::invalid_argument("series shorter than an event");
    sp.offset[c + 1] = sp.offset[c] + sp.valid[c];
  }
  if (t.order == "cell-major") {
    sp.order = TrialOrder::CellMajor;
    sp.per_cell = t.per_cell;
    sp.total = t.per_cell * sp.cells;
  } else if (t.order == "enumerate") {
    sp.order = TrialOrder::Enumerate;
    sp.total = sp.offset[sp.cells];
  } else {
    sp.order = TrialOrder::RoundRobin;
    sp.total = INT64_MAX;
  }
  return sp;
}

// Cache key of a prepared base state (SPEC §2.2).
inline std::string base_key(const Task& t, const std::string& input_sha, const std::string& backend,
                            const BackendConfig& cfg) {
  std::ostringstream k;
  k << input_sha << "|prefix=" << t.prefix << "|ladder=";
  for (int32_t w : t.ladder) k << w << ",";
  k << "|policies=";
  double s_floor = t.policies.at(0).select.s_min;
  for (const auto& p : t.policies) {
    k << p.K << "/" << p.select.s_min << "/" << p.select.episode_gap << "/" << p.select.top_k << ",";
    s_floor = std::min(s_floor, p.select.s_min);
  }
  k << "|s_floor=" << s_floor << "|max_length=" << t.max_length << "|backend=" << backend << "|" << cfg.describe();
  return k.str();
}

}  // namespace sf

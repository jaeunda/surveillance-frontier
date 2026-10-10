// Phase 2 task runner (SPEC §2). Every Phase 2 timing runs through it; the external harness
// (experiments/phase2-gpu/p2/harness.py) owns the end-to-end clock.
//
//   sf_run TASK --out DIR [backend options]                         cold: one task, then exit
//   sf_run TASK --out DIR --deadline-ns T --drain-ms M [...]         deadline (H4): T is absolute CLOCK_MONOTONIC
//   sf_run --serve SOCKET [backend options]                          warm: persistent, one JSON request per line
//   sf_run TASK --bench SECONDS [--diag] [backend options]           steady-state throughput (calibration, T-diag)
//   sf_run TASK --spot DIR [backend options]                         spot verification of a finished fixed-n run
//   sf_run --cp N,ALPHA                                              Clopper-Pearson bounds for k = 0..N (CSV), for
//                                                                    the SciPy cross-check (check_intervals.py)
//
// Back end options:
//   --device cpu|emul|cuda   (default cpu; emul = the GPU device code on the host, never timed)
//   --threads N              CPU workers, or host threads of a GPU path
//   --parity PC1,PC2,...     CPU parity options (SPEC §7); "none" (default)
//   --path inc|hyb           GPU path (G-inc / G-hyb)
//   --microbatch B|auto      GPU trials resident at once (auto = b_max); --submission S microbatches per round trip
//   --warmup-trials W        trials of stream purpose warmup before the measured trials, inside S5
//   --replicate R            override the task's stream replicate (H4 seeds)
//   --chunk N                trials per evaluate call (default 65536)
//   --fault NAME             gate negative controls only
//
// Outputs (fsynced, file and directory, before exit or acknowledgement): fixed n -> cells.csv, hits.bin;
// enumerate -> ref_bits.bin, rates.csv; deadline -> prefix.json; always stages.json with S0..S8 timestamps
// (CLOCK_MONOTONIC ns), resource records and output SHA-256s.
#include <fcntl.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "sf/check.hpp"
#include "sf/interval.hpp"
#include "sf/sha256.hpp"
#include "task.hpp"

namespace sf {
std::string affinity_text();
}

using namespace sf;

namespace {

int64_t mono_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

[[noreturn]] void fail(const std::string& msg) {
  std::fprintf(stderr, "sf_run: %s\n", msg.c_str());
  std::exit(2);
}

struct Options {
  std::string task, out, serve, spot, device = "cpu";
  BackendConfig cfg;
  int64_t deadline_ns = 0, drain_ms = 0, warmup = 0, chunk = 65536;
  double bench = 0;
  int64_t replicate = -1;
  std::string cp;
};

Options parse(int argc, char** argv) {
  Options o;
  int i = 1;
  if (argc > 1 && argv[1][0] != '-') o.task = argv[i++];
  for (; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--diag") {
      o.cfg.eval.diag = true;
      continue;
    }
    if (i + 1 >= argc) fail("missing value for " + a);
    const std::string v = argv[++i];
    if (a == "--device") o.device = v;
    else if (a == "--out") o.out = v;
    else if (a == "--serve") o.serve = v;
    else if (a == "--spot") o.spot = v;
    else if (a == "--threads") o.cfg.threads = std::atoi(v.c_str());
    else if (a == "--path") o.cfg.path = v;
    else if (a == "--microbatch") o.cfg.microbatch = v == "auto" ? 0 : std::atoll(v.c_str());
    else if (a == "--submission") o.cfg.submission = std::atoll(v.c_str());
    else if (a == "--memory-fraction") o.cfg.memory_fraction = std::atof(v.c_str());
    else if (a == "--gpu") o.cfg.device = std::atoi(v.c_str());
    else if (a == "--fault") o.cfg.fault = v;
    else if (a == "--warmup-trials") o.warmup = std::atoll(v.c_str());
    else if (a == "--deadline-ns") o.deadline_ns = std::atoll(v.c_str());
    else if (a == "--drain-ms") o.drain_ms = std::atoll(v.c_str());
    else if (a == "--bench") o.bench = std::atof(v.c_str());
    else if (a == "--replicate") o.replicate = std::atoll(v.c_str());
    else if (a == "--chunk") o.chunk = std::atoll(v.c_str());
    else if (a == "--cp") o.cp = v;
    else if (a == "--parity") {
      std::stringstream ss(v);
      for (std::string item; std::getline(ss, item, ',');) {
        if (item == "none") continue;
        else if (item == "PC1") o.cfg.event_table = true;
        else if (item == "PC2") o.cfg.eval.s_table = true;
        else if (item == "PC3") o.cfg.eval.cap_table = true;
        else if (item == "PC4") o.cfg.eval.sparse_both = true;
        else if (item == "PC5") continue;  // scratch reuse: already how the CPU back end works
        else if (item == "PC6") o.cfg.group_by_cell = true;
        else fail("unknown parity item " + item);
      }
    } else fail("unknown option " + a);
  }
  if (!backend_available(o.device)) fail("back end not available in this build: " + o.device);
  if (o.cfg.threads < 1 || o.chunk < 1) fail("invalid options");
  return o;
}

// ---- durable output ----

void fsync_path(const std::string& path, bool dir) {
  const int fd = open(path.c_str(), dir ? O_RDONLY | O_DIRECTORY : O_RDONLY);
  if (fd < 0 || fsync(fd) != 0) fail("fsync failed: " + path + ": " + std::strerror(errno));
  close(fd);
}

// Writes and fsyncs a file; returns its SHA-256.
std::string write_file(const std::string& path, const std::string& data) {
  const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) fail("cannot write " + path);
  for (size_t done = 0; done < data.size();) {
    const ssize_t w = write(fd, data.data() + done, data.size() - done);
    if (w < 0) fail("write failed: " + path);
    done += static_cast<size_t>(w);
  }
  if (fsync(fd) != 0) fail("fsync failed: " + path);
  close(fd);
  return sha256_hex(data.data(), data.size());
}

void make_dir(const std::string& d) {
  if (d.empty()) fail("--out is required");
  std::string acc;
  std::stringstream ss(d);
  if (d[0] == '/') acc = "/";
  for (std::string part; std::getline(ss, part, '/');) {
    if (part.empty()) continue;
    acc += part + "/";
    if (mkdir(acc.c_str(), 0755) != 0 && errno != EEXIST) fail("cannot create " + acc);
  }
}

std::string num(double v) {
  char b[40];
  std::snprintf(b, sizeof b, "%.17g", v);
  return b;
}

// Bit t * P + p of the packed file, least significant bit first within each byte.
std::string pack_bits(const std::vector<uint64_t>& words, int nwords, int64_t trials, size_t P) {
  std::string out((static_cast<size_t>(trials) * P + 7) / 8, '\0');
  for (int64_t t = 0; t < trials; ++t)
    for (size_t p = 0; p < P; ++p)
      if (words[t * nwords + p / 64] >> (p % 64) & 1) {
        const size_t bit = static_cast<size_t>(t) * P + p;
        out[bit / 8] = static_cast<char>(out[bit / 8] | (1 << (bit % 8)));
      }
  return out;
}

// ---- prepared state (warm mode keeps it) ----

struct Prepared {
  std::string key, input_sha;
  std::unique_ptr<Series> series;
  BaseState base;
  std::unique_ptr<BatchEvaluator> be;
};

struct Stamps {
  int64_t s[10] = {0};  // S0 .. S8, end
  bool warm = false;
};

std::string stage_json(const Stamps& st) {
  std::string j = "{";
  for (int i = 0; i <= 8; ++i) j += (i ? "," : "") + std::string("\"S") + std::to_string(i) + "\":" + std::to_string(st.s[i]);
  return j + ",\"end\":" + std::to_string(st.s[9]) + "}";
}

std::string rusage_json() {
  rusage ru{};
  getrusage(RUSAGE_SELF, &ru);
  char b[200];
  std::snprintf(b, sizeof b, "{\"max_rss_mib\":%.1f,\"major_faults\":%ld,\"minor_faults\":%ld}", ru.ru_maxrss / 1024.0,
                ru.ru_majflt, ru.ru_minflt);
  return b;
}

std::string stats_json(const BackendStats& s) {
  std::ostringstream j;
  const auto& t = s.sum;
  j << "{\"trials\":" << s.trials << ",\"stage_ms_sum\":{\"scan\":" << num(t.scan_ms) << ",\"rank\":" << num(t.rank_ms)
    << ",\"select\":" << num(t.select_ms) << ",\"peak\":" << num(t.peak_ms) << ",\"collect\":" << num(t.collect_ms)
    << ",\"order\":" << num(t.order_ms) << ",\"merge\":" << num(t.merge_ms) << ",\"reset\":" << num(t.reset_ms)
    << "},\"counts_sum\":{\"both\":" << s.counts.both << ",\"neighbours\":" << s.counts.neighbours
    << ",\"peaks\":" << s.counts.peaks << ",\"candidates\":" << s.counts.candidates << "},\"bytes_h2d\":" << s.bytes_h2d
    << ",\"bytes_d2h\":" << s.bytes_d2h << ",\"peak_device_bytes\":" << s.peak_device_bytes << ",\"workers\":[";
  for (size_t w = 0; w < s.workers.size(); ++w) {
    const auto& r = s.workers[w];
    j << (w ? "," : "") << "{\"cpu_start\":" << r.cpu_start << ",\"cpu_end\":" << r.cpu_end << ",\"affinity_start\":"
      << json_quote(r.affinity_start) << ",\"affinity_end\":" << json_quote(r.affinity_end)
      << ",\"trials\":" << r.trials << ",\"busy_s\":" << num(r.busy_s) << ",\"last_done_s\":" << num(r.last_done_s)
      << "}";
  }
  j << "]}";
  return j.str();
}

void merge_stats(BackendStats& a, const BackendStats& b) {
  a.trials += b.trials;
  a.sum.scan_ms += b.sum.scan_ms;
  a.sum.rank_ms += b.sum.rank_ms;
  a.sum.select_ms += b.sum.select_ms;
  a.sum.peak_ms += b.sum.peak_ms;
  a.sum.collect_ms += b.sum.collect_ms;
  a.sum.order_ms += b.sum.order_ms;
  a.sum.merge_ms += b.sum.merge_ms;
  a.sum.reset_ms += b.sum.reset_ms;
  a.counts.both += b.counts.both;
  a.counts.neighbours += b.counts.neighbours;
  a.counts.peaks += b.counts.peaks;
  a.counts.candidates += b.counts.candidates;
  a.bytes_h2d += b.bytes_h2d;
  a.bytes_d2h += b.bytes_d2h;
  a.peak_device_bytes = std::max(a.peak_device_bytes, b.peak_device_bytes);
  // workers: keep the first and the last call's records (loop start and end)
  if (a.workers.empty()) a.workers = b.workers;
  else
    for (size_t w = 0; w < std::min(a.workers.size(), b.workers.size()); ++w) {
      a.workers[w].cpu_end = b.workers[w].cpu_end;
      a.workers[w].affinity_end = b.workers[w].affinity_end;
      a.workers[w].trials += b.workers[w].trials;
      a.workers[w].busy_s += b.workers[w].busy_s;
    }
}

// S1: read the input and verify its hash. S3: base state and back end.
std::unique_ptr<Series> load_input(const Task& t, std::string& sha) {
  sha = sha256_file(t.series_path);
  if (!t.series_sha256.empty() && sha != t.series_sha256)
    throw std::runtime_error("input SHA-256 mismatch for " + t.series_path + ": " + sha);
  auto s = std::make_unique<Series>(load_series(t.series_path).head(t.prefix));
  if (t.prefix > 0 && s->size() != t.prefix) throw std::runtime_error("series shorter than the task prefix");
  return s;
}

void prepare(Prepared& p, const Task& t, const Options& o) {
  p.base = prepare_base(*p.series, t.ladder, t.policies, t.max_length, true,
                        o.device == "cpu" ? o.cfg.threads : std::max(1, o.cfg.threads));
  p.be = make_backend(o.device, p.base, o.cfg);
}

// S4: one median range per event length (a pass over the series), then the factor table of every cell
std::vector<CellPlan> plan_cells(const Task& t, const Series& s) {
  std::map<int32_t, double> med;
  for (const auto& c : t.cells)
    if (!med.count(c.length)) med[c.length] = median_range(view(s), c.length);
  std::vector<CellPlan> plans;
  for (const auto& c : t.cells) {
    CellPlan p;
    p.length = c.length;
    p.strength = c.strength;
    p.kind = c.kind;
    p.median_range = med[c.length];
    p.table = make_event_table(c.length, c.strength, p.median_range);
    plans.push_back(p);
  }
  return plans;
}

void warmup(Prepared& p, const Task& t, int64_t n) {
  if (n <= 0) return;
  TrialSpace sp = make_space(t, p.series->size(), kPurposeWarmup, 0);
  sp.order = TrialOrder::RoundRobin;
  const int words = policy_words(p.base.policies.size());
  std::vector<uint64_t> hit(n * words), ovl(n * words);
  std::vector<uint8_t> done(n);
  BatchOutput out{words, hit.data(), ovl.data(), done.data()};
  p.be->evaluate(TrialBatch{&sp, 0, n}, out, [] { return false; });
  p.be->take_stats();
}

// ---- one task (cold or warm) ----

struct Result {
  std::string status = "ok", error, result_sha, out_dir;
};

Result run_task(Prepared& p, const Task& t, const Options& o, Stamps& st) {
  Result res;
  res.out_dir = o.out;
  const size_t P = p.base.policies.size();
  const int words = policy_words(P);
  const int64_t n_bars = p.series->size();
  const uint32_t replicate = o.replicate >= 0 ? static_cast<uint32_t>(o.replicate) : t.replicate;
  const TrialSpace sp = make_space(t, n_bars, purpose_code(t.purpose), replicate);
  const bool deadline = o.deadline_ns > 0;
  if (deadline != (sp.order == TrialOrder::RoundRobin)) throw std::invalid_argument("round-robin tasks need a deadline");

  // S4: per-length preparation
  const auto plans = plan_cells(t, *p.series);
  p.be->set_cells(plans);
  p.be->sync();
  st.s[4] = mono_ns();

  // S5: trial loop (warm-up trials included)
  warmup(p, t, o.warmup);
  const int64_t stop_at = deadline ? o.deadline_ns - o.drain_ms * 1000000LL : INT64_MAX;
  const auto stop = [stop_at] { return mono_ns() >= stop_at; };
  std::vector<uint64_t> hit, ovl;
  std::vector<uint8_t> done;
  BackendStats stats;
  int64_t G = 0, completed = 0, stop_ns = 0;
  if (!deadline) {
    const int64_t total = sp.total;
    hit.assign(total * words, 0);
    ovl.assign(total * words, 0);
    done.assign(total, 0);
    for (int64_t g0 = 0; g0 < total; g0 += o.chunk) {
      const int64_t g1 = std::min(total, g0 + o.chunk);
      BatchOutput out{words, hit.data() + g0 * words, ovl.data() + g0 * words, done.data() + g0};
      p.be->evaluate(TrialBatch{&sp, g0, g1}, out, [] { return false; });
      merge_stats(stats, p.be->take_stats());
    }
    for (int64_t g = 0; g < total; ++g)
      if (!done[g]) throw std::runtime_error("trial " + std::to_string(g) + " not evaluated");
    G = completed = total;
  } else {
    for (int64_t g0 = 0;; g0 += o.chunk) {
      if (stop()) break;
      const int64_t g1 = g0 + o.chunk;
      hit.resize(g1 * words, 0);
      ovl.resize(g1 * words, 0);
      done.resize(g1, 0);
      BatchOutput out{words, hit.data() + g0 * words, ovl.data() + g0 * words, done.data() + g0};
      p.be->evaluate(TrialBatch{&sp, g0, g1}, out, stop);
      merge_stats(stats, p.be->take_stats());
      bool full = true;
      for (int64_t g = g0; g < g1; ++g) full &= done[g] != 0;
      if (!full) break;
    }
    stop_ns = mono_ns();
    while (G < static_cast<int64_t>(done.size()) && done[G]) ++G;  // contiguous prefix
    for (uint8_t d : done) completed += d;
  }
  p.be->sync();
  st.s[5] = mono_ns();

  // S6: aggregation per cell and policy
  const int C = sp.cells;
  std::vector<int64_t> n_c(C, 0), k_c(C * P, 0), ov_c(C * P, 0);
  std::vector<int64_t> ck_n;  // deadline: per cell, k at the largest reached checkpoint
  std::vector<int64_t> k_at;
  int64_t n_star = 0;
  {
    if (deadline) {
      int64_t min_n = INT64_MAX;
      for (int c = 0; c < C; ++c) min_n = std::min(min_n, G / C + (c < G % C ? 1 : 0));
      for (int64_t cp : t.checkpoints)
        if (cp <= min_n) n_star = std::max(n_star, cp);
      k_at.assign(C * P, 0);
    }
    for (int64_t g = 0; g < G; ++g) {
      int32_t c;
      int64_t j, s;
      if (sp.order == TrialOrder::RoundRobin) {
        c = static_cast<int32_t>(g % C);
        j = g / C;
      } else if (sp.order == TrialOrder::CellMajor) {
        c = static_cast<int32_t>(g / sp.per_cell);
        j = g - c * sp.per_cell;
      } else {
        resolve_trial(sp, g, c, j, s);
      }
      ++n_c[c];
      const uint64_t* hw = hit.data() + g * words;
      const uint64_t* ow = ovl.data() + g * words;
      for (size_t q = 0; q < P; ++q) {
        const int64_t h = hw[q / 64] >> (q % 64) & 1;
        k_c[c * P + q] += h;
        ov_c[c * P + q] += ow[q / 64] >> (q % 64) & 1;
        if (deadline && j < n_star) k_at[c * P + q] += h;
      }
    }
  }
  st.s[6] = mono_ns();

  // S7: intervals and width check
  std::vector<Interval> iv;
  bool width_ok = true;
  double worst_half = 0;
  if (sp.order == TrialOrder::CellMajor) {
    for (int c = 0; c < C; ++c)
      for (size_t q = 0; q < P; ++q) {
        iv.push_back(clopper_pearson(k_c[c * P + q], n_c[c], t.alpha));
        worst_half = std::max(worst_half, iv.back().half_width());
        width_ok &= t.half_width <= 0 || iv.back().half_width() <= t.half_width;
      }
  } else if (deadline && n_star > 0) {
    for (int c = 0; c < C; ++c)
      for (size_t q = 0; q < P; ++q) iv.push_back(clopper_pearson(k_at[c * P + q], n_star, t.alpha_per_interval));
  }
  st.s[7] = mono_ns();

  // S8: write and fsync
  make_dir(o.out);
  std::map<std::string, std::string> files;
  const auto cell_cols = [&](int c, size_t q) {
    const auto& tc = t.cells[c];
    const auto& pol = t.policies[q];
    return std::to_string(c) + "," + std::to_string(tc.length) + "," + num(tc.strength) + "," + kind_name(tc.kind) +
           "," + std::to_string(q) + "," + std::to_string(pol.K) + "," + num(pol.select.s_min) + "," +
           std::to_string(pol.select.episode_gap);
  };
  if (sp.order == TrialOrder::CellMajor) {
    std::string csv = "cell,L,q,kind,policy,K,s_min,episode_gap,n,k,overlap_k,lower,upper,half_width,width_ok\n";
    for (int c = 0; c < C; ++c)
      for (size_t q = 0; q < P; ++q) {
        const Interval& v = iv[c * P + q];
        csv += cell_cols(c, q) + "," + std::to_string(n_c[c]) + "," + std::to_string(k_c[c * P + q]) + "," +
               std::to_string(ov_c[c * P + q]) + "," + num(v.lower) + "," + num(v.upper) + "," + num(v.half_width()) +
               "," + (t.half_width <= 0 || v.half_width() <= t.half_width ? "1" : "0") + "\n";
      }
    files["cells.csv"] = write_file(o.out + "/cells.csv", csv);
    files["hits.bin"] = write_file(o.out + "/hits.bin", pack_bits(hit, words, G, P));
    res.result_sha = files["cells.csv"];
  } else if (sp.order == TrialOrder::Enumerate) {
    std::string csv = "cell,L,q,kind,policy,K,s_min,episode_gap,starts,k,overlap_k,rate\n";
    for (int c = 0; c < C; ++c)
      for (size_t q = 0; q < P; ++q)
        csv += cell_cols(c, q) + "," + std::to_string(n_c[c]) + "," + std::to_string(k_c[c * P + q]) + "," +
               std::to_string(ov_c[c * P + q]) + "," + num(static_cast<double>(k_c[c * P + q]) / n_c[c]) + "\n";
    files["rates.csv"] = write_file(o.out + "/rates.csv", csv);
    files["ref_bits.bin"] = write_file(o.out + "/ref_bits.bin", pack_bits(hit, words, G, P));
    res.result_sha = files["ref_bits.bin"];
  } else {
    std::ostringstream j;
    j << "{\"G\":" << G << ",\"completed\":" << completed << ",\"completed_unused\":" << completed - G
      << ",\"replicate\":" << replicate << ",\"stop_ns\":" << stop_ns << ",\"deadline_ns\":" << o.deadline_ns
      << ",\"drain_ms\":" << o.drain_ms << ",\"checkpoint\":" << n_star << ",\"cells\":[";
    for (int c = 0; c < C; ++c) {
      j << (c ? "," : "") << "{\"cell\":" << c << ",\"L\":" << t.cells[c].length << ",\"q\":" << num(t.cells[c].strength)
        << ",\"N\":" << n_c[c] << ",\"k\":[";
      for (size_t q = 0; q < P; ++q) j << (q ? "," : "") << k_c[c * P + q];
      j << "],\"k_at_checkpoint\":[";
      for (size_t q = 0; q < P && n_star > 0; ++q) j << (q ? "," : "") << k_at[c * P + q];
      j << "],\"lower\":[";
      for (size_t q = 0; q < P && n_star > 0; ++q) j << (q ? "," : "") << num(iv[c * P + q].lower);
      j << "],\"upper\":[";
      for (size_t q = 0; q < P && n_star > 0; ++q) j << (q ? "," : "") << num(iv[c * P + q].upper);
      j << "]}";
    }
    j << "]}\n";
    files["prefix.json"] = write_file(o.out + "/prefix.json", j.str());
    res.result_sha = files["prefix.json"];
  }
  fsync_path(o.out, true);
  st.s[8] = mono_ns();

  // run record of the runner side (outside S0..S8; "end" is taken just before it is written)
  st.s[9] = mono_ns();
  std::ostringstream j;
  j << "{\"task\":" << json_quote(t.name) << ",\"task_file\":" << json_quote(t.path_json) << ",\"series\":"
    << json_quote(t.series_name) << ",\"input_sha256\":" << json_quote(p.input_sha) << ",\"bars\":" << n_bars
    << ",\"device\":" << json_quote(o.device) << ",\"device_info\":" << json_quote(p.be->device_info())
    << ",\"config\":" << json_quote(o.cfg.describe()) << ",\"base_key_sha256\":"
    << json_quote(sha256_hex(p.key.data(), p.key.size())) << ",\"warm\":" << (st.warm ? "true" : "false")
    << ",\"stream\":{\"name\":\"sf-stream-v1\",\"purpose\":" << json_quote(t.purpose) << ",\"replicate\":" << replicate
    << "},\"warmup_trials\":" << o.warmup << ",\"trials\":" << G << ",\"completed\":" << completed
    << ",\"stages_ns\":" << stage_json(st) << ",\"width_ok\":" << (width_ok ? "true" : "false")
    << ",\"worst_half_width\":" << num(worst_half) << ",\"outputs\":{";
  bool first = true;
  for (const auto& f : files) j << (first ? "" : ",") << json_quote(f.first) << ":" << json_quote(f.second), first = false;
  j << "},\"result_sha256\":" << json_quote(res.result_sha) << ",\"rusage\":" << rusage_json()
    << ",\"backend\":" << stats_json(stats) << "}\n";
  write_file(o.out + "/stages.json", j.str());
  fsync_path(o.out, true);
  if (!width_ok) res.status = "width_fail";
  return res;
}

// ---- modes ----

int cold(const Options& o) {
  Stamps st;
  st.s[0] = mono_ns();
  const Task t = load_task(o.task);
  Prepared p;
  p.series = load_input(t, p.input_sha);
  st.s[1] = mono_ns();
  device_init(o.device, o.cfg);
  st.s[2] = mono_ns();
  prepare(p, t, o);
  p.key = base_key(t, p.input_sha, o.device, o.cfg);
  p.be->sync();
  st.s[3] = mono_ns();
  const Result r = run_task(p, t, o, st);
  std::printf("{\"status\":%s,\"result_sha256\":%s,\"out\":%s}\n", json_quote(r.status).c_str(),
              json_quote(r.result_sha).c_str(), json_quote(r.out_dir).c_str());
  std::fflush(stdout);
  // every output is durable at this point: exit without tearing down the prepared state (the same for every back
  // end), so teardown time is not part of the end-to-end time
  std::_Exit(r.status == "ok" ? 0 : 3);
}

int bench(const Options& o) {
  const auto setup0 = mono_ns();
  const Task t = load_task(o.task);
  Prepared p;
  p.series = load_input(t, p.input_sha);
  device_init(o.device, o.cfg);
  prepare(p, t, o);
  p.be->set_cells(plan_cells(t, *p.series));
  warmup(p, t, std::max<int64_t>(o.warmup, 1));
  p.be->sync();
  const double setup_s = (mono_ns() - setup0) / 1e9;
  TrialSpace sp = make_space(t, p.series->size(), kPurposeCalibration, 0);
  sp.order = TrialOrder::RoundRobin;
  const int words = policy_words(p.base.policies.size());
  std::vector<uint64_t> hit(o.chunk * words), ovl(o.chunk * words);
  std::vector<uint8_t> done(o.chunk);
  const int64_t t0 = mono_ns(), stop_at = t0 + static_cast<int64_t>(o.bench * 1e9);
  int64_t trials = 0, t_last = t0;
  BackendStats stats;
  for (int64_t g0 = 0; mono_ns() < stop_at; g0 += o.chunk) {
    std::fill(done.begin(), done.end(), 0);
    BatchOutput out{words, hit.data(), ovl.data(), done.data()};
    p.be->evaluate(TrialBatch{&sp, g0, g0 + o.chunk}, out, [stop_at] { return mono_ns() >= stop_at; });
    p.be->sync();
    t_last = mono_ns();
    for (uint8_t d : done) trials += d;
    merge_stats(stats, p.be->take_stats());
  }
  const double wall = (t_last - t0) / 1e9;
  std::printf("{\"mode\":\"bench\",\"task_file\":%s,\"device\":%s,\"device_info\":%s,\"config\":%s,\"bars\":%lld,"
              "\"policies\":%zu,\"cells\":%zu,\"seconds\":%g,\"setup_s\":%.6f,\"wall_s\":%.6f,\"trials\":%lld,"
              "\"trials_per_s\":%.6f,\"rusage\":%s,\"backend\":%s}\n",
              json_quote(o.task).c_str(), json_quote(o.device).c_str(), json_quote(p.be->device_info()).c_str(),
              json_quote(o.cfg.describe()).c_str(), static_cast<long long>(p.series->size()),
              p.base.policies.size(), t.cells.size(), o.bench, setup_s, wall, static_cast<long long>(trials),
              wall > 0 ? trials / wall : 0.0, rusage_json().c_str(), stats_json(stats).c_str());
  return 0;
}

// Spot verification (SPEC §2.5): 256 trial ids from purpose spot, re-evaluated by the run's back end and by the CPU
// tail method, compared in all fields with each other and with the stored bits.
int spot(const Options& o) {
  const Task t = load_task(o.task);
  Prepared p;
  p.series = load_input(t, p.input_sha);
  device_init(o.device, o.cfg);
  prepare(p, t, o);
  const auto plans = plan_cells(t, *p.series);
  p.be->set_cells(plans);
  const uint32_t replicate = o.replicate >= 0 ? static_cast<uint32_t>(o.replicate) : t.replicate;
  const TrialSpace sp = make_space(t, p.series->size(), purpose_code(t.purpose), replicate);
  if (sp.order == TrialOrder::RoundRobin) fail("spot verification is for fixed-n and enumeration runs");
  const std::string bits = read_text(o.spot + (sp.order == TrialOrder::Enumerate ? "/ref_bits.bin" : "/hits.bin"));
  const size_t P = p.base.policies.size();
  if (bits.size() != (static_cast<size_t>(sp.total) * P + 7) / 8) fail("stored bits have the wrong size");
  constexpr int kSpot = 256;
  std::vector<int64_t> ids(kSpot), starts(kSpot);
  std::vector<int32_t> cells(kSpot);
  const StreamKey key{t.task_code, t.series_code, 0, 0, kPurposeSpot};
  for (int i = 0; i < kSpot; ++i) {
    uint64_t g;
    if (!stream_draw(key, i, static_cast<uint64_t>(sp.total), g)) fail("spot draw failed");
    ids[i] = static_cast<int64_t>(g);
    int64_t j;
    resolve_trial(sp, ids[i], cells[i], j, starts[i]);
  }
  const int words = policy_words(P);
  std::vector<uint64_t> hit(kSpot * words), ovl(kSpot * words);
  std::vector<uint8_t> done(kSpot);
  std::vector<TrialTrace> traces;
  BatchOutput out{words, hit.data(), ovl.data(), done.data(), &traces};
  p.be->evaluate(TrialBatch{nullptr, 0, kSpot, cells.data(), starts.data()}, out, [] { return false; });
  EvalOptions eo;
  eo.trace = true;
  Evaluator tail(p.base, Method::Tail, eo);
  int mismatches = 0;
  std::string first;
  for (int i = 0; i < kSpot; ++i) {
    const CellPlan& c = plans[cells[i]];
    const TestEvent e{starts[i], c.length, c.strength, c.kind};
    const auto& got = tail.run(e, c.median_range, 1);
    std::vector<uint64_t> th(words, 0), to(words, 0), sh(words, 0);
    for (size_t q = 0; q < P; ++q) {
      const Detection d = detection(got[q], p.base.episodes[q], e);
      if (d.hit) th[q / 64] |= uint64_t{1} << (q % 64);
      if (d.overlap) to[q / 64] |= uint64_t{1} << (q % 64);
      const size_t bit = static_cast<size_t>(ids[i]) * P + q;
      if (bits[bit / 8] >> (bit % 8) & 1) sh[q / 64] |= uint64_t{1} << (q % 64);
    }
    CheckResult r = compare_traces(tail.trace(), traces[i], true);
    if (r.ok()) r = compare_bits(th.data(), to.data(), hit.data() + i * words, ovl.data() + i * words, P);
    if (r.ok()) r = compare_bits(th.data(), to.data(), sh.data(), to.data(), P);  // stored: hit bits only
    if (!r.ok()) {
      ++mismatches;
      if (first.empty()) first = "trial " + std::to_string(ids[i]) + ": " + check_stage_name(r.first) + " " + r.detail;
    }
  }
  std::ostringstream j;
  j << "{\"spot_trials\":" << kSpot << ",\"mismatches\":" << mismatches << ",\"first\":" << json_quote(first)
    << ",\"device\":" << json_quote(o.device) << ",\"config\":" << json_quote(o.cfg.describe()) << "}\n";
  write_file(o.spot + "/spot.json", j.str());
  fsync_path(o.spot, true);
  std::fputs(j.str().c_str(), stdout);
  return mismatches ? 1 : 0;
}

// Warm mode: one request per connection, a JSON line {"task": PATH, "out": DIR, "deadline_ns": T, "drain_ms": M,
// "warmup_trials": W, "replicate": R} or {"cmd": "quit"}; the answer is a JSON line sent after the output fsync.
int serve(const Options& base_opt) {
  const int srv = socket(AF_UNIX, SOCK_STREAM, 0);
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  if (base_opt.serve.size() >= sizeof addr.sun_path) fail("socket path too long");
  std::strcpy(addr.sun_path, base_opt.serve.c_str());
  unlink(base_opt.serve.c_str());
  if (srv < 0 || bind(srv, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0 || listen(srv, 4) != 0)
    fail("cannot listen on " + base_opt.serve);
  device_init(base_opt.device, base_opt.cfg);
  std::map<std::string, std::unique_ptr<Prepared>> cache;  // by base-state key; at most one entry
  std::fprintf(stdout, "{\"serving\":%s}\n", json_quote(base_opt.serve).c_str());
  std::fflush(stdout);
  for (;;) {
    const int c = accept(srv, nullptr, nullptr);
    if (c < 0) continue;
    std::string line;
    char ch;
    while (read(c, &ch, 1) == 1 && ch != '\n') line.push_back(ch);
    Stamps st;
    st.warm = true;
    st.s[0] = mono_ns();
    std::string reply;
    bool quit = false;
    try {
      const Json req = Json::parse(line);
      if (req.str_or("cmd", "") == "quit") {
        quit = true;
        reply = "{\"status\":\"bye\"}";
      } else {
        Options o = base_opt;
        o.task = req["task"].str();
        o.out = req["out"].str();
        o.deadline_ns = req.int_or("deadline_ns", 0);
        o.drain_ms = req.int_or("drain_ms", 0);
        o.warmup = req.int_or("warmup_trials", base_opt.warmup);
        o.replicate = req.int_or("replicate", -1);
        const Task t = load_task(o.task);
        std::string sha = sha256_file(t.series_path);
        if (!t.series_sha256.empty() && sha != t.series_sha256) throw std::runtime_error("input SHA-256 mismatch");
        const std::string key = base_key(t, sha, o.device, o.cfg);
        // one prepared state at a time: another key's state would hold memory (and GPU memory sizes the batch)
        for (auto it = cache.begin(); it != cache.end();) it = it->first == key ? std::next(it) : cache.erase(it);
        auto& slot = cache[key];
        const bool hit = static_cast<bool>(slot);
        st.s[1] = st.s[2] = mono_ns();
        if (!hit) {
          slot = std::make_unique<Prepared>();
          slot->series = load_input(t, slot->input_sha);
          prepare(*slot, t, o);
          slot->key = key;
          slot->be->sync();
        }
        st.s[3] = mono_ns();
        const Result r = run_task(*slot, t, o, st);
        reply = "{\"status\":" + json_quote(r.status) + ",\"result_sha256\":" + json_quote(r.result_sha) +
                ",\"base_cached\":" + (hit ? "true" : "false") + ",\"out\":" + json_quote(r.out_dir) + "}";
      }
    } catch (const std::exception& e) {
      reply = "{\"status\":\"error\",\"error\":" + json_quote(e.what()) + "}";
    }
    reply += "\n";
    if (write(c, reply.data(), reply.size()) < 0) std::fprintf(stderr, "sf_run: reply failed\n");
    close(c);
    if (quit) break;
  }
  close(srv);
  unlink(base_opt.serve.c_str());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: sf_run TASK --out DIR [options] | sf_run --serve SOCKET [options] (see source)\n");
    return 2;
  }
  const Options o = parse(argc, argv);
  if (!o.cp.empty()) {
    const int64_t n = std::atoll(o.cp.c_str());
    const double alpha = std::atof(o.cp.substr(o.cp.find(',') + 1).c_str());
    std::vector<Interval> iv(n + 1);
#pragma omp parallel for schedule(dynamic, 64)
    for (int64_t k = 0; k <= n; ++k) iv[k] = clopper_pearson(k, n, alpha);
    std::printf("k,lower,upper\n");
    for (int64_t k = 0; k <= n; ++k) std::printf("%lld,%.17g,%.17g\n", static_cast<long long>(k), iv[k].lower, iv[k].upper);
    return 0;
  }
  try {
    if (!o.serve.empty()) return serve(o);
    if (o.task.empty()) fail("task file required");
    if (o.bench > 0) return bench(o);
    if (!o.spot.empty()) return spot(o);
    return cold(o);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "sf_run: error: %s\n", e.what());
    std::printf("{\"status\":\"error\",\"error\":%s}\n", json_quote(e.what()).c_str());
    return 1;
  }
}

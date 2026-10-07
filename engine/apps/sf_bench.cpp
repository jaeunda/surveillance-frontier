// Throughput of stress-test trials on the CPU. A trial applies one random test event (sf/scenario.hpp) to the base
// series and evaluates every policy of a set on it with one method (sf/evaluate.hpp). Prints one JSON line.
//
//   sf_bench <series.bin> [options]
//     --method full|shared|tail|incremental   (default full)
//     --mode within|across  within: one trial at a time, parallel inside over window lengths (default)
//                           across: one trial per thread, P trials in flight
//     --threads P           worker threads (default 1)
//     --n N                 use the first N bars (default: all)
//     --k-set 15            policy grid: window counts (comma list) ...
//     --s-set 2             ... s_min values ...
//     --gap-set 60          ... episode gaps; policies = K x s_min x gap
//     --length L --strength q --event both|price|volume   test event (default 30, 8, both)
//     --max-length L        longest event the incremental base is sized for (default: --length)
//     --seconds B           run each worker until B seconds have passed (default 5)
//     --trials M            instead: exactly M trials in total
//     --warmup W            untimed trials per worker first (default 2)
//     --verify              also run every trial with the full method and count mismatching trials
//     --enumerate STRIDE    instead of random positions: every STRIDE-th start, once (exact detection rate)
//     --hits-out FILE       with --enumerate: one byte per (position, policy), 1 = detected
//     --seed S
//
// Throughput is the steady-state rate sum over workers of (trials completed / time of its last completion); a worker
// finishes the trial in flight at the deadline, so slow trials are not cut off. trials_by_deadline counts only those
// completed within the budget. Each worker allocates and first touches its own memory on its own thread.
#include <omp.h>
#include <sched.h>
#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "sf/evaluate.hpp"

namespace {
using Clock = std::chrono::steady_clock;
double seconds_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

struct Options {
  std::string path, method = "full", mode = "within", event = "both", hits_out;
  int threads = 1, warmup = 2;
  int64_t n = 0, trials = 0, enumerate = 0;
  double seconds = 5, strength = 8.0;
  int32_t length = 30, max_length = 0;
  bool verify = false;
  std::vector<int> ks = {15};
  std::vector<double> s_mins = {2.0};
  std::vector<int64_t> gaps = {60};
  uint64_t seed = 20261004;
};

template <typename T>
std::vector<T> list(const std::string& v) {
  std::vector<T> out;
  std::stringstream ss(v);
  for (std::string item; std::getline(ss, item, ',');) out.push_back(static_cast<T>(std::atof(item.c_str())));
  return out;
}

[[noreturn]] void fail(const std::string& msg) {
  std::fprintf(stderr, "%s\n", msg.c_str());
  std::exit(2);
}

Options parse(int argc, char** argv) {
  Options o;
  o.path = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--verify") {
      o.verify = true;
      continue;
    }
    if (i + 1 >= argc) fail("missing value for " + a);
    const std::string v = argv[++i];
    if (a == "--method") o.method = v;
    else if (a == "--mode") o.mode = v;
    else if (a == "--threads") o.threads = std::atoi(v.c_str());
    else if (a == "--n") o.n = std::atoll(v.c_str());
    else if (a == "--k-set") o.ks = list<int>(v);
    else if (a == "--s-set") o.s_mins = list<double>(v);
    else if (a == "--gap-set") o.gaps = list<int64_t>(v);
    else if (a == "--length") o.length = std::atoi(v.c_str());
    else if (a == "--strength") o.strength = std::atof(v.c_str());
    else if (a == "--event") o.event = v;
    else if (a == "--max-length") o.max_length = std::atoi(v.c_str());
    else if (a == "--seconds") o.seconds = std::atof(v.c_str());
    else if (a == "--trials") o.trials = std::atoll(v.c_str());
    else if (a == "--warmup") o.warmup = std::atoi(v.c_str());
    else if (a == "--enumerate") o.enumerate = std::atoll(v.c_str());
    else if (a == "--hits-out") o.hits_out = v;
    else if (a == "--seed") o.seed = std::strtoull(v.c_str(), nullptr, 10);
    else fail("unknown option " + a);
  }
  if (o.max_length < o.length) o.max_length = o.length;
  if ((o.mode != "within" && o.mode != "across") || o.threads < 1 || o.ks.empty() || o.s_mins.empty() ||
      o.gaps.empty() || (o.event != "both" && o.event != "price" && o.event != "volume"))
    fail("invalid options");
  return o;
}

double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const size_t m = v.size() / 2;
  return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

sf::EventKind event_kind(const std::string& e) {
  return e == "price" ? sf::EventKind::Price : e == "volume" ? sf::EventKind::Volume : sf::EventKind::Both;
}

struct Worker {
  sf::Evaluator eval;
  std::unique_ptr<sf::Evaluator> ref;  // --verify
  std::mt19937_64 rng;
  std::vector<double> trial_ms, scan_ms, rank_ms, select_ms;
  std::vector<int64_t> hits, overlaps;  // per policy
  int64_t done = 0, by_deadline = 0, mismatches = 0;
  double last_s = 0;
  int cpu = -1;

  Worker(const sf::BaseState& b, sf::Method m, bool verify, uint64_t seed)
      : eval(b, m), rng(seed), hits(b.policies.size()), overlaps(b.policies.size()) {
    if (verify) ref = std::make_unique<sf::Evaluator>(b, sf::Method::Full);
  }

  // One trial at position start; records detection per policy (and the hit bytes when asked).
  void run(const sf::BaseState& b, const sf::TestEvent& e, double med, int threads, bool record,
           uint8_t* hit_bytes = nullptr) {
    const auto t0 = Clock::now();
    sf::StageTimes st;
    const auto& got = eval.run(e, med, threads, &st);
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    if (!record) return;
    for (size_t p = 0; p < got.size(); ++p) {
      const auto d = sf::detection(got[p], b.episodes[p], e);
      hits[p] += d.hit;
      overlaps[p] += d.overlap;
      if (hit_bytes) hit_bytes[p] = d.hit;
    }
    if (ref) {
      const auto& expected = ref->run(e, med, threads);
      bool same = expected.size() == got.size();
      for (size_t p = 0; same && p < got.size(); ++p) {
        same = expected[p].size() == got[p].size();
        for (size_t i = 0; same && i < got[p].size(); ++i)
          same = expected[p][i].k == got[p][i].k && expected[p][i].start == got[p][i].start &&
                 expected[p][i].S == got[p][i].S;
      }
      mismatches += !same;
    }
    trial_ms.push_back(ms);
    scan_ms.push_back(st.scan_ms);
    rank_ms.push_back(st.rank_ms);
    select_ms.push_back(st.select_ms);
  }
};

std::string json_list(const std::vector<double>& v, const char* fmt = "%.6f") {
  std::string s = "[";
  char buf[64];
  for (size_t i = 0; i < v.size(); ++i) {
    std::snprintf(buf, sizeof buf, fmt, v[i]);
    s += (i ? "," : "") + std::string(buf);
  }
  return s + "]";
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <series.bin> [options]  (see source header)\n", argv[0]);
    return 2;
  }
  const Options o = parse(argc, argv);
  const auto setup0 = Clock::now();
  const sf::Series base = sf::load_series(o.path).head(o.n);
  const sf::Method method = sf::parse_method(o.method);
  const auto policies = sf::policy_grid(o.ks, o.s_mins, o.gaps);
  for (int k : o.ks)
    if (k < 1 || k > static_cast<int>(sf::kDefaultWindows.size())) fail("K out of range");
  const bool across = o.mode == "across";
  const int workers = across ? o.threads : 1;
  const int inner = across ? 1 : o.threads;
  const sf::BaseState b =
      sf::prepare_base(base, sf::kDefaultWindows, policies, o.max_length, method == sf::Method::Incremental, o.threads);
  const double med = sf::median_range(sf::view(base), o.length);
  const int64_t positions = base.size() - o.length + 1;
  const auto event_at = [&](int64_t start) { return sf::TestEvent{start, o.length, o.strength, event_kind(o.event)}; };

  // workers are created, first touched and warmed up on the thread that will run them
  std::vector<std::unique_ptr<Worker>> pool(workers);
#pragma omp parallel num_threads(workers)
  {
    const int t = omp_get_thread_num();
    pool[t] = std::make_unique<Worker>(b, method, o.verify, o.seed + t);
    pool[t]->cpu = sched_getcpu();
    std::uniform_int_distribution<int64_t> pos(0, positions - 1);
    for (int i = 0; i < o.warmup; ++i) pool[t]->run(b, event_at(pos(pool[t]->rng)), med, inner, false);
  }
  const double setup_s = seconds_since(setup0);

  std::vector<uint8_t> hit_bytes;
  const int64_t enum_count = o.enumerate > 0 ? (positions + o.enumerate - 1) / o.enumerate : 0;
  if (enum_count && !o.hits_out.empty()) hit_bytes.assign(enum_count * policies.size(), 0);

  const auto t0 = Clock::now();
  const auto one = [&](Worker& w, int64_t start, uint8_t* bytes) {
    w.run(b, event_at(start), med, inner, true, bytes);
    ++w.done;
    w.last_s = seconds_since(t0);
    if (w.last_s <= o.seconds) ++w.by_deadline;
  };
#pragma omp parallel num_threads(workers)
  {
    Worker& w = *pool[omp_get_thread_num()];
    std::uniform_int_distribution<int64_t> pos(0, positions - 1);
    if (enum_count) {
#pragma omp for schedule(dynamic, 64)
      for (int64_t i = 0; i < enum_count; ++i)
        one(w, i * o.enumerate, hit_bytes.empty() ? nullptr : hit_bytes.data() + i * policies.size());
    } else if (o.trials > 0) {
#pragma omp for schedule(dynamic, 1)
      for (int64_t i = 0; i < o.trials; ++i) one(w, pos(w.rng), nullptr);
    } else {
      // a worker starts trials until the deadline and always completes at least one
      while (seconds_since(t0) < o.seconds || w.done == 0) one(w, pos(w.rng), nullptr);
    }
  }
  const double wall_s = seconds_since(t0);
  if (!hit_bytes.empty()) {
    std::ofstream f(o.hits_out, std::ios::binary);
    f.write(reinterpret_cast<const char*>(hit_bytes.data()), static_cast<std::streamsize>(hit_bytes.size()));
  }

  // merge per-worker samples
  std::vector<double> tr, sc, rk, se, hit_rate(policies.size()), overlap_rate(policies.size()), cpus;
  int64_t done = 0, by_deadline = 0, mismatches = 0;
  double rate = 0;
  for (const auto& w : pool) {
    tr.insert(tr.end(), w->trial_ms.begin(), w->trial_ms.end());
    sc.insert(sc.end(), w->scan_ms.begin(), w->scan_ms.end());
    rk.insert(rk.end(), w->rank_ms.begin(), w->rank_ms.end());
    se.insert(se.end(), w->select_ms.begin(), w->select_ms.end());
    for (size_t p = 0; p < policies.size(); ++p) {
      hit_rate[p] += w->hits[p];
      overlap_rate[p] += w->overlaps[p];
    }
    done += w->done;
    by_deadline += w->by_deadline;
    mismatches += w->mismatches;
    if (w->done) rate += w->done / w->last_s;
    cpus.push_back(w->cpu);
  }
  for (size_t p = 0; p < policies.size(); ++p) {
    hit_rate[p] /= std::max<int64_t>(done, 1);
    overlap_rate[p] /= std::max<int64_t>(done, 1);
  }
  const bool timed = !enum_count && o.trials == 0;
  if (!timed) rate = done / wall_s;
  rusage ru{};
  getrusage(RUSAGE_SELF, &ru);
  std::vector<double> ks(o.ks.begin(), o.ks.end()), gaps(o.gaps.begin(), o.gaps.end());
  std::printf(
      "{\"method\":\"%s\",\"mode\":\"%s\",\"threads\":%d,\"n\":%lld,\"k_set\":%s,\"s_set\":%s,\"gap_set\":%s,"
      "\"policies\":%zu,\"length\":%d,\"strength\":%g,\"event\":\"%s\",\"enumerate\":%lld,\"trials\":%lld,"
      "\"trials_by_deadline\":%lld,\"wall_s\":%.6f,\"setup_s\":%.6f,\"trials_per_s\":%.6f,\"trial_ms_median\":%.4f,"
      "\"scan_ms_median\":%.4f,\"rank_ms_median\":%.4f,\"select_ms_median\":%.4f,\"verified\":%s,"
      "\"mismatches\":%lld,\"max_rss_mb\":%.1f,\"cpus\":%s,\"hit_rate\":%s,\"overlap_rate\":%s}\n",
      o.method.c_str(), o.mode.c_str(), o.threads, static_cast<long long>(base.size()), json_list(ks, "%g").c_str(),
      json_list(o.s_mins, "%g").c_str(), json_list(gaps, "%g").c_str(), policies.size(), o.length, o.strength,
      o.event.c_str(), static_cast<long long>(o.enumerate), static_cast<long long>(done),
      static_cast<long long>(timed ? by_deadline : done), wall_s, setup_s, rate, median(tr), median(sc), median(rk),
      median(se), o.verify ? "true" : "false", static_cast<long long>(mismatches), ru.ru_maxrss / 1024.0,
      json_list(cpus, "%g").c_str(), json_list(hit_rate).c_str(), json_list(overlap_rate).c_str());
  return o.verify && mismatches ? 1 : 0;
}

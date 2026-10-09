# Phase 1 results - aws-c7i-16xlarge_2026-10-09

CPU: Intel(R) Xeon(R) Platinum 8488C (32 physical / 64 logical cores), compiler: c++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0, commit bc9cd9c, OMP_PLACES=cores, OMP_PROC_BIND=spread, 3 repetitions.

## Gates and hypotheses

| id   | measured                                                                                                                               | criterion                                                                       | supported   |
|:-----|:---------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------|:------------|
| G    | 0 mismatching trials in 2112 verified trials                                                                                           | 0 (real data, all methods vs full, 24 policies)                                 | True        |
| F1   | incremental / full at N = 604,800, K = 15: 349x                                                                                        | >= 10x                                                                          | True        |
| F2   | full 14%, incremental 19%, tail 15%                                                                                                    | < 20% at N = 604,800 for every K and method                                     | True        |
| F3   | lowest efficiency at 32 threads: 0.72 (tail, N = 604,800)                                                                              | >= 0.7 for every method and N                                                   | True        |
| F4   | shared -2.1%, tail +0.4%, incremental -4.4%                                                                                            | |error| <= 15% for every method                                                 | True        |
| E1   | h = 0.05: pointwise min 0.945, all at once 0.111, Bonferroni 0.995; h = 0.02: pointwise min 0.940, all at once 0.042, Bonferroni 0.994 | pointwise >= 0.931 (0.95 - 4 replay SE) in every cell, Bonferroni joint >= 0.95 | True        |

## Single-thread ms per trial, K = 15, one policy

|      n |   full (ms) |   incremental (ms) |   tail (ms) |
|-------:|------------:|-------------------:|------------:|
|  16000 |       22.92 |               0.15 |        5.79 |
|  32000 |       48.13 |               0.25 |       11.11 |
|  64000 |      102.75 |               0.41 |       22.95 |
| 128000 |      218.71 |               0.71 |       46.33 |
| 256000 |      467.47 |               1.53 |       89.46 |
| 604800 |     1180.55 |               3.36 |      224.27 |

Cost models (ms), fitted on N <= 64,000: full: 3.95·1 + -0.000107·N·K + 1.33e-05·N·K·log2 N; incremental: 0.00452·1 + 4.2e-07·N·K; tail: 0.374·1 + 2.36e-05·N·K

## Stage times at N = 604,800, K = 15 (1 thread, ms)

| method      |   scan_ms_median |   rank_ms_median |   select_ms_median |   trial_ms_median |
|:------------|-----------------:|-----------------:|-------------------:|------------------:|
| full        |           147.10 |          1025.57 |               7.87 |           1180.55 |
| incremental |             0.04 |             1.87 |               1.44 |              3.36 |
| tail        |           147.16 |            70.12 |               7.16 |            224.27 |

## Stage slowdown per trial at 32 threads vs 1 (across mode)

| method      |      n |   scan / update |   rank |   select |
|:------------|-------:|----------------:|-------:|---------:|
| full        |  64000 |            1.17 |   1.17 |     1.11 |
| full        | 604800 |            1.17 |   1.20 |     1.48 |
| incremental |  64000 |            1.18 |   1.18 |     1.18 |
| incremental | 604800 |            1.19 |   1.22 |     1.17 |
| tail        |  64000 |            1.17 |   1.24 |     1.38 |
| tail        | 604800 |            1.17 |   1.89 |     1.38 |

## Time-to-solution per task (projected from measured batch throughput)

| task                          | factor         | value         | series     |   policies |   cells |   trials_per_cell |   trials | full_s       | shared_s     | tail_s   | incremental_s   |
|:------------------------------|:---------------|:--------------|:-----------|-----------:|--------:|------------------:|---------:|:-------------|:-------------|:---------|:----------------|
| T1-curve                      | task           | T1-curve      | 1m-quarter |          1 |     136 |              2449 |   333064 | 48.5 min     | not measured | 12.1 min | 11.1 s          |
| T1-curve                      | task           | T1-curve      | 1s-week    |          1 |     136 |              2449 |   333064 | 4.0 h        | not measured | 52.6 min | 42.9 s          |
| T2-policy-map                 | task           | T2-policy-map | 1m-quarter |         24 |     136 |             11726 |  1594736 | 3.1 days     | 3.9 h        | 1.0 h    | 3.0 min         |
| T2-policy-map                 | task           | T2-policy-map | 1s-week    |         24 |     136 |             11726 |  1594736 | 15.2 days    | 19.4 h       | 4.7 h    | 12.1 min        |
| base, half_width = 0.05       | half_width     | 0.05          | 1m-quarter |          1 |     136 |               402 |    54672 | 8.0 min      | not measured | 2.0 min  | 1.9 s           |
| base, half_width = 0.05       | half_width     | 0.05          | 1s-week    |          1 |     136 |               402 |    54672 | 39.9 min     | not measured | 8.6 min  | 7.3 s           |
| base, half_width = 0.02       | half_width     | 0.02          | 1m-quarter |          1 |     136 |              2449 |   333064 | 48.5 min     | not measured | 12.1 min | 11.1 s          |
| base, half_width = 0.02       | half_width     | 0.02          | 1s-week    |          1 |     136 |              2449 |   333064 | 4.0 h        | not measured | 52.6 min | 42.9 s          |
| base, half_width = 0.01       | half_width     | 0.01          | 1m-quarter |          1 |     136 |              9701 |  1319336 | 3.2 h        | not measured | 47.8 min | 43.8 s          |
| base, half_width = 0.01       | half_width     | 0.01          | 1s-week    |          1 |     136 |              9701 |  1319336 | 16.0 h       | not measured | 3.5 h    | 2.8 min         |
| base, strength_ratio = x2     | strength_ratio | x2            | 1m-quarter |          1 |      72 |              2449 |   176328 | 25.7 min     | not measured | 6.4 min  | 5.9 s           |
| base, strength_ratio = x2     | strength_ratio | x2            | 1s-week    |          1 |      72 |              2449 |   176328 | 2.1 h        | not measured | 27.9 min | 22.8 s          |
| base, strength_ratio = x1.41  | strength_ratio | x1.41         | 1m-quarter |          1 |     136 |              2449 |   333064 | 48.5 min     | not measured | 12.1 min | 11.1 s          |
| base, strength_ratio = x1.41  | strength_ratio | x1.41         | 1s-week    |          1 |     136 |              2449 |   333064 | 4.0 h        | not measured | 52.6 min | 42.9 s          |
| base, strength_ratio = x1.19  | strength_ratio | x1.19         | 1m-quarter |          1 |     264 |              2449 |   646536 | 1.6 h        | not measured | 23.4 min | 21.5 s          |
| base, strength_ratio = x1.19  | strength_ratio | x1.19         | 1s-week    |          1 |     264 |              2449 |   646536 | 7.9 h        | not measured | 1.7 h    | 1.4 min         |
| base, lengths = 1 lengths     | lengths        | 1 lengths     | 1m-quarter |          1 |      17 |              2449 |    41633 | 6.1 min      | not measured | 1.5 min  | 1.4 s           |
| base, lengths = 1 lengths     | lengths        | 1 lengths     | 1s-week    |          1 |      17 |              2449 |    41633 | 30.7 min     | not measured | 6.6 min  | 5.5 s           |
| base, lengths = 4 lengths     | lengths        | 4 lengths     | 1m-quarter |          1 |      68 |              2449 |   166532 | 24.2 min     | not measured | 6.0 min  | 5.6 s           |
| base, lengths = 4 lengths     | lengths        | 4 lengths     | 1s-week    |          1 |      68 |              2449 |   166532 | 2.0 h        | not measured | 26.3 min | 21.6 s          |
| base, lengths = 8 lengths     | lengths        | 8 lengths     | 1m-quarter |          1 |     136 |              2449 |   333064 | 48.5 min     | not measured | 12.1 min | 11.1 s          |
| base, lengths = 8 lengths     | lengths        | 8 lengths     | 1s-week    |          1 |     136 |              2449 |   333064 | 4.0 h        | not measured | 52.6 min | 42.9 s          |
| base, lengths = 15 lengths    | lengths        | 15 lengths    | 1m-quarter |          1 |     255 |              2449 |   624495 | 1.5 h        | not measured | 22.6 min | 20.9 s          |
| base, lengths = 15 lengths    | lengths        | 15 lengths    | 1s-week    |          1 |     255 |              2449 |   624495 | 7.6 h        | not measured | 1.6 h    | 1.3 min         |
| base, policies = P1           | policies       | P1            | 1m-quarter |          1 |     136 |              2449 |   333064 | 48.5 min     | not measured | 12.1 min | 11.1 s          |
| base, policies = P1           | policies       | P1            | 1s-week    |          1 |     136 |              2449 |   333064 | 4.0 h        | not measured | 52.6 min | 42.9 s          |
| base, policies = P24          | policies       | P24           | 1m-quarter |         24 |     136 |              2449 |   333064 | 15.5 h       | 48.8 min     | 12.8 min | 37.1 s          |
| base, policies = P24          | policies       | P24           | 1s-week    |         24 |     136 |              2449 |   333064 | 3.2 days     | 4.0 h        | 59.4 min | 2.5 min         |
| base, policies = P105         | policies       | P105          | 1m-quarter |        105 |     136 |              2449 |   333064 | not measured | 48.8 min     | 13.1 min | 42.4 s          |
| base, policies = P105         | policies       | P105          | 1s-week    |        105 |     136 |              2449 |   333064 | not measured | 4.0 h        | 56.8 min | 2.6 min         |
| base, coverage = pointwise    | coverage       | pointwise     | 1m-quarter |          1 |     136 |              2449 |   333064 | 48.5 min     | not measured | 12.1 min | 11.1 s          |
| base, coverage = pointwise    | coverage       | pointwise     | 1s-week    |          1 |     136 |              2449 |   333064 | 4.0 h        | not measured | 52.6 min | 42.9 s          |
| base, coverage = simultaneous | coverage       | simultaneous  | 1m-quarter |          1 |     136 |              7975 |  1084600 | 2.6 h        | not measured | 39.3 min | 36.0 s          |
| base, coverage = simultaneous | coverage       | simultaneous  | 1s-week    |          1 |     136 |              7975 |  1084600 | 13.2 h       | not measured | 2.9 h    | 2.3 min         |

## Projection check on the validation task

| method      |   cells |   trials |   actual_s |   projected_s |   error |
|:------------|--------:|---------:|-----------:|--------------:|--------:|
| shared      |       9 |     3618 |    161.537 |       158.067 |  -0.021 |
| tail        |       9 |     3618 |     38.238 |        38.378 |   0.004 |
| incremental |       9 |     3618 |      1.729 |         1.652 |  -0.044 |

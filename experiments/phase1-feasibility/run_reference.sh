#!/usr/bin/env bash
# Phase 1 reference run, in protocol order. Run from anywhere on the reference machine:
#
#   experiments/phase1-feasibility/run_reference.sh <env-label> [--shutdown]
#
# Stops at the first failing gate. --shutdown powers the machine off at the end (also after a failure), so an
# unattended cloud instance does not keep running.
set -euo pipefail

ENV_LABEL=${1:?usage: run_reference.sh <env-label> [--shutdown]}
SHUTDOWN=${2:-}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
if [[ "$SHUTDOWN" == "--shutdown" ]]; then trap 'sudo shutdown -h +1' EXIT; fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "working tree is dirty; commit first so env.json records the exact code" >&2
  exit 1
fi

export OMP_PLACES=cores OMP_PROC_BIND=spread
PY=.venv/bin/python
HERE=experiments/phase1-feasibility

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/sf_test
[[ -x $PY ]] || { python3 -m venv .venv && .venv/bin/pip install -q -r $HERE/requirements.txt; }
[[ -f data/BTCUSDT_1m_2024Q1.bin ]] || $PY cases/finance/fetch_klines.py 1m-2024q1
[[ -f data/BTCUSDT_1s_20240108_14.bin ]] || $PY cases/finance/fetch_klines.py 1s-2024-01-08
$PY $HERE/check_phase0.py

$PY $HERE/run_sweep.py --env "$ENV_LABEL"
$PY $HERE/validate_estimator.py --env "$ENV_LABEL"
RUN_DIR=$(ls -d $HERE/results/"$ENV_LABEL"_* | tail -1)
$PY $HERE/make_figures.py "$RUN_DIR"
$PY $HERE/scenario_space.py --sweep "$RUN_DIR/cpu_sweep.jsonl" --pmap "$RUN_DIR/estimator_cells.csv" \
  --out "$RUN_DIR/tasks_variance_aware.csv" > /dev/null
echo "done: $RUN_DIR"

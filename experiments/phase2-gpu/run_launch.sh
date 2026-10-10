#!/usr/bin/env bash
# One formal launch on a fresh instance (Part III). Run from anywhere in the repository:
#
#   experiments/phase2-gpu/run_launch.sh --stage A|D --role cpu|gpu|size-control --launch I --label LABEL \
#                                        [--upload s3://bucket/prefix] [--shutdown] [--dry-run]
#
# Builds on the instance (-O3 -march=native; CUDA for --role gpu), checks inputs and the committed task files, then
# runs launch.py. --upload copies the launch directory to S3 and verifies the copy before anything is shut down;
# --shutdown powers the machine off at the end, also after a failure.
set -euo pipefail

STAGE="" ROLE="" LAUNCH="" LABEL="" UPLOAD="" SHUTDOWN="" DRY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE=$2; shift 2 ;;
    --role) ROLE=$2; shift 2 ;;
    --launch) LAUNCH=$2; shift 2 ;;
    --label) LABEL=$2; shift 2 ;;
    --upload) UPLOAD=$2; shift 2 ;;
    --shutdown) SHUTDOWN=1; shift ;;
    --dry-run) DRY=--dry-run; shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ -n "$STAGE" && -n "$ROLE" && -n "$LAUNCH" && -n "$LABEL" ]] || { echo "usage: see header" >&2; exit 2; }

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
if [[ -n "$SHUTDOWN" ]]; then trap 'sudo shutdown -h +1' EXIT; fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "working tree is dirty; commit first so env.json records the exact code" >&2
  exit 1
fi

HERE=experiments/phase2-gpu
PY=.venv/bin/python
if [[ "$ROLE" == "gpu" ]]; then
  sudo nvidia-smi -pm 1 >/dev/null || echo "warning: persistence mode could not be enabled (recorded in env.json)" >&2
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSF_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="75-real;80-real;89-real;120-real"
else
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
fi
cmake --build build -j

[[ -x $PY ]] || { python3 -m venv .venv && .venv/bin/pip install -q -r $HERE/requirements.txt; }
[[ -f data/BTCUSDT_1m_2024Q1.bin ]] || $PY cases/finance/fetch_klines.py 1m-2024q1
[[ -f data/BTCUSDT_1s_20240108_14.bin ]] || $PY cases/finance/fetch_klines.py 1s-2024-01-08
$PY $HERE/make_tasks.py --check

$PY $HERE/launch.py --stage "$STAGE" --role "$ROLE" --launch "$LAUNCH" --label "$LABEL" $DRY

if [[ -n "$UPLOAD" ]]; then
  if [[ "$ROLE" == "size-control" ]]; then NAME=size-control; else NAME=launch-$LAUNCH-$ROLE; fi
  DIR=$HERE/results/formal/$LABEL/$NAME
  aws s3 cp --recursive "$DIR" "$UPLOAD/$LABEL/$NAME/"
  REMOTE=$(aws s3 ls --recursive "$UPLOAD/$LABEL/$NAME/" | wc -l)
  LOCAL=$(find "$DIR" -type f | wc -l)
  [[ "$REMOTE" -ge "$LOCAL" ]] || { echo "upload incomplete ($REMOTE of $LOCAL files)" >&2; exit 1; }
fi
echo "done: $HERE/results/formal/$LABEL"

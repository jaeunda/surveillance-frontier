#!/usr/bin/env bash
# One formal launch on a fresh instance (Part III). Run from anywhere in the repository:
#
#   experiments/phase2-gpu/run_launch.sh --stage A|D --role cpu|gpu|size-control --launch I --label LABEL \
#                                        [--spent-usd X] [--upload s3://bucket/prefix] [--terminate] [--dry-run]
#
# Builds on the instance (-O3 -march=native; CUDA for --role gpu), installs the locked Python packages, checks inputs
# and the committed task files, then runs launch.py. On exit, also after a failure or a stopped launch:
#   --upload     copies the slot directory (every attempt) to S3, downloads it again and verifies every file against
#                a local SHA-256 list; a failed verification keeps the instance running (nothing is lost)
#   --terminate  terminates this instance through the EC2 API (aws ec2 terminate-instances), and only after a
#                verified upload; an OS shutdown would only stop an EBS-backed instance and keep its storage billed
# Launch the instance with --instance-initiated-shutdown-behavior terminate as a second safeguard.
set -euo pipefail

STAGE="" ROLE="cpu" LAUNCH="" LABEL="" UPLOAD="" TERMINATE="" DRY="" SPENT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE=$2; shift 2 ;;
    --role) ROLE=$2; shift 2 ;;
    --launch) LAUNCH=$2; shift 2 ;;
    --label) LABEL=$2; shift 2 ;;
    --spent-usd) SPENT=$2; shift 2 ;;
    --upload) UPLOAD=$2; shift 2 ;;
    --terminate) TERMINATE=1; shift ;;
    --dry-run) DRY=--dry-run; shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ -n "$STAGE" && -n "$LAUNCH" && -n "$LABEL" ]] || { echo "usage: see header" >&2; exit 2; }
if [[ -n "$TERMINATE" && -z "$UPLOAD" ]]; then echo "--terminate needs --upload (results first)" >&2; exit 2; fi

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
HERE=experiments/phase2-gpu
if [[ "$STAGE" == "A" ]]; then REL=stage-a/launch-$LAUNCH
elif [[ "$ROLE" == "size-control" ]]; then REL=stage-d/size-control
else REL=stage-d/pair-$LAUNCH/$ROLE; fi
DIR=$HERE/results/formal/$LABEL/$REL

finish() {
  local code=$?
  set +e
  if [[ -n "$UPLOAD" && -d "$DIR" ]]; then
    local dest="$UPLOAD/$LABEL/$REL/" check
    check=$(mktemp -d)
    (cd "$DIR" && find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$check/local.sha256"
    aws s3 cp --recursive --only-show-errors "$DIR" "$dest" &&
      aws s3 cp --recursive --only-show-errors "$dest" "$check/copy" &&
      (cd "$check/copy" && sha256sum --quiet -c "$check/local.sha256")
    if [[ $? -ne 0 ]]; then
      echo "upload NOT verified: the instance is kept running; copy $DIR by hand" >&2
      exit 1
    fi
    echo "upload verified: $(wc -l < "$check/local.sha256") files at $dest"
  fi
  if [[ -n "$TERMINATE" ]]; then
    local tok id region
    tok=$(curl -s -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 300")
    id=$(curl -s -H "X-aws-ec2-metadata-token: $tok" http://169.254.169.254/latest/meta-data/instance-id)
    region=$(curl -s -H "X-aws-ec2-metadata-token: $tok" http://169.254.169.254/latest/meta-data/placement/region)
    echo "terminating $id ($region); afterwards: collect_env.py --ec2-times $id --out <attempt>/ec2.json"
    aws ec2 terminate-instances --region "$region" --instance-ids "$id" >/dev/null || sudo shutdown -h now
  fi
  exit $code
}
trap finish EXIT

if [[ -n "$(git status --porcelain)" ]]; then
  echo "working tree is dirty; commit first so env.json records the exact code" >&2
  exit 1
fi

PY=.venv/bin/python
if [[ "$ROLE" == "gpu" ]]; then
  sudo nvidia-smi -pm 1 >/dev/null || echo "warning: persistence mode could not be enabled (recorded in env.json)" >&2
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSF_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="75-real;80-real;89-real;120-real"
else
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
fi
cmake --build build -j

[[ -x $PY ]] || python3 -m venv .venv
.venv/bin/pip install -q -r $HERE/requirements.txt
[[ -f data/BTCUSDT_1m_2024Q1.bin ]] || $PY cases/finance/fetch_klines.py 1m-2024q1
[[ -f data/BTCUSDT_1s_20240108_14.bin ]] || $PY cases/finance/fetch_klines.py 1s-2024-01-08
$PY $HERE/make_tasks.py --check

$PY $HERE/launch.py --stage "$STAGE" --role "$ROLE" --launch "$LAUNCH" --label "$LABEL" \
  ${SPENT:+--spent-usd "$SPENT"} $DRY
echo "done: $DIR"

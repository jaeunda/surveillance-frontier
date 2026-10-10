"""Machine manifest and AWS records (p2/machine.py), for use outside a launch.

    python experiments/phase2-gpu/collect_env.py --profile lab-desktop [--out env.json]
    python experiments/phase2-gpu/collect_env.py --price c7i.16xlarge [--out price.json]   official on-demand price
    python experiments/phase2-gpu/collect_env.py --ec2-times INSTANCE_ID --out .../ec2.json  after termination
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from p2 import machine  # noqa: E402
from p2.common import ROOT, write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="lab-desktop")
    ap.add_argument("--build", type=Path, default=ROOT / "build")
    ap.add_argument("--price")
    ap.add_argument("--ec2-times")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.price:
        rec = machine.on_demand_price(a.price)
    elif a.ec2_times:
        rec = machine.ec2_times(a.ec2_times)
    else:
        rec = machine.environment(a.profile, a.build)
    if a.out:
        write_json(a.out, rec)
    else:
        print(json.dumps(rec, indent=1, default=str))


if __name__ == "__main__":
    main()

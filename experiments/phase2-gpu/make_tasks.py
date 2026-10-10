"""Generate the Phase 2 task files (SPEC §2) from protocol.json into tasks/ (layout: p2/tasks.py).

    python experiments/phase2-gpu/make_tasks.py [--check]   # --check: exit 1 if the committed files would change
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from p2.common import TASKS  # noqa: E402
from p2.tasks import generate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    files = generate()
    changed = []
    for rel, task in sorted(files.items()):
        path = TASKS / rel
        text = json.dumps(task, indent=1) + "\n"
        if not path.exists() or path.read_text() != text:
            changed.append(rel)
            if not a.check:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
    stale = sorted(str(p.relative_to(TASKS)) for p in TASKS.rglob("*.json") if str(p.relative_to(TASKS)) not in files)
    if stale and not a.check:
        for rel in stale:
            (TASKS / rel).unlink()
    if any(not t["series"]["sha256"] for t in files.values()):
        print("warning: task files without input SHA-256 (data/ not present)", file=sys.stderr)
    print(f"{len(files)} task files; {len(changed)} {'differ' if a.check else 'written'}; "
          f"{len(stale)} stale {'present' if a.check else 'removed'}")
    sys.exit(1 if a.check and (changed or stale) else 0)


if __name__ == "__main__":
    main()

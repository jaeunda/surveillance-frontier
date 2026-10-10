"""Trial stream sf-stream-v1 (SPEC §3), written independently of engine/include/sf/stream.hpp.

Used to (1) generate stream_vectors.json, the expected rejection sequence the gate (sf_gate item 8) compares the
C++/CUDA stream with, and (2) re-derive trial starts in analyze.py (H4: every live run's (N_c, k_c) must equal what the
reference bits give for the same trial ids). Pure Python integers for the reference functions; a vectorised NumPy
path for many draws, checked against them.

    python experiments/phase2-gpu/stream.py --write   # regenerate stream_vectors.json
    python experiments/phase2-gpu/stream.py --check   # self-test: known answers, NumPy == pure Python
"""

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEED = 20261010
MASK32 = 0xFFFFFFFF
MASK64 = (1 << 64) - 1
M0, M1 = 0xD2511F53, 0xCD9E8D57
W0, W1 = 0x9E3779B9, 0xBB67AE85
TASKS = {"T1-curve": 1, "T2-policy-map": 2, "T-ref-est": 3, "T-diag": 4, "gate": 5}
PURPOSES = {"measured": 0, "warmup": 1, "spot": 2, "oracle": 3, "calibration": 4, "gate": 5}


def philox4x32_10(ctr, key):
    """Random123 Philox4x32 with 10 rounds on Python ints."""
    c0, c1, c2, c3 = ctr
    k0, k1 = key
    for r in range(10):
        if r:
            k0, k1 = (k0 + W0) & MASK32, (k1 + W1) & MASK32
        p0, p1 = M0 * c0, M1 * c2
        c0, c1, c2, c3 = ((p1 >> 32) ^ c1 ^ k0) & MASK32, p1 & MASK32, ((p0 >> 32) ^ c3 ^ k1) & MASK32, p0 & MASK32
    return c0, c1, c2, c3


def counter(task, series, cell, replicate, purpose, j, attempt):
    return (j & MASK32, (j >> 32) & MASK32, (task & 0xFF) << 24 | (series & 0xF) << 20 | (cell & 0xFFFFF),
            (purpose & 0xF) << 28 | (replicate & 0xFFFFF) << 8 | (attempt & 0xFF))


def draw(task, series, cell, replicate, purpose, j, M):
    """(start, rejected candidates) of trial j: uniform in [0, M) by Lemire's method."""
    key = (SEED & MASK32, SEED >> 32)
    t = ((1 << 64) - M) % M
    rejected = 0
    for attempt in range(256):
        x = philox4x32_10(counter(task, series, cell, replicate, purpose, j, attempt), key)
        for u in (x[0] | x[1] << 32, x[2] | x[3] << 32):
            m = u * M
            lo = m & MASK64
            if lo < M and lo < t:
                rejected += 1
                continue
            return m >> 64, rejected
    raise RuntimeError("no accepted draw within 256 attempts")


# ---- vectorised path (NumPy uint64 arithmetic) ----

def _mulhilo32(a, b):
    p = a.astype(np.uint64) * np.uint64(b)
    return (p >> np.uint64(32)).astype(np.uint64), (p & np.uint64(MASK32)).astype(np.uint64)


def philox_np(c0, c1, c2, c3):
    k0, k1 = np.uint64(SEED & MASK32), np.uint64(SEED >> 32)
    m32 = np.uint64(MASK32)
    for r in range(10):
        if r:
            k0, k1 = (k0 + np.uint64(W0)) & m32, (k1 + np.uint64(W1)) & m32
        h0, l0 = _mulhilo32(c0, M0)
        h1, l1 = _mulhilo32(c2, M1)
        c0, c1, c2, c3 = (h1 ^ c1 ^ k0) & m32, l1, (h0 ^ c3 ^ k1) & m32, l0
    return c0, c1, c2, c3


def _mul64(u, M):
    """High and low 64 bits of u * M for uint64 arrays u and a Python int M < 2^63."""
    m_lo, m_hi = np.uint64(M & MASK32), np.uint64(M >> 32)
    u_lo, u_hi = u & np.uint64(MASK32), u >> np.uint64(32)
    ll, lh, hl, hh = u_lo * m_lo, u_lo * m_hi, u_hi * m_lo, u_hi * m_hi
    mid = (ll >> np.uint64(32)) + (lh & np.uint64(MASK32)) + (hl & np.uint64(MASK32))
    lo = (ll & np.uint64(MASK32)) | (mid << np.uint64(32))
    hi = hh + (lh >> np.uint64(32)) + (hl >> np.uint64(32)) + (mid >> np.uint64(32))
    return hi, lo


def starts_np(task, series, cell, replicate, purpose, js, M):
    """Starts of trials js (int array) of one cell. Falls back to the pure-Python draw for rejected ones."""
    js = np.asarray(js, dtype=np.uint64)
    c2 = np.uint64((task & 0xFF) << 24 | (series & 0xF) << 20 | (cell & 0xFFFFF))
    c3 = np.uint64((purpose & 0xF) << 28 | (replicate & 0xFFFFF) << 8)  # attempt 0
    x0, x1, _, _ = philox_np(js & np.uint64(MASK32), js >> np.uint64(32), np.full_like(js, c2), np.full_like(js, c3))
    u = x0 | (x1 << np.uint64(32))
    hi, lo = _mul64(u, M)
    t = ((1 << 64) - M) % M
    rej = (lo < np.uint64(M)) & (lo < np.uint64(t))
    out = hi.astype(np.int64)
    for i in np.nonzero(rej)[0]:
        out[i] = draw(task, series, cell, replicate, purpose, int(js[i]), M)[0]
    return out


KAT = [((0, 0, 0, 0), (0, 0), (0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8)),
       ((MASK32,) * 4, (MASK32, MASK32), (0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd)),
       ((0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344), (0xa4093822, 0x299f31d0),
        (0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1))]

REJECTION_KEY = {"task": 5, "series": 0, "cell": 7, "replicate": 3, "purpose": 5}
REJECTION_M = (1 << 62) + 1  # t = 2^62 - 3: about a quarter of the candidates are rejected


def vectors(n=4096):
    k = REJECTION_KEY
    starts, rejected = [], 0
    for j in range(n):
        s, r = draw(k["task"], k["series"], k["cell"], k["replicate"], k["purpose"], j, REJECTION_M)
        starts.append(str(s))
        rejected += r
    return {"_about": "Generated by stream.py (independent of the C++ stream). sf_gate item 8 compares the engine's "
                      "draws with these. Starts are strings (they exceed 2^53).",
            "stream": "sf-stream-v1", "seed": SEED,
            "kat": [{"counter": list(c), "key": list(key), "out": list(o)} for c, key, o in KAT],
            "rejection": {"key": k, "M": REJECTION_M, "rejected": rejected, "starts": starts}}


def self_check():
    for c, key, want in KAT:
        assert philox4x32_10(c, key) == want, "Philox known answer"
    rng = np.random.default_rng(1)
    for M in (604793, 131039, REJECTION_M):
        js = rng.integers(0, 1 << 40, size=2000)
        got = starts_np(2, 1, 17, 0, 0, js, M)
        want = [draw(2, 1, 17, 0, 0, int(j), M)[0] for j in js]
        assert list(got) == want, f"NumPy path differs from pure Python at M = {M}"
    print("stream.py: known answers and NumPy == pure Python: ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check or not args.write:
        self_check()
    if args.write:
        (HERE / "stream_vectors.json").write_text(json.dumps(vectors(), indent=1) + "\n")
        print(f"wrote {HERE / 'stream_vectors.json'}")


if __name__ == "__main__":
    main()

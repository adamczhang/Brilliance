#!/usr/bin/env python3
"""calib_st2gram.py -- turn the Hessian files of a `--calib-dir` run into what a recipe's `calib=` reads.

A plugin that records with CalibFull (arch/common/rad_calib_full.h) writes one safetensors file a
distinct linear input, `<DIR>/<site>.safetensors`, holding `hessian_upper` (f32, the packed upper
triangle of the raw sum H = x^T x) and metadata `k`, `rows`, `linears` (the declared weight stems
that read this input, comma-separated) and, for an int8 trunk, `domain`. libquant reads none of
that: a recipe's `gptq calib=DIR` looks for `gram.r<rank>.<the weight's own stem>.bin` (RADGRAM2: a
32-byte header `RADGRAM2`, u32 k, u32 rank, u64 rows, u32 domain, a pad word, then k*k f32 row-major,
libquant/lq_hess.cpp). This writes that file for every stem in `linears` (one copy, hard-linked
under the other names), into DIR itself unless -o names another directory. Files that are already
RADGRAM2 (the routed experts' Grams the MoE block writes itself) are left as they are.

  tools/calib_st2gram.py DIR [-o OUT] [--check]

--check converts nothing and lists, for the weight stems given with --weights FILE (one stem a
line, as `rad-info --weights` or a recipe's plan prints them without `.weight`), those without a
Gram. Needs numpy only.
"""
import argparse, json, os, struct, sys
import numpy as np


def read_site(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        meta = hdr.get("__metadata__", {})
        t = hdr["hessian_upper"]
        a, b = t["data_offsets"]
        f.seek(8 + n + a)
        tri = np.frombuffer(f.read(b - a), dtype="<f4")
    k = int(meta["k"])
    if tri.size != k * (k + 1) // 2:
        raise ValueError(f"{path}: {tri.size} entries for k={k}")
    return k, int(meta.get("rows", 0)), int(meta.get("domain", 0)), \
        [s for s in meta.get("linears", "").split(",") if s], tri


def full_matrix(tri, k):
    """The symmetric k x k matrix of a packed upper triangle (row i holds columns i..k-1)."""
    m = np.empty((k, k), dtype="<f4")
    off = 0
    for i in range(k):
        m[i, i:] = tri[off:off + k - i]
        off += k - i
    for i0 in range(0, k, 512):          # mirror, a block of columns at a time
        i1 = min(i0 + 512, k)
        for i in range(i0, i1):
            m[i + 1:, i] = m[i, i + 1:]
    return m


def write_gram(path, m, rows, domain):
    k = m.shape[0]
    with open(path, "wb") as f:
        f.write(b"RADGRAM2" + struct.pack("<IIQII", k, 0, rows, domain, 0))
        m.tofile(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("-o", "--out", help="where the .bin files go (default DIR)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--weights", help="with --check: weight stems, one a line")
    a = ap.parse_args()
    out = a.out or a.dir
    os.makedirs(out, exist_ok=True)
    have, n_sites = set(), 0
    for name in sorted(os.listdir(a.dir)):
        p = os.path.join(a.dir, name)
        if name.startswith("gram.r0.") and name.endswith(".bin"):
            have.add(name[len("gram.r0."):-len(".bin")])
        if not name.endswith(".safetensors"):
            continue
        k, rows, domain, linears, tri = read_site(p)
        if not linears:
            print(f"{name}: no `linears` in its metadata, skipped", file=sys.stderr)
            continue
        n_sites += 1
        have.update(linears)
        if a.check:
            continue
        first = None
        for stem in linears:
            dst = os.path.join(out, f"gram.r0.{stem}.bin")
            if os.path.lexists(dst):
                os.unlink(dst)
            if first is None:
                write_gram(dst, full_matrix(tri, k), rows, domain)
                first = dst
            else:
                os.link(first, dst)
        print(f"{name}: k={k} rows={rows} domain={domain} -> {len(linears)} Gram(s)")
    if a.weights:
        miss = [s.strip() for s in open(a.weights) if s.strip() and s.strip() not in have]
        for s in miss:
            print(f"no Gram for {s}")
        print(f"{n_sites} site file(s); {len(miss)} weight(s) without a Gram")
        return 1 if miss else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())

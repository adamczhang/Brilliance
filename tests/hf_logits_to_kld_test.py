#!/usr/bin/env python3
"""tools/hf_logits_to_kld.py writes what core/kld.h documents: file sizes, the manifest, log-probabilities, the head
(descending, a tie keeping the lower id), rows.f32 (NLL, entropy, tail mass)."""
import json, os, subprocess, sys, tempfile
try:
    import numpy as np
except ImportError:   # Brilliance: the tool needs numpy, which a Windows Python may not have
    print("hf_logits_to_kld_test: skipped (numpy is not installed)")
    sys.exit(77)

TOOL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "hf_logits_to_kld.py")


def lse(z):
    m = z.max()
    return m + np.log(np.exp(z - m).sum())


def main():
    rng = np.random.default_rng(7)
    V = 50
    docs = [(rng.integers(0, V, 9), 0), (rng.integers(0, V, 6), 2)]
    base = os.environ.get("RAD_TEST_TMPDIR") or None   # Brilliance: scratch inside the build tree
    if base:
        os.makedirs(base, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base) as t:
        lines = []
        for i, (ids, sf) in enumerate(docs):
            L = (rng.normal(size=(len(ids), V)) * 3).astype(np.float32)
            L[1, 4] = L[1, 9] = L[1].max() + 1.0               # a tie at the top: the lower id first
            np.save(f"{t}/doc{i}.npy", L)
            lines.append(json.dumps({"ids": ids.tolist(), "source": f"d{i}", "score_from": sf}))
        open(f"{t}/c.jsonl", "w").write("\n".join(lines) + "\n")
        for top in (0, 6):
            out = f"{t}/ref{top}"
            subprocess.run([sys.executable, "-I", TOOL, out, "--docs", f"{t}/c.jsonl", "--logits-dir", t, "--top", str(top)],
                           check=True, stdout=subprocess.DEVNULL)
            m = json.load(open(f"{out}/kld.json"))
            rows = sum(len(ids) - 1 - sf for ids, sf in docs)
            k = top or 16
            assert m["format"] == ("radiance-kld-top-1" if top else "radiance-kld-1") and m["n_vocab"] == V and m["rows"] == rows, m
            assert [d["row0"] for d in m["docs"]] == [0, 8] and m["docs"][1]["score_from"] == 2
            rw = 3 if top else 2
            assert os.path.getsize(f"{out}/tokens.i32") == 4 * 15
            assert os.path.getsize(f"{out}/top.bin") == rows * k * 8
            assert os.path.getsize(f"{out}/rows.f32") == rows * rw * 4
            assert os.path.exists(f"{out}/logp.f16") == (not top)
            if not top:
                assert os.path.getsize(f"{out}/logp.f16") == rows * V * 2
            tb = np.fromfile(f"{out}/top.bin", np.dtype([("id", "<i4"), ("lp", "<f4")])).reshape(rows, k)
            rf = np.fromfile(f"{out}/rows.f32", "<f4").reshape(rows, rw)
            r = 0
            for i, (ids, sf) in enumerate(docs):
                L = np.load(f"{t}/doc{i}.npy").astype(np.float64)
                for p in range(sf, len(ids) - 1):
                    lp = L[p] - lse(L[p])
                    order = np.lexsort((np.arange(V), -lp))[:k]
                    assert (tb["id"][r] == order).all(), (r, tb["id"][r], order)
                    assert np.allclose(tb["lp"][r], lp[order], atol=1e-5)
                    assert abs(rf[r, 0] - (-lp[ids[p + 1]])) < 1e-4
                    assert abs(rf[r, 1] - (-(np.exp(lp) * lp).sum())) < 1e-4
                    if top:
                        assert abs(rf[r, 2] - (1 - np.exp(lp[order]).sum())) < 1e-5
                    else:
                        h = np.fromfile(f"{out}/logp.f16", "<f2", count=V, offset=r * V * 2).astype(np.float64)
                        assert np.abs(h - lp).max() < 5e-3
                    r += 1
        # never overwritten
        r = subprocess.run([sys.executable, "-I", TOOL, f"{t}/ref0", "--docs", f"{t}/c.jsonl", "--logits-dir", t],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        assert r.returncode != 0
        # too narrow a vocabulary is refused
        r = subprocess.run([sys.executable, "-I", TOOL, f"{t}/refw", "--docs", f"{t}/c.jsonl", "--logits-dir", t, "--n-vocab", "60"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        assert r.returncode != 0
    print("hf_logits_to_kld_test: ok")


if __name__ == "__main__":
    main()

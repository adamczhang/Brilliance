#!/usr/bin/env python3
"""Turn HF logits for a token corpus into a radiance KL reference directory (`--kld-ref DIR`).

radiance's own `--kld-record` writes the same directory from a radiance run (core/kld.h). This tool
writes it from logits computed elsewhere, e.g. the transformers port of Hy4-flash (PR 50903, c2ec8505)
in bf16, so a quantised container is measured against the HF reference:

    python3 tools/hf_logits_to_kld.py OUT_DIR --docs corpus-ids.jsonl --logits-dir hf-logits/ [--top N]
    radiance --model hy4-q.rad --kld-ref OUT_DIR --kld-out kl.json     # (yaml keys: kld-ref / kld-out)

Inputs
  --docs FILE      JSONL, one document a line: {"ids": [token ids], "source": "name", "score_from": 0,
                   "logits": "path"} ("source", "score_from" and "logits" optional). The ids are the
                   HF tokenisation; radiance scores exactly these ids (it reads tokens.i32 from the
                   reference), so no tokeniser parity is needed. `logits` defaults to
                   LOGITS_DIR/doc<i>.npy (i from 0). The ids must include whatever the HF run saw
                   (BOS, chat markup).
  --npz FILE...    instead of --docs: ref/ harness dumps (`input_ids` and `logits` arrays), one document
                   a file, source = the file's stem.
  logits file      .npy (or .npz with a `logits` array): [n, V] or [1, n, V], f32 / f16 / bf16 stored as
                   f32 (numpy has no bf16: save `logits.float().numpy()`). Row p are the logits AFTER
                   token p, i.e. predicting token p + 1, exactly what an HF forward returns.
  --hf-model DIR   a document without a logits file is run through transformers on the CPU (needs torch
                   and the PR's transformers on the path); `--hf-dtype bfloat16|float32`. Slow on the
                   real model: compute the logits on the GPU box and pass files instead.

Document d scores positions [score_from, n - 1) (core/kld.h): row r of the reference is the log-softmax
of logits[score_from + r]. Output files are byte-compatible with radiance-kld-1 (f16 log-probabilities,
the 16 most likely tokens exactly, rows.f32 = next-token NLL and entropy) or, with `--top N`,
radiance-kld-top-1 (the N most likely tokens exactly plus the tail mass; no logp.f16).

--n-vocab V must equal the model's logits width in radiance (Hy4-flash: 120832 summed over ranks); the
default is the logits' width. A narrower HF width is refused, a wider one is cut with --crop-vocab.
"""
import argparse, json, os, sys, time, zlib  # noqa: E401
import numpy as np

TOP_FULL = 16          # kKldTop
TOP_MAX = 8192         # kKldTopMax
FORMAT, FORMAT_TOP = "radiance-kld-1", "radiance-kld-top-1"
TOP_DTYPE = np.dtype([("id", "<i4"), ("lp", "<f4")])


def fnv1a64(data: bytes) -> int:
    h = 1469598103934665603
    for b in data:
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def load_logits(path):
    if path.endswith(".npz"):
        z = np.load(path)
        a = z["logits"]
    else:
        a = np.load(path, mmap_mode="r")
    if a.ndim == 3:
        if a.shape[0] != 1:
            sys.exit(f"{path}: batch of {a.shape[0]}; one document a file")
        a = a[0]
    if a.ndim != 2:
        sys.exit(f"{path}: logits must be [n, V], got {a.shape}")
    return a


def hf_logits(model_dir, ids, dtype):
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=getattr(torch, dtype))
    m.eval()
    with torch.no_grad():
        out = m(input_ids=torch.tensor([ids]), use_cache=False).logits
    return out[0].float().numpy()


def head(lp, k):
    """The k largest of a row, descending, a tie keeping the lower id (radiance's argmax rule)."""
    idx = np.argpartition(-lp, k - 1)[:k]
    # sort by (-lp, id)
    order = np.lexsort((idx, -lp[idx]))
    idx = idx[order]
    # a tie at the cut must also keep the lower ids: take every token >= the k-th value, then cut
    thr = lp[idx[-1]]
    ties = np.nonzero(lp >= thr)[0]
    if len(ties) > k:
        ties = ties[np.lexsort((ties, -lp[ties]))][:k]
        idx = ties
    return idx


def convert(out, docs, n_vocab, top, model, corpus_name, corpus_hash, crop):
    sparse = top > 0
    k = top if sparse else TOP_FULL
    if k > n_vocab:
        sys.exit(f"--top {k} exceeds the vocabulary {n_vocab}")
    plan, n_rows = [], 0
    for d in docs:
        n = len(d["ids"])
        sf = d.get("score_from", 0)
        rows = n - 1 - sf
        if sf < 0 or rows < 1:
            sys.exit(f"document {d.get('source')!r}: {n} tokens leave nothing to score from {sf}")
        plan.append((d, n, sf, n_rows))
        n_rows += rows
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, "kld.json")):
        sys.exit(f"{out} already holds a reference; never overwritten")
    rw = 3 if sparse else 2
    logp_f = None if sparse else open(os.path.join(out, "logp.f16"), "wb")
    top_f = open(os.path.join(out, "top.bin"), "wb")
    rows_f = open(os.path.join(out, "rows.f32"), "wb")
    tok_f = open(os.path.join(out, "tokens.i32"), "wb")
    manifest_docs = []
    t0 = time.time()
    for d, n, sf, row0 in plan:
        ids = np.asarray(d["ids"], dtype="<i4")
        tok_f.write(ids.tobytes())
        L = d["_logits"]
        if L.shape[0] != n:
            sys.exit(f"document {d.get('source')!r}: {n} ids but {L.shape[0]} logit rows")
        if L.shape[1] < n_vocab:
            sys.exit(f"document {d.get('source')!r}: logits are {L.shape[1]} wide, --n-vocab is {n_vocab}")
        if L.shape[1] > n_vocab and not crop:
            sys.exit(f"logits are {L.shape[1]} wide, radiance's vocabulary {n_vocab}: --crop-vocab cuts the padding")
        for p in range(sf, n - 1):
            z = np.asarray(L[p, :n_vocab], dtype=np.float64)
            m = z.max()
            lse = m + np.log(np.exp(z - m).sum())
            lp = z - lse
            lp32 = lp.astype("<f4")
            pr = np.exp(lp)
            ent = float(-(pr[pr > 0] * lp[pr > 0]).sum())
            nll = float(lse - z[ids[p + 1]])
            idx = head(lp32, k)
            ent_rec = np.zeros(k, TOP_DTYPE)
            ent_rec["id"], ent_rec["lp"] = idx, lp32[idx]
            if sparse:
                mask = np.ones(n_vocab, bool)
                mask[idx] = False
                tail = float(pr[mask].sum())
                rows_f.write(np.array([nll, ent, tail], "<f4").tobytes())
            else:
                logp_f.write(lp32.astype("<f2").tobytes())
                rows_f.write(np.array([nll, ent], "<f4").tobytes())
            top_f.write(ent_rec.tobytes())
        manifest_docs.append({"source": d.get("source", ""), "tokens": n, "score_from": sf, "row0": row0})
    for f in (logp_f, top_f, rows_f, tok_f):
        if f:
            f.flush(); os.fsync(f.fileno()); f.close()
    man = {"format": FORMAT_TOP if sparse else FORMAT, "model": model, "corpus": corpus_name,
           "corpus_fnv1a64": "%016x" % corpus_hash, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "n_vocab": n_vocab, "top": k, "rows": n_rows, "docs": manifest_docs}
    tmp = os.path.join(out, "kld.json.tmp")
    with open(tmp, "w") as f:
        json.dump(man, f, indent=1); f.write("\n")
    os.rename(tmp, os.path.join(out, "kld.json"))
    print(f"wrote {n_rows} positions of {len(docs)} documents to {out} ({'top-%d' % k if sparse else 'full'}) in {time.time() - t0:.1f} s")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--docs")
    ap.add_argument("--npz", nargs="+")
    ap.add_argument("--logits-dir")
    ap.add_argument("--hf-model")
    ap.add_argument("--hf-dtype", default="bfloat16")
    ap.add_argument("--top", type=int, default=0, help="sparse form: keep the N most likely tokens (default: full f16 rows)")
    ap.add_argument("--n-vocab", type=int, default=0)
    ap.add_argument("--crop-vocab", action="store_true")
    ap.add_argument("--model", default="hf-bf16", help="name stored in the manifest")
    a = ap.parse_args(argv)
    if bool(a.docs) == bool(a.npz):
        ap.error("give --docs or --npz")
    if not 0 <= a.top <= TOP_MAX:
        ap.error(f"--top must be 0..{TOP_MAX}")
    docs = []
    if a.npz:
        h = 1469598103934665603
        for f in a.npz:
            z = np.load(f)
            L = z["logits"]
            L = L[0] if L.ndim == 3 else L
            ids = np.asarray(z["input_ids"]).reshape(-1).tolist()
            docs.append({"ids": ids, "source": os.path.splitext(os.path.basename(f))[0], "_logits": L})
        corpus_name, corpus_hash = "npz:" + ",".join(a.npz), fnv1a64(json.dumps([d["ids"] for d in docs]).encode())
    else:
        raw = open(a.docs, "rb").read()
        corpus_name, corpus_hash = a.docs, fnv1a64(raw)
        for i, line in enumerate(raw.decode().splitlines()):
            if not line.strip():
                continue
            d = json.loads(line)
            if "ids" not in d:
                sys.exit(f"{a.docs}: line {i + 1} has no \"ids\"")
            d.setdefault("source", "")
            path = d.get("logits") or (os.path.join(a.logits_dir, f"doc{len(docs)}.npy") if a.logits_dir else None)
            if path and os.path.exists(path):
                d["_logits"] = load_logits(path)
            elif a.hf_model:
                d["_logits"] = hf_logits(a.hf_model, d["ids"], a.hf_dtype)
            else:
                sys.exit(f"{a.docs}: document {len(docs)} has no logits file ({path}); --logits-dir or --hf-model")
            docs.append(d)
    if not docs:
        sys.exit("no documents")
    width = docs[0]["_logits"].shape[1]
    convert(a.out, docs, a.n_vocab or width, a.top, a.model, corpus_name, corpus_hash, a.crop_vocab)


if __name__ == "__main__":
    main()

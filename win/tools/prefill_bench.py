"""prefill-bench: prefill speed on fixed, fresh text, with the spread over repeats.

Brilliance (counters-and-prefill). BetterBench's prefill points at 32K and below moved 6-11% between
two warm runs on one server (2026-11-20 tp2-baseline-a2), too noisy to screen a 2-7% change. This
sends the same text at every size and repeat, so a repeat differs from the last only in the run:

  * fixed: one body of English-like prose, generated from a seed by this file (no corpus, no
    download), cut to each size at a token boundary through the server's /tokenize;
  * fresh: every request starts with a line naming the run, the size and the repeat, so the prefix
    cache never holds its first block and the whole prompt is prefilled (timings.cache_n is checked
    to be 0). The n-gram rows the body touches are the same each time, so after the first pass at a
    size they are in the page cache, as on a server warmed by --ngram-warm.

The speed is the server's: timings.prompt_n / timings.prompt_ms (admission to first token), with
the client's wall time beside it. Requests run one at a time, sizes interleaved (8K, 32K, 64K, 128K,
8K, ...) so a drift spreads over every size, after one discarded warm-up pass. With --stats it
reads /stats around every request and records link.ar_bytes, ar_calls, h2d_bytes and the expert
stager's staged_bytes (rank 0).

  python -I prefill_bench.py --port 8000 --sizes 32K,64K,128K --repeats 8 --out run.json

Any running server, any model with a tokenizer; nothing about the model is assumed. It is a
measurement: run it alone on the machine (run class M). Exit 0, 1 on a failed request, 2 on usage.
"""
import argparse
import http.client
import json
import os
import statistics
import sys
import time
import uuid

WORDS = (
    "the of and to in is was for that with as on by at from his her their its which this be are were "
    "have has had not but or an one all would there been will more when who what about if can into "
    "time year people way day man woman child world life hand part place case week company system "
    "program question work government number night point home water room mother area money story "
    "fact month lot right study book eye job word business issue side kind head house service friend "
    "father power hour game line end member law car city community name president team minute idea "
    "kid body information back parent face others level office door health person art war history "
    "party result change morning reason research girl guy moment air teacher force education foot boy "
    "age policy process music market sense nation plan college interest death experience effect use "
    "class control care field development role effort rate heart drug show leader light voice wife "
    "police mind price report decision son view relationship town road arm difference value building "
    "action model season society tax director position player record paper space ground form event "
    "official matter center couple site project activity star table need court oil situation cost "
    "industry figure street image phone data picture practice piece land product doctor wall patient "
    "worker news test movie north love support technology step baby computer type attention film tree "
    "source organization hair window evidence population site river garden bridge harbor valley "
    "mountain island forest engine signal circuit memory bandwidth kernel buffer thread lattice "
    "measured carried built opened found kept turned moved gave held brought began seemed showed "
    "heard played ran wrote stood lost paid met included continued set learned changed led understood "
    "watched followed stopped created spoke read allowed added spent grew offered remembered loved "
    "considered appeared bought waited served died sent expected stayed fell cut reached killed "
    "remained suggested raised passed sold required reported decided pulled quiet early late public "
    "simple careful narrow broad bright heavy gentle rapid ancient modern local central western "
    "northern several various certain recent whole strong possible special clear different large "
    "small great little old young long short high low early important national free real best "
    "better true full major social political economic available likely hard final main"
).split()

KB = 1024


def size_of(s):
    s = s.strip().upper()
    if s.endswith("K"):
        return int(float(s[:-1]) * KB)
    return int(s)


class Rng:
    """xorshift64*: the same stream in every Python, so the body is the same text everywhere."""

    def __init__(self, seed):
        self.s = (seed * 0x9E3779B97F4A7C15 + 0x1234567) & 0xFFFFFFFFFFFFFFFF or 1

    def next(self):
        x = self.s
        x ^= (x >> 12)
        x ^= (x << 25) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 27)
        self.s = x
        return (x * 0x2545F4914F6CDD1D) & 0xFFFFFFFFFFFFFFFF

    def below(self, n):
        return self.next() % n


def make_body(seed, chars):
    """Paragraphs of 4-8 sentences of 6-20 words, a number now and then; about `chars` long."""
    r = Rng(seed)
    out, n = [], 0
    while n < chars:
        para = []
        for _ in range(4 + r.below(5)):
            k = 6 + r.below(15)
            w = [WORDS[r.below(len(WORDS))] for _ in range(k)]
            if r.below(5) == 0:
                w.insert(r.below(k), str(1 + r.below(2000)))
            if k > 10 and r.below(3) == 0:
                w[r.below(k - 2) + 1] += ","
            w[0] = w[0].capitalize()
            para.append(" ".join(w) + ".")
        p = " ".join(para)
        out.append(p)
        n += len(p) + 2
    return "\n\n".join(out)


class Client:
    def __init__(self, host, port, timeout):
        self.host, self.port, self.timeout = host, port, timeout

    def call(self, method, path, body=None):
        c = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        try:
            data = json.dumps(body).encode() if body is not None else None
            hdr = {"Content-Type": "application/json"} if data else {}
            t = time.perf_counter()
            c.request(method, path, body=data, headers=hdr)
            r = c.getresponse()
            raw = r.read()
            dt = time.perf_counter() - t
            if r.status != 200:
                raise RuntimeError(f"{method} {path}: HTTP {r.status}: {raw[:300]!r}")
            return json.loads(raw), dt
        finally:
            c.close()


def now():
    return time.strftime("%I:%M:%S %p").lstrip("0")


def say(*a):
    print(f"[prefill-bench {now()}]", *a, flush=True)


def stats_link(cl):
    s, _ = cl.call("GET", "/stats")
    ln = s.get("link") or {}
    ex = s.get("experts") or {}
    return {"ar_bytes": ln.get("ar_bytes"), "ar_calls": ln.get("ar_calls"),
            "h2d_bytes": ln.get("h2d_bytes"), "staged_bytes": ex.get("staged_bytes"),
            "in_flight": s.get("in_flight")}


def build_prompts(cl, model, sizes, seed, run_id):
    """One body, cut per size so header + body + trailer is the size in tokens (within 0.5%)."""
    biggest = max(sizes)
    body = make_body(seed, int(biggest * 6.5) + 20000)
    j, _ = cl.call("POST", "/tokenize", {"model": model, "content": body, "with_pieces": True,
                                         "add_special": False})
    pieces = j["tokens"]
    if len(pieces) < biggest:
        raise RuntimeError(f"the body is {len(pieces)} tokens, short of {biggest}")
    ends, at = [], 0
    for p in pieces:
        s = p["piece"] if isinstance(p.get("piece"), str) else bytes(p["piece"]).decode("utf-8", "replace")
        at += len(s)
        ends.append(at)
    trailer = "\n\nIn one word, what is the text above about?"
    out = {}
    for n in sizes:
        head = header(run_id, n, 0)
        over = count(cl, model, head + trailer)
        k = max(1, n - over)
        text = body[:ends[k - 1]]
        for _ in range(4):
            got = count(cl, model, head + text + trailer)
            if abs(got - n) <= max(2, n * 0.005):
                break
            k = max(1, k + (n - got))
            text = body[:ends[min(k, len(ends)) - 1]]
        out[n] = {"body": text, "trailer": trailer, "tokens": got}
    return out


def header(run_id, n, rep):
    """The cache-busting first line: fixed width, so every repeat's header is the same length."""
    return f"Run {run_id} size {n:06d} pass {rep:03d}. Read the text below.\n\n"


def count(cl, model, text):
    j, _ = cl.call("POST", "/tokenize", {"model": model, "prompt": text})
    return int(j["count"]) if "count" in j else len(j["tokens"])


def one(cl, model, prompt, use_stats):
    s0 = stats_link(cl) if use_stats else None
    body = {"model": model, "prompt": prompt, "max_tokens": 1, "temperature": 0, "stream": False}
    j, wall = cl.call("POST", "/v1/completions", body)
    s1 = stats_link(cl) if use_stats else None
    t = j.get("timings") or {}
    u = j.get("usage") or {}
    r = {"prompt_tokens": u.get("prompt_tokens"), "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
         "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"), "prompt_ms": t.get("prompt_ms"),
         "ttft_ms": t.get("ttft_ms"), "wall_s": wall}
    pn = r["prompt_n"] or r["prompt_tokens"]
    r["tps"] = pn * 1000.0 / r["prompt_ms"] if pn and r["prompt_ms"] else None
    r["tps_client"] = (r["prompt_tokens"] or 0) / wall if wall > 0 else None
    if use_stats:
        r["d"] = {k: (s1[k] - s0[k]) if isinstance(s0.get(k), (int, float)) and isinstance(s1.get(k), (int, float))
                  else None for k in ("ar_bytes", "ar_calls", "h2d_bytes", "staged_bytes")}
        r["in_flight_before"] = s0.get("in_flight")
    return r


def summarize(rows):
    v = [r["tps"] for r in rows if r.get("tps")]
    if not v:
        return {"n": 0}
    med = statistics.median(v)
    sd = statistics.stdev(v) if len(v) > 1 else 0.0
    spread = (max(v) - min(v)) / med * 100
    out = {"n": len(v), "median": med, "mean": statistics.fmean(v), "sd": sd,
           "cv_pct": sd / statistics.fmean(v) * 100, "min": min(v), "max": max(v),
           "spread_pct": spread, "floor_b4_pct": max(3.0, 2 * spread),
           "median_client": statistics.median([r["tps_client"] for r in rows if r.get("tps_client")])}
    ar = [r["d"]["ar_bytes"] / r["prompt_n"] for r in rows
          if r.get("d") and r["d"].get("ar_bytes") is not None and r.get("prompt_n")]
    if ar:
        out["ar_bytes_per_token"] = statistics.median(ar)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="", help="default: the server's first model")
    ap.add_argument("--sizes", default="8K,32K,64K,128K", help="prompt sizes in tokens, K = 1024")
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=1, help="discarded passes over every size first")
    ap.add_argument("--seed", type=int, default=20261120, help="the body's seed: change it for other text")
    ap.add_argument("--stats", action="store_true", help="record /stats link counters around each request")
    ap.add_argument("--deadline", default="", help="unix time to stop starting new passes by")
    ap.add_argument("--label", default="")
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sizes = [size_of(s) for s in a.sizes.split(",") if s.strip()]
    if not sizes or a.repeats < 1:
        ap.error("need sizes and at least one repeat")
    cl = Client(a.host, a.port, a.timeout)
    model = a.model
    if not model:
        j, _ = cl.call("GET", "/v1/models")
        model = j["data"][0]["id"]
    run_id = uuid.uuid4().hex[:12]
    meta = {"label": a.label, "model": model, "sizes": sizes, "repeats": a.repeats, "warmup": a.warmup,
            "seed": a.seed, "run_id": run_id, "started": time.time(), "host": a.host, "port": a.port}
    try:
        meta["version"], _ = cl.call("GET", "/version")
    except Exception as e:  # an older server: not needed for the measurement
        meta["version"] = str(e)
    say(f"model {model}, sizes {sizes}, {a.repeats} repeats after {a.warmup} warm-up, run {run_id}")
    prompts = build_prompts(cl, model, sizes, a.seed, run_id)
    for n in sizes:
        say(f"size {n}: {prompts[n]['tokens']} tokens with header and question")
    rows, failed = [], 0
    deadline = float(a.deadline) if a.deadline else None
    plan = [("warmup", w) for w in range(a.warmup)] + [("rep", r) for r in range(a.repeats)]
    for kind, i in plan:
        if deadline and kind == "rep" and time.time() > deadline:
            say(f"deadline reached before pass {i}; stopping")
            break
        for n in sizes:
            p = prompts[n]
            rep = i if kind == "rep" else 900 + i
            prompt = header(run_id, n, rep) + p["body"] + p["trailer"]
            try:
                r = one(cl, model, prompt, a.stats)
            except Exception as e:
                failed += 1
                say(f"{kind} {i} size {n}: FAILED: {e}")
                rows.append({"kind": kind, "pass": i, "size": n, "error": str(e)})
                continue
            r.update({"kind": kind, "pass": i, "size": n, "t": time.time()})
            rows.append(r)
            warn = ""
            if r.get("cache_n"):
                warn = f"  WARNING: {r['cache_n']} tokens came from the prefix cache"
            ar = ""
            if r.get("d") and r["d"].get("ar_bytes") is not None:
                ar = f", ar {r['d']['ar_bytes'] / 2**30:.2f} GiB in {r['d']['ar_calls']} calls"
            tps = f"{r['tps']:,.0f}" if r.get("tps") else "?"
            say(f"{kind} {i} size {n}: {tps} tok/s ({r.get('prompt_n')} tok in "
                f"{(r.get('prompt_ms') or 0) / 1000:.2f} s, wall {r['wall_s']:.2f} s){ar}{warn}")
    meta["finished"] = time.time()
    summ = {str(n): summarize([r for r in rows if r.get("kind") == "rep" and r["size"] == n and "error" not in r])
            for n in sizes}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": summ, "rows": rows}, f, indent=1)
    for n in sizes:
        s = summ[str(n)]
        if s.get("n"):
            say(f"SUMMARY size {n}: median {s['median']:,.0f} tok/s over {s['n']}, spread {s['spread_pct']:.1f}%, "
                f"cv {s['cv_pct']:.1f}%, floor (B.4) {s['floor_b4_pct']:.1f}%"
                + (f", ar {s['ar_bytes_per_token']:,.0f} B/token" if "ar_bytes_per_token" in s else ""))
    say(f"wrote {a.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

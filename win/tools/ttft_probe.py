"""ttft-probe: time to first token under a steady load of short chat requests, for its tail.

Brilliance (ttft-tail). The alpha.3 final bench put p99 TTFT 7.9% (c1) and 5.2% (c8) above alpha.2,
from one run. A full BetterBench is too slow to repeat per arm; this sends short streamed chat
requests from `--concurrency` workers for `--seconds`, each worker starting its next request as its
last ends, and reports the TTFT distribution with a bootstrap interval on p99, so two arms' tails
can be told apart from noise.

  * TTFT is the client's: request sent to the first streamed chunk that carries text (content or
    reasoning). The server's own `timings.ttft_ms` (body received to first token) from the final
    chunk is kept beside it.
  * Prompts are 48 fixed short tasks, each with a unique first line (run, worker, request), so the
    prefix cache serves only the chat template's shared head, as for real traffic. Seeds are fixed
    per request number, so arms see the same request stream.
  * Output is capped at --max-tokens (default 96) with ignore_eos, so every request decodes as long
    and the c8 streams overlap evenly.

  python -I ttft_probe.py --port 8000 --concurrency 8 --seconds 300 --out c8.json

Any running server with a chat template. A measurement: run it alone on the machine (class M).
Exit 0, 1 if more than 1% of requests failed, 2 on usage.
"""
import argparse
import http.client
import json
import random
import statistics
import sys
import threading
import time
import uuid

TASKS = [
    "Explain in two sentences why the sky is blue.",
    "Write a haiku about a river in winter.",
    "List three uses of a paperclip other than holding paper.",
    "What is the difference between a list and a tuple in Python?",
    "Summarize the plot of a heist story in three sentences.",
    "Give me a short packing list for a weekend hiking trip.",
    "Translate 'good morning, how are you?' into French and Spanish.",
    "Write a SQL query that counts orders per customer.",
    "Suggest a name for a coffee shop run by cats.",
    "Explain what a hash table is to a ten-year-old.",
    "Write a limerick about a forgetful robot.",
    "What are two pros and two cons of remote work?",
    "Describe the smell of rain in one paragraph.",
    "Give a regular expression that matches a US ZIP code.",
    "Write a polite email declining a meeting invitation.",
    "What causes the seasons on Earth?",
    "Write a JavaScript function that reverses a string.",
    "Name three famous bridges and where they are.",
    "Explain recursion with a short example.",
    "Write a two-line motivational quote about learning.",
    "What is the capital of Australia, and why is it not Sydney?",
    "Describe a sunset over the ocean for a novel's opening line.",
    "Give three tips for writing clear documentation.",
    "Explain the difference between TCP and UDP briefly.",
    "Write a short product description for a reusable water bottle.",
    "What is photosynthesis? Answer in two sentences.",
    "Write a bash one-liner that counts lines in all .txt files.",
    "Invent a new board game and describe its goal in three sentences.",
    "Explain big-O notation briefly with one example.",
    "Write a short dialogue between a cat and a dog about breakfast.",
    "What are the three primary colors of light?",
    "Give a recipe idea using rice, eggs and spinach.",
    "Explain what an API is in plain words.",
    "Write a four-line poem about city lights.",
    "Why do we have leap years?",
    "List four planets in order from the sun.",
    "Write a C function that returns the maximum of two ints.",
    "What is the main idea of supply and demand?",
    "Write a short thank-you note to a teacher.",
    "Explain the rules of tic-tac-toe in three sentences.",
    "Describe a medieval market square in a few sentences.",
    "What is the boiling point of water at sea level in Fahrenheit?",
    "Give two ways to improve sleep quality.",
    "Write a Python list comprehension for squares of even numbers.",
    "What does a compiler do?",
    "Write a tweet announcing a library's new opening hours.",
    "Explain why ice floats on water.",
    "Name three instruments in a string quartet.",
]


def now():
    return time.strftime("%I:%M:%S %p").lstrip("0")


def say(*a):
    print(f"[ttft-probe {now()}]", *a, flush=True)


def pct(v, p):
    """The p-th percentile by linear interpolation (numpy's default)."""
    if not v:
        return None
    s = sorted(v)
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def boot_ci(v, p, n=1000, seed=7):
    if len(v) < 20:
        return None
    r = random.Random(seed)
    est = sorted(pct([v[r.randrange(len(v))] for _ in v], p) for _ in range(n))
    return [est[int(0.025 * n)], est[int(0.975 * n) - 1]]


def stream_one(host, port, model, msg, max_tokens, seed, timeout):
    body = {"model": model, "messages": [{"role": "user", "content": msg}], "max_tokens": max_tokens,
            "temperature": 0.7, "seed": seed, "ignore_eos": True, "stream": True}
    c = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        t0 = time.perf_counter()
        c.request("POST", "/v1/chat/completions", body=json.dumps(body).encode(),
                  headers={"Content-Type": "application/json"})
        r = c.getresponse()
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {r.read()[:200]!r}")
        first, timings, n_chunks = None, None, 0
        while True:
            line = r.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            j = json.loads(data)
            if "error" in j:
                raise RuntimeError(str(j["error"])[:200])
            n_chunks += 1
            for ch in j.get("choices") or []:
                d = ch.get("delta") or {}
                if first is None and (d.get("content") or d.get("reasoning_content") or d.get("reasoning")):
                    first = time.perf_counter() - t0
            if j.get("timings"):
                timings = j["timings"]
        total = time.perf_counter() - t0
        if first is None:
            raise RuntimeError("no text in the stream")
        return {"ttft_ms": first * 1000.0, "total_s": total, "chunks": n_chunks,
                "server_ttft_ms": (timings or {}).get("ttft_ms"),
                "prompt_ms": (timings or {}).get("prompt_ms"),
                "cache_n": (timings or {}).get("cache_n"), "prompt_n": (timings or {}).get("prompt_n"),
                "predicted_n": (timings or {}).get("predicted_n")}
    finally:
        c.close()


def summarize(v):
    if not v:
        return {"n": 0}
    return {"n": len(v), "mean": statistics.fmean(v), "p50": pct(v, 50), "p90": pct(v, 90),
            "p99": pct(v, 99), "max": max(v), "p99_ci95": boot_ci(v, 99), "p50_ci95": boot_ci(v, 50)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=300)
    ap.add_argument("--warmup", type=float, default=15, help="seconds of requests discarded first")
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--label", default="")
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.concurrency < 1 or a.seconds <= 0:
        ap.error("--concurrency and --seconds must be positive")
    model = a.model
    if not model:
        c = http.client.HTTPConnection(a.host, a.port, timeout=30)
        c.request("GET", "/v1/models")
        model = json.loads(c.getresponse().read())["data"][0]["id"]
        c.close()
    run_id = uuid.uuid4().hex[:8]
    say(f"model {model}, c{a.concurrency}, {a.seconds:.0f} s after {a.warmup:.0f} s warm-up, "
        f"max_tokens {a.max_tokens}, run {run_id}")
    t_start = time.time()
    t_meas = t_start + a.warmup
    t_end = t_meas + a.seconds
    rows, lock = [], threading.Lock()
    counter = [0]

    def worker(w):
        while time.time() < t_end:
            with lock:
                i = counter[0]
                counter[0] += 1
            msg = f"[{run_id}-{w}-{i:05d}] " + TASKS[i % len(TASKS)]
            sent = time.time()
            try:
                r = stream_one(a.host, a.port, model, msg, a.max_tokens, 1000 + i, a.timeout)
            except Exception as e:
                r = {"error": str(e)}
            r.update({"i": i, "worker": w, "sent": sent, "measured": sent >= t_meas})
            with lock:
                rows.append(r)

    th = [threading.Thread(target=worker, args=(w,), daemon=True) for w in range(a.concurrency)]
    for t in th:
        t.start()
    last = time.time()
    while any(t.is_alive() for t in th):
        time.sleep(1)
        if time.time() - last >= 60:
            last = time.time()
            with lock:
                v = [r["ttft_ms"] for r in rows if r.get("measured") and "ttft_ms" in r]
                bad = sum(1 for r in rows if "error" in r)
            p99 = pct(v, 99)
            say(f"{len(v)} measured, p50 {pct(v, 50) or 0:.1f} ms, p99 {p99 or 0:.1f} ms, {bad} failed")
    meas = [r for r in rows if r.get("measured")]
    ok = [r for r in meas if "ttft_ms" in r]
    bad = [r for r in rows if "error" in r]
    summ = {"client": summarize([r["ttft_ms"] for r in ok]),
            "server": summarize([r["server_ttft_ms"] for r in ok if r.get("server_ttft_ms") is not None]),
            "failed": len(bad), "requests": len(meas),
            "rate_per_s": len(ok) / a.seconds if a.seconds else None,
            "tokens_per_s": sum(r.get("predicted_n") or 0 for r in ok) / a.seconds}
    meta = {"label": a.label, "model": model, "concurrency": a.concurrency, "seconds": a.seconds,
            "warmup": a.warmup, "max_tokens": a.max_tokens, "run_id": run_id, "started": t_start,
            "finished": time.time()}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": summ, "rows": rows}, f, indent=1)
    c = summ["client"]
    if c.get("n"):
        ci = c.get("p99_ci95") or [0, 0]
        say(f"SUMMARY c{a.concurrency}: {c['n']} requests, TTFT p50 {c['p50']:.1f} ms, p90 {c['p90']:.1f} ms, "
            f"p99 {c['p99']:.1f} ms (95% CI {ci[0]:.1f}-{ci[1]:.1f}), max {c['max']:.1f} ms; "
            f"{len(bad)} failed; {summ['tokens_per_s']:.0f} output tok/s")
    say(f"wrote {a.out}")
    return 1 if len(bad) > max(1, 0.01 * len(rows)) else 0


if __name__ == "__main__":
    sys.exit(main())

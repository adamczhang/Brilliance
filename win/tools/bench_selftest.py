"""Host check of prefill_bench.py and ttft_probe.py against a fake server: no model, no card.

The fake answers what the two clients use: /v1/models, /version, /stats (link counters that grow
with every prompt token), /tokenize (one token a word or space, with pieces), /v1/completions (a
prefill timing proportional to the prompt) and a streamed /v1/chat/completions. It checks that the
prefill bench cuts every size to within 0.5%, never hits the prefix cache, reports the spread and
the per-token ar_bytes the fake billed; and that the probe streams, measures TTFT and summarizes
its tail.

  python -I bench_selftest.py         exit 0 when every check holds
"""
import http.server
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TOK = re.compile(r"\s+|[^\s]+")
AR_PER_TOKEN = 501760


def toks(text):
    return TOK.findall(text)


class Fake(http.server.BaseHTTPRequestHandler):
    seen = set()
    ar = 0
    calls = 0

    def log_message(self, *a):
        pass

    def send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/v1/models":
            return self.send({"data": [{"id": "fake"}]})
        if self.path == "/version":
            return self.send({"version": "fake"})
        if self.path == "/stats":
            return self.send({"in_flight": 0, "link": {"ar_bytes": Fake.ar, "ar_calls": Fake.calls, "h2d_bytes": 0},
                              "experts": {"staged_bytes": 0}})
        self.send({"error": "no"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        b = json.loads(self.rfile.read(n))
        if self.path == "/tokenize":
            text = b.get("content", b.get("prompt", ""))
            t = toks(text)
            if b.get("with_pieces"):
                return self.send({"tokens": [{"id": i, "piece": p} for i, p in enumerate(t)], "count": len(t)})
            return self.send({"tokens": list(range(len(t))), "count": len(t)})
        if self.path == "/v1/completions":
            t = toks(b["prompt"])
            head = "".join(t[:16])   # the first block: the prefix cache's key
            cached = 16 if head in Fake.seen else 0
            Fake.seen.add(head)
            pn = len(t) - cached
            Fake.ar += pn * AR_PER_TOKEN
            Fake.calls += 1
            ms = pn / 6.0 * (1.0 + 0.01 * ((len(Fake.seen) * 7) % 3))   # ~6,000 tok/s, a 2% spread
            return self.send({"usage": {"prompt_tokens": len(t), "completion_tokens": 1,
                                        "prompt_tokens_details": {"cached_tokens": cached}},
                              "timings": {"prompt_n": pn, "cache_n": cached, "prompt_ms": ms, "ttft_ms": ms + 1}})
        if self.path == "/v1/chat/completions":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            time.sleep(0.02)
            for k in range(min(5, b.get("max_tokens", 5))):
                d = {"choices": [{"delta": {"reasoning_content" if k == 0 else "content": "x"}}]}
                self.wfile.write(b"data: " + json.dumps(d).encode() + b"\n\n")
                self.wfile.flush()
                time.sleep(0.002)
            fin = {"choices": [{"delta": {}, "finish_reason": "length"}],
                   "timings": {"ttft_ms": 19.0, "prompt_ms": 15.0, "cache_n": 0, "prompt_n": 30, "predicted_n": 5}}
            self.wfile.write(b"data: " + json.dumps(fin).encode() + b"\n\ndata: [DONE]\n\n")
            return
        self.send({"error": "no"}, 404)


def main():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    bad = []
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "p.json")
        r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "prefill_bench.py"), "--port", str(port),
                            "--sizes", "1K,4K", "--repeats", "3", "--stats", "--out", out],
                           capture_output=True, text=True, timeout=300)
        print(r.stdout[-1500:], r.stderr[-1500:])
        if r.returncode != 0:
            bad.append(f"prefill_bench exit {r.returncode}")
        else:
            j = json.load(open(out, encoding="utf-8"))
            for row in j["rows"]:
                if row.get("cache_n"):
                    bad.append(f"prefix cache hit: {row}")
                if row["kind"] == "rep" and abs(row["prompt_tokens"] - row["size"]) > max(2, row["size"] * 0.005):
                    bad.append(f"size {row['size']} cut to {row['prompt_tokens']}")
            for n, s in j["summary"].items():
                if s.get("n") != 3:
                    bad.append(f"size {n}: {s.get('n')} repeats summarized")
                if abs(s.get("ar_bytes_per_token", 0) - AR_PER_TOKEN) > 1:
                    bad.append(f"size {n}: ar {s.get('ar_bytes_per_token')} B/token, not {AR_PER_TOKEN}")
                if not (0 < s.get("spread_pct", -1) < 5):
                    bad.append(f"size {n}: spread {s.get('spread_pct')}")
        out2 = os.path.join(d, "t.json")
        r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "ttft_probe.py"), "--port", str(port),
                            "--concurrency", "4", "--seconds", "3", "--warmup", "1", "--out", out2],
                           capture_output=True, text=True, timeout=120)
        print(r.stdout[-1000:], r.stderr[-1000:])
        if r.returncode != 0:
            bad.append(f"ttft_probe exit {r.returncode}")
        else:
            j = json.load(open(out2, encoding="utf-8"))
            c = j["summary"]["client"]
            if not c.get("n") or c["n"] < 20:
                bad.append(f"ttft_probe measured {c.get('n')}")
            elif not (15 < c["p50"] < 500) or c["p99"] < c["p50"]:
                bad.append(f"ttft_probe percentiles {c}")
            if j["summary"]["server"].get("p50") != 19.0:
                bad.append(f"server ttft {j['summary']['server']}")
    srv.shutdown()
    for b in bad:
        print("FAIL:", b)
    print("bench_selftest:", "ok" if not bad else f"{len(bad)} failure(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

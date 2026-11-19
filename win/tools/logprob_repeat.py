"""Brilliance (the sampling investigation): The same prompt's next-token top-5 logprobs, asked N times: does the forward pass repeat?"""
import argparse
import json
import math
import sys
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8100")
ap.add_argument("--body", required=True)
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--cont", default="")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def post(path, body):
    req = urllib.request.Request(a.url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


chat = json.load(open(a.body, encoding="utf-8"))
prompt = post("/apply-template", {k: chat[k] for k in ("model", "messages", "tools") if k in chat})["prompt"] + a.cont
seen = {}
for i in range(a.n):
    r = post("/v1/completions", {"model": chat["model"], "prompt": prompt, "max_tokens": 1, "temperature": 0,
                                 "logprobs": 5})
    tl = r["choices"][0]["logprobs"]["top_logprobs"][0]
    key = tuple((t, round(lp, 4)) for t, lp in sorted(tl.items(), key=lambda kv: -kv[1]))
    seen.setdefault(key, []).append(i)
    u = r.get("usage", {})
    print(i, " ".join(f"{t!r}:{math.exp(lp):.4f}" for t, lp in key), u.get("prompt_tokens_details", ""), flush=True)
print(f"{len(seen)} distinct distributions over {a.n} asks")

"""Brilliance (the sampling investigation): does a batched row see the same distribution
as a solo one? The fixture's first-token top-5 logprobs (temperature 0) asked alone, then W at once for
R rounds; prints each distinct distribution with how often it came back."""
import argparse
import collections
import concurrent.futures as cf
import json
import math
import sys
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8100")
ap.add_argument("--body", required=True)
ap.add_argument("--model", default="")
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--rounds", type=int, default=10)
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
body = json.load(open(a.body, encoding="utf-8"))
if a.model:
    body["model"] = a.model
body.update(stream=False, max_tokens=1, temperature=0, logprobs=True, top_logprobs=5)
body.pop("stream_options", None)
data = json.dumps(body).encode()


def ask(_):
    req = urllib.request.Request(a.url + "/v1/chat/completions", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        c = json.load(r)["choices"][0]["logprobs"]["content"][0]
    return tuple((t["token"], round(math.exp(t["logprob"]), 4)) for t in c["top_logprobs"])


solo = ask(0)
print("solo:", solo)
seen = collections.Counter()
with cf.ThreadPoolExecutor(a.workers) as ex:
    for _ in range(a.rounds):
        for d in ex.map(ask, range(a.workers)):
            seen[d] += 1
for d, n in seen.most_common():
    print(f"x{n} {'== solo' if d == solo else 'DIFFERS'}: {d}")

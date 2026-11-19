"""Brilliance (the sampling investigation): The sampler against the model, through /v1/chat/completions itself: the first generated token's
top-20 logprobs (greedy, max_tokens 1), then N one-token samples at temperature/top-k/top-p."""
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
ap.add_argument("--n", type=int, default=400)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--top-k", type=int, default=20)
ap.add_argument("--top-p", type=float, default=0.95)
ap.add_argument("--omit", action="store_true", help="send no sampling fields (the client's way)")
ap.add_argument("--model", default="")
ap.add_argument("--workers", type=int, default=4)
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
base = json.load(open(a.body, encoding="utf-8"))
if a.model:
    base["model"] = a.model
base["stream"] = False
base.pop("stream_options", None)
base["max_tokens"] = 1


def post(body):
    req = urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


r = post(dict(base, temperature=0, logprobs=True, top_logprobs=20))
c = r["choices"][0]["logprobs"]["content"][0]
tl = {t["token"]: t["logprob"] for t in c["top_logprobs"]}
print("greedy first token:", repr(c["token"]), "usage", r.get("usage"))
ranked = sorted(tl.items(), key=lambda kv: -kv[1])[: a.top_k]
z = [math.exp(lp / a.temperature) for _, lp in ranked]
p = [v / sum(z) for v in z]
keep, cum = 0, 0.0
for v in p:
    keep += 1
    cum += v
    if cum >= a.top_p:
        break
want = {ranked[i][0]: p[i] / sum(p[:keep]) for i in range(keep)}

samp = dict(base, logprobs=True, top_logprobs=1)
if not a.omit:
    samp.update(temperature=a.temperature, top_k=a.top_k, top_p=a.top_p)


def one(_):
    rr = post(samp)
    return rr["choices"][0]["logprobs"]["content"][0]["token"]


with cf.ThreadPoolExecutor(a.workers) as ex:
    got = collections.Counter(ex.map(one, range(a.n)))
print(f"workers {a.workers}")
print(f"{a.n} samples ({'no sampling fields' if a.omit else f'T {a.temperature} k {a.top_k} p {a.top_p}'}); "
      f"{keep} token(s) kept by top-p")

def loglik(T):
    zz = {t: math.exp(lp / T) for t, lp in ranked}
    s = sum(zz.values())
    return sum(c * math.log(zz[t] / s) for t, c in got.items() if t in zz)


grid = [0.5 + 0.01 * i for i in range(251)]
t_eff = max(grid, key=loglik) if a.top_p >= 1.0 else float("nan")
top_tok = ranked[0][0]
print(f"top token {top_tok!r}: expected share {want.get(top_tok, 0):.4f}, got {got.get(top_tok, 0) / a.n:.4f}; "
      f"effective temperature (top-p off only) {t_eff:.2f}")
for tok in list(dict.fromkeys(list(want) + [t for t, _ in ranked[:6]] + list(got))):
    e = want.get(tok, 0.0) * a.n
    raw = math.exp(tl[tok]) if tok in tl else float("nan")
    print(f"  {tok!r:>16}  model p {raw:7.4f}  expected {e:6.1f}  got {got.get(tok, 0):4d}"
          + ("" if tok in want else "   (outside the kept set)"))

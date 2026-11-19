"""Brilliance (the sampling investigation): Replays one captured chat request N times (streamed, as the client sent it), with W at a time,
and counts answers that lack an expected string. Prints each bad answer, its finish reason and the
token counts the server reported."""
import argparse
import collections
import concurrent.futures as cf
import json
import sys
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8100")
ap.add_argument("--body", required=True)
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--workers", type=int, default=1)
ap.add_argument("--expect", default="123.4")
ap.add_argument("--set", action="append", default=[], help="key=json, merged into the body")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
body = json.load(open(a.body, encoding="utf-8"))
for kv in a.set:
    k, v = kv.split("=", 1)
    body[k] = json.loads(v)
data = json.dumps(body).encode()


def one(i):
    req = urllib.request.Request(a.url + "/v1/chat/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    text, reason, fin, usage = [], [], None, None
    with urllib.request.urlopen(req, timeout=300) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content"):
                    text.append(d["content"])
                if d.get("reasoning_content"):
                    reason.append(d["reasoning_content"])
                if ch.get("finish_reason"):
                    fin = ch["finish_reason"]
    return i, "".join(text), "".join(reason), fin, usage


bad, fins, outs, first = 0, collections.Counter(), collections.Counter(), {}
with cf.ThreadPoolExecutor(a.workers) as ex:
    for i, text, reason, fin, usage in ex.map(one, range(a.n)):
        fins[fin] += 1
        outs[(reason, text)] += 1
        first.setdefault((reason, text), i)
        if a.expect not in text:
            bad += 1
            ct = (usage or {}).get("completion_tokens")
            print(f"#{i}: finish={fin} completion_tokens={ct} text={text!r} reasoning={reason[-120:]!r}", flush=True)
print(f"{a.n} replays, {bad} without {a.expect!r}; finish reasons {dict(fins)}; settings {a.set}")
print(f"{len(outs)} distinct outputs; the most common:")
for (reason, text), c in outs.most_common(6):
    print(f"  x{c} (first #{first[(reason, text)]}): reasoning {reason[:160]!r} text {text[:100]!r}")

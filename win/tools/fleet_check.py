#!/usr/bin/env python3
"""Brilliance: the fleet's lifecycle on a live server, against upstream's own statement of
it (docs/MULTIMODEL.md sections 5.4, 8, 10 and 11). Standard library only:

  .toolchain\\python\\python.exe -I win\\tools\\fleet_check.py --url http://127.0.0.1:18490 \\
      --live out\\fleet\\live.yaml --next out\\fleet\\next.yaml

The fleet it expects (the two files are written to match it): `small` with the aliases `tiny` and
`mini`, `big`, the two in an exclusive group; `broken`, whose file is not a container; the warm-up
off, so every load is one this check asks for. --next keeps big as it is, changes small's
max-model-len (--small-len-after), drops broken and adds `extra`.

Steps, each printed PASS or FAIL:
  aliases      a request naming an alias is answered by its model, which becomes resident
  health       /health is 200; /health?model=NAME answers for a resident model and a cold one
  admin        POST /admin/models/big/load loads it and sends small away (the exclusive group);
               /admin/models/big/unload leaves it cold
  failure      a request for `broken` fails with an error body, the entry reads failed, and the
               rest of the fleet serves on; a name the fleet does not have is a 404
  reload       --next is copied over --live and POST /admin/reload applied: small changed (drained,
               unloaded, cold, its new max_model_len), broken removed (404), big untouched, extra
               added (cold), and the changed model answers again
"""
import argparse
import json
import shutil
import sys
import time
import urllib.error
import urllib.request

ARGS = None
RESULTS = []


def call(method, path, body=None, timeout=600):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(ARGS.url + path, method=method, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            text = r.read().decode("utf-8", "replace")
            return r.status, text
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def js(text):
    try:
        return json.loads(text)
    except ValueError:
        return None


def record(name, ok, detail):
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)


def models():
    """{id: entry} from /v1/models."""
    s, t = call("GET", "/v1/models", timeout=30)
    return {m["id"]: m for m in (js(t) or {}).get("data", [])} if s == 200 else {}


def fleet_states():
    """{id: state} from /admin/fleet, which also says `failed` and `waiting`."""
    s, t = call("GET", "/admin/fleet", timeout=30)
    return {m["id"]: m.get("state") for m in (js(t) or {}).get("models", [])} if s == 200 else {}


def chat(model, max_tokens=64):
    body = {"model": model, "max_tokens": max_tokens, "temperature": 0,
            "messages": [{"role": "user", "content": "Name one primary colour. One word."}],
            "chat_template_kwargs": {"enable_thinking": False}}
    s, t = call("POST", "/v1/chat/completions", body)
    j = js(t) or {}
    text = ""
    if s == 200 and j.get("choices"):
        msg = j["choices"][0].get("message", {})
        text = (msg.get("content") or msg.get("reasoning_content") or "").strip()
    return s, j, text


def step_aliases():
    s, j, text = chat("tiny")
    st = fleet_states()
    record("aliases", s == 200 and bool(text) and st.get("small") == "resident",
           f"'tiny' -> {s}, answered {text[:40]!r} as model {j.get('model')!r}; small is {st.get('small')}")


def step_health():
    s0, _ = call("GET", "/health", timeout=30)
    s1, t1 = call("GET", "/health?model=small", timeout=30)
    s2, t2 = call("GET", "/health?model=big", timeout=30)
    # A cold model is not ready: 503 with its state, which is what a load balancer acts on.
    cold = s2 == 503 and (js(t2) or {}).get("status") == "cold"
    record("health", s0 == 200 and s1 == 200 and (js(t1) or {}).get("status") == "ok" and cold,
           f"/health {s0}; ?model=small (resident) {s1} {t1.strip()[:80]!r}; "
           f"?model=big (cold) {s2} {t2.strip()[:80]!r}")


def step_admin():
    s1, t1 = call("POST", "/admin/models/big/load")
    st1 = fleet_states()
    s2, t2 = call("POST", "/admin/models/big/unload")
    st2 = fleet_states()
    ok = (s1 == 200 and st1.get("big") == "resident" and st1.get("small") != "resident" and
          s2 == 200 and st2.get("big") != "resident")
    record("admin", ok, f"load big -> {s1} {t1.strip()[:60]!r}: big {st1.get('big')}, small {st1.get('small')}; "
                        f"unload big -> {s2} {t2.strip()[:60]!r}: big {st2.get('big')}")


def step_failure():
    s, j, _ = chat("broken")
    err = (j.get("error") or {}).get("message", "") if isinstance(j.get("error"), dict) else str(j.get("error", ""))
    st = fleet_states()
    sh, _ = call("GET", "/health", timeout=30)
    s2, _, text2 = chat("small")
    s3, j3, _ = chat("no-such-model")
    ok = s >= 400 and bool(err) and st.get("broken") == "failed" and sh == 200 and s2 == 200 and text2 and s3 == 404
    record("failure", ok, f"'broken' -> {s}: {err[:120]!r}; its state {st.get('broken')}; /health {sh}; "
                          f"small after it -> {s2} {text2[:30]!r}; an unknown name -> {s3}")


def step_reload():
    before = models()
    st_before = fleet_states()
    shutil.copyfile(ARGS.next, ARGS.live)
    s, t = call("POST", "/admin/reload", {}, timeout=600)
    deadline = time.time() + 120
    while time.time() < deadline:
        st = fleet_states()
        if st.get("small") not in ("closing", "loading") and "broken" not in st:
            break
        time.sleep(0.5)
    after = models()
    st = fleet_states()
    sb, _, _ = chat("broken")
    # The changed model went cold (a cold entry states no max_model_len); a request loads it again
    # with the new settings.
    small_cold = st.get("small") == "cold"
    ss, _, text = chat("small")
    small_len = models().get("small", {}).get("max_model_len")
    ok = (s == 200 and "broken" not in after and "extra" in after and st.get("extra") != "resident" and
          small_cold and small_len == ARGS.small_len_after and st_before.get("big") == st.get("big") and
          sb == 404 and ss == 200 and bool(text))
    record("reload", ok, f"POST /admin/reload -> {s} {t.strip()[:160]!r}; models {sorted(before)} -> "
                         f"{sorted(after)}; small {st_before.get('small')} -> {st.get('small')}, its "
                         f"max_model_len {before.get('small', {}).get('max_model_len')} -> {small_len} once "
                         f"loaded again; big {st_before.get('big')} -> {st.get('big')}; extra {st.get('extra')}; "
                         f"'broken' -> {sb}; small again -> {ss} {text[:30]!r}")


IDENTITY_TEXT = "".join(f"Entry {i}: the keeper of light {i % 7} logged ship {i * 37 % 1009} at hour {i % 24}.\n"
                        for i in range(160))


def chat_cached(model, user):
    """A long conversation, the same every time: (status, text, prompt tokens, cached tokens)."""
    body = {"model": model, "max_tokens": 48, "temperature": 0, "user": user,
            "messages": [{"role": "user", "content": IDENTITY_TEXT + "\nWhich ship did entry 12 log? Answer briefly."}],
            "chat_template_kwargs": {"enable_thinking": False}}
    s, t = call("POST", "/v1/chat/completions", body)
    j = js(t) or {}
    u = j.get("usage") or {}
    msg = ((j.get("choices") or [{}])[0].get("message") or {})
    text = (msg.get("content") or msg.get("reasoning_content") or "").strip()
    return s, text, u.get("prompt_tokens"), (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0


def step_identity_reload():
    """A conversation served twice comes from the prefix cache the second time; after the model is
    unloaded and loaded again nothing of it is cached (the cache belonged to the engine that left)."""
    m = ARGS.identity_model
    a = chat_cached(m, "keeper-1")
    b = chat_cached(m, "keeper-1")
    su, tu = call("POST", f"/admin/models/{m}/unload")
    c = chat_cached(m, "keeper-1")
    ok = a[0] == b[0] == c[0] == 200 and b[3] > 0 and c[3] == 0 and su == 200 and bool(c[1])
    record("identity-reload", ok, f"{m}: first {a[0]}, {a[3]} of {a[2]} cached; again {b[3]} of {b[2]} cached; "
                                  f"unload -> {su}; after the reload {c[3]} of {c[2]} cached, answer {c[1][:40]!r}")


def step_identity_name():
    """--next keeps the name and gives it another container: after POST /admin/reload the name answers
    from the new model (its own prompt length, nothing cached) and /v1/models names the new file."""
    m = ARGS.identity_model
    before = models().get(m, {})
    a = chat_cached(m, "keeper-2")
    shutil.copyfile(ARGS.next, ARGS.live)
    sr, tr = call("POST", "/admin/reload", {}, timeout=600)
    b = chat_cached(m, "keeper-2")
    after = models().get(m, {})   # read once it is loaded again: a cold entry states no file
    s_sess, t_sess = call("GET", f"/sessions?model={m}", timeout=30)
    ok = (sr == 200 and a[0] == 200 and b[0] == 200 and before.get("root") and after.get("root")
          and before.get("root") != after.get("root") and b[3] == 0 and bool(b[1]) and a[2] != b[2])
    record("identity-name", ok, f"{m}: {before.get('root', '?').split('/')[-1]} -> {after.get('root', '?').split('/')[-1]} "
                                f"(reload {sr}); before {a[2]} prompt tokens, answer {a[1][:30]!r}; after {b[2]} prompt "
                                f"tokens, {b[3]} cached, answer {b[1][:30]!r}; /sessions?model={m} -> {s_sess} "
                                f"{t_sess.strip()[:120]!r}")


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", required=True)
    ap.add_argument("--live", required=True, help="the YAML the server was started with")
    ap.add_argument("--next", required=True, help="the YAML the reload step copies over --live")
    ap.add_argument("--small-len-after", type=int, default=2048, help="small's max_model_len in --next")
    ap.add_argument("--identity", action="store_true",
                    help="run the identity steps instead: the cache across an unload, and a name reused for "
                         "another container by --next")
    ap.add_argument("--identity-model", default="m", help="the model name the identity steps use")
    ARGS = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    steps = ((step_identity_reload, step_identity_name) if ARGS.identity else
             (step_aliases, step_health, step_admin, step_failure, step_reload))
    for step in steps:
        try:
            step()
        except Exception as e:      # a step that throws is a failed step, and the rest still run
            record(step.__name__[5:], False, f"{type(e).__name__}: {e}")
    print(f"\n{sum(RESULTS)} of {len(RESULTS)} steps passed")
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())

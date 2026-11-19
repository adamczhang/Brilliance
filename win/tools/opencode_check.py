"""Brilliance: tool round trips through OpenCode against a served model, standard library only.

  .toolchain\\python\\python.exe -I win\\tools\\opencode_check.py --opencode <opencode.exe> \\
      --work out\\clients\\oc-work --home out\\clients\\oc-home \\
      --out out\\clients\\opencode-run1 [--repeat 3]

--work holds the project OpenCode runs in: opencode.json (the server as an OpenAI-compatible provider;
bash, edit and webfetch denied), .opencode\\tools\\disk_free.ts (a controlled tool with a fixed answer,
"123.4 GB free") and notes\\secret.txt (the word HELIOTROPE). --home is where OpenCode keeps its
config, data and cache (the XDG variables), so nothing lands in the user's profile.

The cases, each a fresh `opencode run --format json` unless it continues a session:
  disk       "How much free disk space ..." (issue #4's question): disk_free is called, the answer
             gives 123.4, and no other tool is tried;
  read       the secret word read from notes\\secret.txt with OpenCode's own read tool;
  followup   the disk session continued: the figure recalled from the session's history;
  isolation  a new session asked for a secret word: it must not know HELIOTROPE;
  disk-N     the disk case again --repeat times (with --disk-only, nothing else).
Every run's events go to --out\\<case>.jsonl and its stderr to <case>.err.
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time


def walk(o):
    if isinstance(o, dict):
        yield o
        for v in o.values():
            yield from walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from walk(v)


def run(a, name, prompt, session=None):
    env = dict(os.environ)
    home = pathlib.Path(a.home).resolve()
    for k, sub in (("XDG_CONFIG_HOME", "config"), ("XDG_DATA_HOME", "data"),
                   ("XDG_CACHE_HOME", "cache"), ("XDG_STATE_HOME", "state")):
        env[k] = str(home / sub)
    cmd = [a.opencode, "run", "--format", "json", "--model", a.model]
    if session:
        cmd += ["--session", session]
    cmd.append(prompt)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=a.work, env=env, capture_output=True, timeout=a.timeout)
    secs = time.time() - t0
    out = pathlib.Path(a.out)
    (out / f"{name}.jsonl").write_bytes(p.stdout)
    (out / f"{name}.err").write_bytes(p.stderr)
    events = []
    for line in p.stdout.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    # A part can arrive several times as it updates (a tool going pending, running, completed); the
    # last copy of each part id wins, in first-seen order.
    tools, texts, sid = {}, {}, None
    for ev in events:
        for d in walk(ev):
            if sid is None and isinstance(d.get("sessionID"), str):
                sid = d["sessionID"]
            if d.get("type") == "tool" and isinstance(d.get("tool"), str):
                st = d.get("state") if isinstance(d.get("state"), dict) else {}
                key = d.get("callID") or d.get("id") or f"{d['tool']}#{len(tools)}"
                tools[key] = [d["tool"], st.get("status")]
            if d.get("type") == "text" and isinstance(d.get("text"), str):
                texts[d.get("id") or d["text"]] = d["text"]
    tools = list(tools.values())
    text = "\n".join(texts.values()).strip()
    return dict(rc=p.returncode, secs=secs, events=len(events), session=sid, tools=tools, text=text)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--opencode", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--home", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="brilliance/flashnext")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--disk-only", action="store_true", help="only the --repeat disk runs")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
    results, ok = {}, True

    def check(name, r, passed, why):
        nonlocal ok
        ok &= passed
        results[name] = dict(r, passed=passed)
        short = r["text"].replace("\n", " ")[:160]
        print(f"{'PASS' if passed else 'FAIL'} {name}: {why}; {r['secs']:.1f} s, exit {r['rc']}, "
              f"tools {r['tools']}, text {short!r}", flush=True)

    disk_q = "How much free disk space does the system have?"

    def disk(name):
        r = run(a, name, disk_q)
        names = [t[0] for t in r["tools"]]
        check(name, r, r["rc"] == 0 and "disk_free" in names and set(names) <= {"disk_free"} and "123.4" in r["text"],
              "disk_free called, nothing else tried, 123.4 in the answer")
        return r

    if not a.disk_only:
        disk_session = disk("disk")["session"]

        r = run(a, "read", "What is the secret word written in notes/secret.txt?")
        names = [t[0] for t in r["tools"]]
        check("read", r, r["rc"] == 0 and "read" in names and "HELIOTROPE" in r["text"].upper(),
              "the read tool used, HELIOTROPE in the answer")

        if disk_session:
            r = run(a, "followup", "According to the tool you used, how many gigabytes were free? Reply with the "
                    "number only.", session=disk_session)
            check("followup", r, r["rc"] == 0 and "123.4" in r["text"], "the session's history recalled (123.4)")
        else:
            ok = False
            print("FAIL followup: the disk run named no session")

        r = run(a, "isolation", "Have I told you a secret word in this conversation? If so, say it; if not, say "
                "no. Do not use any tools.")
        check("isolation", r, r["rc"] == 0 and "HELIOTROPE" not in r["text"].upper(),
              "a new session does not know the other session's word")

    for i in range(1, a.repeat + 1):
        disk(f"disk-{i}")
    failed = [n for n, r in results.items() if not r["passed"]]
    print(f"{len(results) - len(failed)} of {len(results)} passed" + (f"; failed: {', '.join(failed)}" if failed else ""))

    (pathlib.Path(a.out) / "summary.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("every OpenCode case passed" if ok else "an OpenCode case FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

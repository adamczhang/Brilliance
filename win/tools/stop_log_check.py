#!/usr/bin/env python3
"""Brilliance: read a stop journal (core/util/stoplog.h) and say whether the stop was bounded.

A stop journal is what a server writes, record by record and through to the disk, when
BRILLIANCE_STOP_LOG names a file or a directory. This reads one and gives a verdict against the
bounded stop contract (win/SUPPORT.md, "Stopping a multi-card server"):

  BOUNDED      every step in the contract's order, each within its bound, every rank left its loop
               without skipping a step its peers ran, every stream found drained, and the journal
               closed. One step may go over its bound by design: a drain that timed out and ended
               the open responses (the retire, "... (retire)", then engine.retired). It is listed
               apart as expected, not as a reason, when the timeout came on time (within
               RETIRE_SLACK_MS of the drain's bound).
  UNBOUNDED    the journal closed, but a step went over its bound, a watcher WAIT was recorded, a
               rank left after a split release, or a stream had not drained.
  TRUNCATED    the journal ends without its CLOSE: the process ended hard (a forced end, a crash)
               or the machine stopped (a hard lock). The last record and every step still open are
               the evidence of where.
  NOT-STOPPED  armed, but no stop began (the server is still running, or the journal is not a
               stop's).

Usage:
  python -I stop_log_check.py JOURNAL [--tp N] [--json]

Exit status: 0 BOUNDED, 1 UNBOUNDED, 2 TRUNCATED or NOT-STOPPED, 3 unreadable.
Standard library only. Read-only: it never touches a server or a card.
"""
import argparse
import json
import re
import sys

# The contract's order for each model (a suffix of the record's name) and for the fleet. A record
# must come after the one before it; a model's steps repeat for each model.
ORDER = [
    ("STOP", "begin"),
    ("NOTE", "fleet.loop_exit"),
    ("END", "fleet.history_join"),
    ("NOTE", "fleet.request_stop"),
    ("NOTE", "engine.at_stop"),
    ("END", ".drain"),
    ("NOTE", "engine.retired"),
    ("NOTE", "engine.stopping"),
    ("END", ".step_loop_stop"),
    ("END", ".rank_join"),
    ("END", ".serve_join"),
    ("END", "fleet.front_stop"),
    ("BEGIN", ".teardown"),
    ("END", ".stream_probe"),
    ("END", ".transfer_sync"),
    ("END", ".rank0.ctx_shutdown"),
    ("END", ".ranks_free"),
    ("END", ".plugins_release"),
    ("END", ".teardown"),
    ("END", "fleet.sweep_buffers"),
    ("END", "fleet.listener_join"),
    ("NOTE", "fleet.run_return"),
    ("NOTE", "process.main_return"),
    ("CLOSE", "journal"),
]

# A drain that times out ends its step at the timeout with this detail, then the server retires the
# open responses (core/engine.cpp, Engine::serve; win/SUPPORT.md, "Stopping a multi-card server").
RETIRE_DETAIL = "(retire)"
# How late the drain's timeout may come and still count as the designed retire.
RETIRE_SLACK_MS = 1000.0

LINE = re.compile(r"^(\S+) (\S+) t=([0-9.]+) tid=(\d+) ([A-Z]+) (\S+)(?: (.*))?$")
END_DETAIL = re.compile(r"^(ok|over) ([0-9.]+) ms bound (-?\d+) ms ?(.*)$")


def parse(text):
    recs, bad = [], 0
    for raw in text.splitlines():
        m = LINE.match(raw.rstrip("\r"))
        if not m:
            if raw.strip():
                bad += 1
            continue
        date, clock, t, tid, kind, name, detail = m.groups()
        recs.append({"date": date, "clock": clock, "t": float(t), "tid": int(tid), "kind": kind,
                     "name": name, "detail": detail or ""})
    return recs, bad


def is_retire(recs, i, m):
    """Whether END record i (its detail matched as m) is the drain's designed timeout: the drain
    step, over by at most RETIRE_SLACK_MS, its detail naming the retire, and the same model's
    engine.retired after it."""
    r = recs[i]
    if not r["name"].endswith(".drain") or not m.group(4).rstrip().endswith(RETIRE_DETAIL):
        return False
    if float(m.group(2)) > float(m.group(3)) + RETIRE_SLACK_MS:
        return False
    model = r["name"][len("model."):-len(".drain")] if r["name"].startswith("model.") else None
    return any(x["kind"] == "NOTE" and x["name"] == "engine.retired" and
               (model is None or x["detail"].startswith("model %s:" % model))
               for x in recs[i + 1:])


def check(recs, tp=None):
    """The verdict, the reasons and a summary for one journal's records."""
    reasons, expected = [], []
    if not recs or recs[0]["kind"] != "ARM":
        return "UNREADABLE", ["no ARM record first"], {}
    stop = next((r for r in recs if r["kind"] == "STOP" and r["name"] == "begin"), None)
    if stop is None:
        return "NOT-STOPPED", ["no STOP begin record"], {"records": len(recs)}

    closed = recs[-1]["kind"] == "CLOSE"
    # Steps begun and not ended, in order: the place a stop that never finished was.
    open_steps = {}
    for r in recs:
        if r["kind"] == "BEGIN":
            open_steps.setdefault(r["name"], []).append(r)
        elif r["kind"] == "END" and open_steps.get(r["name"]):
            open_steps[r["name"]].pop()
        elif r["kind"] == "NOTE" and r["detail"].startswith("still open at close"):
            reasons.append("step %s was still open when the journal closed" % r["name"])
    still_open = [(n, b[-1]) for n, b in open_steps.items() if b]

    for i, r in enumerate(recs):
        if r["kind"] == "END":
            m = END_DETAIL.match(r["detail"])
            if not m:
                reasons.append("END %s: unreadable detail %r" % (r["name"], r["detail"]))
            elif m.group(1) != "ok" and is_retire(recs, i, m):
                expected.append("%s timed out as designed and retired the open responses: %s ms "
                                "against %s ms" % (r["name"], m.group(2), m.group(3)))
            elif m.group(1) != "ok":
                reasons.append("%s went over its bound: %s ms against %s ms (%s)" %
                               (r["name"], m.group(2), m.group(3), m.group(4)))
        elif r["kind"] == "WAIT":
            reasons.append("%s was still waiting at t=%.1f: %s" % (r["name"], r["t"], r["detail"]))
        elif r["kind"] == "NOTE" and r["name"] == "rank.exit" and "SPLIT" in r["detail"]:
            reasons.append("a rank left after a split release: " + r["detail"])
        elif r["kind"] == "NOTE" and r["name"] == "stream.probe" and ": drained after" not in r["detail"]:
            reasons.append("a stream was not found drained: " + r["detail"])

    # The order, from the STOP on.
    at = recs.index(stop)
    for kind, name in ORDER[1:]:
        hit = next((i for i in range(at, len(recs))
                    if recs[i]["kind"] == kind and recs[i]["name"].endswith(name)), None)
        if hit is None:
            if closed:
                reasons.append("no %s %s after record %d (the contract's order)" % (kind, name, at))
            break
        at = hit

    exits = [r for r in recs if r["kind"] == "NOTE" and r["name"] == "rank.exit"]
    if tp is not None and closed and len(exits) < tp:
        reasons.append("%d rank exit(s) recorded for tp %d" % (len(exits), tp))

    at_stop = [r["detail"] for r in recs if r["kind"] == "NOTE" and r["name"] == "engine.at_stop"]
    stop_t = stop["t"]
    last = recs[-1]
    summary = {
        "records": len(recs),
        "stop_cause": stop["detail"],
        "stop_clock": stop["date"] + " " + stop["clock"],
        "at_stop": at_stop,
        "stop_to_last_ms": round(last["t"] - stop_t, 1),
        "last_record": "%s %s %s" % (last["kind"], last["name"], last["detail"]),
        "open_steps": ["%s (begun t=%.1f, %.1f ms before the last record)" %
                       (n, b["t"], last["t"] - b["t"]) for n, b in still_open],
        "rank_exits": [r["detail"] for r in exits],
        "steps": [{"name": r["name"], "detail": r["detail"]} for r in recs if r["kind"] == "END"],
        "expected": expected,
    }
    if not closed:
        reasons.insert(0, "the journal ends without its CLOSE: last record %s" % summary["last_record"])
        for s in summary["open_steps"]:
            reasons.append("open step: " + s)
        return "TRUNCATED", reasons, summary
    return ("BOUNDED" if not reasons else "UNBOUNDED"), reasons, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("journal")
    ap.add_argument("--tp", type=int, default=None, help="ranks of the one model served")
    ap.add_argument("--json", action="store_true", help="print the verdict as JSON")
    a = ap.parse_args(argv)
    try:
        with open(a.journal, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError as e:
        print("UNREADABLE: %s" % e)
        return 3
    recs, bad = parse(text)
    verdict, reasons, summary = check(recs, a.tp)
    if bad:
        reasons.append("%d line(s) did not parse" % bad)
    if a.json:
        print(json.dumps({"verdict": verdict, "reasons": reasons, "summary": summary}, indent=1))
    else:
        print(verdict)
        for r in reasons:
            print("  - " + r)
        for e in summary.get("expected", []) if summary else []:
            print("  expected: " + e)
        if summary:
            print("  stop: %s (%s); %s ms to the last record" %
                  (summary.get("stop_cause"), summary.get("stop_clock"), summary.get("stop_to_last_ms")))
            for s in summary.get("at_stop", []):
                print("  at the stop: " + s)
    return {"BOUNDED": 0, "UNBOUNDED": 1, "TRUNCATED": 2, "NOT-STOPPED": 2}.get(verdict, 3)


if __name__ == "__main__":
    sys.exit(main())

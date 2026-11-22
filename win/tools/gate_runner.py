#!/usr/bin/env python3
"""Brilliance: run the release gates against one candidate and write one evidence record.

    python -I win/tools/gate_runner.py --plan PLAN.json --zip <package zip> --out <evidence folder>
        [--inputs INPUTS.json] [--pkg <unpacked package>] [--source <build clone>]
        [--criteria CRITERIA.json] [--allow-cards 3,9] [--solo-window <note>]
        [--build-lease <folder>] [--only ID,ID] [--keep-going] [--resume] [--list]

(win/tools/gate-runner.ps1 is the same command from PowerShell.) How it works, the plan and inputs
formats, the verdicts and the two extension points are in win/tools/GATE-RUNNER.md.

The candidate is the ZIP. Its identity (the ZIP's SHA-256 and MANIFEST.json's source commit) is
established first, and every gate result in the record is tied to it: the runner ran the gate
against it ("run"), a log names it ("verified"), or the inputs file only says so ("declared").
Every gate gets one verdict: pass, fail, skip, carryover, blocked or error. A skip and a carryover
each carry a reason, never pass silently, and only pass or an accepted carryover counts toward a
"ready" record.

Exit status: 0 ready, 1 fail (a gate failed or a gate could not be judged), 2 incomplete (a required
gate was skipped or blocked), 3 a usage error (nothing judged). Standard library only. Every child
process runs without a window. No GPU is touched by the runner itself; what a gate runs is the
plan's, and a gate that names cards runs only on cards --allow-cards leases.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_check_lib

RUNNER_VERSION = "1"
PLAN_SCHEMA = "brilliance-gate-plan/1"
INPUTS_SCHEMA = "brilliance-gate-inputs/1"
RECORD_SCHEMA = "brilliance-gate-record/1"
CRITERIA_SCHEMA = "brilliance-perf-criteria/1"
MEASURE_SCHEMA = "brilliance-perf-measurements/1"

VERDICTS = ("pass", "fail", "skip", "carryover", "blocked", "error")
BINDING_RANK = {"none": 0, "declared": 1, "verified": 2, "run": 3}
KINDS = ("identity", "manifest", "command", "log", "checks", "criteria")
CLASSES = ("C", "D", "F", "M", "S")
CREATE_NO_WINDOW = 0x08000000
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ZIP_NAME_RE = re.compile(r"^Brilliance-(?P<ver>.+)-(?P<short>[0-9a-f]{7,12})-win64\.zip$")


class UsageError(Exception):
    pass


# ----------------------------------------------------------------- small helpers
def now():
    return datetime.datetime.now().astimezone()


def stamp(t):
    """ISO time with the machine's offset, and the 12-hour clock form the reports use."""
    return {"iso": t.isoformat(timespec="seconds"),
            "clock": t.strftime("%I:%M:%S %p").lstrip("0")}


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def read_text(path):
    """A log as text, whatever wrote it: UTF-16 (Windows PowerShell 5.1's redirection), UTF-8 with or
    without a BOM, or a code page (bytes that are not UTF-8 are replaced, never fatal)."""
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = raw.decode("utf-16", errors="replace")
    elif len(raw) >= 4 and raw[1:2] == b"\x00" and raw[3:4] == b"\x00":
        text = raw.decode("utf-16-le", errors="replace")
    else:
        if raw[:3] == b"\xef\xbb\xbf":
            raw = raw[3:]
        text = raw.decode("utf-8", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def load_json(path, what):
    try:
        with open(path, "rb") as f:
            raw = f.read()
        if raw[:3] == b"\xef\xbb\xbf":
            raw = raw[3:]
        return json.loads(raw.decode("utf-8"))
    except (OSError, ValueError) as e:
        raise UsageError("%s %s: %s" % (what, path, e))


def write_json(path, obj):
    data = json.dumps(obj, indent=2, ensure_ascii=False) + "\n"
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(data)
    os.replace(tmp, path)


def evidence_entry(path):
    if path and os.path.isfile(path):
        return {"path": os.path.abspath(path), "sha256": sha256_file(path), "bytes": os.path.getsize(path)}
    return {"path": os.path.abspath(path) if path else path, "missing": True}


def git(repo, *args):
    p = subprocess.run(["git", "-C", repo] + list(args), capture_output=True, text=True,
                       creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


# ----------------------------------------------------------------- the candidate
class Candidate:
    """What every gate is tied to: the ZIP, its SHA-256 and its MANIFEST's identity."""

    def __init__(self, zip_path, pkg, source, out):
        self.zip = os.path.abspath(zip_path)
        self.pkg = os.path.abspath(pkg) if pkg else ""
        self.source = os.path.abspath(source) if source else ""
        self.out = os.path.abspath(out)
        self.zip_name = os.path.basename(self.zip)
        self.zip_sha256 = ""
        self.manifest = None
        self.manifest_name = ""
        self.facts = {}
        self.vars = {}

    @property
    def commit(self):
        return (self.manifest or {}).get("source_commit", "")

    def tokens(self):
        """Placeholders a plan's argv, paths and patterns may use."""
        c = self.commit
        return {"zip": self.zip, "zip_name": self.zip_name, "zip_sha256": self.zip_sha256,
                "pkg": self.pkg, "source": self.source, "out": self.out, "commit": c,
                "commit7": c[:7], "commit12": c[:12],
                "version": (self.manifest or {}).get("brilliance", ""),
                "package": (self.manifest or {}).get("package", ""),
                "python": sys.executable, "repo": REPO,
                **{"var." + k: v for k, v in self.vars.items()}}

    def record(self):
        m = self.manifest or {}
        d = {"zip": self.zip, "zip_name": self.zip_name, "zip_sha256": self.zip_sha256,
             "package": m.get("package"), "version": m.get("brilliance"),
             "source_commit": m.get("source_commit"), "source_dirty": m.get("source_dirty"),
             "upstream": m.get("upstream"), "pkg": self.pkg or None, "source_tree": self.source or None}
        d.update(self.facts)
        return d


def manifest_from_zip(zf):
    names = [n for n in zf.namelist() if n.rsplit("/", 1)[-1] == "MANIFEST.json"]
    names.sort(key=lambda n: n.count("/"))
    if not names:
        return None, None, None
    raw = zf.read(names[0])
    return names[0], raw, json.loads(raw[3:] if raw[:3] == b"\xef\xbb\xbf" else raw)


# ----------------------------------------------------------------- judging text
def expand(s, tok, regex=False):
    """Fill {token} placeholders; in a pattern (regex=True) each value is matched literally."""
    if not isinstance(s, str):
        return s
    for k, v in tok.items():
        s = s.replace("{" + k + "}", re.escape(str(v)) if regex else str(v))
    return s


def judge_text(text, judge, tok, exit_code=None):
    """Apply a plan's judge to a log. Returns (problems, details, counts, skips)."""
    problems, details, counts = [], [], {}
    skips = []
    flags = re.MULTILINE
    want_exit = judge.get("exit")
    if exit_code is not None and want_exit is not None and exit_code != want_exit:
        problems.append("exit %d, expected %d" % (exit_code, want_exit))
    for pat in judge.get("require", []):
        if not re.search(expand(pat, tok, True), text, flags):
            problems.append("missing required line /%s/" % pat)
    allow_after = judge.get("allow_after") or {}
    for pat in judge.get("forbid", []):
        hits = [m for m in re.finditer(expand(pat, tok, True), text, flags)]
        if not hits:
            continue
        rule = allow_after.get(pat)
        if rule:
            first = re.search(expand(rule, tok, True), text, flags)
            bad = [m for m in hits if not first or m.start() < first.start()]
            if bad:
                problems.append("/%s/ %d time(s) before /%s/" % (pat, len(bad), rule))
            else:
                details.append("/%s/ %d time(s), all after /%s/ (allowed)" % (pat, len(hits), rule))
            continue
        line = text[text.rfind("\n", 0, hits[0].start()) + 1:].split("\n", 1)[0].strip()
        problems.append("forbidden /%s/ %d time(s), first: %s" % (pat, len(hits), line[:160]))
    for name, pat in (judge.get("count") or {}).items():
        counts[name] = len(re.findall(expand(pat, tok, True), text, flags))
    for name, rng in (judge.get("expect_counts") or {}).items():
        n = counts.get(name)
        lo, hi = rng[0], rng[1] if len(rng) > 1 else None
        if n is None:
            problems.append("expect_counts names %s, which count does not define" % name)
        elif n < lo or (hi is not None and n > hi):
            problems.append("%s counted %d, expected %s" % (name, n, "%d..%s" % (lo, hi if hi is not None else "")))
    if judge.get("ctest"):
        p, d, c, s = judge_ctest(text, judge)
        problems += p
        details += d
        counts.update(c)
        skips = s
    return problems, details, counts, skips


CTEST_SUMMARY = re.compile(r"^(\d+)% tests passed(?:, (\d+) tests? failed)? out of (\d+)", re.MULTILINE)
CTEST_NOTRUN = re.compile(r"^\s*(\d+) - (\S+) \((Skipped|Not Run|Disabled)\)", re.MULTILINE)
CTEST_FAILED = re.compile(r"^\s*(\d+) - (\S+) \((Failed|Timeout|SEGFAULT|Exception|[^)]*)\)", re.MULTILINE)


def judge_ctest(text, judge):
    problems, details, counts = [], [], {}
    m = list(CTEST_SUMMARY.finditer(text))
    if not m:
        return ["no ctest summary line ('N% tests passed ... out of M')"], [], {}, []
    pct, failed, total = m[-1].group(1), int(m[-1].group(2) or 0), int(m[-1].group(3))
    counts.update({"tests": total, "failed": failed})
    if failed:
        tail = text[m[-1].end():]
        names = [x.group(2) for x in CTEST_FAILED.finditer(tail) if x.group(3) not in ("Skipped", "Not Run", "Disabled")]
        problems.append("%d of %d tests failed: %s" % (failed, total, ", ".join(names) or "(names not in the log)"))
    minimum = judge.get("min_tests")
    if minimum and total < minimum:
        problems.append("%d tests ran, fewer than the plan's %d" % (total, minimum))
    notrun = sorted({x.group(2) for x in CTEST_NOTRUN.finditer(text)})
    counts["skipped"] = len(notrun)
    expected = set(judge.get("expected_skips", []))
    unexpected = [n for n in notrun if n not in expected]
    if unexpected:
        problems.append("skipped and not in the expected list: %s" % ", ".join(unexpected))
    gone = sorted(expected - set(notrun))
    if gone:
        details.append("expected skips that ran this time: %s" % ", ".join(gone))
    skips = [{"test": n, "expected": n in expected} for n in notrun]
    details.append("ctest: %s%% of %d passed, %d failed, %d skipped (%s)"
                   % (pct, total, failed, len(notrun), ", ".join(notrun) or "none"))
    return problems, details, counts, skips


# ----------------------------------------------------------------- the runner
class Runner:
    def __init__(self, args, plan, inputs, plan_path, inputs_path):
        self.a = args
        self.plan = plan
        self.inputs = inputs
        self.plan_path = plan_path
        self.inputs_path = inputs_path
        self.cand = Candidate(args.zip, args.pkg or inputs.get("pkg"), args.source or inputs.get("source"), args.out)
        self.cand.vars = {k: str(v) for k, v in (inputs.get("vars") or {}).items()}
        self.allow_cards = parse_cards(args.allow_cards if args.allow_cards is not None else inputs.get("allow_cards", ""))
        self.solo = args.solo_window or inputs.get("solo_window") or ""
        self.results = []
        self.prev = {}
        self.logdir = os.path.join(self.cand.out, "logs")
        self.plugins_base = os.path.dirname(os.path.abspath(plan_path))
        # hashed once, at the start: the files that drove this run, as they were when it began
        crit = self.criteria_path()
        self.fixed = {"script_sha256": sha256_file(__file__), "plan": evidence_entry(plan_path),
                      "inputs": evidence_entry(inputs_path) if inputs_path else None,
                      "criteria": evidence_entry(crit) if crit else None}

    # --- record
    def record(self, started, finished=None, summary=None):
        return {
            "schema": RECORD_SCHEMA,
            "runner": {"version": RUNNER_VERSION, "script": os.path.abspath(__file__),
                       "script_sha256": self.fixed["script_sha256"],
                       "python": sys.version.split()[0],
                       "plan": self.fixed["plan"], "plan_name": self.plan.get("name"),
                       "inputs": self.fixed["inputs"], "criteria": self.fixed["criteria"],
                       "argv": sys.argv[1:]},
            "started": stamp(started), "finished": stamp(finished) if finished else None,
            "leases": {"cards": sorted(self.allow_cards), "solo_window": self.solo or None,
                       "build_lease": self.a.build_lease or None},
            "candidate": self.cand.record(),
            "gates": self.results,
            "summary": summary,
        }

    def criteria_path(self):
        p = self.a.criteria or self.inputs.get("criteria") or self.plan.get("criteria") or ""
        return os.path.abspath(expand(p, self.cand.tokens())) if p else ""

    # --- main loop
    def run(self):
        os.makedirs(self.logdir, exist_ok=True)
        started = now()
        rec_path = os.path.join(self.cand.out, "evidence.json")
        if self.a.resume and os.path.isfile(rec_path):
            old = load_json(rec_path, "previous record")
            for g in old.get("gates", []):
                self.prev[g["id"]] = (g, old.get("candidate", {}))
        only = set(x.strip() for x in (self.a.only or "").split(",") if x.strip())
        stop_reason = None
        identity_ok = True
        for gate in self.plan["gates"]:
            gid = gate["id"]
            if only and gid not in only and gate["kind"] != "identity":
                r = self.result(gate, "skip", "not selected (--only %s)" % ",".join(sorted(only)))
            elif not identity_ok and gate["kind"] != "identity":
                r = self.result(gate, "blocked", "not started: the candidate's identity was not established")
            elif stop_reason:
                r = self.result(gate, "blocked", "not started: %s" % stop_reason)
            else:
                r = self.one(gate)
            self.results.append(r)
            print("%-10s %-24s %s" % (r["verdict"].upper(), gid, r["reason"]), flush=True)
            if gate["kind"] == "identity" and r["verdict"] != "pass":
                identity_ok = False
            if r["verdict"] in ("fail", "error") and gate.get("required", True) and not self.a.keep_going \
                    and gate["kind"] != "identity":
                stop_reason = "%s %s" % (gid, "failed" if r["verdict"] == "fail" else "could not be judged")
            write_json(rec_path, self.record(started))
        summary = self.summarize()
        finished = now()
        rec = self.record(started, finished, summary)
        write_json(rec_path, rec)
        md = os.path.join(self.cand.out, "EVIDENCE.md")
        with open(md, "w", encoding="utf-8", newline="\n") as f:
            f.write(render_md(rec))
        print("record: %s (%s)" % (rec_path, summary["verdict"]))
        return {"ready": 0, "fail": 1, "incomplete": 2}[summary["verdict"]]

    def summarize(self):
        counts = {v: 0 for v in VERDICTS}
        not_ready, failed = [], []
        for r in self.results:
            counts[r["verdict"]] += 1
            if r["verdict"] in ("fail", "error"):
                failed.append(r["id"])
            elif r["required"] and r["verdict"] not in ("pass", "carryover"):
                not_ready.append("%s: %s (%s)" % (r["id"], r["verdict"], r["reason"]))
            elif r["required"] and r["verdict"] == "pass" \
                    and BINDING_RANK.get(r["binding"], 0) < BINDING_RANK[r["min_binding"]]:
                not_ready.append("%s: bound only as '%s', the plan asks '%s'" % (r["id"], r["binding"], r["min_binding"]))
        verdict = "fail" if failed else ("incomplete" if not_ready else "ready")
        return {"verdict": verdict, "counts": counts, "failed": failed, "not_ready": not_ready,
                "candidate": {"zip_sha256": self.cand.zip_sha256, "source_commit": self.cand.commit}}

    def result(self, gate, verdict, reason, **kw):
        r = {"id": gate["id"], "title": gate.get("title", ""), "kind": gate["kind"],
             "class": gate.get("class", "D"), "required": bool(gate.get("required", True)),
             "verdict": verdict, "reason": reason, "binding": kw.pop("binding", "none"),
             "min_binding": gate.get("min_binding", "verified" if gate["kind"] == "log" else "run"),
             "candidate": {"zip_sha256": self.cand.zip_sha256, "source_commit": self.cand.commit},
             "gate_sha256": sha256_bytes(json.dumps(gate, sort_keys=True).encode()),
             "started": None, "finished": None, "seconds": None, "log": None, "evidence": [],
             "details": [], "counts": {}}
        r.update(kw)
        return r

    def one(self, gate):
        r = self.one_(gate)
        if gate["kind"] == "criteria" and "cases" not in r:
            crit, why = self.load_criteria()
            if crit is not None:
                r["cases"] = criteria_cases(crit)
                r["cards_needed"] = sorted(criteria_buses(crit))
        return r

    def one_(self, gate):
        gid = gate["id"]
        skip = next((s for s in self.inputs.get("skips", []) if s.get("gate") == gid), None)
        carry = next((c for c in self.inputs.get("carryovers", []) if c.get("gate") == gid), None)
        if skip and carry:
            return self.result(gate, "error", "the inputs both skip and carry over this gate")
        if skip:
            if not (skip.get("reason") or "").strip():
                return self.result(gate, "error", "a skip without a reason is refused")
            return self.result(gate, "skip", skip["reason"].strip(), details=["skipped by the inputs file"])
        if carry:
            return self.carryover(gate, carry)
        why = self.preconditions(gate)
        if why:
            return self.result(gate, "skip", why, details=["skipped by the runner: a precondition is not met"])
        reused = self.reuse(gate)
        if reused:
            return reused
        t0 = now()
        try:
            r = getattr(self, "gate_" + gate["kind"])(gate)
        except UsageError as e:
            r = self.result(gate, "error", str(e))
        except Exception as e:  # a gate the runner could not judge is an error, never a pass
            r = self.result(gate, "error", "%s: %s" % (type(e).__name__, e))
        t1 = now()
        r["started"], r["finished"] = stamp(t0), stamp(t1)
        r["seconds"] = round((t1 - t0).total_seconds(), 1)
        return r

    def preconditions(self, gate):
        cls = gate.get("class", "D")
        cards = parse_cards(gate.get("cards", []))
        if gate["kind"] == "criteria":
            crit, _ = self.load_criteria()
            if crit is not None:
                cards |= criteria_buses(crit)
        if cards and not cards <= self.allow_cards:
            return "needs cards on buses %s; this run leases %s" % (
                ",".join(map(str, sorted(cards))), ",".join(map(str, sorted(self.allow_cards))) or "none")
        if cls in ("M", "S") and not self.solo:
            return "class %s runs alone in a solo window the manager records; none given (--solo-window)" % cls
        for need in gate.get("needs", []):
            if need == "source" and not self.cand.source:
                return "needs the source tree (--source: a clone at the snapshot commit)"
            if need == "pkg" and not self.cand.pkg:
                return "needs the unpacked package (--pkg)"
            if need.startswith("var:") and not self.cand.vars.get(need[4:]):
                return "needs vars.%s in the inputs file" % need[4:]
            if need.startswith("input:"):
                key = need[6:]
                if not self.inputs.get(key):
                    return "needs '%s' in the inputs file" % key
        return ""

    def reuse(self, gate):
        prev = self.prev.get(gate["id"])
        if not prev:
            return None
        g, cand = prev
        if g.get("verdict") != "pass" or g.get("gate_sha256") != self.result(gate, "pass", "")["gate_sha256"]:
            return None
        if cand.get("zip_sha256") != self.cand.zip_sha256 or cand.get("source_commit") != self.cand.commit:
            return None
        for e in g.get("evidence", []):
            if e.get("missing") or not os.path.isfile(e["path"]) or sha256_file(e["path"]) != e["sha256"]:
                return None
        g = dict(g)
        g["details"] = list(g.get("details", [])) + ["reused from the previous run's record (--resume): same "
                                                     "candidate, same gate definition, evidence unchanged"]
        g["reused"] = True
        return g

    # --- kinds
    def gate_identity(self, gate):
        c, details, problems = self.cand, [], []
        if not os.path.isfile(c.zip):
            return self.result(gate, "fail", "no ZIP at %s" % c.zip)
        c.zip_sha256 = sha256_file(c.zip)
        c.facts["zip_bytes"] = os.path.getsize(c.zip)
        details.append("ZIP SHA-256 %s (%d bytes)" % (c.zip_sha256, c.facts["zip_bytes"]))
        with zipfile.ZipFile(c.zip) as zf:
            name, raw, m = manifest_from_zip(zf)
        if m is None:
            return self.result(gate, "fail", "the ZIP has no MANIFEST.json", details=details)
        c.manifest, c.manifest_name = m, name
        c.facts["manifest_sha256"] = sha256_bytes(raw)
        if raw[:3] == b"\xef\xbb\xbf":
            problems.append("MANIFEST.json starts with a BOM")
        commit = m.get("source_commit", "")
        if not re.fullmatch(r"[0-9a-f]{40}", commit or ""):
            problems.append("MANIFEST source_commit is not a full commit id: %r" % commit)
        if m.get("source_dirty") is not False:
            problems.append("MANIFEST source_dirty is %r, not false" % m.get("source_dirty"))
        mm = ZIP_NAME_RE.match(c.zip_name)
        if gate.get("check_name", True):
            if not mm:
                problems.append("ZIP name %s is not Brilliance-<version>-<commit>-win64.zip" % c.zip_name)
            else:
                if not commit.startswith(mm.group("short")):
                    problems.append("ZIP name's commit %s is not MANIFEST's %s" % (mm.group("short"), commit[:12]))
                if mm.group("ver") != m.get("brilliance"):
                    problems.append("ZIP name's version %s is not MANIFEST's %s" % (mm.group("ver"), m.get("brilliance")))
                if m.get("package") and m["package"] + ".zip" != c.zip_name:
                    problems.append("MANIFEST package %s does not name this ZIP" % m["package"])
        exp = self.inputs.get("expect", {})
        for key, field in (("commit", "source_commit"), ("version", "brilliance"), ("upstream", "upstream")):
            if exp.get(key) and exp[key] != m.get(field):
                problems.append("expected %s %s, MANIFEST says %s" % (key, exp[key], m.get(field)))
        if exp.get("zip_sha256") and exp["zip_sha256"].lower() != c.zip_sha256:
            problems.append("expected ZIP SHA-256 %s, the file is %s" % (exp["zip_sha256"], c.zip_sha256))
        side = c.zip + ".sha256"
        if os.path.isfile(side):
            t = read_text(side).split()
            ok = bool(t) and t[0].lower() == c.zip_sha256
            c.facts["sidecar_sha256"] = "match" if ok else "MISMATCH"
            if not ok:
                problems.append("%s does not hold the ZIP's hash" % os.path.basename(side))
        else:
            c.facts["sidecar_sha256"] = "none beside the ZIP"
        if c.source:
            rc, head, err = git(c.source, "rev-parse", "HEAD")
            c.facts["source_head"] = head if rc == 0 else None
            if rc != 0:
                problems.append("source tree %s: not a git checkout (%s)" % (c.source, err[:120]))
            elif head != commit:
                problems.append("source tree HEAD %s is not the snapshot commit %s" % (head[:12], commit[:12]))
            else:
                rc, st, _ = git(c.source, "status", "--porcelain", "--untracked-files=no")
                c.facts["source_clean"] = (rc == 0 and st == "")
                if not c.facts["source_clean"]:
                    problems.append("source tree has tracked changes")
                else:
                    details.append("source tree at the snapshot commit, no tracked changes")
        for k in ("private_source", "public_parent", "notes"):
            if self.inputs.get(k):
                c.facts[k] = self.inputs[k]
        details.append("MANIFEST: %s, commit %s, dirty %s, upstream %s"
                       % (m.get("package"), commit[:12], m.get("source_dirty"), m.get("upstream")))
        if problems:
            return self.result(gate, "fail", "; ".join(problems), details=details, binding="run",
                               evidence=[evidence_entry(c.zip)])
        return self.result(gate, "pass", "ZIP %s... = commit %s" % (c.zip_sha256[:16], commit[:12]),
                           details=details, binding="run", evidence=[evidence_entry(c.zip)])

    def gate_manifest(self, gate):
        """Every file in the ZIP (and, given --pkg, in the unpacked folder) matches MANIFEST."""
        c = self.cand
        files = (c.manifest or {}).get("files") or {}
        if not files:
            return self.result(gate, "fail", "MANIFEST lists no files")
        prefix = c.manifest_name[: -len("MANIFEST.json")]
        problems, details = [], []
        with zipfile.ZipFile(c.zip) as zf:
            seen = {}
            for info in zf.infolist():
                if info.is_dir() or not info.filename.startswith(prefix):
                    continue
                rel = info.filename[len(prefix):]
                if rel == "MANIFEST.json":
                    continue
                h = hashlib.sha256()
                with zf.open(info) as f:
                    for b in iter(lambda: f.read(1 << 20), b""):
                        h.update(b)
                seen[rel] = h.hexdigest()
        missing = sorted(set(files) - set(seen))
        extra = sorted(set(seen) - set(files))
        changed = sorted(k for k in files if k in seen and seen[k] != files[k].lower())
        for what, lst in (("listed but not in the ZIP", missing), ("in the ZIP but not listed", extra),
                          ("hash differs", changed)):
            if lst:
                problems.append("%d %s (%s)" % (len(lst), what, ", ".join(lst[:5])))
        details.append("ZIP: %d files, MANIFEST: %d" % (len(seen), len(files)))
        if c.pkg:
            on_disk = {}
            for root, _, names in os.walk(c.pkg):
                for n in names:
                    p = os.path.join(root, n)
                    on_disk[os.path.relpath(p, c.pkg).replace(os.sep, "/")] = p
            pm = os.path.join(c.pkg, "MANIFEST.json")
            if not os.path.isfile(pm) or sha256_file(pm) != c.facts.get("manifest_sha256"):
                problems.append("the unpacked folder's MANIFEST.json is not the ZIP's")
            bad = [k for k in files if k not in on_disk or sha256_file(on_disk[k]) != files[k].lower()]
            if bad:
                problems.append("unpacked %s: %d missing or changed (%s)" % (c.pkg, len(bad), ", ".join(bad[:5])))
            details.append("unpacked folder: %d files on disk, %d listed, %d missing or changed"
                           % (len(on_disk), len(files), len(bad)))
        if problems:
            return self.result(gate, "fail", "; ".join(problems), details=details, binding="run")
        return self.result(gate, "pass", "%d files, every hash matches MANIFEST" % len(files), details=details,
                           binding="run")

    def gate_command(self, gate):
        tok = self.cand.tokens()
        ev = self.inputs.get("evidence", {}).get(gate["id"])
        if ev and gate.get("intake", False):
            return self.intake(gate, ev)
        if ev and not gate.get("intake", False):
            raise UsageError("the inputs give a log for a gate the plan says must run here (intake false)")
        argv = [expand(x, tok) for x in gate["argv"]]
        log = os.path.join(self.logdir, gate["id"] + ".log")
        env = dict(os.environ)
        # Test scratch goes under the evidence folder: the test harness's default is /tmp, outside the
        # project. A plan's own env may name another folder.
        env["RAD_TEST_TMPDIR"] = os.path.join(self.cand.out, "tmp", gate["id"])
        for k, v in (gate.get("env") or {}).items():
            env[k] = expand(v, tok)
        os.makedirs(env["RAD_TEST_TMPDIR"], exist_ok=True)
        cwd = expand(gate.get("cwd", ""), tok) or None
        # No timeout unless the plan sets one, and never on a gate that uses cards: a timeout ends the
        # child process, and a GPU process is never force-ended (its gate stops its own server).
        timeout = None if gate.get("cards") else (gate.get("timeout") or None)
        lease, waited = None, 0.0
        if gate.get("lease") == "build" and self.a.build_lease:
            t_wait = time.time()
            lease, why = take_lease(self.a.build_lease, gate["id"], self.a.lease_wait)
            waited = round(time.time() - t_wait, 1)
            if not lease:
                return self.result(gate, "skip", "build lease not taken: %s" % why, lease_wait_s=waited)
        t_run = now()
        code = None
        try:
            with open(log, "wb") as f:
                head = ("# gate %s\n# argv %s\n# cwd %s\n# started %s\n\n"
                        % (gate["id"], json.dumps(argv), cwd or os.getcwd(), now().isoformat(timespec="seconds")))
                f.write(head.encode("utf-8"))
                f.flush()
                try:
                    p = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                       cwd=cwd, env=env, timeout=timeout,
                                       creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0)
                    code = p.returncode
                except subprocess.TimeoutExpired:
                    # subprocess.run ends the child on a timeout; plans set no timeout on a gate whose
                    # process must never be force-ended (a GPU server): such gates run their own stop.
                    code = None
                except OSError as e:
                    f.write(("\n# could not start: %s\n" % e).encode("utf-8"))
                    return self.result(gate, "error", "could not start %s: %s" % (argv[0], e),
                                       log=evidence_entry(log), binding="run")
        finally:
            if lease:
                release_lease(lease)
        text = read_text(log)
        if code is None:
            return self.result(gate, "fail", "no exit within %s s" % timeout, log=evidence_entry(log), binding="run")
        problems, details, counts, skips = judge_text(text, gate.get("judge", {}), tok, exit_code=code)
        details.insert(0, "exit %d; verdict read from the log" % code)
        run_s = round((now() - t_run).total_seconds(), 1)
        if lease:
            details.insert(1, "build lease: waited %.0f s for it, then held it %.0f s for this command" % (waited, run_s))
        r = dict(log=evidence_entry(log), binding="run", details=details, counts=counts, run_s=run_s)
        if lease:
            r["lease_wait_s"] = waited
        if skips:
            r["skips"] = skips
        if problems:
            return self.result(gate, "fail", "; ".join(problems), **r)
        return self.result(gate, "pass", gate.get("pass_text", "every check in the log passed"), **r)

    def intake(self, gate, ev, for_carry=None):
        """Judge a log made elsewhere (an earlier step of this release, or a carried-over run)."""
        tok = self.cand.tokens()
        paths = ev.get("logs") or ([ev["log"]] if ev.get("log") else [])
        if not paths:
            raise UsageError("evidence for %s names no log" % gate["id"])
        entries, problems, details, counts, skips, texts = [], [], [], {}, [], []
        for p in paths:
            e = evidence_entry(p)
            entries.append(e)
            if e.get("missing"):
                problems.append("no log at %s" % p)
                continue
            texts.append(read_text(p))
        if problems:
            return self.result(gate, "fail", "; ".join(problems), evidence=entries)
        text = "\n".join(texts)
        judge = dict(gate.get("judge", {}))
        judge.pop("exit", None)
        if judge.get("min_logs") and len(texts) < judge["min_logs"]:
            problems.append("%d log(s) given, the plan asks at least %d" % (len(texts), judge["min_logs"]))
        if judge.get("per_log"):
            # every log must pass on its own (one smoke summary per layout, say), not just the whole
            pr, de, co, sk = [], [], {}, []
            for p, t in zip(paths, texts):
                a, b, c, d = judge_text(t, judge, tok)
                pr += ["%s: %s" % (os.path.basename(p), x) for x in a]
                de += ["%s: %s" % (os.path.basename(p), x) for x in b]
                for k, v in c.items():
                    co[k] = co.get(k, 0) + v
                sk += d
        else:
            pr, de, co, sk = judge_text(text, judge, tok)
        problems += pr
        details += de
        counts.update(co)
        skips = sk
        if ev.get("exit") is not None and gate.get("judge", {}).get("exit") is not None \
                and ev["exit"] != gate["judge"]["exit"]:
            problems.append("the inputs record exit %s, expected %s" % (ev["exit"], gate["judge"]["exit"]))
        binding = "declared"
        if for_carry is None:
            binds = [expand(b, tok) for b in gate.get("binds", [])]
            hit = [b for b in binds if b and b in text]
            if hit:
                binding = "verified"
                details.append("the log names the candidate (%s)" % ", ".join(hit))
            elif binds:
                details.append("the log names none of %s: tied to the candidate by the inputs file only"
                               % ", ".join(binds))
            declared = ev.get("for", "")
            if declared and declared not in (self.cand.commit, self.cand.zip_sha256) \
                    and not self.cand.commit.startswith(declared):
                problems.append("the inputs say this log is for %s, not this candidate" % declared)
        r = dict(evidence=entries, binding=binding, details=["intake: verdict read from %d log(s) made outside "
                                                             "this run" % len(entries)] + details, counts=counts)
        if ev.get("note"):
            r["details"].append("note: " + ev["note"])
        if skips:
            r["skips"] = skips
        if problems:
            return self.result(gate, "fail", "; ".join(problems), **r)
        return self.result(gate, "pass", gate.get("pass_text", "every check in the log passed"), **r)

    def gate_log(self, gate):
        ev = self.inputs.get("evidence", {}).get(gate["id"])
        if not ev:
            return self.result(gate, "skip", "no evidence given for this gate (inputs 'evidence.%s')" % gate["id"])
        return self.intake(gate, ev)

    def carryover(self, gate, carry):
        """Evidence from an earlier candidate, accepted for this one only with a component-specific reason."""
        problems = []
        if not gate.get("carryover", True) or gate["kind"] in ("identity", "manifest"):
            return self.result(gate, "error", "the plan does not allow a carryover for this gate")
        reason = (carry.get("reason") or "").strip()
        frm = carry.get("from") or {}
        changed = carry.get("changed_paths")
        if not reason:
            problems.append("no reason")
        if not (frm.get("commit") or frm.get("zip_sha256")):
            problems.append("no 'from' identity (commit or zip_sha256 of the earlier candidate)")
        if changed is None:
            problems.append("no 'changed_paths' (what differs between the two candidates)")
        if problems:
            return self.result(gate, "error", "carryover refused: " + "; ".join(problems))
        if frm.get("commit") == self.cand.commit or (frm.get("zip_sha256") or "").lower() == self.cand.zip_sha256:
            return self.result(gate, "error", "carryover refused: 'from' is this candidate; give its log as evidence instead")
        details = ["carried over from commit %s / ZIP %s" % ((frm.get("commit") or "?")[:12],
                                                            (frm.get("zip_sha256") or "?")[:16]),
                   "reason: " + reason, "changed paths (declared): %s" % (", ".join(changed) or "none")]
        diff = carry.get("diff")
        if diff:
            rc, out, err = git(expand(diff["repo"], self.cand.tokens()), "diff", "--name-only", diff["from"], diff["to"])
            if rc != 0:
                return self.result(gate, "error", "carryover diff could not be read: %s" % err[:160], details=details)
            actual = sorted(x for x in out.splitlines() if x)
            if actual != sorted(changed):
                return self.result(gate, "fail", "carryover's changed_paths %s differ from git diff %s..%s: %s"
                                   % (sorted(changed), diff["from"][:8], diff["to"][:8], actual), details=details)
            details.append("changed paths verified: git diff --name-only %s %s" % (diff["from"][:12], diff["to"][:12]))
        else:
            details.append("changed paths not verified (no 'diff' given)")
        if carry.get("covered_by"):
            details.append("covered by: " + carry["covered_by"])
        if carry.get("approved_by"):
            details.append("approved by: " + carry["approved_by"])
        ev = {"logs": carry.get("evidence") or []}
        if not ev["logs"]:
            return self.result(gate, "error", "carryover refused: no evidence log from the earlier candidate", details=details)
        r = self.intake(gate, ev, for_carry=True)
        details += r["details"]
        if r["verdict"] != "pass":
            return self.result(gate, "fail", "the carried-over evidence does not pass this gate's judge: " + r["reason"],
                               details=details, evidence=r["evidence"])
        return self.result(gate, "carryover", reason, details=details, evidence=r["evidence"],
                           counts=r["counts"], binding="declared", carried_from=frm, skips=r.get("skips", []))

    def gate_checks(self, gate):
        """The plug-in point for checks the runner runs (the licence checks): every *.py in a folder."""
        folder = expand(gate["folder"], self.cand.tokens())
        if not os.path.isabs(folder):
            folder = os.path.join(self.plugins_base, folder)
        ctx = CheckContext(self, gate)
        subs, details = [], []
        evidence = [evidence_entry(gate_check_lib.__file__)]
        for name, path, res in gate_check_lib.run_checks(folder, ctx):
            evidence.append(evidence_entry(path))
            sub = {"check": name, "verdict": res["verdict"], "reason": res.get("reason", ""),
                   "details": res.get("details", [])}
            evidence.extend(evidence_entry(p) for p in res.get("evidence", []))
            subs.append(sub)
            details.append("%s %s: %s" % (sub["verdict"].upper(), sub["check"], sub["reason"]))
        if not subs:
            return self.result(gate, "skip", "no checks registered in %s" % folder)
        verdicts = [s["verdict"] for s in subs]
        kw = dict(details=details, evidence=evidence, checks=subs, binding="run")
        if "error" in verdicts:
            return self.result(gate, "error", "a check could not be judged", **kw)
        if "fail" in verdicts:
            return self.result(gate, "fail", "%d of %d checks failed" % (verdicts.count("fail"), len(subs)), **kw)
        if "skip" in verdicts:
            return self.result(gate, "skip", "%d of %d checks skipped" % (verdicts.count("skip"), len(subs)), **kw)
        return self.result(gate, "pass", "%d checks passed" % len(subs), **kw)

    def gate_criteria(self, gate):
        """The plug-in point for the performance criteria: judge measurements against a criteria file."""
        path = self.criteria_path()
        if not path:
            return self.result(gate, "skip", "no performance criteria file given (--criteria, or 'criteria' in "
                                             "the plan or inputs)")
        crit, why = self.load_criteria()
        if crit is None:
            return self.result(gate, "error", why)
        ev = self.inputs.get("evidence", {}).get(gate["id"]) or {}
        mpath = ev.get("measurements")
        if not mpath:
            return self.result(gate, "skip", "no measurements given (inputs 'evidence.%s.measurements')" % gate["id"],
                               evidence=[evidence_entry(path)])
        meas = load_json(mpath, "measurements")
        evidence = [evidence_entry(path), evidence_entry(mpath)]
        if crit.get("evaluator"):
            ep = crit["evaluator"]
            if not os.path.isabs(ep):
                ep = os.path.join(os.path.dirname(path), ep)
            spec = importlib.util.spec_from_file_location("gate_perf_evaluator", ep)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            evidence.append(evidence_entry(ep))
            res = mod.evaluate(crit, meas, self.cand.record())
        else:
            res = evaluate_criteria(crit, meas, self.cand)
            if crit.get("cases") and res["verdict"] != "skip":
                res = merge_case_checks(res, check_cases(crit, meas))
                for c in (meas.get("cases") or {}).values():
                    for p in (c.get("written_verdict") or {}).get("evidence", []) + c.get("evidence", []):
                        evidence.append(evidence_entry(p))
        v = res["verdict"]
        if v not in ("pass", "fail", "skip"):
            return self.result(gate, "error", "evaluator gave verdict %r" % v, evidence=evidence)
        return self.result(gate, v, res["reason"], details=res.get("details", []), evidence=evidence,
                           binding=res.get("binding", "declared"), cases=criteria_cases(crit, meas),
                           cards_needed=sorted(criteria_buses(crit)))

    def load_criteria(self):
        path = self.criteria_path()
        if not path:
            return None, "no criteria file given"
        if not os.path.isfile(path):
            return None, "criteria file %s does not exist" % path
        crit = load_json(path, "criteria")
        if crit.get("schema") != CRITERIA_SCHEMA:
            return None, "criteria schema %r, expected %s" % (crit.get("schema"), CRITERIA_SCHEMA)
        return crit, ""


class CheckContext:
    """What a check plug-in gets: the candidate, its paths, and helpers that read the ZIP."""

    def __init__(self, runner, gate):
        self.candidate = runner.cand.record()
        self.zip = runner.cand.zip
        self.pkg = runner.cand.pkg
        self.source = runner.cand.source
        self.manifest = runner.cand.manifest or {}
        self.inputs = runner.inputs
        self.options = gate.get("options", {})
        self.out = os.path.join(runner.cand.out, "checks")
        self._prefix = runner.cand.manifest_name[: -len("MANIFEST.json")]
        os.makedirs(self.out, exist_ok=True)

    def zip_names(self):
        with zipfile.ZipFile(self.zip) as zf:
            return [n[len(self._prefix):] for n in zf.namelist() if n.startswith(self._prefix) and not n.endswith("/")]

    def zip_read(self, rel):
        with zipfile.ZipFile(self.zip) as zf:
            return zf.read(self._prefix + rel)

    def source_read(self, rel):
        proc = subprocess.run(["git", "-C", self.source, "cat-file", "blob",
                               self.manifest["source_commit"] + ":" + rel], capture_output=True,
                              creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0)
        if proc.returncode:
            raise FileNotFoundError(rel)
        return proc.stdout


def evaluate_criteria(crit, meas, cand):
    """The default evaluator: each metric against its target, beyond the noise floor."""
    details, failed, missing = [], [], []
    if meas.get("schema") != MEASURE_SCHEMA:
        return {"verdict": "fail", "reason": "measurements schema %r, expected %s" % (meas.get("schema"), MEASURE_SCHEMA)}
    mc = meas.get("candidate") or {}
    if mc.get("zip_sha256", "").lower() != cand.zip_sha256:
        return {"verdict": "fail", "reason": "the measurements are for ZIP %s, not this candidate"
                % (mc.get("zip_sha256") or "(none named)")[:16]}
    floor_all = float(crit.get("noise_floor_pct", 0))
    values = meas.get("metrics") or {}
    for m in crit.get("metrics", []):
        mid, better = m["id"], m.get("better", "higher")
        if m.get("target") is None:
            failed.append("%s has no target in the criteria" % mid)
            continue
        target = float(m["target"])
        floor = float(m.get("noise_floor_pct", floor_all))
        got = values.get(mid)
        if got is None:
            (missing if m.get("required", True) else details).append(mid if m.get("required", True) else
                                                                      "%s: not measured (optional)" % mid)
            continue
        val = float(got["value"] if isinstance(got, dict) else got)
        spread = float(got.get("spread_pct", 0)) if isinstance(got, dict) else 0.0
        if spread > floor > 0:
            failed.append("%s spread %.1f%% exceeds the noise floor %.1f%%" % (mid, spread, floor))
            continue
        delta = (val - target) / target * 100.0 if target else 0.0
        ok = val >= target if better == "higher" else val <= target
        near = abs(delta) <= floor
        line = "%s: %g %s against target %g (%+.1f%%, floor %.1f%%)" % (mid, val, m.get("unit", ""), target, delta, floor)
        if ok:
            details.append(line + ": met")
        elif near:
            details.append(line + ": within the noise floor, met")
        else:
            failed.append(line)
    if missing:
        failed.append("not measured: " + ", ".join(missing))
    if failed:
        return {"verdict": "fail", "reason": "; ".join(failed), "details": details, "binding": "verified"}
    return {"verdict": "pass", "reason": "%d metrics met their targets" % len(crit.get("metrics", [])),
            "details": details, "binding": "verified"}


# ----------------------------------------------------------------- criteria cases
GATING_MODES = ("measured", "claim", "screen", "matched")


def case_buses(crit, case):
    """The card set a case's profile names, and its buses (None when the criteria do not say)."""
    prof = (crit.get("profiles") or {}).get(case.get("profile"), {})
    sets = (crit.get("run_conditions") or {}).get("card_sets") or {}
    name = prof.get("card_set")
    return name, (sorted(int(b) for b in sets[name]) if name in sets else None)


def criteria_buses(crit):
    """Every bus a gating case (one that feeds a runner metric) runs on: the lease the gate needs."""
    out = set()
    for c in crit.get("cases") or []:
        if c.get("mode") in GATING_MODES:
            out |= set(case_buses(crit, c)[1] or [])
    return out


def criteria_cases(crit, meas=None):
    """Each case of the criteria with its bus set, for the record; with what the measurements say."""
    mc = (meas or {}).get("cases") or {}
    rows = []
    for c in crit.get("cases") or []:
        name, buses = case_buses(crit, c)
        row = {"id": c["id"], "profile": c.get("profile"), "protocol": c.get("protocol"), "mode": c.get("mode"),
               "comparison": c.get("comparison"), "pipeline": c.get("pipeline"), "card_set": name, "buses": buses,
               "gating": c.get("mode") in GATING_MODES}
        if meas is not None:
            got = mc.get(c["id"]) or {}
            row["measured_buses"] = got.get("buses")
            if got.get("rank_buses"):
                row["measured_rank_buses"] = got["rank_buses"]
            wv = got.get("written_verdict")
            if wv:
                row["written_verdict"] = {"reviewer": wv.get("reviewer"), "reason": wv.get("reason")}
            if got.get("baseline"):
                row["baseline"] = {"zip_sha256": got["baseline"].get("zip_sha256")}
        rows.append(row)
    return rows


def check_cases(crit, meas):
    """What the criteria's cases ask beyond the numbers: the bus set each measured case ran on, the
    tuning, and a matched previous-release baseline or a written verdict where the mode asks one."""
    problems, details = [], []
    metrics = meas.get("metrics") or {}
    mc = meas.get("cases") or {}
    prev = crit.get("previous_release") or {}
    want_tuning = (crit.get("run_conditions") or {}).get("tuning") or {}
    got_tuning = (meas.get("conditions") or {}).get("tuning")
    if want_tuning:
        if got_tuning is None:
            problems.append("the measurements record no tuning (conditions.tuning)")
        else:
            bad = ["%s %s, criteria %s" % (k, got_tuning.get(k), v) for k, v in want_tuning.items()
                   if got_tuning.get(k) != v]
            if bad:
                problems.append("tuning differs: " + "; ".join(bad))
            else:
                details.append("tuning as the criteria: " + ", ".join("%s %s" % kv for kv in want_tuning.items()))
    for c in crit.get("cases") or []:
        cid, mode = c["id"], c.get("mode")
        if mode not in GATING_MODES:
            continue
        name, buses = case_buses(crit, c)
        fed = [k for k in metrics if k.startswith(cid + ".")]
        got = mc.get(cid) or {}
        if not fed and not got:
            continue  # nothing measured: the case's required metrics fail in the numeric check
        if buses is None:
            problems.append("%s: profile %s has no card set in the criteria" % (cid, c.get("profile")))
        elif got.get("buses") is None:
            problems.append("%s: the measurements do not record its bus set (cases.%s.buses; expected %s)"
                            % (cid, cid, ",".join(map(str, buses))))
        elif sorted(int(b) for b in got["buses"]) != buses:
            problems.append("%s: ran on buses %s; the criteria's %s set is %s"
                            % (cid, ",".join(map(str, got["buses"])), name, ",".join(map(str, buses))))
        else:
            details.append("%s: buses %s (%s)" % (cid, ",".join(map(str, buses)), name))
        wv = got.get("written_verdict")
        if wv is not None:
            missing = [k for k in ("reviewer", "reason", "evidence") if not wv.get(k)]
            if missing:
                problems.append("%s: written verdict without %s" % (cid, ", ".join(missing)))
            else:
                gone = [p for p in wv["evidence"] if not os.path.isfile(p)]
                if gone:
                    problems.append("%s: written verdict evidence missing: %s" % (cid, ", ".join(gone)))
        vm = metrics.get(cid + ".verdict_pass")
        vval = vm.get("value") if isinstance(vm, dict) else vm
        if mode == "matched" and vval is not None:
            base = got.get("baseline")
            if base:
                if (base.get("zip_sha256") or "").lower() != (prev.get("zip_sha256") or "").lower():
                    problems.append("%s: the matched baseline is ZIP %s, not the previous release's %s"
                                    % (cid, (base.get("zip_sha256") or "none")[:16], (prev.get("zip_sha256") or "?")[:16]))
                elif not base.get("evidence"):
                    problems.append("%s: the matched baseline names no raw runs (baseline.evidence)" % cid)
                else:
                    details.append("%s: matched baseline on the previous release (%s)" % (cid, prev.get("tag")))
            elif not wv:
                problems.append("%s: mode matched needs a matched previous-release baseline (cases.%s.baseline) "
                                "or a written verdict for the gap (cases.%s.written_verdict)" % (cid, cid, cid))
            else:
                details.append("%s: no matched baseline; written verdict by %s: %s"
                               % (cid, wv.get("reviewer"), wv.get("reason")))
        if mode == "screen" and vval is not None and not wv:
            problems.append("%s: mode screen needs a written verdict (cases.%s.written_verdict)" % (cid, cid))
        if mode == "claim" and vval is not None and not (got.get("evidence") or wv):
            problems.append("%s: a claim verdict names no raw evidence (cases.%s.evidence)" % (cid, cid))
    return problems, details


def merge_case_checks(res, cc):
    problems, details = cc
    out = dict(res)
    out["details"] = list(res.get("details", [])) + details
    if problems:
        reasons = [] if res["verdict"] == "pass" else [res["reason"]]
        out["verdict"] = "fail"
        out["reason"] = "; ".join(reasons + problems)
    return out


# ----------------------------------------------------------------- leases
def take_lease(folder, gate_id, wait_s):
    deadline = time.time() + max(0, wait_s)
    while True:
        try:
            os.mkdir(folder)
            with open(os.path.join(folder, "holder.txt"), "w", encoding="utf-8", newline="\n") as f:
                f.write("gate-runner %s %s\n" % (gate_id, now().strftime("%I:%M %p").lstrip("0")))
            return folder, ""
        except FileExistsError:
            if time.time() >= deadline:
                holder = ""
                try:
                    holder = read_text(os.path.join(folder, "holder.txt")).strip()
                except OSError:
                    pass
                return None, "%s is held%s" % (folder, " by: " + holder if holder else "")
            time.sleep(min(30, max(1, deadline - time.time())))


def release_lease(folder):
    try:
        os.remove(os.path.join(folder, "holder.txt"))
    except OSError:
        pass
    os.rmdir(folder)


def parse_cards(v):
    if isinstance(v, (list, tuple, set)):
        return {int(x) for x in v}
    return {int(x) for x in str(v).replace(" ", "").split(",") if x}


# ----------------------------------------------------------------- the readable record
def render_md(rec):
    c, s = rec["candidate"], rec["summary"]
    L = ["# Gate record: %s" % (c.get("package") or c.get("zip_name")), "",
         "| | |", "|---|---|",
         "| Verdict | **%s** |" % s["verdict"].upper(),
         "| ZIP | `%s` |" % c.get("zip_name"),
         "| ZIP SHA-256 | `%s` |" % c.get("zip_sha256"),
         "| Source snapshot | `%s` (dirty: %s) |" % (c.get("source_commit"), c.get("source_dirty")),
         "| Version | %s |" % c.get("version"),
         "| Plan | %s (`%s`) |" % (rec["runner"].get("plan_name"), rec["runner"]["plan"].get("sha256", "")[:16]),
         "| Run | %s to %s |" % (rec["started"]["clock"], (rec.get("finished") or {}).get("clock", "?")),
         "| Leases | cards %s; solo window: %s |" % (",".join(map(str, rec["leases"]["cards"])) or "none",
                                                     rec["leases"]["solo_window"] or "none"),
         "", "| Gate | Verdict | Bound | Reason |", "|---|---|---|---|"]
    for g in rec["gates"]:
        L.append("| %s%s | %s | %s | %s |" % (g["id"], "" if g["required"] else " (optional)", g["verdict"].upper(),
                                              g["binding"], g["reason"].replace("|", "\\|")))
    if s["failed"] or s["not_ready"]:
        L += ["", "## Not ready", ""]
        L += ["- failed: %s" % x for x in s["failed"]]
        L += ["- %s" % x for x in s["not_ready"]]
    L += ["", "## Details", ""]
    for g in rec["gates"]:
        L.append("### %s: %s" % (g["id"], g.get("title") or ""))
        L.append("")
        L.append("- verdict **%s** (class %s, %s): %s" % (g["verdict"], g["class"],
                                                          "required" if g["required"] else "optional", g["reason"]))
        if g.get("seconds") is not None:
            L.append("- ran %s to %s (%.1f s%s)" % (g["started"]["clock"], g["finished"]["clock"], g["seconds"],
                                                    ", of it %.0f s waiting for the build lease" % g["lease_wait_s"]
                                                    if g.get("lease_wait_s") else ""))
        if g.get("log"):
            L.append("- log `%s` sha256 `%s`" % (g["log"].get("path"), (g["log"].get("sha256") or "missing")[:16]))
        for e in g.get("evidence", []):
            L.append("- evidence `%s` sha256 `%s`" % (e.get("path"), (e.get("sha256") or "MISSING")[:16]))
        for d in g.get("details", []):
            L.append("- " + d)
        if g.get("cases"):
            L += ["", "Cases, with the bus set the criteria give each (this gate leases %s):"
                  % ",".join(map(str, g.get("cards_needed", []))),
                  "", "| Case | Mode | Profile | Buses |", "|---|---|---|---|"]
            for c in g["cases"]:
                measured = " (measured on %s)" % ",".join(map(str, c["measured_buses"])) if c.get("measured_buses") else ""
                L.append("| %s | %s | %s | %s%s |" % (c["id"], c.get("mode"), c.get("profile"),
                                                     ",".join(map(str, c.get("buses") or [])) or "?", measured))
        L.append("")
    return "\n".join(L)


# ----------------------------------------------------------------- the plan
def check_plan(plan):
    if plan.get("schema") != PLAN_SCHEMA:
        raise UsageError("plan schema %r, expected %s" % (plan.get("schema"), PLAN_SCHEMA))
    gates = plan.get("gates")
    if not gates or gates[0].get("kind") != "identity":
        raise UsageError("a plan's first gate is the identity gate (kind 'identity')")
    seen = set()
    for g in gates:
        if g.get("id") in seen or not g.get("id"):
            raise UsageError("gate id %r missing or repeated" % g.get("id"))
        seen.add(g["id"])
        if g.get("kind") not in KINDS:
            raise UsageError("gate %s: kind %r is not one of %s" % (g["id"], g.get("kind"), ", ".join(KINDS)))
        if g.get("class", "D") not in CLASSES:
            raise UsageError("gate %s: class %r is not one of %s" % (g["id"], g.get("class"), ", ".join(CLASSES)))
        if g.get("min_binding", "run") not in BINDING_RANK:
            raise UsageError("gate %s: min_binding %r" % (g["id"], g.get("min_binding")))
        if g["kind"] == "command" and not g.get("argv"):
            raise UsageError("gate %s: a command gate needs argv" % g["id"])
        if g["kind"] == "checks" and not g.get("folder"):
            raise UsageError("gate %s: a checks gate needs folder" % g["id"])
        for pat in g.get("judge", {}).get("require", []) + g.get("judge", {}).get("forbid", []):
            re.compile(pat)
    return gates


def check_inputs(inputs, gates):
    if inputs and inputs.get("schema") != INPUTS_SCHEMA:
        raise UsageError("inputs schema %r, expected %s" % (inputs.get("schema"), INPUTS_SCHEMA))
    ids = {g["id"] for g in gates}
    for key in ("skips", "carryovers"):
        for e in inputs.get(key, []):
            if e.get("gate") not in ids:
                raise UsageError("inputs %s names gate %r, which the plan does not have" % (key, e.get("gate")))
    for gid in inputs.get("evidence", {}):
        if gid not in ids:
            raise UsageError("inputs evidence names gate %r, which the plan does not have" % gid)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plan", required=True, help="the gate plan (JSON, %s)" % PLAN_SCHEMA)
    ap.add_argument("--zip", required=True, help="the candidate package ZIP")
    ap.add_argument("--out", required=True, help="the evidence folder (evidence.json, EVIDENCE.md, logs/)")
    ap.add_argument("--inputs", default="", help="this candidate's evidence, skips and carryovers (JSON)")
    ap.add_argument("--pkg", default="", help="a fresh unpack of the ZIP")
    ap.add_argument("--source", default="", help="a clone at the snapshot commit (the build clone)")
    ap.add_argument("--criteria", default="", help="the performance criteria file (JSON, %s)" % CRITERIA_SCHEMA)
    ap.add_argument("--allow-cards", default=None, help="PCI buses this run holds, e.g. 3,9 (default: none)")
    ap.add_argument("--solo-window", default="", help="the manager's note of the solo window (M and S gates)")
    ap.add_argument("--build-lease", default="", help="a lease folder taken (mkdir) around gates marked lease: build")
    ap.add_argument("--lease-wait", type=int, default=0, help="seconds to wait for the build lease (default 0)")
    ap.add_argument("--only", default="", help="run these gate ids; the rest are recorded as skipped")
    ap.add_argument("--keep-going", action="store_true", help="judge every gate after a failure")
    ap.add_argument("--resume", action="store_true", help="reuse passes in --out's record for the same candidate")
    ap.add_argument("--list", action="store_true", help="print the plan's gates and exit")
    a = ap.parse_args(argv)
    try:
        plan = load_json(a.plan, "plan")
        gates = check_plan(plan)
        inputs = load_json(a.inputs, "inputs") if a.inputs else {}
        check_inputs(inputs, gates)
        if a.list:
            for g in gates:
                print("%-24s %-9s class %s%s  %s" % (g["id"], g["kind"], g.get("class", "D"),
                                                     "" if g.get("required", True) else " (optional)", g.get("title", "")))
            return 0
        os.makedirs(a.out, exist_ok=True)
        if os.name == "nt":
            try:  # below normal, as the other release tools
                import ctypes
                ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
            except Exception:
                pass
        return Runner(a, plan, inputs, os.path.abspath(a.plan), os.path.abspath(a.inputs) if a.inputs else "").run()
    except UsageError as e:
        print("gate_runner: " + str(e), file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())

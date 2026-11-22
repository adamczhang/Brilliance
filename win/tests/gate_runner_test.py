#!/usr/bin/env python3
"""win/tools/gate_runner.py with fixture candidates: pass, fail, skip and carryover paths.

    python -I win/tests/gate_runner_test.py [--tmp DIR]

Builds small fixture packages (a ZIP with MANIFEST.json, its unpack, logs, check plug-ins and
criteria files) in a fresh folder under --tmp (default: RAD_TEST_TMPDIR, else the build's tmp beside
this checkout), runs the runner on them as a child process, and reads the evidence record it writes.
No GPU, no server, no network. Prints "ok"/"FAIL" per check; exit 0 when every check passed.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(os.path.dirname(HERE), "tools", "gate_runner.py")
FAILS = []


def check(what, cond, detail=""):
    if cond:
        print("  ok    " + what)
    else:
        FAILS.append(what)
        print("  FAIL  %s%s" % (what, ": " + detail if detail else ""))


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return path


def wjson(path, obj):
    return write(path, json.dumps(obj, indent=2))


COMMIT = "1234567abcdef0123456789abcdef0123456789a"
OLD_COMMIT = "fedcba9876543210fedcba9876543210fedcba98"


def make_candidate(root, commit=COMMIT, version="9.9.9-test", tamper=False, sidecar=True):
    """A ZIP shaped like a package: <name>/MANIFEST.json and a few files; and its unpack."""
    name = "Brilliance-%s-%s-win64" % (version, commit[:7])
    files = {"bin/radiance.exe": b"not really an exe\n", "serve.ps1": b"# launcher\n",
             "share/doc/radiance/NOTICE": b"notice\n"}
    manifest = {"package": name, "brilliance": version, "source_commit": commit, "source_dirty": False,
                "upstream": "radiance v0.0.0 test",
                "files": {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}}
    mraw = json.dumps(manifest, indent=2).encode()
    os.makedirs(root, exist_ok=True)
    zpath = os.path.join(root, name + ".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for k, v in files.items():
            z.writestr(name + "/" + k, v + (b"tampered" if tamper and k == "serve.ps1" else b""))
        z.writestr(name + "/MANIFEST.json", mraw)
    pkg = os.path.join(root, "pkg")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(pkg)
    pkg = os.path.join(pkg, name)
    digest = hashlib.sha256(open(zpath, "rb").read()).hexdigest()
    if sidecar:
        write(zpath + ".sha256", "%s  %s\n" % (digest, os.path.basename(zpath)))
    return zpath, pkg, digest


PY = sys.executable


def cmd_gate(gid, script, judge, **kw):
    g = {"id": gid, "title": gid, "kind": "command", "class": "C", "argv": [PY, "-I", "-c", script],
         "judge": judge}
    g.update(kw)
    return g


SUITE_LOG = """Test project
100% tests passed out of 12

The following tests did not run:
\t 48 - host_experts_test (Skipped)
\t 85 - vhead_test (Skipped)
"""


def base_plan(extra):
    return {"schema": "brilliance-gate-plan/1", "name": "fixture",
            "gates": [{"id": "identity", "kind": "identity", "title": "candidate identity"},
                      {"id": "manifest", "kind": "manifest", "title": "files against MANIFEST"}] + extra}


def run(tmp, tag, plan, inputs=None, zpath=None, pkg=None, extra=()):
    d = os.path.join(tmp, tag)
    os.makedirs(d, exist_ok=True)
    pp = wjson(os.path.join(d, "plan.json"), plan)
    argv = [PY, "-I", RUNNER, "--plan", pp, "--zip", zpath, "--out", os.path.join(d, "out")]
    if pkg:
        argv += ["--pkg", pkg]
    if inputs is not None:
        inputs = dict(inputs)
        inputs.setdefault("schema", "brilliance-gate-inputs/1")
        argv += ["--inputs", wjson(os.path.join(d, "inputs.json"), inputs)]
    argv += list(extra)
    p = subprocess.run(argv, capture_output=True, text=True)
    rec = None
    rp = os.path.join(d, "out", "evidence.json")
    if os.path.isfile(rp):
        rec = json.load(open(rp, encoding="utf-8"))
    return p, rec


def gate(rec, gid):
    return next((g for g in rec["gates"] if g["id"] == gid), None) if rec else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmp", default=os.environ.get("RAD_TEST_TMPDIR", ""))
    a = ap.parse_args()
    base = a.tmp or os.path.join(os.path.dirname(os.path.dirname(HERE)), "build-win", "host", "tmp")
    tmp = os.path.join(base, "gate-runner-test-" + uuid.uuid4().hex[:8])
    os.makedirs(tmp)
    print("fixtures in " + tmp)
    zpath, pkg, digest = make_candidate(os.path.join(tmp, "cand"))
    logs = os.path.join(tmp, "logs")
    good_log = write(os.path.join(logs, "suite-good.log"), SUITE_LOG)
    old_log = write(os.path.join(logs, "suite-old.log"), SUITE_LOG)
    bad_log = write(os.path.join(logs, "suite-bad.log"), SUITE_LOG.replace("100% tests passed out of 12",
                                                                             "91% tests passed, 1 tests failed out of 12")
                    + "\nThe following tests FAILED:\n\t 7 - terms_test (Failed)\n")
    surprise_log = write(os.path.join(logs, "suite-surprise.log"),
                         SUITE_LOG + "\t 99 - new_device_test (Skipped)\n")
    bound_log = write(os.path.join(logs, "pkgcheck.log"),
                      "PASS unzip %s\nPASS serve\nevery check passed\n" % os.path.basename(zpath))
    suite_gate = {"id": "suite", "kind": "log", "class": "C", "title": "host suite",
                  "min_binding": "declared",
                  "judge": {"ctest": True, "expected_skips": ["host_experts_test", "vhead_test"]}}
    pkgcheck_gate = {"id": "pkgcheck", "kind": "log", "class": "C", "binds": ["{zip_name}"],
                     "judge": {"require": ["^every check passed$"], "forbid": ["^FAIL "]}}

    # 1. the pass path ---------------------------------------------------------------------------
    print("1. pass: identity, manifest, a command, a bound log, a ctest log with expected skips")
    plan = base_plan([cmd_gate("docs", "print('docs_check: 9 release documents; 0 errors, 0 warnings')",
                               {"exit": 0, "require": [r"0 errors, 0 warnings$"], "forbid": ["^error: "]}),
                      pkgcheck_gate, suite_gate])
    p, rec = run(tmp, "pass", plan, {"evidence": {"pkgcheck": {"log": bound_log},
                                                  "suite": {"log": good_log, "for": COMMIT[:12]}}},
                 zpath, pkg)
    check("exit 0 (ready)", p.returncode == 0, p.stdout + p.stderr)
    check("record verdict ready", rec and rec["summary"]["verdict"] == "ready", rec and str(rec["summary"]))
    check("record carries the ZIP SHA-256", rec and rec["candidate"]["zip_sha256"] == digest)
    check("record carries the snapshot commit", rec and rec["candidate"]["source_commit"] == COMMIT)
    check("every gate tied to the same identity",
          rec and all(g["candidate"] == {"zip_sha256": digest, "source_commit": COMMIT} for g in rec["gates"]))
    check("sidecar .sha256 matched", rec and rec["candidate"].get("sidecar_sha256") == "match")
    check("unpacked folder checked", rec and any("unpacked folder" in d for d in gate(rec, "manifest")["details"]))
    check("command gate bound 'run', its log hashed",
          rec and gate(rec, "docs")["binding"] == "run" and len(gate(rec, "docs")["log"]["sha256"]) == 64)
    check("log naming the ZIP is bound 'verified'", rec and gate(rec, "pkgcheck")["binding"] == "verified")
    check("expected ctest skips recorded by name",
          rec and [s["test"] for s in gate(rec, "suite").get("skips", [])] == ["host_experts_test", "vhead_test"])
    check("EVIDENCE.md written", os.path.isfile(os.path.join(tmp, "pass", "out", "EVIDENCE.md")))

    # 2. fail paths ------------------------------------------------------------------------------
    print("2. fail: a FAIL line under exit 0, a failed ctest, a non-zero exit, later gates blocked")
    plan = base_plan([cmd_gate("quiet", "print('PASS a'); print('FAIL b'); print('every check passed')",
                               {"exit": 0, "require": ["^every check passed$"], "forbid": ["^FAIL "]}),
                      cmd_gate("later", "print('ok')", {"exit": 0})])
    p, rec = run(tmp, "fail-quiet", plan, None, zpath)
    check("exit 1 (fail)", p.returncode == 1, p.stdout)
    check("FAIL line under exit 0 fails the gate", rec and gate(rec, "quiet")["verdict"] == "fail",
          rec and gate(rec, "quiet")["reason"])
    check("the next gate is blocked, named", rec and gate(rec, "later")["verdict"] == "blocked"
          and "quiet" in gate(rec, "later")["reason"])
    p, rec = run(tmp, "fail-keep", plan, None, zpath, extra=["--keep-going"])
    check("--keep-going judges the next gate", rec and gate(rec, "later")["verdict"] == "pass")
    plan = base_plan([cmd_gate("exit3", "print('every check passed'); raise SystemExit(3)",
                               {"exit": 0, "require": ["every check passed"]})])
    p, rec = run(tmp, "fail-exit", plan, None, zpath)
    check("a PASS log with exit 3 fails", rec and gate(rec, "exit3")["verdict"] == "fail"
          and "exit 3" in gate(rec, "exit3")["reason"])
    plan = base_plan([suite_gate])
    p, rec = run(tmp, "fail-suite", plan, {"evidence": {"suite": {"log": bad_log}}}, zpath)
    check("a failed ctest test fails the gate, named", rec and gate(rec, "suite")["verdict"] == "fail"
          and "terms_test" in gate(rec, "suite")["reason"], rec and gate(rec, "suite")["reason"])
    p, rec = run(tmp, "fail-surprise", plan, {"evidence": {"suite": {"log": surprise_log}}}, zpath)
    check("an unexpected ctest skip fails the gate", rec and gate(rec, "suite")["verdict"] == "fail"
          and "new_device_test" in gate(rec, "suite")["reason"])
    p, rec = run(tmp, "fail-for", plan, {"evidence": {"suite": {"log": good_log, "for": OLD_COMMIT}}}, zpath)
    check("a log the inputs give for another candidate fails", rec and gate(rec, "suite")["verdict"] == "fail")
    tz, tpkg, _ = make_candidate(os.path.join(tmp, "tampered"), tamper=True)
    p, rec = run(tmp, "fail-manifest", base_plan([]), None, tz, tpkg)
    check("a changed file fails the manifest gate", rec and gate(rec, "manifest")["verdict"] == "fail"
          and "serve.ps1" in gate(rec, "manifest")["reason"])
    p, rec = run(tmp, "fail-identity", base_plan([cmd_gate("x", "print(1)", {"exit": 0})]),
                 {"expect": {"commit": OLD_COMMIT}}, zpath)
    check("an identity mismatch fails identity", rec and gate(rec, "identity")["verdict"] == "fail")
    check("... and blocks every other gate", rec and all(g["verdict"] == "blocked" for g in rec["gates"][1:]))
    check("... exit 1", p.returncode == 1)

    # 3. skip paths ------------------------------------------------------------------------------
    print("3. skip: by the inputs with a reason, by card lease, by run class, by missing input; never a pass")
    plan = base_plan([{"id": "smoke", "kind": "log", "class": "F", "cards": [3, 9], "title": "card smoke"},
                      {"id": "bench", "kind": "log", "class": "M", "title": "bench"},
                      {"id": "walk", "kind": "log", "class": "F", "cards": [32], "required": False},
                      {"id": "notes", "kind": "log", "class": "D"}])
    p, rec = run(tmp, "skip", plan, {"skips": [{"gate": "notes", "reason": "not written yet"}]}, zpath,
                 extra=["--allow-cards", "32"])
    check("exit 2 (incomplete)", p.returncode == 2, p.stdout + p.stderr)
    check("verdict incomplete, not ready", rec and rec["summary"]["verdict"] == "incomplete")
    check("a gate needing unleased cards is skipped with the buses named",
          rec and gate(rec, "smoke")["verdict"] == "skip" and "3,9" in gate(rec, "smoke")["reason"])
    check("an M gate without a solo window is skipped", rec and gate(rec, "bench")["verdict"] == "skip"
          and "solo window" in gate(rec, "bench")["reason"])
    check("a log gate with no evidence is skipped", rec and gate(rec, "walk")["verdict"] == "skip")
    check("the inputs' skip keeps its reason", rec and gate(rec, "notes")["reason"] == "not written yet")
    check("every required skip is listed as not ready",
          rec and sorted(x.split(":")[0] for x in rec["summary"]["not_ready"]) == ["bench", "notes", "smoke"])
    p, rec = run(tmp, "skip-noreason", plan, {"skips": [{"gate": "notes", "reason": "  "}]}, zpath)
    check("a skip without a reason is an error", rec and gate(rec, "notes")["verdict"] == "error")
    check("... and the record fails", p.returncode == 1)
    plan = base_plan([cmd_gate("docs", "print('ok')", {"exit": 0}, needs=["source"])])
    p, rec = run(tmp, "skip-source", plan, None, zpath)
    check("a gate needing --source is skipped, saying so", rec and gate(rec, "docs")["verdict"] == "skip"
          and "--source" in gate(rec, "docs")["reason"])
    p, rec = run(tmp, "skip-only", base_plan([cmd_gate("a", "print(1)", {"exit": 0}),
                                              cmd_gate("b", "print(1)", {"exit": 0})]), None, zpath,
                 extra=["--only", "a"])
    check("--only records the others as skipped", rec and gate(rec, "b")["verdict"] == "skip"
          and gate(rec, "a")["verdict"] == "pass" and p.returncode == 2)

    # 4. carryover paths -------------------------------------------------------------------------
    print("4. carryover: accepted with a reason and passing evidence; refused otherwise")
    plan = base_plan([suite_gate])
    good_carry = {"gate": "suite", "reason": "the new candidate changes serve.ps1 only; its tests passed on the fix",
                  "from": {"commit": OLD_COMMIT}, "changed_paths": ["win/serve.ps1"], "evidence": [old_log],
                  "approved_by": "the manager (fixture)"}
    p, rec = run(tmp, "carry", plan, {"carryovers": [good_carry]}, zpath)
    g = gate(rec, "suite")
    check("verdict carryover, record ready", g and g["verdict"] == "carryover" and p.returncode == 0,
          g and g["reason"])
    check("carryover keeps reason, source and evidence hash",
          g and g["reason"].startswith("the new candidate") and g["carried_from"]["commit"] == OLD_COMMIT
          and len(g["evidence"][0]["sha256"]) == 64)
    check("carryover is counted as such, not as a pass",
          rec and rec["summary"]["counts"]["carryover"] == 1 and rec["summary"]["counts"]["pass"] == 2)
    bad = dict(good_carry, evidence=[bad_log])
    p, rec = run(tmp, "carry-bad", plan, {"carryovers": [bad]}, zpath)
    check("carried-over evidence that fails the judge fails", gate(rec, "suite")["verdict"] == "fail")
    for tag, change, want in (("noreason", {"reason": ""}, "no reason"),
                              ("nofrom", {"from": {}}, "'from'"),
                              ("nopaths", {"changed_paths": None}, "changed_paths"),
                              ("self", {"from": {"commit": COMMIT}}, "this candidate"),
                              ("noev", {"evidence": []}, "no evidence")):
        c = dict(good_carry)
        c.update(change)
        if change.get("changed_paths", 1) is None:
            del c["changed_paths"]
        p, rec = run(tmp, "carry-" + tag, plan, {"carryovers": [c]}, zpath)
        g = gate(rec, "suite")
        check("carryover refused: %s" % tag, g and g["verdict"] == "error" and want in g["reason"],
              g and g["reason"])
    p, rec = run(tmp, "carry-identity", plan, {"carryovers": [dict(good_carry, gate="manifest")]}, zpath)
    check("the manifest gate cannot be carried over", gate(rec, "manifest")["verdict"] == "error")
    p, rec = run(tmp, "carry-and-skip", plan, {"carryovers": [good_carry],
                                               "skips": [{"gate": "suite", "reason": "x"}]}, zpath)
    check("a gate both skipped and carried over is an error", gate(rec, "suite")["verdict"] == "error")

    # 5. the extension points --------------------------------------------------------------------
    print("5. extension points: a checks folder (licence checks) and a criteria file (performance)")
    chk = os.path.join(tmp, "checks")
    write(os.path.join(chk, "notice_present.py"),
          "CHECK_ID = 'notice-present'\n"
          "def run(ctx):\n"
          "    ok = 'share/doc/radiance/NOTICE' in ctx.zip_names()\n"
          "    return {'verdict': 'pass' if ok else 'fail', 'reason': 'NOTICE ships' if ok else 'no NOTICE'}\n")
    write(os.path.join(chk, "_helper.py"), "raise SystemExit('a leading underscore is not a check')\n")
    plan = base_plan([{"id": "licences", "kind": "checks", "class": "D", "folder": chk}])
    p, rec = run(tmp, "checks", plan, None, zpath)
    g = gate(rec, "licences")
    check("a plug-in check runs and passes", g and g["verdict"] == "pass" and g["checks"][0]["check"] == "notice-present",
          g and str(g))
    write(os.path.join(chk, "broken.py"), "def run(ctx):\n    return {'verdict': 'maybe'}\n")
    p, rec = run(tmp, "checks-broken", plan, None, zpath)
    check("a check with a bad verdict is an error, not a pass", gate(rec, "licences")["verdict"] == "error")
    empty = os.path.join(tmp, "checks-empty")
    os.makedirs(empty)
    p, rec = run(tmp, "checks-empty", base_plan([{"id": "licences", "kind": "checks", "folder": empty}]), None, zpath)
    check("an empty checks folder is a skip", gate(rec, "licences")["verdict"] == "skip")

    crit = wjson(os.path.join(tmp, "criteria.json"), {
        "schema": "brilliance-perf-criteria/1", "noise_floor_pct": 3,
        "metrics": [{"id": "prefill_32k", "unit": "tok/s", "better": "higher", "target": 5000},
                    {"id": "decode_c1", "unit": "ms/token", "better": "lower", "target": 40}]})

    def meas(name, prefill, decode, zsha=digest, spread=1.0):
        return wjson(os.path.join(tmp, name), {
            "schema": "brilliance-perf-measurements/1", "candidate": {"zip_sha256": zsha},
            "metrics": {"prefill_32k": {"value": prefill, "spread_pct": spread}, "decode_c1": {"value": decode}}})
    cplan = base_plan([{"id": "perf", "kind": "criteria", "class": "M", "min_binding": "verified"}])
    solo = ["--solo-window", "fixture window"]
    p, rec = run(tmp, "crit-none", cplan, None, zpath, extra=solo)
    check("no criteria file: skip, saying so", gate(rec, "perf")["verdict"] == "skip"
          and "criteria" in gate(rec, "perf")["reason"])
    p, rec = run(tmp, "crit-pass", cplan, {"evidence": {"perf": {"measurements": meas("m1.json", 5100, 39)}}},
                 zpath, extra=solo + ["--criteria", crit])
    check("measurements meeting the targets pass", gate(rec, "perf")["verdict"] == "pass" and p.returncode == 0,
          gate(rec, "perf")["reason"])
    p, rec = run(tmp, "crit-near", cplan, {"evidence": {"perf": {"measurements": meas("m2.json", 4900, 41)}}},
                 zpath, extra=solo + ["--criteria", crit])
    check("within the noise floor passes, said so", gate(rec, "perf")["verdict"] == "pass"
          and any("noise floor" in d for d in gate(rec, "perf")["details"]))
    p, rec = run(tmp, "crit-fail", cplan, {"evidence": {"perf": {"measurements": meas("m3.json", 4000, 39)}}},
                 zpath, extra=solo + ["--criteria", crit])
    check("a missed target fails", gate(rec, "perf")["verdict"] == "fail")
    p, rec = run(tmp, "crit-noisy", cplan, {"evidence": {"perf": {"measurements": meas("m4.json", 5100, 39, spread=5)}}},
                 zpath, extra=solo + ["--criteria", crit])
    check("a spread over the noise floor fails", gate(rec, "perf")["verdict"] == "fail")
    p, rec = run(tmp, "crit-other", cplan, {"evidence": {"perf": {"measurements": meas("m5.json", 5100, 39, zsha="0" * 64)}}},
                 zpath, extra=solo + ["--criteria", crit])
    check("measurements of another ZIP fail", gate(rec, "perf")["verdict"] == "fail")

    print("5b. criteria cases: bus set per case, tuning, matched baseline or written verdict")
    prev_zip = "ab" * 32
    tuning = {"board_power_offset_pct": -20, "voltage_offset_mv": -50, "vram_mhz": 2518}
    ccrit = wjson(os.path.join(tmp, "criteria-cases.json"), {
        "schema": "brilliance-perf-criteria/1",
        "previous_release": {"tag": "v0", "zip_sha256": prev_zip},
        "run_conditions": {"tuning": tuning, "card_sets": {"tp2": [3, 9], "tp4": [3, 6, 9, 32], "odd": [7]}},
        "profiles": {"p2": {"card_set": "tp2"}, "p4": {"card_set": "tp4"}, "px": {"card_set": "odd"}},
        "cases": [{"id": "c2", "profile": "p2", "mode": "measured", "metrics": []},
                  {"id": "c4m", "profile": "p4", "mode": "matched", "metrics": []},
                  {"id": "hist", "profile": "px", "mode": "historical", "metrics": []}],
        "metrics": [{"id": "c2.x", "better": "higher", "target": 100, "noise_floor_pct": 3, "unit": "tok/s"},
                    {"id": "c4m.verdict_pass", "better": "higher", "target": 1, "noise_floor_pct": 0, "unit": "check"}]})
    wv_file = write(os.path.join(tmp, "verdict.md"), "TP4: no matched alpha.5 run; gap accepted (fixture)\n")

    def cmeas(name, buses2=(3, 9), c4=None, tun=tuning):
        cases = {"c2": {"buses": list(buses2)}}
        if c4 is not None:
            cases["c4m"] = c4
        return wjson(os.path.join(tmp, name), {
            "schema": "brilliance-perf-measurements/1", "candidate": {"zip_sha256": digest},
            "conditions": {"tuning": tun}, "cases": cases,
            "metrics": {"c2.x": 101, "c4m.verdict_pass": 1}})
    written = {"buses": [3, 6, 9, 32], "written_verdict": {"reviewer": "fixture", "reason": "no matched run",
                                                            "evidence": [wv_file]}}
    four = ["--allow-cards", "3,6,9,32"]
    p, rec = run(tmp, "case-pass", cplan, {"evidence": {"perf": {"measurements": cmeas("cm1.json", c4=written)}}},
                 zpath, extra=solo + four + ["--criteria", ccrit])
    g = gate(rec, "perf")
    check("cases with their bus sets and a written verdict pass", g and g["verdict"] == "pass", g and g["reason"])
    check("the record lists each case's bus set",
          g and {c["id"]: c["buses"] for c in g["cases"]} == {"c2": [3, 9], "c4m": [3, 6, 9, 32], "hist": [7]})
    check("the gate needs the union of the gating cases' buses", g and g["cards_needed"] == [3, 6, 9, 32])
    md = open(os.path.join(tmp, "case-pass", "out", "EVIDENCE.md"), encoding="utf-8").read()
    check("EVIDENCE.md has the case table with bus sets", "| c4m | matched | p4 | 3,6,9,32 (measured on 3,6,9,32) |" in md)
    check("the written verdict's file is hashed into the record",
          g and any(e.get("path", "").endswith("verdict.md") and e.get("sha256") for e in g["evidence"]))
    p, rec = run(tmp, "case-twobus", cplan, {"evidence": {"perf": {"measurements": cmeas("cm1.json", c4=written)}}},
                 zpath, extra=solo + ["--allow-cards", "3,9", "--criteria", ccrit])
    g = gate(rec, "perf")
    check("a two-card lease skips the gate, naming 3,6,9,32", g and g["verdict"] == "skip" and "3,6,9,32" in g["reason"],
          g and g["reason"])
    check("... and still records the cases' bus sets", g and len(g.get("cases", [])) == 3)
    p, rec = run(tmp, "case-wrongbus", cplan, {"evidence": {"perf": {"measurements": cmeas("cm2.json", (3, 6), written)}}},
                 zpath, extra=solo + four + ["--criteria", ccrit])
    check("a case run on the wrong buses fails", gate(rec, "perf")["verdict"] == "fail"
          and "c2: ran on buses 3,6" in gate(rec, "perf")["reason"], gate(rec, "perf")["reason"])
    p, rec = run(tmp, "case-nobus", cplan, {"evidence": {"perf": {"measurements": wjson(os.path.join(tmp, "cm3.json"), {
        "schema": "brilliance-perf-measurements/1", "candidate": {"zip_sha256": digest}, "conditions": {"tuning": tuning},
        "metrics": {"c2.x": 101, "c4m.verdict_pass": 1}, "cases": {"c4m": written}})}}},
        zpath, extra=solo + four + ["--criteria", ccrit])
    check("a measured case with no bus set recorded fails", gate(rec, "perf")["verdict"] == "fail"
          and "c2: the measurements do not record its bus set" in gate(rec, "perf")["reason"])
    p, rec = run(tmp, "case-nomatch", cplan, {"evidence": {"perf": {"measurements": cmeas("cm4.json", c4={"buses": [3, 6, 9, 32]})}}},
                 zpath, extra=solo + four + ["--criteria", ccrit])
    check("matched mode without a baseline or a written verdict fails", gate(rec, "perf")["verdict"] == "fail"
          and "matched previous-release baseline" in gate(rec, "perf")["reason"])
    base_bad = {"buses": [3, 6, 9, 32], "baseline": {"zip_sha256": "cd" * 32, "evidence": [wv_file]}}
    p, rec = run(tmp, "case-basebad", cplan, {"evidence": {"perf": {"measurements": cmeas("cm5.json", c4=base_bad)}}},
                 zpath, extra=solo + four + ["--criteria", ccrit])
    check("a baseline on another ZIP than the previous release fails", gate(rec, "perf")["verdict"] == "fail")
    base_ok = {"buses": [3, 6, 9, 32], "baseline": {"zip_sha256": prev_zip, "evidence": [wv_file]}}
    p, rec = run(tmp, "case-baseok", cplan, {"evidence": {"perf": {"measurements": cmeas("cm6.json", c4=base_ok)}}},
                 zpath, extra=solo + four + ["--criteria", ccrit])
    check("a matched baseline on the previous release passes", gate(rec, "perf")["verdict"] == "pass",
          gate(rec, "perf")["reason"])
    p, rec = run(tmp, "case-tuning", cplan, {"evidence": {"perf": {"measurements": cmeas(
        "cm7.json", c4=written, tun=dict(tuning, voltage_offset_mv=0))}}}, zpath, extra=solo + four + ["--criteria", ccrit])
    check("measurements at other tuning fail", gate(rec, "perf")["verdict"] == "fail"
          and "tuning differs" in gate(rec, "perf")["reason"])
    real = os.path.join(os.path.dirname(HERE), "tools", "perf-criteria.json")
    if os.path.isfile(real):
        p, rec = run(tmp, "case-real", cplan, None, zpath, extra=solo + ["--allow-cards", "3,9", "--criteria", real])
        g = gate(rec, "perf")
        rows = {c["id"]: c for c in (g or {}).get("cases", [])}
        check("the shipped criteria: G6 needs buses 3,6,9,32", g and g["cards_needed"] == [3, 6, 9, 32]
              and g["verdict"] == "skip", g and g["reason"])
        check("the shipped criteria: TP4 cases on 3,6,9,32 and TP2 cases on 3,9",
              rows.get("flashnext_tp4_standard", {}).get("buses") == [3, 6, 9, 32]
              and rows.get("flashnext_tp2_default_prefill", {}).get("buses") == [3, 9], str(list(rows)[:3]))
    else:
        check("win/tools/perf-criteria.json is present", False, real)

    # 6. resume and the build lease --------------------------------------------------------------
    print("6. resume and the build lease")
    plan = base_plan([cmd_gate("docs", "print('0 errors, 0 warnings')", {"exit": 0})])
    run(tmp, "resume", plan, None, zpath)
    p, rec = run(tmp, "resume", plan, None, zpath, extra=["--resume"])
    check("--resume reuses a pass for the same candidate", gate(rec, "docs").get("reused") is True)
    lease = os.path.join(tmp, "build-lease")
    os.mkdir(lease)
    write(os.path.join(lease, "holder.txt"), "someone-else build 1:00 AM\n")
    plan = base_plan([cmd_gate("build", "print('built')", {"exit": 0}, lease="build")])
    p, rec = run(tmp, "lease-held", plan, None, zpath, extra=["--build-lease", lease])
    check("a held lease skips the gate, naming the holder", gate(rec, "build")["verdict"] == "skip"
          and "someone-else" in gate(rec, "build")["reason"])
    check("a held lease is left alone", os.path.isfile(os.path.join(lease, "holder.txt")))
    shutil.rmtree(lease)
    p, rec = run(tmp, "lease-free", plan, None, zpath, extra=["--build-lease", lease])
    check("a free lease is taken and released", gate(rec, "build")["verdict"] == "pass" and not os.path.exists(lease))
    plan = base_plan([cmd_gate("tmpdir", "import os; print(os.environ.get('RAD_TEST_TMPDIR', ''))", {"exit": 0})])
    p, rec_t = run(tmp, "tmpdir", plan, None, zpath)
    logtext = open(gate(rec_t, "tmpdir")["log"]["path"], encoding="utf-8").read()
    want = os.path.join(tmp, "tmpdir", "out", "tmp", "tmpdir")
    check("a command gate gets RAD_TEST_TMPDIR under the evidence folder",
          os.path.normcase(os.path.abspath(want)) in os.path.normcase(logtext) and os.path.isdir(want), logtext[-200:])
    check("the lease wait is recorded apart from the command's time",
          "lease_wait_s" in gate(rec, "build") and "run_s" in gate(rec, "build"))
    plan = base_plan([cmd_gate("cardgate", "import time; time.sleep(2); print('done')", {"exit": 0, "require": ["^done$"]},
                               cards=[32], timeout=1),
                      cmd_gate("cpugate", "import time; time.sleep(2); print('done')", {"exit": 0}, timeout=1)])
    p, rec = run(tmp, "timeouts", plan, None, zpath, extra=["--allow-cards", "32", "--keep-going"])
    check("a gate that uses cards is never given a timeout (nothing force-ended)",
          gate(rec, "cardgate")["verdict"] == "pass", gate(rec, "cardgate")["reason"])
    check("a CPU gate past its timeout fails", gate(rec, "cpugate")["verdict"] == "fail"
          and "no exit within" in gate(rec, "cpugate")["reason"])

    # 7. refused plans and inputs ----------------------------------------------------------------
    print("7. usage errors: nothing judged")
    p, rec = run(tmp, "bad-plan", {"schema": "brilliance-gate-plan/1", "gates": [{"id": "x", "kind": "command"}]},
                 None, zpath)
    check("a plan not starting with identity exits 3", p.returncode == 3 and rec is None)
    p, rec = run(tmp, "bad-inputs", base_plan([]), {"skips": [{"gate": "nope", "reason": "r"}]}, zpath)
    check("inputs naming an unknown gate exit 3", p.returncode == 3)

    if not FAILS:
        shutil.rmtree(tmp, ignore_errors=True)
    print("gate_runner_test: %d failed" % len(FAILS))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())

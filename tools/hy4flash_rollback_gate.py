#!/usr/bin/env python3
"""WP13.0 rollback gate of the Hy4-flash trunk (kernels/dfly2.md section 6, 7.4, 10): no drafter.

Prompt lookup (--spec-ngram K) with --debug-draft-oracle proposes the plain greedy run's own continuation, shaped by
RADIANCE_DEBUG_ORACLE_MODE so every rejection pattern of section 6.2 happens:
    right   all drafts right                  wrong   all wrong (reject-all, the maximum of rollbacks)
    random  right up to a hashed index        close   wrong exactly at the row that closes an indexer group of 4
    close2  as close, at the second such row  after   wrong at the row after a closing row (the closing row is accepted)
    miss    30 % wrong by hash (WP8's)
and two modes with the DFly2 drafter instead of the oracle (WP13.3; not in the default list):
    dfly            --num-speculative-tokens K: the drafter proposes
    dfly-sabotage   the same with RADIANCE_HY4_DFLY_SABOTAGE=1 (every proposal the mask id: nearly every draft rejected)
The oracle modes run with RADIANCE_HY4_DFLY=0, since prompt lookup alone (--spec-ngram) is refused beside a drafter.
For each (prompt, mode) the speculative run must give, against the plain greedy run of the same prompt:
  * the generated ids and their logprobs byte-equal (production path, no taps), and with the debug taps on;
  * state probes equal on every committed position: the debug taps (RADIANCE_HY4_TAPS=1, read by the engine's buffer
    probe) attn_kv_latent (sparse and sliding latents), index_token_keys (what the tail ring is fed and later pools from),
    index_group_keys (the pooled keys of closed groups), index_sel / index_nsel (what selection read from tail and
    pooled keys), engram_hash_ids (the CONV history through engram_ids' num_accepted), final_hidden.
    A tap row of a rejected row is stale by design and masked; every committed row is compared.
It also reports how many rows were rejected, how many of them closed an indexer group, and how many lay past the window
(--window) and the dense limit (--dense-limit), so a green run says what it covered.

    tools/hy4flash_rollback_gate.py --build build --rad model.rad --preset mini --prompt-lens 30,31,32,33 --gen 24
    ... --gpu --tp 2 --kernels libr4d,libavx --rad gpu-mini.rad --preset gpu-mini --prompt-ids P.json:p2052 ...

Host runs set the fake-rank environment; --gpu runs the servers under flock gpu.lock (held while a server lives).
--test-ring DELTA (RADIANCE_HY4_TEST_RING) resizes the indexer tail ring of the speculative runs: the probes must then
differ for a ring that is too small. Stdlib only.
"""
import argparse, glob, json, os, random, re, shutil, signal, socket, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
SP, SL = "sparse", "sliding"
PRESETS = {      # layer kinds and Engram layer indices (ids are one above): ref/minicfg.py
    "mini": ([SP, SL, SP], [1]),
    "mini4": ([SP, SL, SP], [1]),
    "mini6": ([SP, SP, SL, SL, SL, SP], [1, 4]),
    "gpu-mini": ([SP, SL, SP, SL], [1]),
}
MODES = ["right", "wrong", "random", "close", "close2", "after", "miss"]
DFLY_MODES = ["dfly", "dfly-sabotage"]


def prompt_text(ids):
    return "".join(f"<t{i}>" for i in ids)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, cmd, env, logpath, gpu):
        self.logpath = logpath
        self.log = open(logpath, "w+")
        pre = ["flock", "/root/work/claude-scratch/gpu.lock"] if gpu else []
        self.proc = subprocess.Popen(pre + [str(c) for c in cmd], env=env, stdout=self.log, stderr=subprocess.STDOUT,
                                     start_new_session=True)

    def text(self):
        self.log.flush()
        return open(self.logpath, errors="replace").read()

    def wait_ready(self, port, timeout):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                return False
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1).close()
                return True
            except Exception:
                time.sleep(0.2)
        return False

    def stop(self):
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        t0 = time.time()
        while time.time() - t0 < 20:       # the whole group must be gone: flock's child holds gpu.lock
            try:
                os.killpg(self.proc.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.2)
        else:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def complete(port, ids, n_gen, timeout=3600):
    body = json.dumps(dict(prompt=prompt_text(ids), max_tokens=n_gen, temperature=0, ignore_eos=True, logprobs=1,
                           add_special_tokens=False, skip_special_tokens=False)).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", body, {"Content-Type": "application/json"})
    ch = json.load(urllib.request.urlopen(req, timeout=timeout))["choices"][0]
    gen = []
    lp = (ch.get("logprobs") or {}).get("content") or []
    for e in lp:
        if isinstance(e.get("id"), int):
            gen.append(e["id"])
            continue
        m = re.fullmatch(r"<t(\d+)>", e.get("token", ""))
        if m:
            gen.append(int(m.group(1)))
    if len(gen) != n_gen:
        gen = [int(x) for x in re.findall(r"<t(\d+)>", ch.get("text", ""))]
    return gen, [e.get("logprob") for e in lp]


class Run:
    """One server: the arguments, the tap set, the dump directory."""

    def __init__(self, a, tag, extra, env_extra, taps):
        self.a, self.tag, self.taps = a, tag, taps
        self.dump = os.path.join(a.work, tag + ".dump")
        shutil.rmtree(self.dump, ignore_errors=True)
        os.makedirs(self.dump)
        env = os.environ.copy()
        env.update(RADIANCE_HOME=os.path.join(a.build, "radiance_home"), OMP_NUM_THREADS=str(a.threads),
                   RADIANCE_LOG_STEPS="1")
        env["RADIANCE_KERNELS"] = a.kernels.replace(",", ":")
        if not a.gpu:
            env.update(ROCR_VISIBLE_DEVICES="99", HIP_VISIBLE_DEVICES="", RADIANCE_HOST_VRAM_MIB=str(a.host_vram_mib),
                       RADIANCE_HOST_DEVICES=str(a.tp))
        if a.engram_f32:
            env["RADIANCE_HY4_ENGRAM_EPS"] = "f32"
        if taps:
            env.update(RADIANCE_HY4_TAPS="1", RADIANCE_DUMP_DIR=self.dump, RADIANCE_DUMP_BUF=",".join(a.tap_names),
                       RADIANCE_DUMP_BYTES="4000000")
        else:
            env["RADIANCE_HY4_TAPS"] = "0"
        env.update(env_extra)
        self.port = free_port()
        cmd = [os.path.join(SRC, "scripts/radiance-flags"), os.path.join(a.build, "bin/radiance"), "--model", a.rad,
               "--host", "127.0.0.1", "--port", self.port, "--kernels", a.kernels, "--max-model-len", a.max_len,
               "--max-num-batched-tokens", a.chunk, "--max-num-seqs", 1, "--tp", a.tp, "-vv"]
        if not a.prefix_cache:
            cmd.append("--no-prefix-cache")
        if "libref" in a.kernels.split(","):
            cmd.append("--debug-accept-reference-kernels")
        self.srv = Server(cmd + extra, env, os.path.join(a.work, tag + ".log"), a.gpu)
        if not self.srv.wait_ready(self.port, a.ready_timeout):
            sys.exit(f"{tag}: server did not come up:\n" + self.srv.text()[-3000:])

    def gen(self, ids, n, warm=False):
        """-> (gen, logprobs, step log text of this request). warm: a prefix-cache warm-up request first."""
        if warm:
            complete(self.port, ids, 1)
            time.sleep(0.3)
        for f in glob.glob(os.path.join(self.dump, "*")):
            os.remove(f)
        t0 = len(self.srv.text())
        g, lp = complete(self.port, ids, n)
        time.sleep(0.5)
        return g, lp, self.srv.text()[t0:]

    def stop(self):
        self.srv.stop()


STEP_RE = re.compile(r"step (\d+) (prefill|decode|MIXED) n_seq=(\d+) n_tok=(\d+)\s+\[0\] slot=\d+ q=(\d+) ctx=(\d+)")


def steps_of(text):
    return [dict(step=int(m.group(1)), kind=m.group(2), n=int(m.group(4)), ctx=int(m.group(6)))
            for m in STEP_RE.finditer(text)]


def committed_rows(steps, P, N):
    """For each decode step: (step, ctx, rows accepted, rows), clipped to positions < P + N - 1 (the last token has no row)."""
    out = []
    dec = [s for s in steps if s["kind"] != "prefill"]
    end = P + N - 1
    for i, s in enumerate(dec):
        c = s["ctx"]
        nxt = dec[i + 1]["ctx"] if i + 1 < len(dec) else end
        acc = max(0, min(nxt, end, c + s["n"]) - c)
        out.append((s["step"], c, acc, s["n"]))
    return out


def rejection_stats(steps, P, N, window, dense):
    st = dict(steps=0, rows=0, rejected=0, closing_rejected=0, past_window=0, past_dense=0, reject_all=0)
    end = P + N - 1
    for step, c, acc, n in committed_rows(steps, P, N):
        st["steps"] += 1
        st["rows"] += n
        n = min(n, end - c) if c + n > end else n      # rows past max_tokens are dropped, not rejections
        for pos in range(c + acc, c + n):
            st["rejected"] += 1
            st["closing_rejected"] += pos % 4 == 3
            st["past_window"] += pos >= window
            st["past_dense"] += pos >= dense
        st["reject_all"] += acc == 1 and n > 1
    return st


def probe(dump, steps, P, N, names, rows, dense=0):
    """-> {(name, kind, pos or group): bytes} of every committed decode row (group keys: by group id)."""
    out = {}
    for step, c, acc, n in committed_rows(steps, P, N):
        for name in names:
            fn = os.path.join(dump, f"{step}--{name}.bin")
            if not os.path.exists(fn):
                out[("MISSING", name, step)] = b""
                continue
            raw = open(fn, "rb").read()
            if not raw or len(raw) % rows:
                out[("BADSIZE", name, step)] = b""
                continue
            w = len(raw) // rows
            if name.startswith("dbg.index_group_keys"):
                for g in range(c // 4, (c + n) // 4):
                    if P <= 4 * g + 3 < c + acc:
                        out[(name, "g", g)] = raw[(g - c // 4) * w:(g - c // 4 + 1) * w]
            else:
                # a step with no selecting row leaves index_sel/nsel unwritten; a mixed one writes dense rows too: below
                # the dense limit the tap is undefined, so it is not compared there
                sel = name.startswith(("dbg.index_sel", "dbg.index_nsel"))
                for j in range(acc):
                    if sel and c + j < dense:
                        continue
                    out[(name, "p", c + j)] = raw[j * w:(j + 1) * w]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rad", required=True, help="converted container (rad-convert --max-ctx/--max-tok covering the cases)")
    ap.add_argument("--preset", required=True, choices=list(PRESETS))
    ap.add_argument("--vocab", type=int, help="token ids drawn below this (default: mini 320, gpu-mini 3584)")
    ap.add_argument("--prompt-lens", help="comma list of random prompt lengths (seeded)")
    ap.add_argument("--prompt-ids", action="append", default=[], help="FILE:name of a json of {name: [ids]} (repeatable)")
    ap.add_argument("--gen", type=int, default=24)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--depth", type=int, default=5, help="--spec-ngram K")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--kernels", default=None, help="default libref (host) / libr4d,libavx (--gpu)")
    ap.add_argument("--prefix-cache", action="store_true", help="prefix cache on; each case resumes from a warm cache (second turn)")
    ap.add_argument("--notap-modes", default=None, help="modes that also run without taps (default: all selected; '' none)")
    ap.add_argument("--no-taps", action="store_true", help="production path only: text and logprobs, no state probes")
    ap.add_argument("--layers", help="restrict the taps to these layers")
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--dense-limit", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--test-ring", type=int, default=None, help="RADIANCE_HY4_TEST_RING delta; the gate must then FAIL")
    ap.add_argument("--engram-f32", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--host-vram-mib", type=int, default=512)
    ap.add_argument("--ready-timeout", type=int, default=900)
    ap.add_argument("--work", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dfly-extra", default="", help="extra server flags of the dfly modes, e.g. "
                    "'--draft-coupling on --draft-verify traversal --draft-depth adaptive' (WP13.8)")
    a = ap.parse_args()
    a.kernels = a.kernels or ("libr4d,libavx" if a.gpu else "libref")
    a.work = a.work or os.path.join(a.build, "rollback_gate")
    os.makedirs(a.work, exist_ok=True)
    kinds, eng = PRESETS[a.preset]
    layers = [int(x) for x in a.layers.split(",")] if a.layers else range(len(kinds))
    names = []
    for l in layers:
        names.append(f"dbg.attn_kv_latent.{l}")
        if kinds[l] == SP:
            names += [f"dbg.index_token_keys.{l}", f"dbg.index_group_keys.{l}", f"dbg.index_sel.{l}", f"dbg.index_nsel.{l}"]
        if l in eng:
            names.append(f"dbg.engram_hash_ids.{l}")
    names.append("dbg.final_hidden")
    rows = a.depth + 1
    a.tap_names = [f"{n}:{rows}" for n in names]
    vocab = a.vocab or (3584 if a.preset == "gpu-mini" else 320)
    prompts = {}
    rng = random.Random(a.seed)
    for L in [int(x) for x in a.prompt_lens.split(",")] if a.prompt_lens else []:
        prompts[f"r{L}"] = [rng.randrange(1, vocab) for _ in range(L)]
    for spec in a.prompt_ids:
        f, n = spec.rsplit(":", 1)
        prompts[n] = json.load(open(f))[n]
    if not prompts:
        sys.exit("no prompts")
    a.max_len = a.max_len or max(len(p) for p in prompts.values()) + a.gen + 8
    a.chunk = a.chunk or max(16, max(len(p) for p in prompts.values()))
    modes = a.modes.split(",")
    notap = set(modes if a.notap_modes is None else a.notap_modes.split(",") if a.notap_modes else [])
    env_ring = {} if a.test_ring is None else {"RADIANCE_HY4_TEST_RING": str(a.test_ring)}
    fails = 0
    total = {}

    def verdict(ok, text):
        nonlocal fails
        fails += not ok
        print(("  PASS " if ok else "  FAIL ") + text, flush=True)

    for pname, ids in prompts.items():
        P, N = len(ids), a.gen
        print(f"== prompt {pname}: {P} tokens, {N} generated, committed end {P + N - 1} (residue {(P + N - 1) % 4}), "
              f"tp {a.tp}, kernels {a.kernels}, prefix cache {'on, second turn' if a.prefix_cache else 'off'}", flush=True)
        tag0 = f"{pname}-tp{a.tp}"
        ref = Run(a, tag0 + "-plain-tap", [], {}, not a.no_taps)
        try:
            g0, lp0, t0 = ref.gen(ids, N, warm=a.prefix_cache)
            steps0 = steps_of(t0)
            ref_probe = probe(ref.dump, steps0, P, N, names, rows, a.dense_limit)
        finally:
            ref.stop()
        if len(g0) != N:
            sys.exit(f"plain run gave {len(g0)} tokens")
        bad0 = [] if a.no_taps else [k for k in ref_probe if k[0] in ("MISSING", "BADSIZE")]
        if bad0:
            sys.exit(f"plain probes missing: {bad0[:5]}")
        if not a.no_taps:
            plain = Run(a, tag0 + "-plain", [], {}, False)
            try:
                g0n, lp0n, _ = plain.gen(ids, N, warm=a.prefix_cache)
            finally:
                plain.stop()
            verdict(g0n == g0 and lp0n == lp0, "plain: taps on/off give the same text and logprobs")
        oracle = os.path.join(a.work, f"oracle-{tag0}.txt")
        open(oracle, "w").write(" ".join(map(str, g0)))
        for mode in modes:
            if mode in DFLY_MODES:
                extra = ["--num-speculative-tokens", a.depth] + a.dfly_extra.split()
                envm = dict(env_ring, **({"RADIANCE_HY4_DFLY_SABOTAGE": "1"} if mode == "dfly-sabotage" else {}))
            else:
                extra = ["--spec-ngram", a.depth, "--debug-draft-oracle", oracle]
                envm = dict(env_ring, RADIANCE_DEBUG_ORACLE_MODE=mode, RADIANCE_HY4_DFLY="0")
            for taps in ([] if a.no_taps else [True]) + ([False] if mode in notap or a.no_taps else []):
                r = Run(a, f"{tag0}-{mode}-{'tap' if taps else 'notap'}", extra, envm, taps)
                try:
                    g1, lp1, t1 = r.gen(ids, N, warm=a.prefix_cache)
                    steps1 = steps_of(t1)
                    got = probe(r.dump, steps1, P, N, names, rows, a.dense_limit) if taps else None
                finally:
                    r.stop()
                st = rejection_stats(steps1, P, N, a.window, a.dense_limit)
                for k, v in st.items():
                    total[k] = total.get(k, 0) + v
                head = (f"{pname} mode {mode:6} {'taps' if taps else 'notap':5}: {st['steps']} verify steps, "
                        f"{st['rejected']} rejected rows ({st['closing_rejected']} closing a group, "
                        f"{st['past_window']} >= {a.window}, {st['past_dense']} >= {a.dense_limit}, "
                        f"{st['reject_all']} reject-all steps)")
                lpd = sum(1 for x, y in zip(lp0, lp1) if x != y)
                verdict(g1 == g0 and lp1 == lp0, f"{head}; text {'equal' if g1 == g0 else 'DIFFERS'}, logprobs "
                        f"{'bit-equal' if lp1 == lp0 else f'DIFFER in {lpd}/{N}'}")
                if taps:
                    bad = [k for k in sorted(set(ref_probe) | set(got), key=str) if ref_probe.get(k) != got.get(k)]
                    byname = {}
                    for k in bad:
                        byname.setdefault(k[0] if k[0] not in ("MISSING", "BADSIZE") else k[0] + ":" + k[1], []).append(k[-1])
                    verdict(not bad, f"  probes: {len(ref_probe)} committed rows/groups compared, {len(bad)} differ"
                            + "".join(f"\n        {n}: first {sorted(v)[:6]}" for n, v in sorted(byname.items())))
                if mode in DFLY_MODES and "'dfly2' proposes" not in r.srv.text():
                    verdict(False, "  the DFly2 drafter was not served: the run is prompt lookup")
                if mode in ("wrong", "dfly-sabotage") and st["rejected"] == 0:
                    verdict(False, "  nothing was rejected in mode wrong: the run proves nothing")
    print("coverage total:", json.dumps(total))
    if a.test_ring is not None:
        print(f"test ring {a.test_ring}: {fails} failing checks (a ring that is too small must fail some)")
        return 0 if fails else 1
    print("RESULT", "FAIL" if fails else "PASS", f"({fails} failing checks)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

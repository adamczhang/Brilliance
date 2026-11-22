# The gate runner

`win/tools/gate_runner.py` runs a release's gates against one candidate package and writes one
evidence record. The record ties every gate result to the candidate's identity: the ZIP's SHA-256
and the source snapshot commit in its `MANIFEST.json`. Each gate gets exactly one verdict. A skip
or a carryover is written down with its reason and is never counted as a pass.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File win\tools\gate-runner.ps1 `
  -Plan <gate-plan.json> -Zip <Brilliance-<ver>-<commit>-win64.zip> -Out <evidence folder> `
  -Inputs <inputs.json> -Pkg <fresh unpack> -Source <build clone> -BuildLease <lease folder>
```

The same command without PowerShell is
`python -I win/tools/gate_runner.py --plan ... --zip ... --out ... [--inputs ...] [--pkg ...] [--source ...]`.
`-List` (`--list`) prints the plan's gates and exits.

It writes these files in `-Out`:

- `evidence.json`, the record. It is rewritten after every gate, so a stopped run still leaves what
  it judged.
- `EVIDENCE.md`, the same record in a readable form: a summary table, then each gate's details.
- `logs/<gate>.log`, the output of every command the runner ran, as UTF-8.

The exit status is 0 for **ready**, 1 for **fail**, 2 for **incomplete** and 3 for a usage error,
which means nothing was judged. The verdict comes from the record, not from this status, as for
every other release gate.

The runner uses the Python standard library only and runs below normal priority. Every child
process runs without a window. The runner touches no card itself. A gate that names cards runs
only when `-AllowCards` leases all of them. A command has no timeout unless the plan sets
one, because a timeout ends the child process. A gate that names cards never gets one, whatever
the plan says: a GPU process is never force-ended, and those gates stop their own servers with
Ctrl-Break.

A gate that takes the build lease records the time it waited for the lease (`lease_wait_s`)
separately from the time its command ran (`run_s`).

Every command gate runs with `RAD_TEST_TMPDIR` set to `<Out>\tmp\<gate id>`, unless the plan's
`env` names another folder. This keeps the test harness's scratch out of its default `/tmp`,
which is outside the project, for the host suite gate and any other test it starts.

## Verdicts

| Verdict | Meaning |
|---|---|
| `pass` | The gate ran (or its log was judged), and every rule held. |
| `fail` | A rule failed. The reason names the rule, and the first offending line for a forbidden pattern. |
| `skip` | The gate was not judged. A skip always has a reason, and three things produce one. The inputs file can skip a gate, and must give the reason. The runner skips a gate when a precondition is missing: cards not leased, an M or S gate with no solo window, no `--source` or `--pkg`, a missing input or var, or a build lease held by someone else. `--only` skips every gate it does not name. |
| `carryover` | The gate's evidence comes from an earlier candidate, and the inputs file justifies it (see below). The carried-over logs must still pass this gate's own rules. |
| `blocked` | Not started. Either the identity gate failed, or an earlier required gate failed and `--keep-going` was not given. |
| `error` | The gate could not be judged. Examples: a skip with no reason, a refused carryover, a check plug-in that raised or returned an unknown verdict, or a command that would not start. An error fails the record. |

The record's verdict is **fail** if any gate failed or errored. It is **ready** only if every
required gate passed or was carried over, and every pass is bound at least as strongly as its
gate's `min_binding`. Otherwise it is **incomplete**, and `summary.not_ready` lists each gate that
held it back, with the reason.

**Binding.** Each result records how it is tied to the candidate:

- `run`: the runner ran the gate itself, against this ZIP, its unpack, or a clone that the identity
  gate found at the snapshot commit with no tracked changes.
- `verified`: a log made elsewhere names the candidate. The plan's `binds` tokens, for example
  `{commit12}` or `{zip_name}`, must appear in the log's text.
- `declared`: only the inputs file says the log belongs to this candidate.
- `none`: the gate has no binding.

## The plan

A plan is a JSON file with the schema `brilliance-gate-plan/1`. Its first gate must be the
identity gate. The release plan lives with the private release scripts, as `gate-plan.json` next
to `make-snapshot.py`. It covers G0 to G7.

```json
{"schema": "brilliance-gate-plan/1", "name": "release", "gates": [
  {"id": "identity", "kind": "identity"},
  {"id": "G1.docs_check", "kind": "command", "class": "D", "needs": ["source"],
   "argv": ["{python}", "-I", "{source}/win/tools/docs_check.py", "--root", "{source}"],
   "judge": {"exit": 0, "require": ["^docs_check: .*; 0 errors, 0 warnings"], "forbid": ["^error: "]}}
]}
```

Every gate has these fields:

- `id`: unique within the plan.
- `title`.
- `kind`: one of the six kinds below.
- `class`: `C`, `D`, `F`, `M` or `S`, the project's run classes.
- `required`: defaults to true. An optional gate never holds back a ready record.
- `min_binding`: the weakest binding a pass may have. The default is `verified` for `log` gates and
  `run` for the others.
- `cards`: the PCI buses the gate needs.
- `needs`: `source`, `pkg`, `var:<name>` or `input:<key>`.
- `carryover`: defaults to true. The identity and manifest gates can never be carried over.

| Kind | What it does |
|---|---|
| `identity` | Hashes the ZIP and reads `MANIFEST.json` from inside it. It checks that `source_commit` is a full commit id, that `source_dirty` is false and that there is no BOM. The ZIP's name must be `Brilliance-<version>-<commit>-win64.zip` and match the manifest. A `.sha256` file beside the ZIP must hold the ZIP's hash. The inputs' `expect` values (`commit`, `version`, `upstream`, `zip_sha256`) must match. With `--source`, the clone must be at that commit with no tracked changes. If this gate fails, every other gate is `blocked`. |
| `manifest` | Checks every file in the ZIP against `MANIFEST.json`'s hashes: none missing, none unlisted, none changed. With `--pkg`, it checks the unpacked folder the same way. |
| `command` | Runs `argv`, with optional `cwd` and `env`, without a window. It writes the output to `logs/<id>.log` and judges that log. With `"intake": true`, the inputs file may supply the log of a run made elsewhere instead; that log is judged by the same rules. `"lease": "build"` takes `--build-lease` around the command. |
| `log` | Judges logs the inputs file names. It never runs anything. Use it for the snapshot report, card smoke summaries, the walkthrough record, asset checks, the notes and the review sign-off. |
| `checks` | Runs every check plug-in in `folder`: the licence checks' extension point. |
| `criteria` | Judges measurements against a performance criteria file: the performance protocol's extension point. |

`argv`, `cwd`, `env` and `folder` strings, and the patterns in `judge` and `binds`, can contain
these tokens:

- `{zip}`, `{zip_name}`, `{zip_sha256}`;
- `{pkg}`, `{source}`, `{out}`, `{repo}` (this checkout);
- `{commit}`, `{commit7}`, `{commit12}`, `{version}`, `{package}`;
- `{python}`, the interpreter running the runner;
- `{var.<name>}`, from the inputs' `vars`.

Inside a pattern, a token's value is matched literally.

**Judge** rules apply to the log text. A UTF-16 log from Windows PowerShell 5.1, a log with a BOM,
or a log in a code page is read as text, and CRLF line endings become LF.

- `exit`: the expected exit code, for a command the runner ran.
- `require`: regexes, matched in multiline mode, that must each match.
- `forbid`: regexes that must not match.
- `allow_after`: for a forbidden pattern, a regex after whose first match the pattern is allowed.
  The release plan uses it for the `lld-link: warning` lines after `nemotron35_test`'s link.
- `count`: named regexes whose match counts the record keeps.
- `expect_counts`: a `[min, max]` range for any of those counts.
- `ctest: true`: parses ctest's summary. Every failed test fails the gate, by name. A test that did
  not run must be in `expected_skips`, or the gate fails. The record lists every skip by name.
  `min_tests` sets a floor on the number of tests.
- `per_log: true`: judges each log on its own, for example one smoke summary per layout.
- `min_logs`: the fewest logs that must be given.

## The inputs file

The inputs file, schema `brilliance-gate-inputs/1`, describes one candidate. It is kept with that
candidate's evidence and is never committed. The record stores its hash.

```json
{"schema": "brilliance-gate-inputs/1",
 "expect": {"commit": "<40 hex>", "version": "0.3.0"},
 "private_source": "<private commit the snapshot was made from>",
 "vars": {"model": "<minicpm .rad>", "port": "8071", "smoke": "<smoke.ps1>", "upstream_tag": "v2.2.0"},
 "evidence": {"G1.snapshot": {"log": "<snapshot report>"},
              "G4.smoke": {"logs": ["<layout 1 summary>", "<layout 2 summary>"]},
              "G6.bench": {"measurements": "<measurements.json>"}},
 "skips": [{"gate": "G5.walkthrough", "reason": "..."}],
 "carryovers": [{"gate": "G3.host_suite", "reason": "...", "from": {"commit": "<earlier snapshot>",
                 "zip_sha256": "<earlier ZIP>"}, "changed_paths": ["win/serve.ps1"],
                 "diff": {"repo": "<a repository with both commits>", "from": "<a>", "to": "<b>"},
                 "covered_by": "...", "approved_by": "...", "evidence": ["<earlier suite log>"]}]}
```

An `evidence` entry may give `"for"`, the commit or ZIP hash the log was made for. A log given for
another candidate fails its gate.

**A carryover** is accepted only if it meets all of these conditions:

- It gives a reason.
- It gives the earlier candidate's identity in `from`, and that identity is not this candidate's.
- It lists what changed between the two candidates in `changed_paths`. With `diff`, the runner
  checks this list against `git diff --name-only` and fails the gate on any difference.
- It names the earlier candidate's evidence, and that evidence passes this gate's rules.

Anything less is an `error`. A gate cannot be both skipped and carried over.

## Extension points

### Licence and distribution checks (`checks` gates)

Each Python file in `win/tools/gate_checks/` is a check. A file whose name starts with `_` is not
run. `_example.py` shows the contract: `run(ctx)` returns a verdict of `pass`, `fail` or `skip`,
plus a reason, optional details and optional evidence paths. `ctx` gives access to:

- the candidate;
- the ZIP, with `zip_names()` and `zip_read(path)`;
- the unpack and the build clone;
- `MANIFEST.json`;
- the gate's `options` and the inputs file;
- an output folder.

The gate's verdict is the worst of its checks' verdicts. A check that raises, or returns a verdict
outside those three, makes the gate an `error`. An empty folder makes the gate a `skip` that says
no checks are registered.

`profile-scope.json` maps each shipped profile's model routes to containers, licence IDs, gated
downloads and required terms. Update it alongside `win/profiles`, the **Models and card counts**
section of `win/SUPPORT.md` (the bold label in the cell with `Profile:` or `Profiles:`), the
model-licence index and the terms template. The check reads the block-style model maps used by
the shipped YAMLs. The model routes in Sandbox inherit their gating from the source profiles.
The terms check verifies the required passages in each applicable section, including the full
quoted licence clauses where the template reproduces them.

`make-snapshot.py` runs plug-ins declaring `PRE_SNAPSHOT = True` before writing a snapshot,
including in `--scan-only` mode. In this mode `ctx.zip` is empty and `ctx.source_names()` and
`ctx.source_read(path)` read the filtered source ref's files. The runtime check verifies the
source notice files and install rules for the two build-supplied texts. The release plan's
`G0.scan` expects both checks in the scan log.

`G7.licences` reads the profiles, support, terms and notices inside the candidate ZIP.
`ctx.source_read(path)` reads a blob at the candidate commit in the build clone, which
`profile-scope.json` requires (`--source`). The ZIP runtime check requires the SDK's COMGR
licence when HIP libraries or kernels are present. A check that needs a different folder or its
own options gets its own `checks` gate in the plan.

### Performance criteria (`criteria` gates)

The criteria file comes from `--criteria`, or from `criteria` in the inputs or the plan. It has the
schema `brilliance-perf-criteria/1`:

```json
{"schema": "brilliance-perf-criteria/1", "protocol": "<where the fixed protocol is written>",
 "noise_floor_pct": 3,
 "metrics": [{"id": "tp2.prefill.32k", "unit": "tok/s", "better": "higher", "target": 5700,
              "noise_floor_pct": 3, "required": true}],
 "evaluator": "optional: a .py beside this file with evaluate(criteria, measurements, candidate)"}
```

The measurements file, schema `brilliance-perf-measurements/1`, is named by the inputs'
`evidence.<gate>.measurements`. It has these fields:

- `candidate.zip_sha256`, which must be this ZIP's;
- `conditions`, the hardware, tuning and window, recorded as given;
- `metrics`, mapping each `id` to a `value` and an optional `spread_pct`.

The default evaluator applies these rules to each metric:

- A target that is met passes.
- A miss within the noise floor passes, and the record says so.
- A miss beyond the floor fails.
- A spread wider than the floor fails as too noisy to judge.
- A required metric that was not measured fails.

The perf-criteria stage owns the protocol, the targets and the floor. It can replace these rules
by naming an `evaluator` module, which returns the same verdict dict as a check.

**The release's criteria file** is `win/tools/perf-criteria.json`. The release plan names it as
`{repo}/win/tools/perf-criteria.json`. Its top-level `metrics` are judged by the rules above. Its
`cases` add checks of their own, and the runner applies them whenever a criteria file has
`cases` and no `evaluator`:

- **Bus sets.** A case's bus set is the card set its profile names (`profiles.<p>.card_set`), as
  listed in `run_conditions.card_sets`: buses 3 and 9 for TP2, and 3, 6, 9 and 32 for TP4. Every
  case, with its bus set, is written into the gate's record as `cases`, whatever the verdict,
  skips included.
- **The gate's lease.** The gate needs every bus of every gating case, meaning any mode other than
  `observation` or `historical`. For the release file that is all four cards, recorded as
  `cards_needed`. A run that leases fewer is a skip that names the missing buses.
- **The measurements' bus sets.** Each measured case must give its buses as `cases.<id>.buses`
  in the measurements file, and they must equal the criteria's set. A missing or different set
  fails. `cases.<id>.rank_buses`, the rank-to-bus order the engine logged, is copied into the
  record when given.
- **Tuning.** `conditions.tuning` must match `run_conditions.tuning` key by key, and is required.
- **Written verdicts.** A written verdict is `cases.<id>.written_verdict`, with a `reviewer`, a
  `reason` and `evidence` paths. Every evidence file must exist; the record hashes them.
- **Mode `matched`** (Flash-Next TP4): a `verdict_pass` is accepted only with one of two things.
  The first is `cases.<id>.baseline`: a matched run of the previous release, whose `zip_sha256`
  must be `previous_release.zip_sha256` and which names its raw runs in `evidence`. The second is
  a written verdict that identifies the gap.
- **Mode `screen`**: a `verdict_pass` needs a written verdict.
- **Mode `claim`**: a `verdict_pass` needs the raw evidence (`cases.<id>.evidence`) or a written
  verdict.

```json
{"schema": "brilliance-perf-measurements/1",
 "candidate": {"zip_sha256": "<this ZIP>"},
 "conditions": {"tuning": {"board_power_offset_pct": -20, "board_power_limit_w": 240,
                           "voltage_offset_mv": -50, "vram_mhz": 2518}, "window": "<the solo window>"},
 "cases": {"flashnext_tp2_default_prefill": {"buses": [3, 9], "rank_buses": [3, 9],
                                             "evidence": ["<a1.json>", "<a2.json>"]},
           "flashnext_tp4_standard": {"buses": [3, 6, 9, 32],
                                      "baseline": {"zip_sha256": "<previous release ZIP>", "evidence": ["..."]}}},
 "metrics": {"flashnext_tp2_default_prefill.prefill_32768": {"value": 6010.2, "spread_pct": 1.4},
             "flashnext_tp4_standard.verdict_pass": 1}}
```

## Tests

`win/tests/gate_runner_test.py` (ctest `gate_runner_test`, label `unit`) builds fixture
candidates and runs the runner on them. A fixture candidate is a small ZIP with a MANIFEST, logs,
check plug-ins, a criteria file and measurements. The test exercises:

- the pass path;
- fail paths:
  - a `FAIL` line under exit 0;
  - a passing log with a non-zero exit;
  - failed and unexpected ctest skips;
  - a changed file;
  - an identity mismatch, which blocks the rest;
- skip paths: from the inputs, by card lease, by run class, by a missing source, by `--only`, and a
  refused skip with no reason;
- carryover paths: one accepted, one whose evidence fails, and each refusal;
- both extension points, including the criteria cases: bus sets per case, the four-bus lease,
  tuning, a matched baseline or a written verdict, and a read of the shipped
  `win/tools/perf-criteria.json`;
- `--resume`;
- the build lease, both free and held;
- refused plans.

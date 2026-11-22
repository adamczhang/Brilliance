#!/usr/bin/env python3
"""Brilliance: docs_check's settings-reference rule (win/tools/docs_check.py, REFERENCE.md).

    python -I win/tests/docs_check_reference_test.py [REPO_ROOT]

1. A fixture in memory: a document and a code table that disagree in every way the rule knows (a
   setting the code reads and the document misses, one the document lists and the code no longer
   reads, a cited file that does not name the setting, a cited line that moved, an unknown table
   kind), and one row that is right.
2. The repository itself: REFERENCE.md against the code is clean; with one real row taken out the
   rule names the missing setting, and with a row for a variable nothing reads it names the stale one.

Reads files only; writes nothing. Exit 0 when every case holds, 1 otherwise.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "win", "tools"))
import docs_check  # noqa: E402

failures = []


def expect(cond, what):
    print(("ok   " if cond else "FAIL ") + what)
    if not cond:
        failures.append(what)


def has(items, *words):
    return any(all(w in "%s %s" % it for w in words) for it in items)


# ------------------------------------------------------------------ 1. the fixture
FILES = {
    "src/a.cpp": 'int x;\nconst char* a = getenv("FOO_A");\nconst char* b = getenv("FOO_B");\n',
    "src/c.cpp": '"alpha" is parsed here\n\n\n\n\n',
}
CODE = {
    "env": {"FOO_A": [("src/a.cpp", 2)], "FOO_B": [("src/a.cpp", 3)]},
    "yaml": {"alpha": [("src/c.cpp", 1)]},
}
DOC = """# fixture

<!-- settings: env -->

| Variable | Meaning | Read at |
|---|---|---|
| `FOO_A` | right | `src/a.cpp:2` |
| `FOO_GONE` | stale | `src/a.cpp:2` |

<!-- settings: yaml -->

| Key | Meaning | Read at |
|---|---|---|
| `alpha` | moved | `src/c.cpp:4` |

<!-- settings: nonsense -->

| Key | Meaning | Read at |
|---|---|---|
| `zeta` | unknown kind | `src/c.cpp:1` |
""".split("\n")

errors, warnings, fixes = docs_check.reference_findings("FIX.md", DOC, CODE, FILES.get)
expect(has(errors, "FOO_B", "missing from FIX.md"), "a setting read in code and missing from the document is an error")
expect(has(errors, "FOO_GONE", "no longer read"), "a setting listed and no longer read is an error")
expect(has(errors, "FOO_GONE", "does not name it"), "a Read at file that does not name the setting is an error")
expect(has(errors, "unknown settings kind 'nonsense'"), "a table of an unknown kind is an error")
expect(not has(errors, "FOO_A"), "a right row raises nothing")
expect(has(warnings, "alpha", "src/c.cpp:4", "line 1"), "a Read at line that moved is a warning naming the new line")
expect(fixes == [(14, "`src/c.cpp:4`", "`src/c.cpp:1`")], "--fix-reference-lines would rewrite that cite: %r" % (fixes,))
expect(len(errors) == 4, "exactly the four errors (got %d: %r)" % (len(errors), errors))

# ------------------------------------------------------------------ 2. the repository
path = os.path.join(ROOT, docs_check.REFERENCE_DOC)
with open(path, encoding="utf-8") as f:
    lines = f.read().split("\n")
code = docs_check.settings_in_code(ROOT)


def read(rel):
    p = os.path.join(ROOT, rel)
    return docs_check._read(ROOT, rel) if os.path.isfile(p) else None


errors, warnings, _ = docs_check.reference_findings("REFERENCE.md", lines, code, read)
expect(not errors, "REFERENCE.md matches the code (%d errors%s)" % (len(errors), ": %r" % errors[:3] if errors else ""))
expect(len(code.get("env", {})) >= 47 and len(code.get("yaml", {})) >= 150,
       "the scan finds the settings (env %d, yaml %d)" % (len(code.get("env", {})), len(code.get("yaml", {}))))

victim = "BRILLIANCE_PIN_LIMIT_GIB"
cut = [l for l in lines if not l.startswith("| `%s`" % victim)]
expect(len(cut) == len(lines) - 1, "the fixture removes one real row (%s)" % victim)
errors, _, _ = docs_check.reference_findings("REFERENCE.md", cut, code, read)
expect(has(errors, "env", victim, "missing from REFERENCE.md"), "a real setting missing from REFERENCE.md is caught")

# After BRILLIANCE_HF_CACHE's row in the env tables (the page's top table, unmarked, names it too).
i = next(n for n, kind, names, _ in docs_check.reference_rows(lines) if kind == "env" and "BRILLIANCE_HF_CACHE" in names)
stale = lines[:i] + ["| `BRILLIANCE_NO_SUCH_SWITCH` | gone | unset | - | developer | `core/hf_model.cpp:122` |"] + lines[i:]
errors, _, _ = docs_check.reference_findings("REFERENCE.md", stale, code, read)
expect(has(errors, "BRILLIANCE_NO_SUCH_SWITCH", "no longer read"), "a stale setting in REFERENCE.md is caught")

print("docs_check_reference_test: %d failed" % len(failures))
sys.exit(1 if failures else 0)

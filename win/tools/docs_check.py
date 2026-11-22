#!/usr/bin/env python3
"""Brilliance: check the release documents' links, paths and internal references.

    python -I win/tools/docs_check.py            from the repository root
    python -I win/tools/docs_check.py --root DIR

The release documents are what a package ships (win/package.ps1 and the install rules) plus the
repository's README.md. For each one this checks:

  - every relative Markdown link resolves to a file in the repository, and its #anchor to a
    heading of the target;
  - every backticked package path (bin/, profiles/, tools/, libexec/, share/, the root files)
    names something the package carries, and every backticked source path (win/..., docs/...)
    exists;
  - every profile a `serve.ps1 <profile>` or `profiles/<name>.yaml` names exists in win/profiles;
  - every BRILLIANCE_* variable it names is read somewhere in the code;
  - no internal material: PLAN.md, spec.md, artifacts/, win/investigations/, the ledger,
    UPSTREAM-UPDATE.md, the build machine's name, run or stage keys;
  - that every document package.ps1 copies is in this list, so a new one is not missed;
  - that every known-issue row in SUPPORT.md has an Owner, a link to evidence, a recognized
    Status and a Next check;
  - that REFERENCE.md lists every setting the code reads (environment variables, YAML keys, the
    command-line flags of radiance, rad-convert and rad-export, the Windows scripts' parameters, a
    profile's marked comment lines) and nothing the code no longer reads, each with a `file:line`
    that names it (settings_in_code, reference_findings). --fix-reference-lines rewrites the cited
    lines that have moved.

Errors exit 1; warnings are printed and exit 0. Reads files only: no GPU, no network. Lowers its
own priority to below normal on Windows. Standard library only.
"""
import argparse
import functools
import os
import re
import sys

BS = chr(92)

# (path in the repository, path in the package or None for the repository only, links checked)
RELEASE_DOCS = [
    ("README.md", None),
    ("REFERENCE.md", "REFERENCE.md"),
    ("win/PREVIEW.md", "PREVIEW.md"),
    ("win/SUPPORT.md", "SUPPORT.md"),
    ("win/SERVICE.md", "SERVICE.md"),
    ("win/TERMS-OF-USE.md", "TERMS-OF-USE.md"),
    ("win/model-licenses/index.md", "share/doc/radiance/model-licenses/index.md"),
    ("win/tools/stacks/README.md", "tools/STACKS.md"),
    ("win/licenses/THIRD-PARTY.md", "share/doc/radiance/third_party/THIRD-PARTY.md"),
    ("NOTICE", "share/doc/radiance/NOTICE"),
]
PACKAGE_DOC_PATHS = {d: pkg for d, pkg in RELEASE_DOCS if pkg}
PROFILE_DIR = "win/profiles"          # shipped as profiles/; their comments are checked too

# What the package carries, by package path, mapped to its source in the repository. Paths under
# bin/ and share/radiance/ come from the build and are listed by name only.
PACKAGE_FILES = {
    "serve.ps1": "win/serve.ps1",
    "PREVIEW.md": "win/PREVIEW.md",
    "REFERENCE.md": "REFERENCE.md",
    "SUPPORT.md": "win/SUPPORT.md",
    "SERVICE.md": "win/SERVICE.md",
    "TERMS-OF-USE.md": "win/TERMS-OF-USE.md",
    "MANIFEST.json": None,
    "tools/serve-bg.ps1": "win/tools/serve-bg.ps1",
    "tools/stop-server.ps1": "win/tools/stop-server.ps1",
    "tools/reload-server.ps1": "win/tools/reload-server.ps1",
    "tools/STACKS.md": "win/tools/stacks/README.md",
    "libexec/brilliance/hfget.ps1": "win/libexec/hfget.ps1",
    "libexec/brilliance/resolve-stacks.py": "win/tools/stacks/resolve-stacks.py",
    "share/doc/radiance": None,
    "share/doc/radiance/LICENSE": "LICENSE",
    "share/doc/radiance/NOTICE": "NOTICE",
    "share/doc/radiance/third_party": "win/licenses",
    "share/doc/radiance/model-licenses": "win/model-licenses",
    "share/radiance": None,
    "share/radiance/recipes": None,
    "profiles": "win/profiles",
    "bin": None,
    "tools": None,
    "libexec/brilliance": None,
}
BIN_NAMES = {
    "radiance.exe", "radiance.map", "brl-stacks.exe", "rad-export", "rad-export.cmd", "rad-convert.exe",
    "amdhip64_7.dll", "amd_comgr.dll", "rocm_kpack.dll", "libomp.dll", "avcodec-62.dll",
    "avformat-62.dll", "avutil-60.dll", "swresample-6.dll", "swscale-9.dll",
}

# Internal material: (pattern, what it is). Matched against each line, case-sensitive.
INTERNAL = [
    (r"\bPLAN\.md\b", "the implementation plan"),
    (r"\bspec\.md\b|\bspec §|\bspec section\b", "the port spec"),
    (r"(^|[^A-Za-z])artifacts[/\x5c]", "an artifacts/ path"),
    (r"investigations[/\x5c]", "an investigation file"),
    (r"LEDGER|\bledger\b", "the hang ledger"),
    (r"UPSTREAM-UPDATE\.md", "the upstream-update procedure"),
    (r"\bRED\b", "the build machine's name"),
    (r"\bM[0-9]+ record\b|\bPLAN\b|\bsection 10\b", "a plan record"),
    (r"\b(stage|fix) C[0-9]\b|\bC1 stage\b", "an internal fix key"),
    (r"`(flush-1|fn-tp[0-9][^`]*|q27-[^`]*)`", "a run name"),
    (r"\bthe owner\b", "an internal decision log (say what holds, not who decided)"),
    (r"\badlx\b", "a private tool"),
    (r"\b[0-9][0-9,]* (checks|passes|rounds|real faults)\b|\b[0-9]+ parks\b|\bsoak ran\b",
     "test-run counts (say the status word instead)"),
    (r"^#+ (Decided|Open, needs)", "a decision-log section"),
]
# Known findings left for a later change, reported as warnings rather than errors: (document,
# text on the line, why it is left). Remove an entry when its line is fixed; an entry whose text is
# no longer found is itself reported.
PENDING = [
]

# Links that are allowed although they point at a file not in the package (repository README only).
README_ALLOWED_LINKS = {"RADIANCE-README.md", "LICENSE", "NOTICE", "docs/", "docs/GUIDE.md",
                        "win/setup.ps1", "win/ffmpeg/", "win/licenses/", "docs/img/banner.png"}

LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")
TICK_RE = re.compile(r"`([^`\n]+)`")
VAR_RE = re.compile(r"\bBRILLIANCE_[A-Z0-9_]+\b")
# A profile name may hold a dot (ltx-2.5-1card) but not end in one: "serve.ps1 x." ends a sentence.
PROFILE_RE = re.compile(r"serve\.ps1\s+([a-z0-9](?:[a-z0-9-]|\.(?=[a-z0-9]))*)"
                        r"|profiles[/\x5c]([a-z0-9][a-z0-9.-]*)\.yaml")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
KNOWN_ISSUES_DOC = "win/SUPPORT.md"
KNOWN_ISSUE_COLUMNS = ["Issue or limit", "Owner", "Evidence", "Status", "Next check"]
KNOWN_ISSUE_STATUSES = {"open", "preview", "not supported", "explained"}
TABLE_PIPE_RE = re.compile(r"(?<!\\)\|")
EVIDENCE_LINK_RE = re.compile(r"\[[^\]\s][^\]]*\]\(([^)\s]+)\)")


def lower_priority():
    if os.name == "nt":
        try:
            import ctypes
            k = ctypes.windll.kernel32
            k.SetPriorityClass(k.GetCurrentProcess(), 0x00004000)   # BELOW_NORMAL_PRIORITY_CLASS
        except Exception:
            pass
    else:
        try:
            os.nice(5)
        except Exception:
            pass


def slug(text):
    t = re.sub(r"<[^>]+>", "", text)
    t = t.replace("`", "").strip().lower()
    t = re.sub(r"[^\w\- ]", "", t)
    return t.replace(" ", "-")


def anchors(path):
    out, seen = set(), {}
    in_code = False
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.lstrip().startswith("```"):
                in_code = not in_code
                continue
            m = None if in_code else HEADING_RE.match(line)
            if not m:
                continue
            s = slug(m.group(2))
            n = seen.get(s, 0)
            seen[s] = n + 1
            out.add(s if n == 0 else "%s-%d" % (s, n))
    return out


class Report:
    def __init__(self):
        self.errors, self.warnings = [], []

    def err(self, where, msg):
        self.errors.append("%s: %s" % (where, msg))

    def warn(self, where, msg):
        self.warnings.append("%s: %s" % (where, msg))


def norm(p):
    return p.replace(BS, "/")


def check_link(root, doc, n, target, rep):
    if re.match(r"^[a-z]+:", target) or target.startswith("mailto:"):
        return
    path, _, frag = target.partition("#")
    where = "%s:%d" % (doc, n)
    if not path:
        dest = os.path.join(root, doc)
    else:
        dest = os.path.normpath(os.path.join(root, os.path.dirname(doc), path))
    if not os.path.exists(dest):
        rep.err(where, "link %s does not resolve" % target)
        return
    rel = norm(os.path.relpath(dest, root))
    if doc in PACKAGE_DOC_PATHS:
        # A relative link in a shipped document must reach the same document in the ZIP.
        pkg_doc = PACKAGE_DOC_PATHS[doc]
        pkg_dest = norm(os.path.normpath(os.path.join(os.path.dirname(pkg_doc), path)))
        if not path:
            pkg_dest = pkg_doc
        if PACKAGE_DOC_PATHS.get(rel) != pkg_dest:
            rep.warn(where, "link %s does not resolve to the same document in the package" % target)
    elif doc == "README.md" and \
            not any(rel == a.rstrip("/") or rel.startswith(a) for a in README_ALLOWED_LINKS) and \
            rel not in {d for d, _ in RELEASE_DOCS}:
        rep.warn(where, "link %s points at %s, not a release document" % (target, rel))
    if frag and dest.endswith(".md") and frag not in anchors(dest):
        rep.err(where, "link %s: no heading #%s in %s" % (target, frag, rel))


def check_tick(root, doc, n, tok, rep):
    t = norm(tok.strip())
    if " " in t or "<" in t or "*" in t or t.startswith("-") or t.startswith("$"):
        return
    t = t[2:] if t.startswith("./") else t
    where = "%s:%d" % (doc, n)
    cite = CITE_RE.fullmatch("`%s`" % t)
    if cite:
        # A source line (`core/config.cpp:701`, REFERENCE.md's Read at): a file in the repository.
        if not os.path.isfile(os.path.join(root, cite.group(1))):
            rep.err(where, "`%s`: no such file in the source" % tok)
        return
    first = t.split("/")[0]
    if t.startswith("bin/"):
        name = t[4:]
        if name and name not in BIN_NAMES:
            rep.warn(where, "`%s`: not a name the package's bin/ is known to carry (BIN_NAMES)" % tok)
        return
    if t.startswith("share/radiance/") and not t.startswith("share/radiance/recipes"):
        return
    if t.startswith("share/radiance/recipes/"):
        # The build installs data/recipes as share/radiance/recipes (win/cmake/RadWindows.cmake),
        # so a recipe a document names must be one in data/recipes.
        name = t[len("share/radiance/recipes/"):]
        if name and not os.path.exists(os.path.join(root, "data/recipes", name)):
            rep.err(where, "`%s`: no such recipe in data/recipes" % tok)
        return
    if first in ("profiles", "tools", "libexec", "share") or t in PACKAGE_FILES:
        key = t.rstrip("/")
        if key in PACKAGE_FILES:
            return
        if key.startswith("profiles/"):
            if not os.path.exists(os.path.join(root, PROFILE_DIR, key[len("profiles/"):])):
                rep.err(where, "`%s`: no such profile in %s" % (tok, PROFILE_DIR))
            return
        if key.startswith("share/doc/radiance/third_party/"):
            name = key[len("share/doc/radiance/third_party/"):]
            if not os.path.exists(os.path.join(root, "win/licenses", name)):
                rep.warn(where, "`%s`: not in win/licenses (it may come from the build)" % tok)
            return
        rep.err(where, "`%s`: not a path the package carries (PACKAGE_FILES)" % tok)
        return
    if first in ("win", "docs", "core", "libr4d", "deploy", "scripts") and "/" in t:
        p = t.split(":")[0].rstrip("/")
        if "." in os.path.basename(p) or p.endswith("/"):
            if not os.path.exists(os.path.join(root, p)):
                rep.err(where, "`%s`: no such file in the source" % tok)


def code_vars(root):
    found = set()
    for top in ("core", "libr4d", "win", "tools", "plugins", "kernels", "arch"):  # arch: model code reads switches too (BRILLIANCE_TP2_PIPELINE)
        base = os.path.join(root, top)
        for d, dirs, files in os.walk(base):
            dirs[:] = [x for x in dirs if x not in ("investigations", "licenses", "ffmpeg")]
            for f in files:
                if os.path.splitext(f)[1].lower() in (".cpp", ".h", ".hpp", ".c", ".hip", ".ps1", ".py", ".cmake"):
                    try:
                        with open(os.path.join(d, f), encoding="utf-8", errors="replace") as fh:
                            found.update(VAR_RE.findall(fh.read()))
                    except OSError:
                        pass
    return found


# ------------------------------------------------------------------ the settings reference
#
# REFERENCE.md lists every setting a person can set: the YAML keys, the command-line flags of the
# server and the shipped tools, the parameters of the Windows scripts, and every environment
# variable the code reads. Each table in it follows a marker line `<!-- settings: KIND -->`, and a
# row's first cell names the setting in backticks (several when they are spellings of one). The
# check compares those names, kind by kind, with what the code reads, both ways, and checks that
# each row's `file:line` cites a file that names the setting.

REFERENCE_DOC = "REFERENCE.md"

# Where the environment is read. C and C++ sources in these trees (what radiance, its plugins and
# the tools are built from), less the test and repro trees, which configure tests, not the product.
ENV_C_TREES = ("core", "libr4d", "libhalo", "libr3d", "libquant", "libavx", "libref", "arch", "plugin",
               "win/device", "win/os", "win/compat", "win/diag", "tools")
ENV_C_SKIP_DIRS = ("repro", "tests", "investigations", "fixtures")
ENV_TEST_TREES = ("tests", "win/tests")
# The Windows scripts a user, an operator or a developer runs; the CMake files the build reads;
# the shipped Python tools.
SCRIPTS = {
    "serve.ps1": "win/serve.ps1",
    "setup.ps1": "win/setup.ps1",
    "package.ps1": "win/package.ps1",
    "build.ps1": "win/build.ps1",
    "serve-bg.ps1": "win/tools/serve-bg.ps1",
    "stop-server.ps1": "win/tools/stop-server.ps1",
    "reload-server.ps1": "win/tools/reload-server.ps1",
    "hfget.ps1": "win/libexec/hfget.ps1",
}
ENV_PS_FILES = tuple(SCRIPTS.values()) + ("win/env.ps1",)
ENV_CMAKE = ("CMakeLists.txt", "cmake", "win/cmake")
ENV_PY_FILES = ("tools/rad-export", "win/tools/stacks/resolve-stacks.py")

C_EXT = (".cpp", ".h", ".hpp", ".c", ".hip", ".cc")
# A literal handed to a function whose name holds "env" (getenv, env_on, set_env_default, ...),
# less the writers. And a literal that is a whole variable name with one of these prefixes, which is
# how a table of names (`for (k : {"GPU_MAX_HW_QUEUES", ...}) getenv(k)`) is read.
ENV_CALL_RE = re.compile(r"\b(\w*env\w*)\s*\(\s*L?\"([A-Za-z_][A-Za-z0-9_]*)\"", re.I)
ENV_NOT_READERS = {"_putenv_s", "putenv", "_putenv", "setenv", "_wputenv_s", "unsetenv"}
ENV_PREFIX_RE = re.compile(r"\"((?:BRILLIANCE|RADIANCE|R4D|R3D|HALO|GPU|AMD|PAL|HIP|HSA|ROCR)_[A-Z0-9_]*[A-Z0-9])\"")
# Literals of that shape that are not variable names.
ENV_NOT_NAMES = {
    "RADIANCE_VALUE_PROBE",   # core/text/chatfmt.cpp: a sentinel value rendered through a chat template
}
PS_ENV_RE = re.compile(r"\$env:([A-Za-z_][A-Za-z0-9_]*)\b(?!\s*=[^=])", re.I)
PS_GETENV_RE = re.compile(r"GetEnvironmentVariable\(\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]", re.I)
CMAKE_ENV_RE = re.compile(r"\$ENV\{([A-Za-z_][A-Za-z0-9_]*)\}")
PY_ENV_RE = re.compile(r"os\.(?:environ\.get|getenv)\(\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]"
                       r"|os\.environ\[\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]\s*\](?!\s*=[^=])"
                       r"|['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]\s+in\s+os\.environ")


def _walk(root, top, exts, skip=()):
    base = os.path.join(root, top)
    if os.path.isfile(base):
        yield norm(top)
        return
    for d, dirs, files in os.walk(base):
        dirs[:] = sorted(x for x in dirs if x not in skip)
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in exts:
                yield norm(os.path.relpath(os.path.join(d, f), root))


def _read(root, rel):
    try:
        with open(os.path.join(root, rel), encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def _add(out, kind, name, rel, text, pos):
    locs = out.setdefault(kind, {}).setdefault(name, [])
    loc = (rel, text.count("\n", 0, pos) + 1)
    if loc not in locs:
        locs.append(loc)


# Left to right, the first of: a string or character literal (kept), a // or /* */ comment (blanked).
C_TOKEN_RE = re.compile(r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|//[^\n]*|/\*.*?(?:\*/|\Z)', re.S)


def _strip_c_comments(text):
    """Comments blanked to spaces, newlines kept, so a name a comment mentions is not a read and
    line numbers still hold. String literals are kept whole."""
    def blank(m):
        t = m.group(0)
        return t if t[0] in "\"'" else re.sub(r"[^\n]", " ", t)
    return C_TOKEN_RE.sub(blank, text)


def _strip_hash_comments(text):
    """# comments blanked (PowerShell, CMake, Python), and PowerShell's <# ... #> blocks."""
    text = re.sub(r"<#.*?#>", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.S)
    return re.sub(r"(?m)(^|\s)#[^\n]*", lambda m: m.group(1), text)


def env_reads(root):
    out = {}
    for top in ENV_C_TREES:
        for rel in _walk(root, top, C_EXT, ENV_C_SKIP_DIRS):
            text = _strip_c_comments(_read(root, rel))
            for m in ENV_CALL_RE.finditer(text):
                if m.group(1) not in ENV_NOT_READERS:
                    _add(out, "env", m.group(2), rel, text, m.start(2))
            for m in ENV_PREFIX_RE.finditer(text):
                if m.group(1) not in ENV_NOT_NAMES:
                    _add(out, "env", m.group(1), rel, text, m.start(1))
    # The test trees: Brilliance's own switches only (BRILLIANCE_*). Upstream's test harness has
    # dozens of its own (RAD_TEST_*, SIM_*, ...), which configure its tests, not the product.
    for top in ENV_TEST_TREES:
        for rel in _walk(root, top, C_EXT + (".ps1", ".py")):
            text = _read(root, rel)
            if rel.endswith(C_EXT):
                text = _strip_c_comments(text)
                found = [m for m in ENV_CALL_RE.finditer(text) if m.group(1) not in ENV_NOT_READERS]
            else:
                text = _strip_hash_comments(text)
                found = list(PS_ENV_RE.finditer(text)) + list(PY_ENV_RE.finditer(text))
            for m in found:
                name = [g for g in m.groups() if g and g.isupper()]
                name = name[-1] if name else None
                if name and name.startswith("BRILLIANCE_"):
                    _add(out, "env", name, rel, text, m.start())
    for rel in ENV_PS_FILES:
        text = _strip_hash_comments(_read(root, rel))
        for rx in (PS_ENV_RE, PS_GETENV_RE):
            for m in rx.finditer(text):
                _add(out, "env", m.group(1), rel, text, m.start(1))
    for top in ENV_CMAKE:
        for rel in _walk(root, top, (".cmake", ".txt")):
            if top != "CMakeLists.txt" and not rel.endswith(".cmake"):
                continue
            text = _strip_hash_comments(_read(root, rel))
            for m in CMAKE_ENV_RE.finditer(text):
                _add(out, "env", m.group(1), rel, text, m.start(1))
    for rel in ENV_PY_FILES:
        text = _strip_hash_comments(_read(root, rel))
        for m in PY_ENV_RE.finditer(text):
            name = m.group(1) or m.group(2) or m.group(3)
            _add(out, "env", name, rel, text, m.start())
    # Names spelled in the source as a case-insensitive Windows variable ($env:Path) are one setting.
    return out


def _ps_params(text):
    """The parameter names of a script's top-level param( ... ) block."""
    m = re.search(r"(?im)^\s*(?:\[CmdletBinding[^\]]*\]\s*)?param\s*\(", text)
    if not m:
        return []
    i, depth, start = m.end(), 1, m.end()
    body = []
    while i < len(text) and depth:
        c = text[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif depth == 1:
            body.append((i, c))
        i += 1
    flat = "".join(c for _, c in body)
    names = []
    # At depth 1, outside attributes and defaults: [type]$Name, or $Name after a comma.
    for pm in re.finditer(r"(^|[,\]\s])\$([A-Za-z_][A-Za-z0-9_]*)\s*(?==|,|$)", flat):
        prev = flat[:pm.start(2)].rstrip("$ \t\r\n")
        if prev.endswith("="):
            continue
        if pm.group(2).lower() in ("true", "false", "null", "psscriptroot"):
            continue
        names.append((pm.group(2), body[pm.start(2)][0]))
    return names


def cli_flags(root):
    out = {}
    # radiance: its own command line (main.cpp) and the service mode's (service_win32.cpp).
    for rel, rx in (("core/main.cpp", r"strcmp\(argv\[i\],\s*\"(-{1,2}[a-z][a-z0-9-]*)\"\)"),
                    ("win/os/service_win32.cpp", r"(?:\bs ==|strcmp\(argv\[i\],)\s*\"(--[a-z][a-z0-9-]*)\"")):
        text = _strip_c_comments(_read(root, rel))
        for m in re.finditer(rx, text):
            _add(out, "radiance", m.group(1), rel, text, m.start(1))
    rel = "tools/rad_convert.cpp"
    text = _strip_c_comments(_read(root, rel))
    for m in re.finditer(r"\ba == \"(-{1,2}[a-z][a-z0-9-]*)\"", text):
        _add(out, "rad-convert", m.group(1), rel, text, m.start(1))
    for kind, rel in (("rad-export", "tools/rad-export"),):
        text = _strip_hash_comments(_read(root, rel))
        for m in re.finditer(r"add_argument\(((?:\s*\"-[^\"]+\",?)+)", text):
            for f in re.finditer(r"\"(-[^\"]+)\"", m.group(1)):
                _add(out, kind, f.group(1), rel, text, m.start(1) + f.start(1))
    for kind, rel in SCRIPTS.items():
        text = _strip_hash_comments(_read(root, rel))
        for name, pos in _ps_params(text):
            _add(out, "ps:" + kind, "-" + name, rel, text, pos)
        # The marked comment lines a profile carries for the launcher (`# brilliance-env: ...`),
        # as the script's code matches them.
        for m in re.finditer(r"#\\s\*(brilliance-[a-z]+)\\s\*:", text):
            _add(out, "profile-comment", "# " + m.group(1) + ":", rel, text, m.start(1))
    return out


def yaml_keys(root):
    """The YAML file's keys: every option of the flag table (core/config.cpp) less those the YAML
    spells differently (kReplaced), and the levels' own keys (core/fleet/config.cpp)."""
    out = {}
    rel = "core/fleet/config.cpp"
    fleet = _strip_c_comments(_read(root, rel))
    # server_overrides compares the launcher's variables' values (k == "off"), not keys.
    fleet = re.sub(r"int Reader::server_overrides\(\)\s*\{.*?\n\}\n",
                   lambda m: re.sub(r"[^\n]", " ", m.group(0)), fleet, count=1, flags=re.S)
    m = re.search(r"kReplaced\[\]\s*=\s*\{(.*?)\n\};", fleet, re.S)
    replaced = set(re.findall(r"\{\s*\"([a-z-]+)\"", m.group(1))) if m else set()
    crel = "core/config.cpp"
    ctext = _strip_c_comments(_read(root, crel))
    m = re.search(r"g_flags\[\]\s*=\s*\{(.*?)\n\};", ctext, re.S)
    if m:
        for f in re.finditer(r"^\s*\{\s*\"-{1,2}([a-z][a-z0-9-]*)\"", m.group(1), re.M):
            if f.group(1) not in replaced:
                _add(out, "yaml", f.group(1), crel, ctext, m.start(1) + f.start(1))
    for arr in ("kTopKeys", "kModelFleetKeys"):
        m = re.search(arr + r"\[\]\s*=\s*\{(.*?)\};", fleet, re.S)
        if m:
            for k in re.finditer(r"\"([a-z][a-z0-9-]*)\"", m.group(1)):
                _add(out, "yaml", k.group(1), rel, fleet, m.start(1) + k.start(1))
    for k in re.finditer(r"\b(?:k|fk|pk|kind) [!=]= \"([a-z][a-z0-9-]*)\"", fleet):
        _add(out, "yaml", k.group(1), rel, fleet, k.start(1))
    return out


@functools.lru_cache(maxsize=4)   # main and check_reference both ask; the scan reads the tree
def settings_in_code(root):
    out = {}
    for part in (env_reads(root), cli_flags(root), yaml_keys(root)):
        for kind, names in part.items():
            for name, locs in names.items():
                out.setdefault(kind, {}).setdefault(name, []).extend(locs)
    return out


MARKER_RE = re.compile(r"^<!--\s*settings:\s*(\S+)\s*-->\s*$")
CITE_RE = re.compile(r"`([A-Za-z0-9_./\x5c-]+\.[A-Za-z0-9]+|tools/rad-export):([0-9]+)`")


def reference_rows(lines):
    """(line number, kind, [names], [(file, line, column of the cite)]) for each row of each marked
    table. A table runs from its marker to the first line that is neither blank nor a table row."""
    rows, kind, in_table = [], None, False
    for n, line in enumerate(lines, 1):
        m = MARKER_RE.match(line.strip())
        if m:
            kind, in_table = m.group(1), False
            continue
        if kind is None:
            continue
        s = line.strip()
        if not s.startswith("|"):
            if s or in_table:
                kind = None
            continue
        in_table = True
        cells = [c.strip() for c in re.split(r"(?<!\x5c)\|", s)[1:-1]]
        if not cells or set(cells[0]) <= set("-: "):
            continue
        names = TICK_RE.findall(cells[0])
        if not names:          # the header row
            continue
        cites = [(norm(c.group(1)), int(c.group(2)), c.start()) for c in CITE_RE.finditer(line)]
        rows.append((n, kind, names, cites))
    return rows


def _bare(name):
    """What a source file spells: the YAML key or flag without its dashes, the parameter without
    its dash, the comment line's word."""
    return name.strip().lstrip("#").strip().rstrip(":").lstrip("-")


def _names_line(text_line, bare):
    return re.search(r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % re.escape(bare), text_line, re.I) is not None


def reference_findings(doc, lines, code, read):
    """The drift check: compare REFERENCE.md's tables (lines) with the settings the code reads
    (code, as settings_in_code returns it). read(rel) gives a source file's text, or None. Returns
    (errors, warnings, fixes); a fix is (doc line, old cite, new cite) for --fix-reference-lines."""
    errors, warnings, fixes = [], [], []
    listed = {}
    for n, kind, names, cites in reference_rows(lines):
        where = "%s:%d" % (doc, n)
        if kind not in code and not (kind.startswith("ps:") and kind[3:] in SCRIPTS):
            errors.append((where, "unknown settings kind '%s' (the marker before this table)" % kind))
            continue
        for name in names:
            listed.setdefault(kind, {}).setdefault(name, n)
        if not cites:
            errors.append((where, "%s: no `file:line` saying where the code reads it" % names[0]))
            continue
        for rel, ln, _ in cites:
            text = read(rel)
            if text is None:
                errors.append((where, "%s: cites %s, which does not exist" % (names[0], rel)))
                continue
            src = text.split("\n")
            want = [_bare(x) for x in names]
            if not any(_names_line(text, b) for b in want):
                errors.append((where, "%s: cites %s, which does not name it" % (names[0], rel)))
                continue
            if 1 <= ln <= len(src) and any(_names_line(src[ln - 1], b) for b in want):
                continue
            # The line moved: the nearest place the scan found it read in that file is the fix, else
            # the nearest line that names it.
            hits = [l for x in names for f, l in code.get(kind, {}).get(x, []) if f == rel]
            hits = hits or [i + 1 for i, t in enumerate(src) if any(_names_line(t, b) for b in want)]
            best = min(hits, key=lambda i: abs(i - ln))
            warnings.append((where, "%s: %s:%d no longer names it; it is at line %d (--fix-reference-lines)"
                             % (names[0], rel, ln, best)))
            fixes.append((n, "`%s:%d`" % (rel, ln), "`%s:%d`" % (rel, best)))
    for kind in sorted(set(code) | set(listed)):
        have, doc_names = code.get(kind, {}), listed.get(kind, {})
        for name in sorted(set(have) - set(doc_names)):
            rel, ln = have[name][0]
            errors.append(("%s:%d" % (rel, ln), "%s %s is read here and missing from %s" % (kind, name, doc)))
        for name in sorted(set(doc_names) - set(have)):
            errors.append(("%s:%d" % (doc, doc_names[name]),
                           "%s %s is listed but no longer read anywhere in the code" % (kind, name)))
    return errors, warnings, fixes


def check_reference(root, rep, fix_lines=False):
    path = os.path.join(root, REFERENCE_DOC)
    if not os.path.exists(path):
        rep.err(REFERENCE_DOC, "missing: the settings reference")
        return
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")

    @functools.lru_cache(maxsize=None)
    def read(rel):
        p = os.path.join(root, rel)
        return _read(root, rel) if os.path.isfile(p) else None

    errors, warnings, fixes = reference_findings(REFERENCE_DOC, lines, settings_in_code(root), read)
    if fix_lines and fixes:
        for n, old, new in fixes:
            lines[n - 1] = lines[n - 1].replace(old, new)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines))
        print("docs_check: %d Read at lines rewritten in %s" % (len(fixes), REFERENCE_DOC))
        warnings = [w for w in warnings if "--fix-reference-lines" not in w[1]]
    for w in warnings:
        rep.warn(*w)
    for e in errors:
        rep.err(*e)


def shipped_docs_from_package(root):
    """The .md files win/package.ps1 copies, as repo paths."""
    out = set()
    p = os.path.join(root, "win/package.ps1")
    with open(p, encoding="utf-8") as f:
        for line in f:
            if not line.startswith("Copy-Item"):
                continue
            m = re.search(r"'([^']+\.md)'", line)    # the first is the source; a second, the name in the package
            if m:
                out.add(norm(os.path.normpath("win/" + norm(m.group(1)))))  # ..\REFERENCE.md: the root
    return out


def lines_of(root, doc, comments_only=False):
    with open(os.path.join(root, doc), encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if comments_only and not line.lstrip().startswith("#"):
                continue
            yield n, line.rstrip("\n")


def check_known_issues(lines, rep):
    """Check the Markdown table directly under SUPPORT.md's Known issues heading."""
    section = []
    heading_line = None
    in_code = False
    for n, line in lines:
        if line.lstrip().startswith("```"):
            in_code = not in_code
        if in_code:
            continue
        heading = HEADING_RE.match(line)
        if heading:
            if heading_line is not None:
                break
            if heading.groups() == ("##", "Known issues"):
                heading_line = n
        elif heading_line is not None:
            section.append((n, line.strip()))

    if heading_line is None:
        rep.err(KNOWN_ISSUES_DOC, "missing Known issues section")
        return

    header = next((i for i, (_, line) in enumerate(section) if line.startswith("|")), len(section))
    table = []
    ended = False
    for n, line in section[header:]:
        if not line:
            ended = True
            continue
        if ended:
            if TABLE_PIPE_RE.search(line):
                rep.err("%s:%d" % (KNOWN_ISSUES_DOC, n), "known issue row is outside the table")
            continue
        cells = [cell.strip() for cell in TABLE_PIPE_RE.split(line.removeprefix("|").removesuffix("|"))]
        table.append((n, cells))

    where = "%s:%d" % (KNOWN_ISSUES_DOC, heading_line)
    if not table or table[0][1] != KNOWN_ISSUE_COLUMNS:
        rep.err(where, "Known issues table must have columns: %s" % ", ".join(KNOWN_ISSUE_COLUMNS))
        return
    if len(table) < 2 or len(table[1][1]) != len(KNOWN_ISSUE_COLUMNS) or any(
            not re.fullmatch(r":?-{3,}:?", cell) for cell in table[1][1]):
        rep.err(where, "Known issues table is missing its Markdown separator")
        return
    if len(table) == 2:
        rep.err(where, "Known issues table has no entries")

    for n, cells in table[2:]:
        where = "%s:%d" % (KNOWN_ISSUES_DOC, n)
        if len(cells) != len(KNOWN_ISSUE_COLUMNS):
            rep.err(where, "known issue must have %d columns (found %d)" % (len(KNOWN_ISSUE_COLUMNS), len(cells)))
            continue
        for column, cell in zip(KNOWN_ISSUE_COLUMNS, cells):
            if not cell.strip(" -–—`*") or re.fullmatch(r"<!--.*-->", cell):
                rep.err(where, "known issue is missing %s" % column)
        evidence = cells[2]
        links = EVIDENCE_LINK_RE.findall(evidence)
        if evidence and not any(
                target.startswith(("https://", "http://")) or (":" not in target and target != "#")
                for target in links):
            rep.err(where, "known issue Evidence must include a Markdown link to the record")
        if cells[3] and cells[3] not in KNOWN_ISSUE_STATUSES:
            rep.err(where, "known issue Status must be open, preview, not supported, or explained")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=".", help="the repository root (default: .)")
    ap.add_argument("--fix-reference-lines", action="store_true",
                    help="rewrite REFERENCE.md's Read at lines that have moved, then check")
    a = ap.parse_args()
    lower_priority()
    root = os.path.abspath(a.root)
    rep = Report()

    listed = {d for d, _ in RELEASE_DOCS}
    for d in sorted(shipped_docs_from_package(root) - listed):
        rep.err("win/package.ps1", "copies %s, which is not in RELEASE_DOCS" % d)

    profiles = sorted(f for f in os.listdir(os.path.join(root, PROFILE_DIR)) if f.endswith(".yaml"))
    profile_names = {f[:-5] for f in profiles}
    # Every variable the code names, and those the settings scan finds read (the test trees too).
    vars_in_code = code_vars(root) | set(settings_in_code(root).get("env", {}))
    docs = [(d, False) for d, _ in RELEASE_DOCS] + [("%s/%s" % (PROFILE_DIR, f), True) for f in profiles]
    pending_seen = set()

    for doc, is_profile in docs:
        if not os.path.exists(os.path.join(root, doc)):
            rep.err(doc, "missing")
            continue
        md = doc.endswith(".md")
        in_code = False
        for n, line in lines_of(root, doc, comments_only=is_profile):
            if md and line.lstrip().startswith("```"):
                in_code = not in_code
            for pat, what in INTERNAL:
                if doc == "NOTICE" and what in ("the build machine's name",):
                    continue
                if re.search(pat, line):
                    msg = "internal reference (%s): %s" % (what, line.strip()[:110])
                    pend = [p for p in PENDING if p[0] == doc and p[1] in line]
                    if pend:
                        pending_seen.add(pend[0][:2])
                        rep.warn("%s:%d" % (doc, n), "pending (%s): %s" % (pend[0][2], msg))
                    else:
                        rep.err("%s:%d" % (doc, n), msg)
            if md and not in_code:
                for m in LINK_RE.finditer(line):
                    check_link(root, doc, n, m.group(1), rep)
            if doc != "NOTICE":
                for m in TICK_RE.finditer(line):
                    check_tick(root, doc, n, m.group(1), rep)
                for m in PROFILE_RE.finditer(line):
                    name = m.group(1) or m.group(2)
                    if name in ("<profile>", "profile") or name.startswith("-"):
                        continue
                    if name not in profile_names and not re.search(r"replac|removed|old ", line):
                        rep.err("%s:%d" % (doc, n), "profile %s is not in %s" % (name, PROFILE_DIR))
                for v in VAR_RE.findall(line):
                    if v not in vars_in_code:
                        rep.err("%s:%d" % (doc, n), "%s is not read anywhere in the code" % v)

    if os.path.isfile(os.path.join(root, KNOWN_ISSUES_DOC)):
        check_known_issues(lines_of(root, KNOWN_ISSUES_DOC), rep)
    check_reference(root, rep, fix_lines=a.fix_reference_lines)

    for p in PENDING:
        if p[:2] not in pending_seen:
            rep.warn(p[0], "PENDING entry no longer found, remove it: %s" % p[1])
    for w in rep.warnings:
        print("warning: " + w)
    for e in rep.errors:
        print("error: " + e)
    print("docs_check: %d release documents, %d profiles; %d errors, %d warnings"
          % (len(RELEASE_DOCS), len(profiles), len(rep.errors), len(rep.warnings)))
    return 1 if rep.errors else 0


if __name__ == "__main__":
    sys.exit(main())

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
  - that every document package.ps1 copies is in this list, so a new one is not missed.

Errors exit 1; warnings are printed and exit 0. Reads files only: no GPU, no network. Lowers its
own priority to below normal on Windows. Standard library only.
"""
import argparse
import os
import re
import sys

BS = chr(92)

# (path in the repository, path in the package or None for the repository only, links checked)
RELEASE_DOCS = [
    ("README.md", None),
    ("win/PREVIEW.md", "PREVIEW.md"),
    ("win/SUPPORT.md", "SUPPORT.md"),
    ("win/SERVICE.md", "SERVICE.md"),
    ("win/TERMS-OF-USE.md", "TERMS-OF-USE.md"),
    ("win/model-licenses/index.md", "share/doc/radiance/model-licenses/index.md"),
    ("win/tools/stacks/README.md", "tools/STACKS.md"),
    ("win/licenses/THIRD-PARTY.md", "share/doc/radiance/third_party/THIRD-PARTY.md"),
    ("NOTICE", "share/doc/radiance/NOTICE"),
]
PROFILE_DIR = "win/profiles"          # shipped as profiles/; their comments are checked too

# What the package carries, by package path, mapped to its source in the repository. Paths under
# bin/ and share/radiance/ come from the build and are listed by name only.
PACKAGE_FILES = {
    "serve.ps1": "win/serve.ps1",
    "PREVIEW.md": "win/PREVIEW.md",
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
PROFILE_RE = re.compile(r"serve\.ps1\s+([a-z0-9][a-z0-9-]*)|profiles[/\x5c]([a-z0-9-]+)\.yaml")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


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
    if doc != "README.md":
        # A shipped document's link must resolve in the package too: only links to other shipped
        # documents, by their package path, do.
        rep.warn(where, "link %s: a Markdown link in a shipped document; check it reads in the package" % target)
    elif not any(rel == a.rstrip("/") or rel.startswith(a) for a in README_ALLOWED_LINKS) and \
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
    first = t.split("/")[0]
    if t.startswith("bin/"):
        name = t[4:]
        if name and name not in BIN_NAMES:
            rep.warn(where, "`%s`: not a name the package's bin/ is known to carry (BIN_NAMES)" % tok)
        return
    if t.startswith("share/radiance/") and not t.startswith("share/radiance/recipes"):
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
    for top in ("core", "libr4d", "win", "tools", "plugins", "kernels"):
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
                out.add("win/" + norm(m.group(1)))
    return out


def lines_of(root, doc, comments_only=False):
    with open(os.path.join(root, doc), encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if comments_only and not line.lstrip().startswith("#"):
                continue
            yield n, line.rstrip("\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=".", help="the repository root (default: .)")
    a = ap.parse_args()
    lower_priority()
    root = os.path.abspath(a.root)
    rep = Report()

    listed = {d for d, _ in RELEASE_DOCS}
    for d in sorted(shipped_docs_from_package(root) - listed):
        rep.err("win/package.ps1", "copies %s, which is not in RELEASE_DOCS" % d)

    profiles = sorted(f for f in os.listdir(os.path.join(root, PROFILE_DIR)) if f.endswith(".yaml"))
    profile_names = {f[:-5] for f in profiles}
    vars_in_code = code_vars(root)
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

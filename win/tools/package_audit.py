#!/usr/bin/env python3
"""Brilliance: audit a staged package folder before it is zipped (win/package.ps1 runs it).

    python -I win/tools/package_audit.py <package folder>             from the repository root
    python -I win/tools/package_audit.py <package folder> --root DIR

Reads files only: it runs nothing from the package, uses no GPU and no network. Standard library
only; lowers its own priority to below normal on Windows. Errors exit 1, warnings exit 0.

  1. licences: every DLL in bin/ has a row in THIRD-PARTY.md's "The DLLs" table; every text that
     index names is in share/doc/radiance/third_party; every text there is named by the index or
     installed by an upstream rule (CMakeLists.txt, win/cmake/RadWindows.cmake); FFmpeg's text is
     the LGPL-2.1, and no GPL-3.0 or OpenH264 text ships; the models' licence texts in
     share/doc/radiance/model-licenses are exactly win/model-licenses, byte for byte, none missing
     or extra, and each .txt is one index.md links (/terms/licenses/<id>), each link one that ships;
  2. encoders: no shipped .dll or .exe carries OpenH264's or x264's code (their API and version
     strings; the H.264 decoder's "x264 - core" SEI check is FFmpeg's own and is not matched);
  3. documents: the Markdown files at the package root, tools/STACKS.md, NOTICE and THIRD-PARTY.md
     are exactly the shipped release documents win/tools/docs_check.py covers, byte for byte their
     sources in the repository;
  4. nothing internal: no file whose path names the plan, the ledger, an investigation, artifacts,
     the port spec or the upstream-update procedure, and no text file (outside the third-party
     texts and the recipes, which are upstream's) that cites them; a warning for build-machine
     paths under a user profile, which a package built from a checkout under C:\\Users carries in
     its binaries and link map;
  5. MANIFEST.json, when present: every file listed with its SHA-256, and none unlisted.
"""
import argparse
import fnmatch
import hashlib
import json
import os
import re
import sys

BS = chr(92)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import docs_check  # noqa: E402  (RELEASE_DOCS: the shipped documents and their package paths)

THIRD = "share/doc/radiance/third_party"
INDEX = THIRD + "/THIRD-PARTY.md"
TEXT_RE = re.compile(r"`([A-Za-z0-9_.+-]+\.(?:LICENSE|LICENSES|NOTICE|EULA|PATENTS|SOURCE)"
                     r"(?:\.[A-Za-z0-9-]+)?|[A-Za-z0-9_.+-]+\.LICENSE\.txt)`")
# The models' licence texts the server shows at /terms/licenses, and their source in the repository.
MODELS = "share/doc/radiance/model-licenses"
MODELS_SRC = "win/model-licenses"
LICENCE_LINK_RE = re.compile(r"/terms/licenses/([A-Za-z0-9_.-]+)")
RENAME_RE = re.compile(r"third_party\s+RENAME\s+([A-Za-z0-9_.+-]+)\)?")
HIP_DLLS = {"amdhip64_7.dll", "amd_comgr.dll", "rocm_kpack.dll"}
HIP_KERNELS = {"share/radiance/kernels/" + name + ".dll" for name in ("libr3d", "libr4d", "libhalo")}
COMGR_LICENCE = "amd_comgr.LICENSE.txt"
FFMPEG_LICENCE = "ffmpeg.LICENSE.txt"
BUILD_LICENCES = {COMGR_LICENCE, FFMPEG_LICENCE}
# OpenH264's and x264's code, by strings only their code carries (lower case; each module is
# lowered before matching): OpenH264's encoder and decoder classes and version line, FFmpeg's
# wrappers' long names, x264's SEI banner, and FFmpeg's configure line enabling either. Plain
# "libopenh264" or "x264" would also match FFmpeg's "--disable-libopenh264", its H.264 decoder's
# "x264 - core %d" check, and radiance's own messages naming encoders it does not use.
ENCODER_MARKS = [b"cwelsh264svcencoder", b"cwelsdecoder", b"openh264 codec version",
                 b"openh264 h.264 / avc", b"--enable-libopenh264",
                 b"libx264 h.264 / avc", b"h.264/mpeg-4 avc codec - copy", b"--enable-libx264"]
FORBIDDEN_TEXTS = {"openh264.LICENSE", "GPL-3.0.txt", "x264.LICENSE"}
# Internal material, by path and by citation.
INTERNAL_PATH = re.compile(r"(^|/)(PLAN\.md|spec\.md|artifacts|investigations|UPSTREAM-UPDATE\.md)(/|$)"
                           r"|LEDGER")
INTERNAL_TEXT = [
    (re.compile(r"\bPLAN\.md\b"), "the plan"),
    (re.compile(r"\bspec\.md\b"), "the port spec"),
    (re.compile(r"(^|[^A-Za-z])artifacts[/\x5c]"), "an artifacts/ path"),
    (re.compile(r"investigations[/\x5c]"), "an investigation"),
    (re.compile(r"LEDGER|\bhang ledger\b"), "the hang ledger"),   # hfget.ps1's "ledger" is its own
    (re.compile(r"UPSTREAM-UPDATE\.md"), "the upstream-update procedure"),
]
TEXT_EXT = {".md", ".ps1", ".py", ".yaml", ".yml", ".cmd", ".txt", ".json"}
USER_PATH = re.compile(rb"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\x00\"]+", re.I)


def lower_priority():
    if os.name == "nt":
        try:
            import ctypes
            k = ctypes.windll.kernel32
            k.SetPriorityClass(k.GetCurrentProcess(), 0x00004000)   # BELOW_NORMAL_PRIORITY_CLASS
        except Exception:
            pass


class Report:
    def __init__(self):
        self.errors, self.warnings, self.notes = [], [], []

    def err(self, check, msg):
        self.errors.append("%s: %s" % (check, msg))

    def warn(self, check, msg):
        self.warnings.append("%s: %s" % (check, msg))

    def ok(self, check, msg):
        self.notes.append("%s: %s" % (check, msg))


def walk(pkg):
    for d, _, files in os.walk(pkg):
        for f in files:
            full = os.path.join(d, f)
            yield os.path.relpath(full, pkg).replace(BS, "/"), full


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dll_table(index_text):
    """The first column's backticked names (globs allowed) in "## The DLLs"."""
    pats, inside = [], False
    for line in index_text.splitlines():
        if line.startswith("## "):
            inside = line.strip() == "## The DLLs"
            continue
        if inside and line.startswith("|") and not line.startswith("|---"):
            first = line.split("|")[1]
            pats += re.findall(r"`([^`]+\.dll)`", first)
    return pats


def check_runtime_index(index_text, dlls, available, rep, *, hip):
    """Match the licence index to a staged package or ZIP."""
    pats = dll_table(index_text)
    if not pats:
        rep.err("licences", "THIRD-PARTY.md has no DLL rows")
    for d in dlls:
        if not any(fnmatch.fnmatch(d.lower(), p.lower()) for p in pats):
            rep.err("licences", "bin/%s has no row in THIRD-PARTY.md's DLL table" % d)
    for p in pats:
        if not hip and p.lower() in HIP_DLLS:
            continue
        if not any(fnmatch.fnmatch(d.lower(), p.lower()) for d in dlls):
            rep.warn("licences", "THIRD-PARTY.md lists %s, which bin/ does not carry" % p)
    named = set(TEXT_RE.findall(index_text))
    required = named.copy()
    if hip:
        required.add(COMGR_LICENCE)
    else:
        # CPU installs still carry the source-tree notices. Only this SDK-supplied
        # text and the HIP DLLs themselves are absent from their install.
        required.discard(COMGR_LICENCE)
    if any(d.lower().startswith("avcodec-") for d in dlls):
        required.add(FFMPEG_LICENCE)
    for n in sorted(required - available):
        rep.err("licences", "%s is required and missing or empty in %s" % (n, THIRD))
    rep.ok("licences", "%d DLLs in bin/ against %d index rows; %d required texts"
           % (len(dlls), len(pats), len(required)))
    return named


def check_licences(pkg, root, rep):
    index_path = os.path.join(pkg, INDEX)
    if not os.path.isfile(index_path):
        rep.err("licences", "%s is missing" % INDEX)
        return
    with open(index_path, encoding="utf-8") as f:
        index_text = f.read()
    bin_dir = os.path.join(pkg, "bin")
    dlls = sorted(f for f in os.listdir(bin_dir) if f.lower().endswith(".dll")
                  and os.path.isfile(os.path.join(bin_dir, f))) if os.path.isdir(bin_dir) else []
    have = set(os.listdir(os.path.join(pkg, THIRD)))
    available = {name for name in have if os.path.isfile(os.path.join(pkg, THIRD, name))
                 and os.path.getsize(os.path.join(pkg, THIRD, name))}
    hip = bool(HIP_DLLS.intersection(d.lower() for d in dlls)
               or any(os.path.isfile(os.path.join(pkg, path)) for path in HIP_KERNELS))
    named = check_runtime_index(index_text, dlls, available, rep, hip=hip)
    upstream = set()
    for rel in ("CMakeLists.txt", "win/cmake/RadWindows.cmake"):
        with open(os.path.join(root, rel), encoding="utf-8") as f:
            upstream |= set(RENAME_RE.findall(f.read()))
    for n in sorted(have - named - upstream - {"THIRD-PARTY.md"}):
        rep.err("licences", "%s/%s is named by no index and installed by no rule" % (THIRD, n))
    for n in sorted(have & FORBIDDEN_TEXTS):
        rep.err("licences", "%s ships; the package carries no GPL-3.0 or OpenH264 code" % n)
    ff = os.path.join(pkg, THIRD, FFMPEG_LICENCE)
    if FFMPEG_LICENCE in available and any(d.lower().startswith("avcodec-") for d in dlls):
        with open(ff, encoding="utf-8", errors="replace") as f:
            t = f.read()
        if "GNU LESSER GENERAL PUBLIC LICENSE" not in t or "Version 2.1" not in t:
            rep.err("licences", "ffmpeg.LICENSE.txt is not the LGPL-2.1 text")


def check_model_licences(pkg, root, rep):
    """The models' licence texts behind /terms/licenses: exactly win/model-licenses, byte for byte,
    and every text one the index names (and every one it names shipped)."""
    errors_before = len(rep.errors)
    shipped_dir = os.path.join(pkg, MODELS)
    source_dir = os.path.join(root, MODELS_SRC)
    if not os.path.isdir(shipped_dir):
        rep.err("model-licences", "%s is missing" % MODELS)
        return
    shipped = set(os.listdir(shipped_dir))
    source = set(os.listdir(source_dir)) if os.path.isdir(source_dir) else set()
    for n in sorted(source - shipped):
        rep.err("model-licences", "%s/%s is not in the package" % (MODELS_SRC, n))
    for n in sorted(shipped - source):
        rep.err("model-licences", "%s/%s ships but is not in %s" % (MODELS, n, MODELS_SRC))
    for n in sorted(shipped & source):
        if sha256(os.path.join(shipped_dir, n)) != sha256(os.path.join(source_dir, n)):
            rep.err("model-licences", "%s/%s differs from %s/%s" % (MODELS, n, MODELS_SRC, n))
    index = os.path.join(shipped_dir, "index.md")
    if not os.path.isfile(index):
        rep.err("model-licences", "%s/index.md is missing" % MODELS)
        return
    named = {i.rstrip(".") for i in LICENCE_LINK_RE.findall(open(index, encoding="utf-8").read())}
    texts = {os.path.splitext(n)[0] for n in shipped if n.endswith(".txt")}
    for i in sorted(texts - named):
        rep.err("model-licences", "%s/%s.txt is named by no link in index.md" % (MODELS, i))
    for i in sorted(named - texts):
        rep.err("model-licences", "index.md links /terms/licenses/%s, which does not ship" % i)
    if len(rep.errors) == errors_before:
        rep.ok("model-licences", "%d texts and the index, each its source's bytes" % len(texts))


def check_encoders(pkg, rep):
    n = 0
    for rel, full in walk(pkg):
        if not rel.lower().endswith((".dll", ".exe")):
            continue
        n += 1
        data = open(full, "rb").read().lower()
        hits = [m.decode() for m in ENCODER_MARKS if m in data]
        if hits:
            rep.err("encoders", "%s carries %s" % (rel, ", ".join(hits)))
    rep.ok("encoders", "%d modules scanned for OpenH264 and x264" % n)


def check_docs(pkg, root, rep):
    shipped = {pk: src for src, pk in docs_check.RELEASE_DOCS if pk}
    for pk, src in sorted(shipped.items()):
        p = os.path.join(pkg, pk)
        if not os.path.isfile(p):
            rep.err("docs", "%s (from %s) is not in the package" % (pk, src))
        elif sha256(p) != sha256(os.path.join(root, src)):
            rep.err("docs", "%s differs from %s" % (pk, src))
    extra = [f for f in os.listdir(pkg) if f.lower().endswith(".md") and f not in shipped]
    tools = os.path.join(pkg, "tools")
    if os.path.isdir(tools):
        extra += ["tools/" + f for f in os.listdir(tools) if f.lower().endswith(".md") and "tools/" + f not in shipped]
    for f in sorted(extra):
        rep.err("docs", "%s ships but is not a release document docs_check.py covers" % f)
    rep.ok("docs", "%d release documents, as docs_check.py lists them" % len(shipped))


def check_internal(pkg, rep):
    users = {}
    for rel, full in walk(pkg):
        if INTERNAL_PATH.search(rel):
            rep.err("internal", "%s ships" % rel)
        ext = os.path.splitext(rel)[1].lower()
        if ext in TEXT_EXT and not rel.startswith(THIRD + "/") and "/recipes/" not in "/" + rel \
                and rel != "MANIFEST.json":
            with open(full, encoding="utf-8", errors="replace") as f:
                for n, line in enumerate(f, 1):
                    for rx, what in INTERNAL_TEXT:
                        if rx.search(line):
                            rep.err("internal", "%s:%d cites %s" % (rel, n, what))
        if ext in (".dll", ".exe", ".map", ".json"):
            for m in set(USER_PATH.findall(open(full, "rb").read())):
                users.setdefault(m.decode("latin-1"), []).append(rel)
    for u, files in sorted(users.items()):
        rep.warn("internal", "%s appears in %d files (%s); build from a checkout outside the user profile"
                 % (u, len(files), ", ".join(sorted(files)[:3]) + (", ..." if len(files) > 3 else "")))
    rep.ok("internal", "paths and text files scanned")


def check_manifest(pkg, rep):
    mp = os.path.join(pkg, "MANIFEST.json")
    if not os.path.isfile(mp):
        return
    raw = open(mp, "rb").read()
    if raw.startswith(b"\xef\xbb\xbf"):
        rep.warn("manifest", "MANIFEST.json starts with a byte-order mark")
    m = json.loads(raw.decode("utf-8-sig"))
    files = m.get("files", {})
    seen = set()
    for rel, full in walk(pkg):
        if rel == "MANIFEST.json":
            continue
        seen.add(rel)
        if rel not in files:
            rep.err("manifest", "%s is not listed" % rel)
        elif files[rel] != sha256(full):
            rep.err("manifest", "%s does not match its SHA-256" % rel)
    for rel in sorted(set(files) - seen):
        rep.err("manifest", "%s is listed but not in the package" % rel)
    rep.ok("manifest", "%d files listed" % len(files))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("package", help="the staged package folder")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(HERE)),
                    help="the repository root (default: this script's)")
    a = ap.parse_args()
    lower_priority()
    pkg, root = os.path.abspath(a.package), os.path.abspath(a.root)
    rep = Report()
    check_licences(pkg, root, rep)
    check_model_licences(pkg, root, rep)
    check_encoders(pkg, rep)
    check_docs(pkg, root, rep)
    check_internal(pkg, rep)
    check_manifest(pkg, rep)
    for n in rep.notes:
        print("ok    " + n)
    for w in rep.warnings:
        print("warn  " + w)
    for e in rep.errors:
        print("ERROR " + e)
    print("package_audit: %d error(s), %d warning(s)" % (len(rep.errors), len(rep.warnings)))
    return 1 if rep.errors else 0


if __name__ == "__main__":
    sys.exit(main())

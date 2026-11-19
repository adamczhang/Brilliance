"""Brilliance: find catch funclets that cannot return.

AMD clang 24 targeting x86_64-pc-windows-msvc was seen (tests/server_test.cpp,
a_quoted_string_is_what_nlohmann_writes, -O2) to compile a catch funclet with EVERY exit path
ending in int3: the optimizer treated the code after the catch as unreachable, so a caught
exception kills the process with 0x80000003 instead of continuing. At -O0 the same source runs.

A healthy catch funclet returns (`ret`, handing the continuation address back to the CRT) or
rethrows (a call to _CxxThrowException, or terminate). This lists every `catch$` funclet in the
given objects/libraries that does neither -- the signature of that miscompile -- so a build can
be checked for it rather than discovered one crash at a time.

    python win/tools/scan_catch_funclets.py <llvm-objdump> <file.obj|file.lib|dir>...

Exit status: 0 none found, 1 some found, 2 usage.
"""
import os
import re
import subprocess
import sys

NORETURN_CALLS = ("_CxxThrowException", "terminate", "abort", "_invoke_watson", "__std_terminate",
                  "__fastfail", "_purecall", "exit")
SYM = re.compile(r"^[0-9a-f]+ <(.*)>:$")


def funclets(objdump, path):
    out = subprocess.run([objdump, "-d", "--no-show-raw-insn", "-r", path],
                         capture_output=True, text=True, errors="replace").stdout
    name, body = None, []
    for line in out.splitlines():
        m = SYM.match(line.strip())
        if m:
            if name is not None:
                yield name, body
            name, body = m.group(1), []
        elif name is not None:
            body.append(line)
    if name is not None:
        yield name, body


def suspicious(body):
    text = "\n".join(body)
    if re.search(r"\bret[qlw]?\b", text):
        return False
    if any(c in text for c in NORETURN_CALLS):
        return False
    return "int3" in text


def files(args):
    for a in args:
        if not os.path.exists(a):
            continue          # a directory this build does not have (libr4d in a host-only tree)
        if os.path.isdir(a):
            for root, _, names in os.walk(a):
                for n in names:
                    if n.endswith((".obj", ".lib")) and not n.endswith(".dll.lib"):
                        yield os.path.join(root, n)
        else:
            yield a


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    objdump, found, scanned, seen = sys.argv[1], 0, 0, 0
    for f in files(sys.argv[2:]):
        scanned += 1
        for name, body in funclets(objdump, f):
            if "catch$" not in name:
                continue
            seen += 1
            if suspicious(body):
                found += 1
                print(f"{f}: {name[:200]}")
    print(f"{scanned} file(s), {seen} catch funclet(s), {found} with no return and no rethrow")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())

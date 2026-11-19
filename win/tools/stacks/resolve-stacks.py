"""Brilliance: turn brl-stacks' module+offset frames into function names, from lld-link map files.

    python -I resolve-stacks.py <stacks.txt> [<module>=]<map> ...

Each map is matched to the module of its base name (radiance.map -> radiance.exe, a plugin's
<name>.map -> <name>.dll; a saved copy's prefix is dropped, so hang3-radiance.map serves
radiance.exe), or to the module named before '='. The build writes radiance.map beside radiance.exe (win/cmake/RadWindows.cmake,
brl_win_host_exports); a map from the same link, and only from it, matches a running binary. Frames
of other modules are kept as brl-stacks printed them. Standard library only.
"""
import bisect
import os
import re
import sys


def load_map(path):
    """(addresses, names, preferred base) of a map's public symbols, sorted by address."""
    base, syms = 0x140000000, []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "Preferred load address is" in line:
                base = int(line.split()[-1], 16)
                continue
            p = line.split()
            if len(p) >= 4 and ":" in p[0] and len(p[2]) == 16:
                try:
                    va = int(p[2], 16)
                except ValueError:
                    continue
                if va >= base:
                    syms.append((va, p[1]))
    syms.sort()
    return [v for v, _ in syms], [n for _, n in syms], base


def short(name):
    """A readable form of an MSVC-mangled name: Class::method, else the name cut short."""
    m = re.match(r"\?([^@?$]+)@([^@]+)@", name)
    if m:
        return f"{m.group(2)}::{m.group(1)}"
    return name[:100]


def main():
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    maps = {}
    for arg in sys.argv[2:]:
        # module=path names the module outright; otherwise the file's stem does, and a saved copy's
        # prefix is dropped (hang3-radiance.map serves radiance.exe).
        mod, eq, path = arg.partition("=")
        if not eq:
            path, mod = arg, os.path.splitext(os.path.basename(arg))[0].rsplit("-", 1)[-1]
        maps[mod.lower()] = load_map(path)
    # A frame anywhere on a line: brl-stacks prints one a line, the stall watch's stream report
    # several in a row (win/device/hipnative.cpp).
    frame = re.compile(r"\b([\w\-]+)\.(exe|dll)\+0x([0-9a-fA-F]+)")

    def name(m):
        if m.group(1).lower() not in maps:
            return m.group(0)
        keys, names, base = maps[m.group(1).lower()]
        va = base + int(m.group(3), 16)
        i = bisect.bisect_right(keys, va) - 1
        if i < 0:
            return m.group(0)
        return f"{m.group(1)}.{m.group(2)}!{short(names[i])}+0x{va - keys[i]:x}"

    with open(sys.argv[1], encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            out = frame.sub(name, line.rstrip("\n"))
            # A frame the map named drops DbgHelp's guess, the nearest export, which for an
            # executable without a PDB is unrelated.
            if "!" in out:
                out = re.sub(r"(\.(?:exe|dll)![^ ]+)\s+\([^)]*\)$", r"\1", out)
            print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

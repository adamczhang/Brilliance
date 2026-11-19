"""Brilliance: make a C/C++/HIP source tree LLP64-clean by giving every `long` 64 bits.

Windows is LLP64: `long` is 32 bits there, on the host AND in HIP device code, where Linux (LP64)
gives it 64. libr4d carries every device pointer and stream as `long` (r4d_args.h's r4d_addr, the
230 parameters of r4d.h, 461 `(long)` casts), so on Windows each one would be silently truncated
above 4 GiB. The compiler confirms it with 13,110 pointer-truncation warnings.

The rewrite is mechanical and ABI-neutral on Linux, where `long` and `long long` are both 64 bits:

  long            -> long long          (a lone `long`, never part of `long long` / `long double`)
  unsigned long   -> unsigned long long (follows from the above)
  123L / 123UL    -> 123LL / 123ULL     (an integer literal whose suffix names `long`)
  %ld %lu %lx %li -> %lld %llu %llx %lli (printf length modifiers inside string literals)
  labs(           -> llabs(

Comments are left exactly as they are -- English uses "long" too -- and so are string literals
apart from their format specifiers. The tool is idempotent: run twice, it changes nothing the
second time. Usage:

  python -I win/tools/llp64_sweep.py [--check] FILE_OR_DIR...

--check changes nothing and exits 1 if any file would change (for CI).
"""
import pathlib
import re
import sys

EXTS = {".h", ".hpp", ".c", ".cc", ".cpp", ".hip", ".cuh", ".cu"}

IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
NUMBER = re.compile(r"(?:0[xX][0-9A-Fa-f']+|0[bB][01']+|[0-9][0-9']*)([uU]?[lL]{1,2}[uU]?)?\b")
FMT = re.compile(r"%([-+ #0]*[0-9*]*(?:\.[0-9*]+)?)l([diuxXo])")


def rewrite_code(code):
    """Rewrite a stretch of code that holds no comments and no literals."""
    out = []
    i = 0
    toks = []
    # Tokenise into identifiers, numbers and everything else, keeping exact spacing.
    while i < len(code):
        m = IDENT.match(code, i)
        if m:
            toks.append(("id", m.group(0)))
            i = m.end()
            continue
        if code[i].isdigit():
            m = NUMBER.match(code, i)
            if m:
                toks.append(("num", m.group(0)))
                i = m.end()
                continue
        toks.append(("ch", code[i]))
        i += 1

    def next_id(k):
        """The next identifier after token k, skipping whitespace only."""
        j = k + 1
        while j < len(toks) and toks[j][0] == "ch" and toks[j][1].isspace():
            j += 1
        return toks[j][1] if j < len(toks) and toks[j][0] == "id" else None

    def prev_id(k):
        j = k - 1
        while j >= 0 and toks[j][0] == "ch" and toks[j][1].isspace():
            j -= 1
        return toks[j][1] if j >= 0 and toks[j][0] == "id" else None

    for k, (kind, text) in enumerate(toks):
        if kind == "id" and text == "long":
            if next_id(k) in ("long", "double") or prev_id(k) == "long":
                out.append(text)            # already part of long long / long double
            else:
                out.append("long long")
        elif kind == "id" and text == "labs":
            out.append("llabs")
        elif kind == "num":
            m = NUMBER.match(text)
            suf = m.group(1) or ""
            low = suf.lower()
            if low in ("l", "ul", "lu"):
                base = text[: len(text) - len(suf)]
                new = "ULL" if "u" in low else "LL"
                out.append(base + new)
            else:
                out.append(text)
        else:
            out.append(text)
    return "".join(out)


def rewrite_string(lit):
    return FMT.sub(lambda m: "%" + m.group(1) + "ll" + m.group(2), lit)


def sweep(src):
    """Split src into code / comments / literals and rewrite the code and the literals."""
    out = []
    code_start = 0
    i = 0
    n = len(src)

    def flush_code(end):
        out.append(rewrite_code(src[code_start:end]))

    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            flush_code(i)
            j = src.find("\n", i)
            j = n if j < 0 else j
            # a // comment continues across a backslash-newline
            while j > 0 and j < n and src[j - 1] == "\\":
                k = src.find("\n", j + 1)
                j = n if k < 0 else k
            out.append(src[i:j])
            i = code_start = j
            continue
        if c == "/" and nxt == "*":
            flush_code(i)
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(src[i:j])
            i = code_start = j
            continue
        if c == "R" and nxt == '"':
            # raw string literal R"delim( ... )delim"
            m = re.match(r'R"([^()\s\\]{0,16})\(', src[i:])
            if m:
                flush_code(i)
                end = src.find(")" + m.group(1) + '"', i + m.end())
                end = n if end < 0 else end + len(m.group(1)) + 2
                out.append(src[i:end])
                i = code_start = end
                continue
        if c in ('"', "'"):
            # a character literal only where it cannot be a digit separator (1'000)
            if c == "'" and i > 0 and (src[i - 1].isalnum() or src[i - 1] == "_"):
                i += 1
                continue
            flush_code(i)
            j = i + 1
            while j < n and src[j] != c:
                if src[j] == "\\":
                    j += 1
                elif src[j] == "\n":
                    break
                j += 1
            j = min(j + 1, n)
            lit = src[i:j]
            out.append(rewrite_string(lit) if c == '"' else lit)
            i = code_start = j
            continue
        i += 1
    flush_code(n)
    return "".join(out)


def files(args):
    for a in args:
        p = pathlib.Path(a)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix in EXTS:
                    yield f
        elif p.is_file():
            yield p


def main(argv):
    check = "--check" in argv
    args = [a for a in argv if a != "--check"]
    if not args:
        print(__doc__)
        return 2
    changed = 0
    for f in files(args):
        src = f.read_text(encoding="utf-8")
        new = sweep(src)
        if new != src:
            changed += 1
            if check:
                print(f"would change: {f}")
                # The sweep never adds or removes a line, so the lines pair up; name where.
                shown = 0
                for no, (a, b) in enumerate(zip(src.splitlines(), new.splitlines()), 1):
                    if a != b and shown < 20:
                        print(f"  {f}:{no}: {a.strip()}")
                        shown += 1
            else:
                f.write_text(new, encoding="utf-8", newline="\n")
                print(f"rewrote: {f}")
    print(f"{changed} file(s) {'would change' if check else 'changed'}")
    return 1 if (check and changed) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

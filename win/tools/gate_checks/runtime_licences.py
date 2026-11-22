"""Verify the runtime licence index against the filtered source tree or candidate ZIP."""
from pathlib import Path, PurePosixPath

from package_audit import (
    BUILD_LICENCES, HERE, HIP_DLLS, HIP_KERNELS, INDEX, RENAME_RE, TEXT_RE, THIRD, Report,
    check_runtime_index, dll_table,
)

CHECK_ID = "runtime-licences"
PRE_SNAPSHOT = True
SOURCE_INDEX = "win/licenses/THIRD-PARTY.md"
INSTALL_RULES = "win/cmake/RadWindows.cmake"


def run(ctx):
    if ctx.zip:
        rep = Report()
        names = set(ctx.zip_names())
        if INDEX not in names:
            return {"verdict": "fail", "reason": INDEX + " is missing"}
        index = ctx.zip_read(INDEX).decode("utf-8")
        dlls = sorted(p.name for p in map(PurePosixPath, names)
                      if p.parent.as_posix() == "bin" and p.suffix.lower() == ".dll")
        hip = bool(HIP_DLLS.intersection(d.lower() for d in dlls)
                   or HIP_KERNELS.intersection(n.lower() for n in names))
        available = {n for n in set(TEXT_RE.findall(index)) | BUILD_LICENCES
                     if THIRD + "/" + n in names and ctx.zip_read(THIRD + "/" + n)}
        check_runtime_index(index, dlls, available, rep, hip=hip)
        errors = rep.errors + rep.warnings
        reason = f"ZIP runtime index covers {len(dlls)} DLLs and their licence/notice files"
    else:
        names = set(ctx.source_names())
        missing = {SOURCE_INDEX, INSTALL_RULES} - names
        if missing:
            return {"verdict": "fail", "reason": "missing source files", "details": sorted(missing)}
        index = ctx.source_read(SOURCE_INDEX).decode("utf-8")
        installed = set(RENAME_RE.findall(ctx.source_read(INSTALL_RULES).decode("utf-8")))
        named = set(TEXT_RE.findall(index))
        errors = []
        if not dll_table(index):
            errors.append(f"{SOURCE_INDEX} has no DLL rows")
        for name in sorted(BUILD_LICENCES - named):
            errors.append(f"{SOURCE_INDEX} does not name {name}")
        for name in sorted(BUILD_LICENCES - installed):
            errors.append(f"{INSTALL_RULES} has no install rule for {name}")
        for name in sorted(named - BUILD_LICENCES):
            path = "win/licenses/" + name
            if path not in names or not ctx.source_read(path):
                errors.append(f"{SOURCE_INDEX} names {path}, which is missing or empty")
        reason = (f"{len(named - BUILD_LICENCES)} indexed source licence/notice files present; "
                  f"{len(BUILD_LICENCES)} build-supplied licence texts have install rules")
    if errors:
        return {"verdict": "fail", "reason": f"{len(errors)} runtime index issue(s)", "details": errors}
    return {"verdict": "pass", "reason": reason,
            "evidence": [str(Path(HERE, "package_audit.py"))] if ctx.zip else []}

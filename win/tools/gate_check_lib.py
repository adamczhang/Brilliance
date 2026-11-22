"""Discover and run release check plug-ins for the gate runner and snapshot maker."""
import importlib.util
from pathlib import Path


def run_checks(folder, ctx, *, pre_snapshot=False):
    for path in sorted(Path(folder).glob("*.py")):
        if path.name.startswith("_"):
            continue
        name = path.stem
        try:
            spec = importlib.util.spec_from_file_location("gate_check_" + name, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if pre_snapshot and not getattr(mod, "PRE_SNAPSHOT", False):
                continue
            name = getattr(mod, "CHECK_ID", name)
            result = mod.run(ctx)
            verdict = result.get("verdict")
            if verdict not in ("pass", "fail", "skip"):
                raise ValueError("verdict %r is not pass, fail or skip" % verdict)
            if verdict == "skip" and not (result.get("reason") or "").strip():
                raise ValueError("a skip without a reason")
        except Exception as e:
            result = {"verdict": "error", "reason": "%s: %s" % (type(e).__name__, e)}
        yield name, path, result

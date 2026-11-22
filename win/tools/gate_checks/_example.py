"""An example gate check: the shape every check in this folder has. Files whose name starts with an
underscore are not run; copy this to a name without one to register a check.

The gate runner (win/tools/gate_runner.py, a gate of kind "checks") imports every *.py here, in name
order, and calls run(ctx). Standard library only; read files, run nothing from the package, no
network, no GPU.

ctx carries:
  ctx.candidate   the record's candidate: zip, zip_sha256, source_commit, version, package, ...
  ctx.zip         the ZIP's path;  ctx.zip_names() its files (package paths);  ctx.zip_read(path) bytes
  ctx.pkg         the fresh unpack, or "" when --pkg was not given
  ctx.source      the build clone at the snapshot commit, or "" when --source was not given
  ctx.manifest    MANIFEST.json as a dict
  ctx.options     the gate's "options" from the plan
  ctx.inputs      the candidate's inputs file
  ctx.out         a folder for this run's check output (under the evidence folder)

run(ctx) returns a dict:
  verdict    "pass", "fail" or "skip" (a skip must give a reason; anything else is an error)
  reason     one line
  details    optional list of lines for the record
  evidence   optional list of file paths the record hashes
An exception is recorded as the check's error, and an error fails the gate: never a pass.
"""

CHECK_ID = "example-notice-ships"


def run(ctx):
    names = set(ctx.zip_names())
    want = "share/doc/radiance/NOTICE"
    if want in names:
        return {"verdict": "pass", "reason": "%s ships" % want}
    return {"verdict": "fail", "reason": "%s is not in the ZIP" % want}

#!/usr/bin/env python3
"""Exercise the pre-snapshot and ZIP licence checks and the staged package audit."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
from gate_check_lib import run_checks
from package_audit import COMGR_LICENCE, HIP_DLLS, INDEX, MODELS, THIRD, Report, check_licences

SCOPE = "win/tools/profile-scope.json"
PROFILE = "win/profiles/tiny.yaml"
TERMS = "win/TERMS-OF-USE.md"
SUPPORT = "win/SUPPORT.md"
MODEL_INDEX = "win/model-licenses/index.md"
MODEL_LICENCE = "win/model-licenses/tiny-licence.txt"
RUNTIME_INDEX = "win/licenses/THIRD-PARTY.md"
NOTICE = "win/licenses/tiny-runtime.NOTICE"
INSTALL = "win/cmake/RadWindows.cmake"


def source_fixture():
    scope = {
        "schema": "brilliance-profile-scope/1",
        "profiles": {"tiny": {"support": "Qualified", "models": {"tiny-route": "tiny-model"}}},
        "models": {"tiny-model": {
            "file": "owner/tiny/tiny.rad", "licence": "tiny-licence", "gated": True,
            "index_row": "Tiny model", "use_restrictions": ["Use with attribution."],
            "terms": {"heading": "Tiny model", "required_text": ["You must attribute the model."],
                      "quoted_licence_sections": [["1. Attribution", "2. Other terms"]]},
        }},
    }
    files = {
        PROFILE: "# The repository is gated: accept its terms.\nmodels:\n"
                 "  tiny-route: # comment\n    model: D:/models/hf/owner/tiny/tiny.rad\n",
        SUPPORT: "## Models and card counts\n| Tiny model | **Qualified**. Profile: `tiny` |\n",
        TERMS: "## The models and their licences\n| Tiny model | Tiny Licence | Yes, below |\n"
               "## Tiny model\nYou must attribute the model. `/terms/licenses/tiny-licence`\n\n"
               "1. Attribution\nRetain the attribution notice.\n",
        MODEL_INDEX: "| Tiny model | Tiny Licence | `/terms/licenses/tiny-licence` |\n",
        MODEL_LICENCE: "1. Attribution\nRetain the attribution notice.\n\n2. Other terms\n",
        RUNTIME_INDEX: "## The DLLs\n"
                       "| `tiny-runtime.dll` | `tiny-runtime.NOTICE` |\n"
                       "| `amdhip64_7.dll`, `rocm_kpack.dll`, `amd_comgr.dll` | `amd_comgr.LICENSE.txt` |\n"
                       "| `avcodec-62.dll` | `ffmpeg.LICENSE.txt`, `ffmpeg.SOURCE` |\n",
        NOTICE: "Tiny Runtime licence notice\n",
        "win/licenses/ffmpeg.SOURCE": "Corresponding source\n",
        INSTALL: 'install(FILES "sdk/COMGR" DESTINATION share/doc/radiance/third_party\n'
                 '        RENAME amd_comgr.LICENSE.txt)\n'
                 'install(FILES "sdk/FFmpeg" DESTINATION share/doc/radiance/third_party\n'
                 '        RENAME ffmpeg.LICENSE.txt)\n',
        SCOPE: json.dumps(scope),
    }
    return {path: text.encode("utf-8") for path, text in files.items()}


def package_fixture(source, hip):
    files = {"bin/tiny-runtime.dll": b"dll", "bin/avcodec-62.dll": b"dll",
             THIRD + "/ffmpeg.LICENSE.txt": b"GNU LESSER GENERAL PUBLIC LICENSE\nVersion 2.1\n"}
    for path, content in source.items():
        if path.startswith("win/model-licenses/"):
            files[MODELS + "/" + Path(path).name] = content
        elif path.startswith("win/licenses/"):
            files[THIRD + "/" + Path(path).name] = content
        elif path.startswith("win/profiles/") or path in (SUPPORT, TERMS):
            files[path.removeprefix("win/")] = content
    if hip:
        files.update({"bin/" + dll: b"hip dll" for dll in HIP_DLLS})
        files[THIRD + "/" + COMGR_LICENCE] = b"COMGR notice\n"
    return files


class LicenceChecksTest(unittest.TestCase):
    def setUp(self):
        self.source = source_fixture()

    def checks(self, package=None):
        ctx = SimpleNamespace(zip="" if package is None else "fixture.zip", source="fixture-source",
                              source_read=self.source.__getitem__, source_names=self.source.keys)
        if package is not None:
            ctx.zip_names = package.keys
            ctx.zip_read = package.__getitem__
        return {name: result for name, _, result in
                run_checks(TOOLS / "gate_checks", ctx, pre_snapshot=package is None)}

    def assert_failure(self, results, check, expected):
        result = results[check]
        self.assertEqual(result["verdict"], "fail", result)
        self.assertIn(expected, "\n".join(result.get("details", [])))

    def test_complete_source_and_package(self):
        for package in (None, package_fixture(self.source, hip=False), package_fixture(self.source, hip=True)):
            with self.subTest(package=package is not None):
                results = self.checks(package)
                self.assertEqual({k: v["verdict"] for k, v in results.items()},
                                 {"profile-scope": "pass", "runtime-licences": "pass"}, results)

    def test_missing_model_licence(self):
        del self.source[MODEL_LICENCE]
        self.assert_failure(self.checks(), "profile-scope", MODEL_LICENCE + " is missing")

    def test_missing_licence_index_link(self):
        self.source[MODEL_INDEX] = self.source[MODEL_INDEX].replace(b"/terms/licenses/tiny-licence", b"no link")
        self.assert_failure(self.checks(), "profile-scope", "must link /terms/licenses/tiny-licence")

    def test_missing_terms_section(self):
        self.source[TERMS] = self.source[TERMS].replace(b"## Tiny model", b"## Different model")
        self.assert_failure(self.checks(), "profile-scope", "missing pass-through section")

    def test_incomplete_quoted_terms(self):
        self.source[TERMS] = self.source[TERMS].replace(b"Retain the attribution notice.", b"")
        self.assert_failure(self.checks(), "profile-scope", "must carry the full licence passage")

    def test_missing_runtime_notice(self):
        for content in (None, b""):
            with self.subTest(content=content):
                self.source[NOTICE] = content or b""
                if content is None:
                    del self.source[NOTICE]
                results = self.checks()
                self.assert_failure(results, "runtime-licences", NOTICE + ", which is missing or empty")

    def test_missing_sdk_install_rule(self):
        self.source[INSTALL] = self.source[INSTALL].replace(COMGR_LICENCE.encode(), b"other.LICENSE.txt")
        self.assert_failure(self.checks(), "runtime-licences", "has no install rule for " + COMGR_LICENCE)

    def test_unlisted_profile(self):
        self.source["win/profiles/other.yaml"] = self.source[PROFILE]
        self.assert_failure(self.checks(), "profile-scope", "other: in shipped profiles")

    def test_duplicate_profile_name(self):
        self.source["win/profiles/tiny.yml"] = self.source[PROFILE]
        self.assert_failure(self.checks(), "profile-scope", "duplicate shipped profile name tiny")

    def test_stale_profile(self):
        del self.source[PROFILE]
        self.assert_failure(self.checks(), "profile-scope", "tiny: in " + SCOPE + ", missing from shipped profiles")

    def test_changed_model_route_or_file(self):
        for original, replacement in ((b"tiny-route", b"new-route"), (b"tiny.rad", b"new.rad")):
            with self.subTest(original=original):
                self.source[PROFILE] = source_fixture()[PROFILE].replace(original, replacement)
                self.assert_failure(self.checks(), "profile-scope", "expected model routes")

    def test_changed_gate_status(self):
        self.source[PROFILE] = self.source[PROFILE].replace(b"The repository is gated:", b"Open access.")
        self.assert_failure(self.checks(), "profile-scope", "repository gating disagrees")

    def test_changed_support_label(self):
        self.source[SUPPORT] = self.source[SUPPORT].replace(b"Qualified", b"Preview")
        self.assert_failure(self.checks(), "profile-scope", "differs from SUPPORT.md ('Preview')")

    def test_zip_reads_the_packaged_profile_and_notices(self):
        package = package_fixture(self.source, hip=False)
        package["profiles/tiny.yaml"] = package["profiles/tiny.yaml"].replace(b"tiny.rad", b"new.rad")
        del package[THIRD + "/tiny-runtime.NOTICE"]
        results = self.checks(package)
        self.assert_failure(results, "profile-scope", "expected model routes")
        self.assert_failure(results, "runtime-licences", "tiny-runtime.NOTICE is required")

    def audit(self, files):
        with tempfile.TemporaryDirectory(dir=os.environ["RAD_TEST_TMPDIR"]) as tmp:
            root = Path(tmp)
            stage = root / "stage"
            for base, tree in ((root, {INSTALL: self.source[INSTALL], "CMakeLists.txt": b""}), (stage, files)):
                for rel, content in tree.items():
                    path = base / rel
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(content)
            report = Report()
            check_licences(stage, root, report)
            return report

    def test_cpu_only_package_audit(self):
        report = self.audit(package_fixture(self.source, hip=False))
        self.assertEqual(report.errors + report.warnings, [])

    def test_hip_package_audit_requires_comgr_text(self):
        files = package_fixture(self.source, hip=True)
        report = self.audit(files)
        self.assertEqual(report.errors + report.warnings, [])
        del files[THIRD + "/" + COMGR_LICENCE]
        report = self.audit(files)
        self.assertIn(COMGR_LICENCE + " is required", "\n".join(report.errors))
        self.assert_failure(self.checks(files), "runtime-licences", COMGR_LICENCE + " is required")

    def test_hip_kernel_prevents_cpu_only_exemption(self):
        files = package_fixture(self.source, hip=False)
        files["share/radiance/kernels/libr4d.dll"] = b"hip kernel library"
        report = self.audit(files)
        self.assertIn(COMGR_LICENCE + " is required", "\n".join(report.errors))

    def test_cpu_only_still_requires_ffmpeg_licence(self):
        files = package_fixture(self.source, hip=False)
        del files[THIRD + "/ffmpeg.LICENSE.txt"]
        report = self.audit(files)
        self.assertIn("ffmpeg.LICENSE.txt is required", "\n".join(report.errors))


if __name__ == "__main__":
    Path(os.environ["RAD_TEST_TMPDIR"]).mkdir(parents=True, exist_ok=True)
    unittest.main(verbosity=2)

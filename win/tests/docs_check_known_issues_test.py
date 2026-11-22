#!/usr/bin/env python3
"""Exercise the known-issue schema and the public evidence links in SUPPORT.md."""
from pathlib import Path
import sys
import unittest

ROOT = Path(sys.argv.pop(1)).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "win" / "tools"))
import docs_check


HEADER = (
    "# Fixture\n"
    "\n"
    "## Known issues\n"
    "\n"
    "One table tracks support limits.\n"
    "\n"
    "| Issue or limit | Owner | Evidence | Status | Next check |\n"
    "|---|---|---|---|---|\n"
)
ROW = ["GPU stop", "Windows lifecycle", "[Stop record](#record)", "open", "Run the stop matrix"]


def findings(*rows):
    text = HEADER + "".join("| " + " | ".join(row) + " |\n" for row in rows)
    return validate(text + "\n### Record\n")


def validate(text):
    report = docs_check.Report()
    docs_check.check_known_issues(enumerate(text.splitlines(), 1), report)
    return report.errors


class KnownIssuesTest(unittest.TestCase):
    def test_valid_rows(self):
        rows = [
            ROW,
            ["A \\| B", "Packaging", "[Walkthrough](PREVIEW.md#quick-start)", "preview", "Unpack"],
            ["Audio", "HTTP", "[Vendor record](https://example.com/record)", "not supported", "Run route tests"],
            ["TP2 parity", "DeepSeek numerics", "[Trace](#record)", "explained", "Keep the fixture"],
        ]
        self.assertEqual(findings(*rows), [])

    def test_each_required_field(self):
        for index, field in enumerate(("Owner", "Evidence", "Status", "Next check"), 1):
            with self.subTest(field=field):
                incomplete = ROW.copy()
                incomplete[index] = ""
                self.assertIn(f"win/SUPPORT.md:10: known issue is missing {field}", findings(ROW, incomplete))

    def test_invalid_values(self):
        for evidence in ("pending", "[No target](#)", "[Email](mailto:record-at-example.invalid)"):
            with self.subTest(evidence=evidence):
                row = ROW[:2] + [evidence] + ROW[3:]
                self.assertIn("Evidence must include a Markdown link", findings(row)[0])
        row = ROW[:3] + ["closed"] + ROW[4:]
        self.assertIn("Status must be open, preview, not supported, or explained", findings(row)[0])
        self.assertIn("missing Next check", findings(ROW[:4] + ["<!-- pending -->"])[0])

    def test_missing_or_malformed_table(self):
        self.assertIn("missing Known issues section", validate("# Fixture\n")[0])
        self.assertIn("must have columns", validate(HEADER.replace("Next check", "Notes"))[0])
        self.assertIn("missing its Markdown separator", validate(HEADER.replace("|---|", "| |"))[0])
        self.assertIn("has no entries", findings()[0])
        self.assertIn("must have 5 columns", findings(ROW[:-1])[0])

    def test_disconnected_row(self):
        text = HEADER + "| " + " | ".join(ROW) + " |\n\n| incomplete |\n"
        self.assertIn("row is outside the table", validate(text)[0])

    def test_repository_table(self):
        path = ROOT / docs_check.KNOWN_ISSUES_DOC
        self.assertEqual(validate(path.read_text(encoding="utf-8")), [])

    def test_shipped_evidence_links(self):
        report = docs_check.Report()
        docs_check.check_link(ROOT, "win/SUPPORT.md", 1, "#known-issues", report)
        docs_check.check_link(ROOT, "win/SUPPORT.md", 2, "PREVIEW.md#quick-start", report)
        self.assertEqual(report.errors + report.warnings, [])
        docs_check.check_link(ROOT, "win/SUPPORT.md", 3, "../README.md", report)
        self.assertIn("does not resolve to the same document in the package", report.warnings[0])


if __name__ == "__main__":
    unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout, verbosity=2))

#!/usr/bin/env python3
"""Brilliance: stop_log_check.py's verdicts on synthetic stop journals (core/util/stoplog.h).

A bounded two-rank stop; the same with a step over its bound, a watcher WAIT, a split rank exit and
a stream not drained; a journal cut off mid-teardown (what a hard lock leaves); a journal with no
stop; and the order check. Then the retire: a drain that timed out and ended the open responses is
the designed path, listed as expected and not as a reason, on synthetic journals and on three saved
ones (fixtures/stoplog/): a four-card mid-prefill stop at 250K tokens that took the retire (the
0.3.0 stop matrix's C6-2), a two-rank host-test stop through the retire, and the same host test
before the SPLIT fix, where the retire must not hide the split. Standard library only; no server,
no card.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stop_log_check as slc  # noqa: E402

T = [0.0]
FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "stoplog")


def rec(kind, name, detail=""):
    T[0] += 1.5
    return "2026-11-22 00:10:00.000 t=%.1f tid=7 %s %s%s" % (T[0], kind, name,
                                                            (" " + detail) if detail else "")


def step(name, ms=1.0, bound=1000, ok=True, outcome="done"):
    return [rec("BEGIN", name, "bound %d ms" % bound),
            rec("END", name, "%s %.1f ms bound %d ms %s" % ("ok" if ok else "over", ms, bound, outcome))]


def bounded(tp=2, split=False, over=None, wait=None, pending=False, drain=None):
    T[0] = 0.0
    m = "model.q"
    L = [rec("ARM", "journal", "pid=1"), rec("STOP", "begin", "Ctrl-Break"),
         rec("NOTE", "fleet.loop_exit")]
    L += step("fleet.history_join")
    L += [rec("NOTE", "fleet.request_stop", "model q, drain 30000 ms"),
          rec("NOTE", "engine.at_stop", "model q tp %d: mid-prefill; running 1" % tp)]
    if drain is None:
        L += step(m + ".drain", bound=30000)
    else:
        # A drain over its bound: drain = (ms, detail).
        L += step(m + ".drain", ms=drain[0], bound=30000, ok=False, outcome=drain[1])
    L += [rec("NOTE", "engine.retired", "model q: responses ended, server retired"),
          rec("NOTE", "engine.stopping")]
    L += step(m + ".step_loop_stop", ok=over != "step_loop_stop", ms=12000 if over else 1)
    if wait:
        L.insert(-1, rec("WAIT", m + ".step_loop_stop", "10250 ms so far, bound 10000 ms"))
    for r in range(tp):
        L.append(rec("NOTE", "rank.exit", ("model q rank %d: left after a full release with step 9 "
                                           "published and not issued by this rank (SPLIT)" % r)
                     if split and r == tp - 1 else "model q rank %d: left on the barrier's abort" % r))
    L += step(m + ".rank_join")
    L += step(m + ".serve_join")
    L += step("fleet.front_stop")
    L.append(rec("BEGIN", m + ".teardown", "bound 60000 ms"))
    L.append(rec("BEGIN", m + ".stream_probe", "bound %d ms" % (2000 * tp)))
    for r in range(tp):
        state = "NOT DRAINED (work still queued or running)" if pending and r == 1 else "drained"
        L.append(rec("NOTE", "stream.probe", "%s rank %d device %d compute: %s after 0.1 ms" % (m, r, r, state)))
    L.append(rec("END", m + ".stream_probe", "ok 1.0 ms bound %d ms every stream drained" % (2000 * tp)))
    L += step(m + ".transfer_sync")
    for r in range(tp):
        L += step("%s.rank%d.ctx_shutdown" % (m, r))
    L += step(m + ".ranks_free") + step(m + ".plugins_release")
    L.append(rec("END", m + ".teardown", "ok 5.0 ms bound 60000 ms done"))
    L += step("fleet.sweep_buffers") + step("fleet.listener_join")
    L += [rec("NOTE", "fleet.run_return", "status 0"), rec("NOTE", "process.main_return", "status 0"),
          rec("CLOSE", "journal", "process exit (atexit)")]
    return "\n".join(L) + "\n"


RETIRE = "the drain timed out; the open responses are ended (retire)"


def fixture(name):
    with open(os.path.join(FIXTURES, name), "r", encoding="utf-8") as f:
        return f.read()


def verdict(text, tp=None):
    recs, bad = slc.parse(text)
    v, reasons, summary = slc.check(recs, tp)
    return v, reasons, summary, bad


class Verdicts(unittest.TestCase):
    def test_a_bounded_stop(self):
        v, reasons, s, bad = verdict(bounded(), tp=2)
        self.assertEqual(bad, 0)
        self.assertEqual(v, "BOUNDED", reasons)
        self.assertEqual(s["at_stop"], ["model q tp 2: mid-prefill; running 1"])

    def test_a_step_over_its_bound(self):
        v, reasons, _, _ = verdict(bounded(over="step_loop_stop", wait=True), tp=2)
        self.assertEqual(v, "UNBOUNDED")
        self.assertTrue(any("step_loop_stop went over its bound" in r for r in reasons), reasons)
        self.assertTrue(any("still waiting" in r for r in reasons), reasons)

    def test_a_split_rank_exit(self):
        v, reasons, _, _ = verdict(bounded(split=True), tp=2)
        self.assertEqual(v, "UNBOUNDED")
        self.assertTrue(any("split release" in r for r in reasons), reasons)

    def test_a_stream_not_drained(self):
        v, reasons, _, _ = verdict(bounded(pending=True), tp=2)
        self.assertEqual(v, "UNBOUNDED")
        self.assertTrue(any("not found drained" in r for r in reasons), reasons)

    def test_too_few_rank_exits(self):
        v, reasons, _, _ = verdict(bounded(tp=2), tp=4)
        self.assertEqual(v, "UNBOUNDED")

    def test_a_journal_cut_off_mid_teardown(self):
        # What a hard lock leaves: everything up to rank 1's ctx_shutdown, which never ended, and
        # the watcher's WAIT for it.
        lines = bounded().splitlines()
        cut = next(i for i, l in enumerate(lines) if "BEGIN model.q.rank1.ctx_shutdown" in l)
        text = "\n".join(lines[:cut + 1] + [rec("WAIT", "model.q.rank1.ctx_shutdown",
                                                 "10250 ms so far, bound 10000 ms")]) + "\n"
        v, reasons, s, _ = verdict(text, tp=2)
        self.assertEqual(v, "TRUNCATED")
        self.assertIn("WAIT model.q.rank1.ctx_shutdown", s["last_record"])
        opened = " ".join(s["open_steps"])
        self.assertIn("model.q.rank1.ctx_shutdown", opened)
        self.assertIn("model.q.teardown", opened)
        self.assertNotIn("rank0.ctx_shutdown", opened)

    def test_no_stop(self):
        T[0] = 0.0
        v, _, _, _ = verdict(rec("ARM", "journal", "pid=1") + "\n")
        self.assertEqual(v, "NOT-STOPPED")

    def test_the_order_is_checked(self):
        lines = bounded().splitlines()
        # Move the step loop's end after the teardown: out of the contract's order.
        i = next(k for k, l in enumerate(lines) if "END model.q.step_loop_stop" in l)
        moved = lines.pop(i)
        j = next(k for k, l in enumerate(lines) if "END model.q.teardown" in l)
        lines.insert(j + 1, moved)
        v, reasons, _, _ = verdict("\n".join(lines) + "\n", tp=2)
        self.assertEqual(v, "UNBOUNDED")
        self.assertTrue(any("contract's order" in r for r in reasons), reasons)

    def test_the_retire_is_expected(self):
        v, reasons, s, _ = verdict(bounded(drain=(30000.5, RETIRE)), tp=2)
        self.assertEqual(v, "BOUNDED", reasons)
        self.assertEqual(reasons, [])
        self.assertEqual(len(s["expected"]), 1, s["expected"])
        self.assertIn("model.q.drain timed out as designed", s["expected"][0])

    def test_the_retire_does_not_hide_another_failure(self):
        v, reasons, s, _ = verdict(bounded(drain=(30000.5, RETIRE), split=True), tp=2)
        self.assertEqual(v, "UNBOUNDED")
        self.assertEqual(len(reasons), 1, reasons)
        self.assertIn("split release", reasons[0])
        self.assertEqual(len(s["expected"]), 1)

    def test_a_drain_over_its_bound_that_is_not_the_retire(self):
        v, reasons, s, _ = verdict(bounded(drain=(30000.5, "done")), tp=2)
        self.assertEqual(v, "UNBOUNDED")
        self.assertTrue(any("model.q.drain went over its bound" in r for r in reasons), reasons)
        self.assertEqual(s["expected"], [])

    def test_a_late_retire_is_still_over(self):
        # The timeout itself came late: more than RETIRE_SLACK_MS past the bound.
        v, reasons, s, _ = verdict(bounded(drain=(30000 + slc.RETIRE_SLACK_MS + 500, RETIRE)), tp=2)
        self.assertEqual(v, "UNBOUNDED")
        self.assertTrue(any("model.q.drain went over its bound" in r for r in reasons), reasons)
        self.assertEqual(s["expected"], [])

    def test_a_retire_without_engine_retired_is_over(self):
        lines = [l for l in bounded(drain=(30000.5, RETIRE)).splitlines() if " engine.retired " not in l]
        v, reasons, s, _ = verdict("\n".join(lines) + "\n", tp=2)
        self.assertEqual(v, "UNBOUNDED")
        self.assertTrue(any("model.q.drain went over its bound" in r for r in reasons), reasons)
        self.assertEqual(s["expected"], [])

    def test_saved_four_card_retire_is_bounded(self):
        # The 0.3.0 stop matrix's C6-2: Ctrl-Break mid-prefill at 8,192 of 250,035 tokens, the
        # drain timed out at 30 s, retire, all four ranks left at step 31, every stream drained.
        v, reasons, s, bad = verdict(fixture("tp4-retire-bounded.log"), tp=4)
        self.assertEqual(bad, 0)
        self.assertEqual(v, "BOUNDED", reasons)
        self.assertEqual(reasons, [])
        self.assertEqual(s["expected"], [
            "model.qwen3.8-next-flash.drain timed out as designed and retired the open responses: "
            "30000.5 ms against 30000 ms"])
        self.assertEqual(len(s["rank_exits"]), 4)

    def test_saved_two_rank_retire_is_bounded(self):
        # The host test after the SPLIT fix: a 250 ms drain set to time out, then retire.
        v, reasons, s, bad = verdict(fixture("tp2-retire-bounded.log"), tp=2)
        self.assertEqual(bad, 0)
        self.assertEqual(v, "BOUNDED", reasons)
        self.assertEqual(len(s["expected"]), 1)

    def test_saved_split_through_the_retire_is_still_caught(self):
        # The same host test before the fix: the retire, then a rank leaving after a split release
        # and the journal cut off. The retire is expected; the split and the cut are not.
        v, reasons, s, bad = verdict(fixture("tp2-retire-split-truncated.log"), tp=2)
        self.assertEqual(bad, 0)
        self.assertEqual(v, "TRUNCATED")
        self.assertTrue(any("split release" in r for r in reasons), reasons)
        self.assertFalse(any("drain" in r for r in reasons), reasons)
        self.assertEqual(len(s["expected"]), 1)

    def test_main_exit_status(self):
        # Written only where the test run says (ctest sets RAD_TEST_TMPDIR in the build tree).
        d = os.environ.get("RAD_TEST_TMPDIR")
        if not d:
            self.skipTest("RAD_TEST_TMPDIR is not set")
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "stop_log_check_test.log")
        with open(p, "w", encoding="utf-8") as f:
            f.write(bounded())
        try:
            self.assertEqual(slc.main([p, "--tp", "2"]), 0)
            self.assertEqual(slc.main([os.path.join(FIXTURES, "tp4-retire-bounded.log"), "--tp", "4"]), 0)
            self.assertEqual(slc.main([os.path.join(d, "stop_log_check_missing.log")]), 3)
        finally:
            os.remove(p)


if __name__ == "__main__":
    unittest.main()

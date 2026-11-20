"""Bounded live-path regressions: fake nets/frames, no camera or training."""
from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config, features, landmarks, live, phrases, predictor  # noqa: E402


class FakeNet:
    def __init__(self, probs=(0.96, 0.03, 0.01), classes=True, version=None):
        self.probs = np.asarray(probs, np.float32)
        self.arch = SimpleNamespace(in_dim=features.feature_dim(version))
        if classes:
            self.arch.num_classes = len(probs)
        self.calls = 0
        self.last_input = None

    def predict(self, x):
        self.calls += 1
        self.last_input = np.array(x, copy=True)
        return self.probs[None]


def row(i, active=True, hands=True):
    out = live.fake_row(active) if hands else live.nohand_row(active)
    out[0, 2] = i  # identity only; the unchanged features do not use landmark z
    return out


def ids(rows):
    return [int(r[0, 2]) for r in rows]


class LiveSegmentsTest(unittest.TestCase):
    def setUp(self):
        # Fail closed if a regression unexpectedly reaches real hardware/assets.
        for target in ("src.live.cv2.VideoCapture", "src.landmarks.Tracker",
                       "src.landmarks.ensure_model", "src.predictor.model.load"):
            guard = patch(target, side_effect=AssertionError("external access"))
            guard.start()
            self.addCleanup(guard.stop)
        book = patch("src.predictor.phrases.load_book",
                     side_effect=lambda _: phrases.PhraseBook())
        book.start()
        self.addCleanup(book.stop)

    def make(self, net=None, labels=None, **kwargs):
        p = predictor.WordPredictor(
            net=net or FakeNet(),
            meta={"labels": labels or ["bye", config.REST_LABEL,
                                       config.OTHER_LABEL]},
            log=Mock(), **kwargs)
        self.addCleanup(p.close)
        return p

    def test_failed_onsets_clear_and_accepted_onset_backfills_context(self):
        p = self.make(stride=10000)
        for i in range(0, 90, 3):
            p.update(row(i, True))
            p.update(row(i + 1, True))
            out = p.update(row(i + 2, False))
            self.assertEqual(p._onset_rows, [])
            self.assertEqual(p.seg_rows, [])
            self.assertFalse(out["segment_cut"])
            self.assertIn("too short", out["reason"])
        p.update(row(90, True))
        p.update(row(91, True))
        out = p.update(row(92, True))
        self.assertEqual(ids(p._decision_rows()), [89, 90, 91, 92])
        self.assertEqual(out["decision_frames"], 4)
        self.assertEqual(out["display"], "bye")
        self.assertEqual(p._onset_rows, [])

    def test_internal_dips_and_one_frame_context_match_offline_window(self):
        p = self.make()
        flags = [False] + [True] * 3 + [False] * 2 + [True] * 2 + [False] * 5
        rows = [row(i, flag) for i, flag in enumerate(flags)]
        with patch.object(features, "sequence", wraps=features.sequence) as seq:
            outputs = [p.update(r) for r in rows]
        self.assertEqual(sum(o["segment_cut"] for o in outputs), 1)
        self.assertEqual(outputs[-1]["committed_word"], "bye")
        actual = seq.call_args.args[0]
        start, end = landmarks.signing_window(np.stack(rows))
        self.assertEqual(ids(actual), list(range(start, end)))
        self.assertEqual(ids(actual), list(range(9)))
        self.assertEqual(outputs[-1]["decision_frames"], 9)
        self.assertEqual(p.seg_rows, [])
        self.assertEqual(p._rest_rows, [])

    def test_three_frame_sign_reads_and_cuts_with_or_without_start_context(self):
        for context in (False, True):
            with self.subTest(context=context):
                p = self.make(stride=10000)
                if context:
                    p.update(row(-1, False))
                for i in range(3):
                    out = p.update(row(i))
                self.assertEqual(out["display"], "bye")
                self.assertEqual(out["decision_frames"], 3 + int(context))
                for i in range(3, 8):
                    out = p.update(row(i, False))
                self.assertTrue(out["segment_cut"])
                self.assertEqual(out["committed_word"], "bye")
                self.assertEqual(out["decision_frames"], 4 + int(context))

    def test_long_segment_state_is_bounded_and_closing_tail_does_not_evict(self):
        p = self.make(buffer_seconds=0.01, stride=100000)
        for i in range(1000):
            p.update(row(i, i < 3 or i % 5 == 0))
            self.assertLessEqual(len(p.seg_rows), p.buffer.maxlen)
            self.assertLessEqual(len(p._rest_rows), p.seg.rest - 1)
            self.assertLessEqual(len(p._onset_rows), p.seg.hold)
            self.assertLessEqual(len(p.buffer), p.buffer.maxlen)
        p.update(row(1000, True))  # backfill the four inactive internal rows
        with patch.object(features, "sequence", wraps=features.sequence) as seq:
            for i in range(1001, 1006):
                cut = p.update(row(i, False))
        self.assertTrue(cut["segment_cut"])
        self.assertEqual(cut["decision_frames"], p.buffer.maxlen)
        self.assertEqual(ids(seq.call_args.args[0]), list(range(994, 1002)))

    def test_manual_boundaries_bypass_shoulders_only_and_are_idempotent(self):
        p = self.make(manual=True)
        for i in range(5):
            p.update(row(i, True))  # even raised hands do not auto-start
        self.assertEqual(p.net.calls, 0)
        self.assertFalse(p.seg.signing)
        p.begin_segment()
        for i in range(5, 9):
            p.update(row(i, False))  # chest-level sign
        p.begin_segment()  # do not erase or restart an already open segment
        self.assertEqual(ids(p._decision_rows()), [5, 6, 7, 8])
        cut = p.end_segment()
        self.assertTrue(cut["segment_cut"])
        self.assertEqual(cut["committed_word"], "bye")
        self.assertEqual(cut["hand_rate"], 1.0)
        again = p.end_segment()
        self.assertFalse(again["segment_cut"])
        self.assertEqual(again["committed_word"], "")
        self.assertEqual(again["text"], "bye")
        self.assertIn("no open", again["reason"])
        for i in range(10):
            self.assertFalse(p.update(row(i, True))["segment_cut"])
        self.assertFalse(p.manual_open)

    def test_begin_switches_from_auto_without_leaking_partial_rows(self):
        p = self.make()
        for i in range(4):
            p.update(row(i))
        p.begin_segment()
        self.assertTrue(p.manual)
        self.assertEqual(p.seg_rows, [])
        for i in range(4, 7):
            p.update(row(i, False))
        self.assertEqual(ids(p._decision_rows()), [4, 5, 6])
        self.assertEqual(p.end_segment()["decision_frames"], 3)

    def test_empty_and_short_manual_segments_are_explicit_cuts(self):
        for n in range(3):
            with self.subTest(frames=n):
                p = self.make(manual=True)
                p.begin_segment()
                for i in range(n):
                    p.update(row(i, False))
                out = p.end_segment()
                self.assertTrue(out["segment_cut"])
                self.assertEqual(out["decision_frames"], n)
                self.assertIn("too short", out["reason"])
                self.assertEqual(out["committed_word"], "")
                self.assertEqual(p.net.calls, 0)

    def test_manual_rows_bounded_reset_and_close_clear_all_segment_state(self):
        p = self.make(manual=True, buffer_seconds=0.01, stride=100000)
        p.begin_segment()
        for i in range(1000):
            p.update(row(i, False))
        self.assertEqual(len(p.seg_rows), p.buffer.maxlen)
        self.assertEqual(ids(p._decision_rows()), list(range(992, 1000)))
        self.assertEqual(p.end_segment()["decision_frames"], 8)
        p.begin_segment()
        p.update(row(1001, False))
        p.reset()
        for stop in (p.reset, p.close):
            p.begin_segment()
            p.update(row(1002, False))
            stop()
            self.assertFalse(p.manual_open)
            self.assertFalse(p.seg.signing)
            self.assertEqual(len(p.buffer), 0)
            self.assertEqual(p.seg_rows + p._onset_rows + p._rest_rows, [])
            self.assertIsNone(p._previous_row)
            self.assertEqual(p.result()["text"], "")
            self.assertIsNone(p.last)

    def test_manual_keeps_confidence_hedge_and_background_rules(self):
        cases = [(["bye", "REST", "OTHER"], (0.80, 0.15, 0.05), "confidence"),
                 (["good", "thanks", "OTHER"], (0.96, 0.03, 0.01), "good/thanks?"),
                 (["REST", "bye", "OTHER"], (0.96, 0.03, 0.01), ""),
                 (["OTHER", "bye", "REST"], (0.96, 0.03, 0.01), "")]
        for labels, probs, reason in cases:
            with self.subTest(labels=labels, probs=probs):
                p = self.make(net=FakeNet(probs), labels=labels, manual=True)
                p.begin_segment()
                for i in range(3):
                    p.update(row(i, False))
                out = p.end_segment()
                self.assertEqual(out["committed_word"], "")
                self.assertEqual(out["text"], "")
                self.assertIn(reason, out["reason"])

    def test_wholly_handless_auto_and_manual_segments_refuse_confident_net(self):
        for manual in (False, True):
            with self.subTest(manual=manual):
                p = self.make(manual=manual)
                if manual:
                    p.begin_segment()
                for i in range(5):
                    p.update(row(i, not manual, hands=False))
                if manual:
                    out = p.end_segment()
                else:
                    for i in range(5, 10):
                        out = p.update(row(i, False, hands=False))
                self.assertTrue(out["segment_cut"])
                self.assertEqual(out["hand_rate"], 0.0)
                self.assertEqual(out["committed_word"], "")
                self.assertEqual(out["display"], "uncertain")
                self.assertIn("no hands", out["reason"])
                self.assertEqual(p.net.calls, 0)

    def test_partial_hand_rate_is_advisory_not_a_new_gate(self):
        p = self.make(manual=True)
        p.begin_segment()
        for i in range(10):
            p.update(row(i, False, hands=i == 0))
        out = p.end_segment()
        self.assertAlmostEqual(out["hand_rate"], 0.1)
        self.assertEqual(out["committed_word"], "bye")
        self.assertEqual(out["reason"], "")
        self.assertEqual(out["hint"], config.FRAMING_HINT)

    def test_cut_hand_rate_uses_decision_rows_not_closing_debounce_buffer(self):
        p = self.make()
        p.update(row(0, False, hands=False))
        for i in range(1, 4):
            p.update(row(i, True))
        p.update(row(4, False, hands=False))
        for i in range(5, 9):
            out = p.update(row(i, False, hands=True))
        self.assertTrue(out["segment_cut"])
        self.assertAlmostEqual(out["hand_rate"], 3 / 5)
        self.assertNotAlmostEqual(out["hand_rate"], landmarks.hand_share(p.buffer.array()))
        self.assertEqual(out["hand_in_frame"], (True, True))
        self.assertEqual(int(p.buffer.array()[-1, 0, 2]), 8)

    def test_cut_reads_again_instead_of_reusing_confident_intermediate(self):
        p = self.make(stride=10000)
        for i in range(3):
            mid = p.update(row(i))
        self.assertEqual(mid["display"], "bye")
        p.net.probs = np.asarray([0.6, 0.3, 0.1], np.float32)
        for i in range(3, 8):
            out = p.update(row(i, False))
        self.assertTrue(out["segment_cut"])
        self.assertEqual(out["committed_word"], "")
        self.assertIn("confidence", out["reason"])
        self.assertAlmostEqual(out["p1"], 0.6)

    def test_rgb_timestamps_are_monotonic_clamped_and_input_not_mirrored(self):
        p = self.make(manual=True)
        tracker = Mock()
        tracker.detect.return_value = row(0, False)
        p.tracker = tracker
        rgb = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
        with patch.object(predictor.time, "monotonic", side_effect=[3.5, 3.4, 3.8]):
            for _ in range(3):
                p.update(rgb)
        p.update(rgb, ts_ms=3790)
        p.update(rgb, ts_ms=4100)
        self.assertEqual([c.args[1] for c in tracker.detect.call_args_list],
                         [3500, 3501, 3800, 3801, 4100])
        for call in tracker.detect.call_args_list:
            np.testing.assert_array_equal(call.args[0], rgb)
        p.reset()
        tracker.restart.assert_called_once()
        p.update(rgb, ts_ms=0)
        self.assertEqual(tracker.detect.call_args.args[1], 0)
        p.close()
        tracker.close.assert_called_once()

    def test_checkpoint_label_order_and_arch_validation_with_legacy_fake_net(self):
        labels = ["bye", "OTHER", "hello", "REST"]
        with patch.object(predictor.vocab, "load_vocab",
                          side_effect=AssertionError("must use checkpoint labels")):
            p = self.make(FakeNet((0.96, 0.02, 0.01, 0.01)), labels=labels)
            self.assertEqual(p.labels, labels)
            self.assertEqual(p.words_only, [0, 2])
            self.assertEqual(live.selftest_words(p, []), ["bye", "hello"])
            self.assertEqual(live.selftest_words(p, ["HELLO", "bye"]), ["hello", "bye"])
            with self.assertRaisesRegex(SystemExit, "checkpoint labels"):
                live.selftest_words(p, ["untrained_word"])
            with self.assertRaisesRegex(SystemExit, "checkpoint labels"):
                live.selftest_words(p, ["REST"])
        with self.assertRaisesRegex(ValueError, "num_classes"):
            self.make(FakeNet((0.9, 0.1)), labels=["bye", "REST", "OTHER"])
        legacy = self.make(FakeNet(classes=False))
        for i in range(3):
            out = legacy.update(row(i))
        self.assertEqual(out["display"], "bye")

    def test_existing_v2_and_current_feature_layouts_are_unchanged(self):
        for version in (2, config.FEATURE_VERSION):
            with self.subTest(version=version):
                net = FakeNet(version=version)
                p = self.make(net, manual=True)
                p.begin_segment()
                rows = [row(i, False) for i in range(3)]
                for r in rows:
                    p.update(r)
                p.end_segment()
                expected = features.sequence(np.stack(rows), version=version)
                np.testing.assert_allclose(net.last_input[0], expected)
                self.assertEqual(net.last_input.shape,
                                 (1, config.NUM_FRAMES, features.feature_dim(version)))
                self.assertEqual(p.fver, version)

    def test_overlay_mirrors_pixels_and_landmarks_without_mutating_inputs(self):
        rgb = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
        before = rgb.copy()
        absent = np.full((config.P_TOTAL, 3), np.nan, np.float32)
        np.testing.assert_array_equal(live.overlay(rgb, absent, False), rgb[..., ::-1])
        np.testing.assert_array_equal(live.overlay(rgb, absent, True), rgb[:, ::-1, ::-1])
        np.testing.assert_array_equal(rgb, before)
        point = absent.copy()
        wrist = landmarks.hand_row("left", 0)
        point[wrist, :2] = (0.25, 0.5)
        original = point.copy()
        image = np.zeros((80, 100, 3), np.uint8)
        normal = live.overlay(image, point, False)
        mirrored = live.overlay(image, point, True)
        np.testing.assert_array_equal(normal[40, 25], live.CYAN)
        np.testing.assert_array_equal(mirrored[40, 74], live.CYAN)
        self.assertEqual(live._xy(point, wrist, 100, 80, True), (74, 40))
        np.testing.assert_array_equal(point, original)

    def test_selftest_reports_cut_separately_and_distinguishes_missing_cuts(self):
        mid = {"top3": [("bye", 0.99)], "p1": 0.99, "margin": 0.98}
        cut = {"top3": [("hello", 0.65)], "p1": 0.65, "margin": 0.3,
               "reason": "confidence too low", "hand_rate": 0.6,
               "committed_word": ""}
        summary = live.selftest_row("bye", cut, mid, True)
        self.assertEqual(summary["top1"], "hello")
        self.assertEqual(summary["p1"], 0.65)
        self.assertEqual(summary["hand"], 0.6)
        self.assertEqual(summary["intermediate_top1"], "bye")
        self.assertEqual(summary["intermediate_p1"], 0.99)
        no_open = live.selftest_row("bye", None, None, False)
        self.assertIn("no open", no_open["reason"])
        short = live.selftest_row("bye", None, None, False, "too short: onset")
        self.assertIn("too short", short["reason"])
        still_open = live.selftest_row("bye", None, mid, True, "too short: old")
        self.assertIn("still open", still_open["reason"])
        self.assertNotIn("too short", still_open["reason"])
        self.assertEqual(still_open["top1"], "")
        self.assertEqual(still_open["p1"], 0.0)
        self.assertIsNone(still_open["hand"])
        short_cut = live.selftest_row("bye", {"reason": "too short: 1 frame"}, mid, True)
        self.assertTrue(short_cut["segment"])
        self.assertIn("too short", short_cut["reason"])

    def test_manual_cli_rejects_noninteractive_modes_before_external_access(self):
        for flag in ("--selftest", "--headless", "--benchmark", "--render-test",
                     "--demo", "--logic-test"):
            with self.subTest(flag=flag), contextlib.redirect_stderr(io.StringIO()):
                with patch.object(live, "build_predictor",
                                  side_effect=AssertionError("must reject first")):
                    with self.assertRaises(SystemExit) as cm:
                        live.main(["--manual", flag])
                    self.assertEqual(cm.exception.code, 2)
        with patch.object(live, "selftest", return_value=0) as drill:
            self.assertEqual(live.main(["--selftest", "bye", "hello"]), 0)
        self.assertEqual(drill.call_args.args[0].words, ["bye", "hello"])

    def test_mocked_live_space_reset_and_quit_do_not_save_or_open_camera(self):
        p = self.make(manual=True)
        p._row = Mock(return_value=row(0, False))
        cap = Mock()
        cap.read.return_value = (True, np.zeros((160, 640, 3), np.uint8))
        args = SimpleNamespace(log=None, camera=0, fps=30, no_mirror=False,
                               headless=False)
        cuts = []
        end = p.end_segment

        def capture_cut():
            out = end()
            cuts.append(out)
            return out

        with patch.object(live, "build_predictor", return_value=p), \
                patch.object(live, "open_camera", return_value=cap), \
                patch.object(live.config, "logger_for", return_value=Mock()), \
                patch.object(live.cv2, "imshow"), \
                patch.object(live.cv2, "destroyAllWindows"), \
                patch.object(live.cv2, "waitKey", side_effect=[32, -1, -1, -1,
                                                              32, 32, ord("r"), ord("q")]), \
                patch.object(p, "end_segment", side_effect=capture_cut):
            self.assertEqual(live.live(args), 0)
        self.assertEqual(len(cuts), 1)
        self.assertEqual(cuts[0]["committed_word"], "bye")
        self.assertEqual(cuts[0]["decision_frames"], 4)
        self.assertFalse(p.manual_open)
        self.assertEqual(len(p.buffer), 0)
        self.assertEqual(p.phrases.text(), "")
        cap.release.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)

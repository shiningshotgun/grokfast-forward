import sys, os, tempfile, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import unittest
import runner


class TestDeadline(unittest.TestCase):
    def test_expiry_before_work(self):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "run")
            os.makedirs(out)
            r = runner.run_arm(
                {"name": "a", "filter": "none", "wd": 0.0},
                0.0,
                out,
                torch.device("cpu"),
                clock=lambda: 1.0,
            )
            self.assertEqual(r["status"], "deadline_expired")
            self.assertEqual(r["updates_done"], 0)
            self.assertFalse(r["checkpoint_saved"])
            self.assertFalse(os.path.exists(os.path.join(out, "checkpoint_a.pt")))

    def test_expiry_during_training(self):
        calls = {"n": 0}

        def clock():
            calls["n"] += 1
            return 0 if calls["n"] <= 3 else 10.0

        model = runner.build_model()
        train_data, valid_data, g_split = runner.build_data()
        g_order = runner.make_order_gen()
        opt = runner.make_optimizer(model, 0.0)
        sched = runner.make_scheduler(opt)
        grads = None
        rec = []
        i, grads, g_order, state = runner.run_updates(
            model,
            opt,
            sched,
            grads,
            train_data,
            g_order,
            100,
            torch.device("cpu"),
            "none",
            5.0,
            rec,
            0,
            None,
            clock=clock,
        )
        self.assertEqual(i, 1)
        self.assertTrue(rec[0]["overrun"])

    def test_expiry_during_validation(self):
        calls = {"n": 0}

        def clock():
            calls["n"] += 1
            return 0 if calls["n"] <= 2 else 10.0

        model = runner.build_model()
        train_data, valid_data, g_split = runner.build_data()
        small = valid_data[:, : 512 * 3]
        vr = runner.evaluate(model, small, 512, 5.0, clock=clock)
        self.assertFalse(vr["complete"])
        self.assertEqual(vr["n"], 1024)
        self.assertTrue(vr["overrun"])

    def test_cap_reached_incomplete_final_validation(self):
        expired = {"flag": False}

        def clock():
            return 100.0 if expired["flag"] else 0.0

        orig_eval = runner.evaluate
        eval_calls = {"n": 0}

        def fake_eval(model, data, batch_size, deadline, clock=time.monotonic):
            eval_calls["n"] += 1
            if eval_calls["n"] == 1:
                return orig_eval(model, data, batch_size, deadline, clock=clock)
            expired["flag"] = True
            return {
                "loss": 2.0,
                "acc": 0.2,
                "n": data.shape[1] // 2,
                "complete": False,
                "full_set": False,
                "overrun": True,
            }

        saved = {"called": False}
        orig_save = runner.save_checkpoint

        def fake_save(*a, **k):
            saved["called"] = True

        old_cap = runner.MAX_UPDATES_PER_ARM
        runner.evaluate = fake_eval
        runner.save_checkpoint = fake_save
        runner.MAX_UPDATES_PER_ARM = 2
        try:
            with tempfile.TemporaryDirectory() as td:
                out = os.path.join(td, "run")
                os.makedirs(out)
                r = runner.run_arm(
                    {"name": "b", "filter": "none", "wd": 0.0},
                    50.0,
                    out,
                    torch.device("cpu"),
                    clock=clock,
                )
                self.assertEqual(
                    r["status"], "completed_100_updates_incomplete_validation"
                )
                self.assertEqual(r["updates_done"], 2)
                self.assertFalse(r["checkpoint_saved"])
                self.assertFalse(saved["called"])
                self.assertFalse(os.path.exists(os.path.join(out, "checkpoint_b.pt")))
        finally:
            runner.evaluate = orig_eval
            runner.save_checkpoint = orig_save
            runner.MAX_UPDATES_PER_ARM = old_cap

    def test_expiry_between_status_and_save(self):
        expired = {"flag": False}

        def clock():
            return 100.0 if expired["flag"] else 0.0

        orig_eval = runner.evaluate
        eval_calls = {"n": 0}

        def fake_eval(model, data, batch_size, deadline, clock=time.monotonic):
            eval_calls["n"] += 1
            if eval_calls["n"] == 1:
                return orig_eval(model, data, batch_size, deadline, clock=clock)
            # complete final validation, then expire before save
            expired["flag"] = True
            return {
                "loss": 1.0,
                "acc": 0.1,
                "n": data.shape[1],
                "complete": True,
                "full_set": True,
                "overrun": False,
            }

        saved = {"called": False}
        orig_save = runner.save_checkpoint

        def fake_save(*a, **k):
            saved["called"] = True

        old_cap = runner.MAX_UPDATES_PER_ARM
        runner.evaluate = fake_eval
        runner.save_checkpoint = fake_save
        runner.MAX_UPDATES_PER_ARM = 2
        try:
            with tempfile.TemporaryDirectory() as td:
                out = os.path.join(td, "run")
                os.makedirs(out)
                r = runner.run_arm(
                    {"name": "d", "filter": "none", "wd": 0.0},
                    50.0,
                    out,
                    torch.device("cpu"),
                    clock=clock,
                )
                self.assertEqual(r["status"], "completed_100_updates")
                self.assertEqual(r["updates_done"], 2)
                self.assertFalse(r["checkpoint_saved"])
                self.assertFalse(saved["called"])
        finally:
            runner.evaluate = orig_eval
            runner.save_checkpoint = orig_save
            runner.MAX_UPDATES_PER_ARM = old_cap

    def test_no_checkpoint_save_after_expiry(self):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "run")
            os.makedirs(out)
            r = runner.run_arm(
                {"name": "c", "filter": "none", "wd": 0.0},
                0.0,
                out,
                torch.device("cpu"),
                clock=lambda: 1.0,
            )
            self.assertEqual(r["status"], "deadline_expired")
            self.assertFalse(r["checkpoint_saved"])
            self.assertFalse(os.path.exists(os.path.join(out, "checkpoint_c.pt")))

    def test_later_arms_not_run(self):
        with tempfile.TemporaryDirectory() as td:
            old_ts = runner.TOTAL_SECONDS
            runner.TOTAL_SECONDS = 0
            try:
                results, elapsed, maxrss = runner.run_pilot(td, clock=lambda: 1.0)
            finally:
                runner.TOTAL_SECONDS = old_ts
            for r in results:
                self.assertEqual(r["status"], "not_run_deadline")
            # elapsed values nonnegative
            self.assertGreaterEqual(elapsed, 0)
            self.assertGreaterEqual(maxrss, 0)

    def test_elapsed_values_nonnegative(self):
        with tempfile.TemporaryDirectory() as td:
            old_ts = runner.TOTAL_SECONDS
            runner.TOTAL_SECONDS = 0
            try:
                results, elapsed, maxrss = runner.run_pilot(td, clock=lambda: 1.0)
            finally:
                runner.TOTAL_SECONDS = old_ts
            self.assertGreaterEqual(elapsed, 0)
            self.assertGreaterEqual(maxrss, 0)


if __name__ == "__main__":
    unittest.main()

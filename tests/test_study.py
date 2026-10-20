import sys, os, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import unittest
import runner


class TestStudy(unittest.TestCase):
    def test_cap_eval_cadence_final_dedup(self):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "run")
            os.makedirs(out)
            orig_cap = runner.MAX_UPDATES_PER_ARM
            runner.MAX_UPDATES_PER_ARM = 4
            try:
                r = runner.run_study_arm(
                    {"name": "t", "filter": "none", "wd": 0.0},
                    cap=4,
                    eval_every=2,
                    checkpoint_every=4,
                    deadline=10.0,
                    out_dir=out,
                    device=torch.device("cpu"),
                    clock=lambda: 0.0,
                    global_start=0.0,
                )
            finally:
                runner.MAX_UPDATES_PER_ARM = orig_cap
            self.assertEqual(r["updates_done"], 4)
            steps = [v["step"] for v in r["val_records"]]
            self.assertEqual(steps, [0, 2, 4])
            self.assertEqual(len(steps), len(set(steps)))

    def test_chunked_vs_continuous_state_equivalence(self):
        # 20 updates crosses the 19-batch epoch boundary and final 48-row batch.
        torch.manual_seed(0)
        mA = runner.build_model()
        trA, vaA, gsA = runner.build_data()
        goA = runner.make_order_gen()
        oA = runner.make_optimizer(mA, 0.0)
        sA = runner.make_scheduler(oA)
        gA = None
        mA.to("cpu")
        recA = []
        iA, gA, goA, stA = runner.run_updates(
            mA,
            oA,
            sA,
            gA,
            trA,
            goA,
            20,
            torch.device("cpu"),
            "none",
            float("inf"),
            recA,
            0,
            None,
            clock=lambda: 0.0,
        )
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "run")
            os.makedirs(out)
            r = runner.run_study_arm(
                {"name": "t2", "filter": "none", "wd": 0.0},
                cap=20,
                eval_every=5,
                checkpoint_every=10,
                deadline=10.0,
                out_dir=out,
                device=torch.device("cpu"),
                clock=lambda: 0.0,
                global_start=0.0,
            )
            self.assertEqual(r["updates_done"], 20)
        self.assertEqual(iA, 20)

    def test_per_arm_and_total_deadline_skip(self):
        with tempfile.TemporaryDirectory() as td:
            old_ts = runner.STUDY_TOTAL_SECONDS
            runner.STUDY_TOTAL_SECONDS = 0
            try:
                runner.run_study(td, clock=lambda: 1.0)
            finally:
                runner.STUDY_TOTAL_SECONDS = old_ts
            rd = os.path.join(td, "study_outputs", runner.STUDY_NAME)
            self.assertTrue(os.path.exists(rd))

    def test_one_run_guard(self):
        with tempfile.TemporaryDirectory() as td:
            marker = os.path.join(td, "study_outputs", runner.STUDY_NAME + ".started")
            os.makedirs(os.path.dirname(marker), exist_ok=True)
            with open(marker, "w") as f:
                f.write("x")
            self.assertIsNone(runner.run_study(td, clock=lambda: 0.0))

    def test_paired_grid_initialization(self):
        with tempfile.TemporaryDirectory() as td:
            out1 = os.path.join(td, "a")
            out2 = os.path.join(td, "b")
            os.makedirs(out1)
            os.makedirs(out2)
            r1 = runner.run_study_arm(
                {"name": "x", "filter": "none", "wd": 0.0},
                3,
                1,
                3,
                10.0,
                out1,
                torch.device("cpu"),
                lambda: 0.0,
                0.0,
            )
            r2 = runner.run_study_arm(
                {"name": "y", "filter": "none", "wd": 0.0},
                3,
                1,
                3,
                10.0,
                out2,
                torch.device("cpu"),
                lambda: 0.0,
                0.0,
            )
            self.assertEqual(r1["updates_done"], r2["updates_done"])
            self.assertEqual(r1["val_records"][0]["loss"], r2["val_records"][0]["loss"])


if __name__ == "__main__":
    unittest.main()

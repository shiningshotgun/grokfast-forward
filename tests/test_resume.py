import sys, os, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import unittest
from runner import (
    build_model,
    build_data,
    make_order_gen,
    make_optimizer,
    make_scheduler,
    run_updates,
    save_checkpoint,
    load_checkpoint,
)


class TestResume(unittest.TestCase):
    def _setup(self, filter_name="ema"):
        torch.manual_seed(0)
        model = build_model()
        train_data, valid_data, g_split = build_data()
        g_order = make_order_gen()
        opt = make_optimizer(model, 0.005 if filter_name == "ema" else 0.0)
        sched = make_scheduler(opt)
        grads = None
        return (
            model,
            train_data,
            valid_data,
            g_split,
            g_order,
            opt,
            sched,
            grads,
            filter_name,
        )

    def _run_case(self, n_total, n_split, filter_name="ema"):
        with tempfile.TemporaryDirectory() as td:
            ckpt_path = os.path.join(td, "ckpt.pt")
            # uninterrupted
            modelA, trainA, validA, gsplitA, gorderA, optA, schedA, gradsA, fn = (
                self._setup(filter_name)
            )
            modelA.to("cpu")
            recA = []
            iA, gradsA, gorderA, stateA = run_updates(
                modelA,
                optA,
                schedA,
                gradsA,
                trainA,
                gorderA,
                n_total,
                torch.device("cpu"),
                fn,
                float("inf"),
                recA,
                0,
                None,
            )
            # split
            modelB, trainB, validB, gsplitB, gorderB, optB, schedB, gradsB, fn = (
                self._setup(filter_name)
            )
            modelB.to("cpu")
            recB = []
            iB, gradsB, gorderB, stateB = run_updates(
                modelB,
                optB,
                schedB,
                gradsB,
                trainB,
                gorderB,
                n_split,
                torch.device("cpu"),
                fn,
                float("inf"),
                recB,
                0,
                None,
            )
            save_checkpoint(
                ckpt_path,
                modelB,
                optB,
                schedB,
                gradsB,
                trainB,
                validB,
                gorderB,
                gsplitB,
                iB,
                recB,
                [],
                "paused",
                epoch=stateB["epoch"],
                resume_state=stateB,
                weight_decay=0.005 if fn == "ema" else 0.0,
            )
            (
                modelC,
                optC,
                schedC,
                gradsC,
                trainC,
                validC,
                gorderC,
                gsplitC,
                ckpt,
                resume,
            ) = load_checkpoint(ckpt_path, torch.device("cpu"))
            recC = list(ckpt["train_records"])
            iC, gradsC, gorderC, stateC = run_updates(
                modelC,
                optC,
                schedC,
                gradsC,
                trainC,
                gorderC,
                n_total,
                torch.device("cpu"),
                fn,
                float("inf"),
                recC,
                ckpt["updates_done"],
                resume,
            )
            # comparisons
            for pa, pc in zip(modelA.parameters(), modelC.parameters()):
                self.assertTrue(torch.allclose(pa, pc, atol=1e-5))
            self.assertAlmostEqual(
                optA.param_groups[0]["lr"], optC.param_groups[0]["lr"], places=9
            )
            self.assertEqual(iA, iC)
            self.assertTrue(torch.equal(gorderA.get_state(), gorderC.get_state()))
            if fn == "ema":
                self.assertIsNotNone(gradsA)
                self.assertIsNotNone(gradsC)
                for k in gradsA:
                    self.assertTrue(torch.allclose(gradsA[k], gradsC[k], atol=1e-5))
            # optimizer state
            sa = optA.state_dict()["state"]
            sc = optC.state_dict()["state"]
            self.assertEqual(len(sa), len(sc))
            for k in sa:
                for kk in sa[k]:
                    if torch.is_tensor(sa[k][kk]):
                        self.assertTrue(torch.allclose(sa[k][kk], sc[k][kk], atol=1e-5))
            # RNG state restored (order gen already); global rng not strictly required

    def test_resume_matches_uninterrupted_ema(self):
        self._run_case(4, 2, "ema")

    def test_resume_across_epoch_boundary(self):
        # 19 batches per epoch at 512; cross from epoch 0 into epoch 1
        self._run_case(21, 19, "ema")


if __name__ == "__main__":
    unittest.main()

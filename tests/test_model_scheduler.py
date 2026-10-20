import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import unittest
from runner import build_model, make_optimizer, make_scheduler


class TestModelScheduler(unittest.TestCase):
    def test_scheduler_warmup(self):
        m2 = build_model()
        opt2 = make_optimizer(m2, 0.0)
        sched2 = make_scheduler(opt2)
        for p in m2.parameters():
            p.grad = torch.zeros_like(p)
        lrs = []
        for step in range(1, 13):
            opt2.step()
            sched2.step()
            lrs.append(opt2.param_groups[0]["lr"])
        self.assertAlmostEqual(lrs[0], 0.0001, places=9)
        self.assertAlmostEqual(lrs[9], 0.001, places=9)
        self.assertAlmostEqual(lrs[10], 0.001, places=9)

    def test_param_count(self):
        m = build_model()
        n = sum(p.numel() for p in m.parameters() if p.requires_grad)
        self.assertGreater(n, 100000)
        self.assertLess(n, 500000)

    def test_causal_mask_shape(self):
        m = build_model()
        x = torch.randint(0, 99, (5, 4))
        y = m(x)
        self.assertEqual(tuple(y.shape), (5, 4, 99))

    def test_lr_used_before_optimizer_step_updates_1_to_12(self):
        m = build_model()
        opt = make_optimizer(m, 0.0)
        sched = make_scheduler(opt)
        for p in m.parameters():
            p.grad = torch.zeros_like(p)
        used = []
        for _ in range(12):
            used.append(opt.param_groups[0]["lr"])
            opt.step()
            sched.step()
        expected = [
            0.0,
            0.0001,
            0.0002,
            0.0003,
            0.0004,
            0.0005,
            0.0006,
            0.0007,
            0.0008,
            0.0009,
            0.001,
            0.001,
        ]
        self.assertEqual([round(x, 6) for x in used], [round(x, 6) for x in expected])
        # First update uses LR 0 but still mutates Adam moment buffers.
        self.assertEqual(used[0], 0.0)


if __name__ == "__main__":
    unittest.main()

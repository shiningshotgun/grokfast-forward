import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import unittest
from grokfast import gradfilter_ema


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 4, bias=False)
        self.b = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.b(self.a(x))


def set_grads(model, names):
    for p in model.parameters():
        p.grad = None
    for n, p in model.named_parameters():
        if n in names:
            p.grad = names[n].clone()


class TestEMA(unittest.TestCase):
    def test_first_step_three_x_raw(self):
        torch.manual_seed(0)
        m = Tiny()
        raw = {}
        vals = {}
        for n, p in m.named_parameters():
            vals[n] = torch.randn(4, 4)
            raw[n] = vals[n].clone()
        set_grads(m, vals)
        grads = gradfilter_ema(m, grads=None, alpha=0.98, lamb=2.0)
        params = dict(m.named_parameters())
        for n in raw:
            self.assertTrue(
                torch.allclose(params[n].grad.data, raw[n] * 3.0, atol=1e-6)
            )
            self.assertTrue(torch.allclose(grads[n], raw[n], atol=1e-6))

    def test_multiple_steps_ema(self):
        torch.manual_seed(1)
        m = Tiny()
        alpha = 0.98
        lamb = 2.0
        vals1 = {n: torch.randn(4, 4) for n, _ in m.named_parameters()}
        set_grads(m, vals1)
        raw1 = {n: v.clone() for n, v in vals1.items()}
        grads = gradfilter_ema(m, grads=None, alpha=alpha, lamb=lamb)
        vals2 = {n: torch.randn(4, 4) for n, _ in m.named_parameters()}
        set_grads(m, vals2)
        raw2 = {n: v.clone() for n, v in vals2.items()}
        expected_ema = {n: raw1[n] * alpha + raw2[n] * (1 - alpha) for n in raw1}
        grads = gradfilter_ema(m, grads=grads, alpha=alpha, lamb=lamb)
        params = dict(m.named_parameters())
        for n in raw1:
            self.assertTrue(torch.allclose(grads[n], expected_ema[n], atol=1e-6))
            self.assertTrue(
                torch.allclose(
                    params[n].grad.data, raw2[n] + lamb * expected_ema[n], atol=1e-6
                )
            )

    def test_missing_grad_retained(self):
        torch.manual_seed(2)
        m = Tiny()
        vals = {n: torch.randn(4, 4) for n, _ in m.named_parameters()}
        set_grads(m, vals)
        grads = gradfilter_ema(m, grads=None, alpha=0.98, lamb=2.0)
        ema_b_before = grads["b.weight"].clone()
        set_grads(m, {"a.weight": torch.randn(4, 4)})
        grads2 = gradfilter_ema(m, grads=grads, alpha=0.98, lamb=2.0)
        self.assertTrue(torch.allclose(grads2["b.weight"], ema_b_before))
        self.assertIn("a.weight", grads2)

    def test_frozen_param_skipped(self):
        torch.manual_seed(3)
        m = Tiny()
        m.b.weight.requires_grad_(False)
        vals = {"a.weight": torch.randn(4, 4)}
        set_grads(m, vals)
        grads = gradfilter_ema(m, grads=None, alpha=0.98, lamb=2.0)
        self.assertIn("a.weight", grads)
        self.assertNotIn("b.weight", grads)

    def test_late_first_gradient_initializes_from_raw(self):
        torch.manual_seed(5)
        m = Tiny()
        # step 1: only a has a gradient; b is missing
        vals1 = {"a.weight": torch.randn(4, 4)}
        set_grads(m, vals1)
        grads = gradfilter_ema(m, grads=None, alpha=0.98, lamb=2.0)
        self.assertIn("a.weight", grads)
        self.assertNotIn("b.weight", grads)
        # step 2: b appears for the first time
        raw_b = torch.randn(4, 4)
        set_grads(m, {"a.weight": torch.randn(4, 4), "b.weight": raw_b})
        grads = gradfilter_ema(m, grads=grads, alpha=0.98, lamb=2.0)
        # memory initialized from first raw gradient for b -> modified grad = 3x raw
        params = dict(m.named_parameters())
        self.assertTrue(
            torch.allclose(params["b.weight"].grad.data, raw_b * 3.0, atol=1e-6)
        )
        self.assertTrue(torch.allclose(grads["b.weight"], raw_b, atol=1e-6))


if __name__ == "__main__":
    unittest.main()

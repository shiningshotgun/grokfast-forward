import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import unittest
from model import Decoder
from data import multiplication_mod_p_data
from runner import build_model, build_data, make_order_gen, P


class TestTask(unittest.TestCase):
    def test_data_shape_and_split(self):
        data = multiplication_mod_p_data(97, 97, 98)
        self.assertEqual(tuple(data.shape), (5, 9312))
        x = data[0]
        y = data[2]
        res = data[4]
        self.assertTrue(torch.all(res == (x * y) % 97))
        self.assertTrue(torch.all(data[1] == 98))
        self.assertTrue(torch.all(data[3] == 97))
        self.assertEqual(int(y.min()), 1)
        self.assertEqual(int(y.max()), 96)
        self.assertEqual(int(x.min()), 0)
        self.assertEqual(int(x.max()), 96)

    def test_vocab(self):
        m = build_model()
        self.assertEqual(m.head.out_features, 99)
        self.assertEqual(m.token_embeddings.num_embeddings, 99)

    def test_scoring(self):
        logits = torch.tensor([[0.0, 10.0], [10.0, 0.0]])
        targets = torch.tensor([1, 0])
        acc = (logits.argmax(-1) == targets).float().mean().item()
        self.assertEqual(acc, 1.0)

    def test_paired_init_identical(self):
        m1 = build_model()
        m2 = build_model()
        for a, b in zip(m1.parameters(), m2.parameters()):
            self.assertTrue(torch.equal(a, b))

    def test_paired_split_identical(self):
        t1, v1, _ = build_data()
        t2, v2, _ = build_data()
        self.assertTrue(torch.equal(t1, t2))
        self.assertTrue(torch.equal(v1, v2))
        self.assertEqual(t1.shape[1], 4656)
        self.assertEqual(v1.shape[1], 4656)

    def test_paired_order_identical(self):
        g1 = make_order_gen()
        g2 = make_order_gen()
        a = torch.randperm(100, generator=g1)
        b = torch.randperm(100, generator=g2)
        self.assertTrue(torch.equal(a, b))

    def test_model_forward_shape(self):
        m = build_model()
        x = torch.randint(0, 99, (5, 8))
        out = m(x)
        self.assertEqual(tuple(out.shape), (5, 8, 99))

    def test_4656_row_batch_layout(self):
        n = 4656
        bs = 512
        data = torch.arange(n)
        batches = list(torch.split(data, bs))
        sizes = [b.shape[0] for b in batches]
        self.assertEqual(sizes, [512] * 9 + [48])
        self.assertEqual(sum(sizes), n)
        self.assertEqual(len(batches), 10)


if __name__ == "__main__":
    unittest.main()

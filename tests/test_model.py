import unittest

import torch

from flowmap.model import ToyMLP, TinyTransformer, make_model, sinusoidal_embedding


class TestModels(unittest.TestCase):
    def test_sinusoidal_shape(self):
        t = torch.rand(5)
        e = sinusoidal_embedding(t, 64)
        self.assertEqual(e.shape, (5, 64))
        self.assertTrue(torch.isfinite(e).all())

    def test_toy_mlp_shapes(self):
        m = ToyMLP(dim_in=2, hidden=64, n_layers=2)
        x = torch.randn(7, 1, 2)
        t = torch.rand(7)
        y = m(x, t)
        self.assertEqual(y.shape, (7, 1, 2))

    def test_two_time_requires_s(self):
        m = ToyMLP(dim_in=2, hidden=32, n_layers=1, two_time=True)
        with self.assertRaises(ValueError):
            m(torch.randn(2, 1, 2), torch.rand(2))
        y = m(torch.randn(2, 1, 2), torch.rand(2), torch.rand(2))
        self.assertEqual(y.shape, (2, 1, 2))

    def test_transformer_shapes(self):
        m = TinyTransformer(dim_in=16, seq_len=8, d_model=32, n_layers=1, n_heads=4)
        x = torch.randn(3, 8, 16)
        y = m(x, torch.rand(3))
        self.assertEqual(y.shape, (3, 8, 16))

    def test_make_model_auto(self):
        toy = make_model({}, dim_in=2, seq_len=1)
        self.assertIsInstance(toy, ToyMLP)
        tok = make_model({}, dim_in=16, seq_len=8)
        self.assertIsInstance(tok, TinyTransformer)


if __name__ == "__main__":
    unittest.main()

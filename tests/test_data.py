import unittest

import torch

from flowmap.data import MarkovCorpus, ToyGaussians, make_data
from flowmap.model import make_model


class TestData(unittest.TestCase):
    def test_toy_gaussians_near_modes(self):
        data = ToyGaussians(seed=0)
        x = data.sample(64)
        self.assertEqual(x.shape, (64, 1, 2))
        d = torch.cdist(x[:, 0], data.centers)
        self.assertTrue(torch.all(d.min(dim=1).values < 0.6))
        self.assertEqual(data.centers.shape, (8, 2))

    def test_markov_shapes_and_validity(self):
        data = MarkovCorpus(vocab=16, seq_len=12, seed=0)
        toks = data.sample_tokens(32)
        self.assertEqual(toks.shape, (32, 12))
        self.assertEqual(data.valid_bigram_rate(toks), 1.0)
        x = data.sample(32)
        self.assertEqual(x.shape, (32, 12, 16))
        x = data.encode(toks)
        back = data.decode(x)
        self.assertTrue(torch.equal(back, toks))

    def test_make_data(self):
        self.assertEqual(make_data({"data": {"kind": "toy"}}).kind, "toy")
        self.assertEqual(
            make_data({"data": {"kind": "markov", "vocab": 8, "seq_len": 5}}).vocab, 8
        )

    def test_markov_model_forward(self):
        data = MarkovCorpus(vocab=16, seq_len=8, seed=0)
        m = make_model({"model": {"arch": "transformer", "d_model": 32, "n_layers": 1,
                                  "n_heads": 4}}, data.dim_in, data.seq_len)
        y = m(torch.randn(4, 8, 16), torch.rand(4))
        self.assertEqual(y.shape, (4, 8, 16))


if __name__ == "__main__":
    unittest.main()

import csv
import os
import tempfile
import unittest

import torch

from flowmap.train import train


def toy_cfg(steps=300, hidden=128, layers=3, batch=256, lr=0.003):
    return {
        "algo": "flm",
        "seed": 0,
        "device": "cpu",
        "data": {"kind": "toy"},
        "model": {"arch": "auto", "hidden": hidden, "n_layers": layers},
        "schedule": {"name": "linear"},
        "train": {
            "steps": steps,
            "batch_size": batch,
            "lr": lr,
            "weight_decay": 0.01,
            "ema_decay": 0.999,
            "log_every": max(1, steps // 5),
            "ckpt_every": steps,
        },
        "eval": {"n": 512, "sample_steps": 8},
    }


class TestFLMTraining(unittest.TestCase):
    def test_toy_training_reduces_loss_and_covers_data(self):
        with tempfile.TemporaryDirectory() as d:
            summary = train(toy_cfg(steps=1500), d)
            self.assertTrue(os.path.exists(os.path.join(d, "ckpt.pt")))
            self.assertTrue(os.path.exists(os.path.join(d, "summary.json")))
            m = summary["metrics_ema"]
            self.assertLess(m["mean_nn_distance"], 0.15)
            self.assertGreater(m["coverage_0.3"], 0.8)
            self.assertGreater(m["mode_hit_0.3"], 0.5)
            self.assertEqual(m["mode_coverage"], 1.0)
            losses = []
            with open(os.path.join(d, "train_log.csv"), encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    losses.append(float(row["loss"]))
            self.assertGreaterEqual(len(losses), 3)
            # The conditional velocity has irreducible variance, so the loss
            # floor is > 0; require a clear relative improvement instead.
            self.assertLess(losses[-1], 0.9 * losses[0])

    def test_determinism_cpu(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            s1 = train(toy_cfg(steps=20), d1)
            s2 = train(toy_cfg(steps=20), d2)
            self.assertEqual(s1["metrics_raw"], s2["metrics_raw"])

    def test_sample_shapes(self):
        from flowmap.flm import flm_sample
        from flowmap.model import ToyMLP

        m = ToyMLP(dim_in=2, hidden=32, n_layers=1)
        x = flm_sample(m, (5, 1, 2), steps=4, device="cpu")
        self.assertEqual(x.shape, (5, 1, 2))
        self.assertTrue(torch.isfinite(x).all())


if __name__ == "__main__":
    unittest.main()

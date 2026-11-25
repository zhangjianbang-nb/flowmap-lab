import os
import tempfile
import unittest

import torch

from flowmap.fmlm import one_step_sample, trajectory_agreement
from flowmap.train import train


def fmlm_cfg(teacher_steps=300, steps=300, hidden=96, layers=2, batch=128):
    return {
        "algo": "fmlm",
        "seed": 0,
        "device": "cpu",
        "data": {"kind": "toy"},
        "model": {"arch": "auto", "hidden": hidden, "n_layers": layers, "two_time": True},
        "schedule": {"name": "linear"},
        "train": {
            "steps": steps,
            "batch_size": batch,
            "lr": 0.003,
            "weight_decay": 0.01,
            "ema_decay": 0.999,
            "log_every": max(1, steps // 4),
            "ckpt_every": steps,
            "teacher_steps": teacher_steps,
            "teacher_ckpt": None,
            "full_jump_prob": 0.5,
        },
        "eval": {"n": 512},
    }


class TestFMLM(unittest.TestCase):
    def test_one_step_matches_teacher_trajectory(self):
        with tempfile.TemporaryDirectory() as d:
            summary = train(fmlm_cfg(), d)
            self.assertTrue(os.path.exists(os.path.join(d, "teacher.pt")))
            self.assertTrue(os.path.exists(os.path.join(d, "ckpt.pt")))
            m = summary["metrics_ema"]
            self.assertLess(m["mean_nn_distance"], 1.0)
            self.assertGreater(m["coverage_0.3"], 0.1)

            from flowmap.model import make_model
            from flowmap.utils import load_checkpoint

            payload = load_checkpoint(os.path.join(d, "ckpt.pt"))
            model = make_model(payload["config"], 2, 1)
            model.load_state_dict(payload["ema"])
            tpayload = load_checkpoint(os.path.join(d, "teacher.pt"))
            teacher = make_model(tpayload["config"], 2, 1)
            teacher.load_state_dict(tpayload["model"])
            teacher.eval()
            torch.manual_seed(1)
            x0 = torch.randn(64, 1, 2)
            cos = trajectory_agreement(model, teacher, x0, None, teacher_steps=8)
            self.assertGreater(cos, 0.5)

    def test_one_step_shape(self):
        from flowmap.model import ToyMLP

        m = ToyMLP(dim_in=2, hidden=32, n_layers=1, two_time=True)
        x = one_step_sample(m, (3, 1, 2))
        self.assertEqual(x.shape, (3, 1, 2))


if __name__ == "__main__":
    unittest.main()

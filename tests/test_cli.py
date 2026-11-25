import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestCLI(unittest.TestCase):
    def test_train_resume_and_eval_cli(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "run")
            env = dict(os.environ, PYTHONPATH=ROOT)
            cmd = [sys.executable, "-m", "flowmap.train",
                   "--config", os.path.join(ROOT, "configs", "flm_tiny.yaml"),
                   "--out", out, "--steps", "12", "--device", "cpu"]
            r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            self.assertTrue(os.path.exists(os.path.join(out, "ckpt.pt")))
            summary = json.load(open(os.path.join(out, "summary.json"), encoding="utf-8"))
            self.assertEqual(summary["steps"], 12)
            self.assertEqual(summary["algo"], "flm")

            cmd2 = [sys.executable, "-m", "flowmap.train",
                    "--config", os.path.join(ROOT, "configs", "flm_tiny.yaml"),
                    "--out", out, "--steps", "20", "--device", "cpu",
                    "--resume", os.path.join(out, "ckpt.pt")]
            r2 = subprocess.run(cmd2, cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
            self.assertEqual(r2.returncode, 0, r2.stderr[-2000:])
            self.assertIn("[resume]", r2.stdout)

            cmd3 = [sys.executable, "-m", "flowmap.eval",
                    "--ckpt", os.path.join(out, "ckpt.pt"), "--device", "cpu"]
            r3 = subprocess.run(cmd3, cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
            self.assertEqual(r3.returncode, 0, r3.stderr[-2000:])
            metrics = json.loads(r3.stdout)
            self.assertIn("mean_nn_distance", metrics)

    def test_set_override(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "run")
            env = dict(os.environ, PYTHONPATH=ROOT)
            cmd = [sys.executable, "-m", "flowmap.train",
                   "--config", os.path.join(ROOT, "configs", "flm_tiny.yaml"),
                   "--out", out, "--steps", "5", "--device", "cpu",
                   "--set", "schedule.k=8.0"]
            r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            summary = json.load(open(os.path.join(out, "summary.json"), encoding="utf-8"))
            self.assertEqual(summary["steps"], 5)


if __name__ == "__main__":
    unittest.main()

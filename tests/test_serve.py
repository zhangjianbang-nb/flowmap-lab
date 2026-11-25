import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class TestServe(unittest.TestCase):
    def test_health_and_generate(self):
        from flowmap.train import train
        from tests.test_flm import toy_cfg

        with tempfile.TemporaryDirectory() as d:
            train(toy_cfg(steps=8, hidden=32, layers=1, batch=32), d)
            port = free_port()
            env = dict(os.environ, PYTHONPATH=ROOT)
            proc = subprocess.Popen(
                [sys.executable, "-m", "flowmap.serve",
                 "--ckpt", os.path.join(d, "ckpt.pt"),
                 "--port", str(port), "--device", "cpu"],
                cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                health = None
                for _ in range(80):
                    try:
                        with urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=2) as r:
                            health = json.loads(r.read())
                        break
                    except Exception:
                        time.sleep(0.1)
                self.assertIsNotNone(health, "server did not start")
                self.assertTrue(health["ok"])
                req = urllib.request.Request(
                    "http://127.0.0.1:%d/generate" % port,
                    data=json.dumps({"n": 3, "steps": 4}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=10) as r:
                    out = json.loads(r.read())
                self.assertEqual(out["kind"], "toy")
                self.assertEqual(len(out["points"]), 3)
                self.assertEqual(len(out["points"][0]), 2)
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                if proc.stdout:
                    proc.stdout.close()


if __name__ == "__main__":
    unittest.main()

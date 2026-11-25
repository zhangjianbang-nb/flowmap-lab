"""Minimal HTTP inference service (stdlib only).

    python -m flowmap.serve --ckpt runs/flm/ckpt.pt --port 8140

    GET  /health
    POST /generate   {"n": 4, "steps": 8}

The service is intentionally dependency-free so the same code can run on a
laptop, a CI runner, or inside a container without a web framework.
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from .data import make_data
from .flm import flm_sample
from .fmlm import one_step_sample
from .model import make_model
from .utils import load_checkpoint, pick_device

STATE = {}


def load_state(ckpt, device, use_ema=False):
    payload = load_checkpoint(ckpt, map_location=device)
    cfg = payload["config"]
    data = make_data(cfg, device=device, seed=int(cfg.get("seed", 0)))
    model = make_model(cfg, data.dim_in, data.seq_len).to(device)
    model.load_state_dict(payload["model"])
    if use_ema and "ema" in payload:
        model.load_state_dict(payload["ema"], strict=False)
    model.eval()
    STATE.update(
        {
            "model": model,
            "data": data,
            "cfg": cfg,
            "algo": cfg.get("algo", "flm"),
            "device": device,
            "lock": threading.Lock(),
        }
    )
    return STATE


def generate(n=4, steps=8):
    model = STATE["model"]
    data = STATE["data"]
    algo = STATE["algo"]
    device = STATE["device"]
    n = max(1, min(64, int(n)))
    shape = (n, data.seq_len, data.dim_in)
    with STATE["lock"], torch.no_grad():
        if algo == "fmlm":
            x = one_step_sample(model, shape, device=device)
        else:
            x = flm_sample(model, shape, steps=max(1, min(64, int(steps))), device=device)
    if data.kind == "markov":
        toks = data.decode(x).tolist()
        return {"kind": "markov", "tokens": toks}
    return {"kind": "toy", "points": x.squeeze(1).tolist()}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True, "algo": STATE["algo"],
                             "kind": STATE["data"].kind})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/generate":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            out = generate(body.get("n", 4), body.get("steps", 8))
            self._send(200, out)
        except Exception as e:  # keep the service alive on bad input
            self._send(400, {"error": "%s: %s" % (type(e).__name__, e)})

    def log_message(self, fmt, *args):
        print("[serve] " + fmt % args, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8140)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ema", action="store_true")
    args = ap.parse_args(argv)
    load_state(args.ckpt, pick_device(args.device), use_ema=args.ema)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print("[serve] http://%s:%d  algo=%s" % (args.host, args.port, STATE["algo"]),
          flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()

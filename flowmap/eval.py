"""Evaluate a trained checkpoint on its configured dataset."""

from __future__ import annotations

import argparse
import json

from .data import make_data
from .model import make_model
from .train import evaluate
from .utils import load_checkpoint, pick_device


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ema", action="store_true", help="use EMA weights")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    device = pick_device(args.device)
    payload = load_checkpoint(args.ckpt, map_location=device)
    cfg = payload["config"]
    data = make_data(cfg, device=device, seed=int(cfg.get("seed", 0)))
    model = make_model(cfg, data.dim_in, data.seq_len).to(device)
    model.load_state_dict(payload["model"])
    if args.ema and "ema" in payload:
        model.load_state_dict(payload["ema"], strict=False)
    model.eval()
    metrics = evaluate(model, cfg.get("algo", "flm"), data, cfg, device)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

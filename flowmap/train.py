"""Training entry point.

Examples
--------
    python -m flowmap.train --config configs/flm_tiny.yaml --out runs/flm
    python -m flowmap.train --config configs/fmlm_tiny.yaml --out runs/fmlm
    python -m flowmap.train --config configs/flm_tiny.yaml --steps 50 --device cpu

The script writes ``ckpt.pt`` (last step, raw + EMA weights), ``train_log.csv``
and ``summary.json``. FMLM runs first train (or load) an FLM teacher, then
distil it into a two-time flow map.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

from .data import make_data
from .flm import flm_loss, flm_sample
from .fmlm import flowmap_loss, one_step_sample
from .model import EMA, make_model
from .schedule import make_schedule
from .utils import (
    CSVLogger,
    load_checkpoint,
    load_config,
    merge,
    pick_device,
    save_checkpoint,
    set_seed,
    write_json,
)


def build(cfg, device, seed):
    data = make_data(cfg, device=device, seed=seed)
    model = make_model(cfg, data.dim_in, data.seq_len)
    if cfg.get("algo") == "fmlm" and not getattr(model, "two_time", False):
        mcfg = copy.deepcopy(cfg.get("model", {}))
        mcfg["two_time"] = True
        cfg = copy.deepcopy(cfg)
        cfg["model"] = mcfg
        model = make_model(cfg, data.dim_in, data.seq_len)
    return data, model, cfg


def train_teacher(cfg, data, device, seed, steps):
    tcfg = copy.deepcopy(cfg)
    tcfg["algo"] = "flm"
    tcfg.setdefault("model", {})
    tcfg["model"]["two_time"] = False
    teacher = make_model(tcfg, data.dim_in, data.seq_len).to(device)
    opt = torch.optim.AdamW(teacher.parameters(), lr=float(cfg["train"].get("lr", 1e-3)))
    sched = make_schedule(cfg.get("schedule"))
    for step in range(1, steps + 1):
        x1 = data.sample(int(cfg["train"].get("batch_size", 128)))
        loss = flm_loss(teacher, x1, sched, cfg["train"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
        opt.step()
        if step % max(1, steps // 5) == 0:
            print("[teacher] step %d/%d loss %.5f" % (step, steps, loss.item()), flush=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def evaluate(model, algo, data, cfg, device):
    sched = make_schedule(cfg.get("schedule"))
    n = int(cfg.get("eval", {}).get("n", 512))
    if algo == "fmlm":
        samples = one_step_sample(model, (n, data.seq_len, data.dim_in), device=device)
    else:
        samples = flm_sample(
            model,
            (n, data.seq_len, data.dim_in),
            steps=int(cfg.get("eval", {}).get("sample_steps", 8)),
            device=device,
            schedule=sched,
        )
    if data.kind == "toy":
        ref = data.reference(4096).to(device)
        flat = samples.reshape(n, -1)
        d = torch.cdist(flat, ref.reshape(ref.shape[0], -1))
        nn_dist = d.min(dim=1).values
        out = {
            "kind": "toy",
            "n": n,
            "mean_nn_distance": round(nn_dist.mean().item(), 4),
            "p95_nn_distance": round(nn_dist.quantile(0.95).item(), 4),
            "coverage_0.3": round((nn_dist < 0.3).float().mean().item(), 4),
        }
        if hasattr(data, "centers"):
            # Mode-level metrics are the discriminative ones for the 8-Gaussian
            # ring: an untrained model emits noise ~radius away from every mode.
            pts = samples[:, 0]
            cd = torch.cdist(pts, data.centers.to(device))
            nearest = cd.argmin(dim=1)
            md = cd.min(dim=1).values
            out.update(
                {
                    "mode_hit_0.3": round((md < 0.3).float().mean().item(), 4),
                    "mode_mean_distance": round(md.mean().item(), 4),
                    "mode_coverage": round(
                        len(set(nearest.tolist())) / float(data.modes), 4
                    ),
                }
            )
        return out
    toks = data.decode(samples)
    ref_toks = data.sample_tokens(n)
    hist = data.token_histogram(toks)
    ref_hist = data.token_histogram(ref_toks)
    kl = torch.sum(ref_hist * (torch.log(ref_hist.clamp_min(1e-9)) - torch.log(hist.clamp_min(1e-9))))
    return {
        "kind": "markov",
        "n": n,
        "valid_bigram_rate": round(data.valid_bigram_rate(toks), 4),
        "unique_sequences": len({tuple(r) for r in toks.tolist()}),
        "token_kl": round(float(kl.item()), 4),
    }


def train(cfg, out_dir, steps=None, device=None, seed=None, resume=None):
    seed = int(seed if seed is not None else cfg.get("seed", 0))
    set_seed(seed)
    device = pick_device(device or cfg.get("device", "auto"))
    data, model, cfg = build(cfg, device, seed)
    model.to(device)
    algo = cfg.get("algo", "flm")
    sched = make_schedule(cfg.get("schedule"))
    tcfg = cfg.get("train", {}) or {}
    steps = int(steps if steps is not None else tcfg.get("steps", 1000))
    batch = int(tcfg.get("batch_size", 128))
    lr = float(tcfg.get("lr", 1e-3))
    wd = float(tcfg.get("weight_decay", 0.01))
    log_every = max(1, int(tcfg.get("log_every", 50)))
    ckpt_every = max(1, int(tcfg.get("ckpt_every", 500)))

    teacher = None
    if algo == "fmlm":
        teacher_steps = int(tcfg.get("teacher_steps", 300))
        teacher_ckpt = tcfg.get("teacher_ckpt")
        if teacher_ckpt and os.path.exists(teacher_ckpt):
            payload = load_checkpoint(teacher_ckpt, map_location=device)
            teacher = make_model(payload.get("config", cfg), data.dim_in, data.seq_len).to(device)
            teacher.load_state_dict(payload["model"])
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
            print("[teacher] loaded %s" % teacher_ckpt, flush=True)
        else:
            teacher = train_teacher(cfg, data, device, seed, teacher_steps)
            tcfg_saved = copy.deepcopy(cfg)
            tcfg_saved["algo"] = "flm"
            tcfg_saved.setdefault("model", {})["two_time"] = False
            save_checkpoint(os.path.join(out_dir, "teacher.pt"), teacher,
                            tcfg_saved, teacher_steps)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    ema = EMA(model, decay=float(tcfg.get("ema_decay", 0.999)))
    start_step = 0
    if resume and os.path.exists(resume):
        payload = load_checkpoint(resume, map_location=device)
        model.load_state_dict(payload["model"])
        start_step = int(payload.get("step", 0))
        if "ema" in payload:
            ema.shadow = {k: v.to(device) for k, v in payload["ema"].items()}
        print("[resume] %s at step %d" % (resume, start_step), flush=True)

    logger = CSVLogger(
        os.path.join(out_dir, "train_log.csv"),
        ["step", "loss", "grad_norm", "lr", "elapsed_s"],
    )
    t0 = time.time()
    running = []
    model.train()
    for step in range(start_step + 1, steps + 1):
        x1 = data.sample(batch)
        if algo == "fmlm":
            loss = flowmap_loss(model, teacher, x1, sched, tcfg)
        else:
            loss = flm_loss(model, x1, sched, tcfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)
        running.append(loss.item())
        if step % log_every == 0 or step == steps:
            avg = sum(running) / len(running)
            running = []
            logger.log(
                {
                    "step": step,
                    "loss": round(avg, 6),
                    "grad_norm": round(float(gn), 4),
                    "lr": lr,
                    "elapsed_s": round(time.time() - t0, 2),
                }
            )
            print("[%s] step %d/%d loss %.5f grad %.3f (%.1fs)"
                  % (algo, step, steps, avg, float(gn), time.time() - t0), flush=True)
        if step % ckpt_every == 0 or step == steps:
            save_checkpoint(os.path.join(out_dir, "ckpt.pt"), model, cfg, step, ema=ema)
    logger.close()

    ema_model = copy.deepcopy(model)
    ema.copy_to(ema_model)
    ema_model.eval()
    metrics_raw = evaluate(model.eval(), algo, data, cfg, device)
    metrics_ema = evaluate(ema_model, algo, data, cfg, device)
    summary = {
        "algo": algo,
        "device": device,
        "steps": steps,
        "seed": seed,
        "params": sum(p.numel() for p in model.parameters()),
        "elapsed_s": round(time.time() - t0, 2),
        "metrics_raw": metrics_raw,
        "metrics_ema": metrics_ema,
    }
    write_json(os.path.join(out_dir, "summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train FLM / FMLM reference models")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="runs/default")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--set", action="append", default=[], help="key=value override, e.g. schedule.k=8")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    for item in args.set:
        key, _, val = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                if val.lower() in ("true", "false"):
                    val = val.lower() == "true"
        node[parts[-1]] = val
    train(cfg, args.out, steps=args.steps, device=args.device, seed=args.seed,
          resume=args.resume)


if __name__ == "__main__":
    main()

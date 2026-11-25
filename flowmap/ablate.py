"""Ablation runner: a small, reproducible experiment matrix.

Config format (``configs/ablate.yaml``)::

    base: configs/flm_tiny.yaml
    train_steps: 300
    max_runs: 12
    axes:
      algo: [flm, fmlm]
      schedule.k: [2.0, 4.0, 8.0]

Each combination is trained into ``<out>/<run_id>`` and one row is appended to
``<out>/ablation.csv`` plus a Markdown table in ``<out>/ablation.md``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import os
import time

from .train import train
from .utils import load_config, write_json


def _apply(cfg, key, value):
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def _fmt(v):
    if isinstance(v, float):
        return "%.4f" % v
    return str(v)


def run_matrix(cfg, out_dir):
    base = load_config(cfg["base"])
    axes = cfg.get("axes", {}) or {}
    keys = list(axes.keys())
    values = [list(axes[k]) for k in keys]
    combos = list(itertools.product(*values))
    max_runs = int(cfg.get("max_runs", len(combos)))
    combos = combos[:max_runs]
    steps = int(cfg.get("train_steps", 300))
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    t0 = time.time()
    for i, combo in enumerate(combos, 1):
        run_cfg = copy.deepcopy(base)
        for k, v in zip(keys, combo):
            _apply(run_cfg, k, v)
        run_id = "run%02d_" % i + "_".join("%s-%s" % (k.replace(".", "-"), _fmt(v)) for k, v in zip(keys, combo))
        run_dir = os.path.join(out_dir, run_id)
        print("[ablate] %d/%d %s" % (i, len(combos), run_id), flush=True)
        summary = train(run_cfg, run_dir, steps=steps)
        row = {"run_id": run_id, "algo": run_cfg.get("algo", "flm"), "steps": steps}
        for k, v in zip(keys, combo):
            row[k] = v
        metrics = summary.get("metrics_ema", {})
        for mk, mv in metrics.items():
            if mk != "kind":
                row["metric_" + mk] = mv
        row["elapsed_s"] = summary.get("elapsed_s")
        rows.append(row)
    fields = sorted({k for r in rows for k in r})
    with open(os.path.join(out_dir, "ablation.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    metric_cols = [k for k in fields if k.startswith("metric_")]
    with open(os.path.join(out_dir, "ablation.md"), "w", encoding="utf-8") as f:
        f.write("# Ablation matrix\n\n")
        f.write("| " + " | ".join(keys + metric_cols + ["elapsed_s"]) + " |\n")
        f.write("|" + "|".join(["---"] * (len(keys) + len(metric_cols) + 1)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(_fmt(r.get(c, "")) for c in keys + metric_cols + ["elapsed_s"]) + " |\n")
    write_json(os.path.join(out_dir, "ablation.json"), {"rows": rows,
                                                        "elapsed_s": round(time.time() - t0, 2)})
    print("[ablate] wrote %d rows to %s" % (len(rows), out_dir), flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="runs/ablation")
    args = ap.parse_args(argv)
    run_matrix(load_config(args.config), args.out)


if __name__ == "__main__":
    main()

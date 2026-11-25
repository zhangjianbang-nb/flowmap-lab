"""Small utilities: seeding, config loading, logging, checkpoints, device."""

from __future__ import annotations

import csv
import json
import os
import random
import time

import torch


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str = "auto") -> str:
    if name and name != "auto":
        return name
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_config(path: str) -> dict:
    """Load YAML without requiring PyYAML (falls back to a tiny parser).

    We do depend on PyYAML in requirements.txt; the fallback exists so the
    reference implementation still runs in a bare torch environment.
    """
    with open(path, encoding="utf-8") as f:
        text = f.read()
    try:
        import yaml

        return yaml.safe_load(text) or {}
    except Exception:
        return _mini_yaml(text)


def _mini_yaml(text: str) -> dict:
    """Parse the small subset of YAML used by configs/ (nested maps + scalars)."""
    root = {}
    stack = [(-1, root)]
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        key, _, val = line.strip().partition(":")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if val.strip() == "":
            node = {}
            parent[key.strip()] = node
            stack.append((indent, node))
        else:
            v = val.strip()
            if v.lower() in ("true", "false"):
                v = v.lower() == "true"
            elif v.lower() in ("null", "none", "~"):
                v = None
            else:
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        v = v.strip("'\"")
            parent[key.strip()] = v
    return root


def merge(base: dict, override: dict) -> dict:
    """Recursive dict merge (override wins)."""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


class CSVLogger:
    def __init__(self, path: str, fields):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        new = not os.path.exists(path)
        self.f = open(path, "a", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=list(fields))
        if new:
            self.w.writeheader()
        self.f.flush()

    def log(self, row: dict):
        self.w.writerow(row)
        self.f.flush()

    def close(self):
        self.f.close()


def save_checkpoint(path, model, cfg, step, ema=None, extra=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "config": cfg,
        "step": int(step),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if ema is not None:
        payload["ema"] = ema.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


def load_checkpoint(path, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

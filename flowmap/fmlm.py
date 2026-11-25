"""Flow Map Language Model (FMLM): two-time distillation and one-step sampling.

A flow map ``f_theta(x_t, t, s)`` predicts the state at time ``s`` directly
from the state at time ``t`` (with ``s > t``, i.e. closer to data). The teacher
is a pretrained FLM whose Euler trajectory provides targets:

    x_s^teacher = Euler(v_FLM, x_t, t -> s)
    L = || f_theta(x_t, t, s) - sg(x_s^teacher) ||^2

At inference a single network evaluation maps noise at t=0 to data at t=1:

    x_1 = f_theta(x_0, 0, 1)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .flm import integrate


def flowmap_loss(model, teacher, x1, schedule, cfg=None):
    """Distillation loss for a two-time flow map.

    Half of the batch is trained on the full jump (t -> 1), which is the regime
    used at inference; the other half uses random intermediate targets to keep
    the map well-defined between arbitrary times.
    """
    cfg = cfg or {}
    b = x1.shape[0]
    device = x1.device
    teacher_steps = int(cfg.get("teacher_steps", 8))

    t = (torch.rand(b, device=device) * 0.95).clamp(1e-3, 0.95)
    eps = torch.randn_like(x1)
    xt = schedule.interpolate(x1, eps, t)

    full = torch.rand(b, device=device) < float(cfg.get("full_jump_prob", 0.5))
    s_rand = t + (1.0 - t) * torch.rand(b, device=device)
    s = torch.where(full, torch.ones_like(t), s_rand)
    s = torch.maximum(s, t + 0.02).clamp(max=1.0)

    with torch.no_grad():
        xs = integrate(teacher, xt, t, s, teacher_steps)

    pred = model(xt, t, s)
    return F.mse_loss(pred, xs)


@torch.no_grad()
def one_step_sample(model, shape, device="cpu", start=None):
    """One network evaluation from t=0 (noise) to t=1 (data)."""
    if start is None:
        start = torch.randn(shape, device=device)
    b = start.shape[0]
    t = torch.zeros(b, device=device)
    s = torch.ones(b, device=device)
    return model(start, t, s)


@torch.no_grad()
def flowmap_sample(model, x, t0, t1):
    """Apply the map to an existing batch at arbitrary times."""
    b = x.shape[0]
    if not torch.is_tensor(t0):
        t0 = torch.full((b,), float(t0), device=x.device)
    if not torch.is_tensor(t1):
        t1 = torch.full((b,), float(t1), device=x.device)
    return model(x, t0, t1)


@torch.no_grad()
def trajectory_agreement(model, teacher, x0, schedule, teacher_steps=8):
    """Compare one-step map output with the teacher's multi-step trajectory.

    Returns mean cosine similarity per sample (flattened over L*D).
    """
    x_one = one_step_sample(model, x0.shape, device=x0.device, start=x0)
    x_multi = integrate(teacher, x0, 0.0, 1.0, teacher_steps)
    a = x_one.reshape(x_one.shape[0], -1)
    b = x_multi.reshape(x_multi.shape[0], -1)
    cos = F.cosine_similarity(a, b, dim=-1)
    return cos.mean().item()

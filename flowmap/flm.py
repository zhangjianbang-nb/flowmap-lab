"""Flow matching (FLM): training objective, ODE integration and sampling."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .schedule import LogLinearSchedule


def flm_loss(model, x1, schedule: LogLinearSchedule, cfg=None):
    """Conditional flow-matching loss.

    L = E_{t~U(0,1), eps~N(0,I)} || v_theta(x_t, t) - (alpha'(t) x1 + sigma'(t) eps) ||^2

    ``x1`` has shape (B, L, D). Returns a scalar tensor.
    """
    cfg = cfg or {}
    b = x1.shape[0]
    device = x1.device
    t = torch.rand(b, device=device)
    # Avoid the exact endpoints: at t=1 sigma=0 and the target is degenerate,
    # at t=0 alpha is exponentially small but still nonzero.
    t = t.clamp(1e-3, 1.0 - 1e-3)
    eps = torch.randn_like(x1)
    xt = schedule.interpolate(x1, eps, t)
    target = schedule.velocity(x1, eps, t)
    pred = model(xt, t)
    loss = F.mse_loss(pred, target)
    if cfg.get("loss_weighting") == "sigma":
        # Optional: down-weight the noisy region where the target is dominated
        # by eps. Kept off by default to stay close to the paper's plain MSE.
        w = schedule.sigma(t).clamp_min(1e-3)
        loss = (F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2)) * w).mean()
    return loss


@torch.no_grad()
def integrate(model, x, t0, t1, steps):
    """Euler-integrate the learned ODE from t0 to t1.

    ``t0``/``t1`` are either scalars or per-sample tensors of shape (B,).
    """
    if not torch.is_tensor(t0):
        t0 = torch.full((x.shape[0],), float(t0), device=x.device)
    if not torch.is_tensor(t1):
        t1 = torch.full((x.shape[0],), float(t1), device=x.device)
    steps = max(1, int(steps))
    dt = (t1 - t0) / steps
    view = (-1,) + (1,) * (x.dim() - 1)
    for k in range(steps):
        tk = t0 + dt * k
        v = model(x, tk)
        x = x + v * dt.view(view)
    return x


@torch.no_grad()
def flm_sample(model, shape, steps=8, device="cpu", schedule=None, start=None):
    """Generate samples by integrating from pure noise (t=0) to data (t=1)."""
    if start is None:
        start = torch.randn(shape, device=device)
    return integrate(model, start, 0.0, 1.0, steps)


@torch.no_grad()
def flm_sample_from(model, x, t, steps=8):
    """Continue an ODE trajectory from an arbitrary (x, t)."""
    return integrate(model, x, t, 1.0, steps)


def sample_dispatch(model, algo, shape, steps, device, schedule=None, start=None):
    if algo == "flm":
        return flm_sample(model, shape, steps=steps, device=device,
                          schedule=schedule, start=start)
    if algo == "fmlm":
        from .fmlm import one_step_sample

        return one_step_sample(model, shape, device=device, start=start)
    raise ValueError("unknown algo: %r" % algo)

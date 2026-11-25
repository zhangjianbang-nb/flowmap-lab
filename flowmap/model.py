"""Denoisers / velocity models.

Two architectures, selected automatically from the data shape:

* ``ToyMLP``  – for low-dimensional continuous data (dim_in small, seq_len 1).
* ``TinyTransformer`` – for token sequences represented as one-hot vectors.

Both accept a continuous state ``x`` of shape ``(B, L, D)`` and a time tensor
``t`` of shape ``(B,)``. Flow-map models additionally accept a target time
``s`` (also ``(B,)``), so the network learns a two-time map f(x_t, t, s).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(t: torch.Tensor, dim: int = 64) -> torch.Tensor:
    """Standard transformer time embedding; t is assumed to lie in [0, 1]."""
    half = dim // 2
    device = t.device
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimeMLP(nn.Module):
    def __init__(self, d_model: int, dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, t):
        return self.net(sinusoidal_embedding(t, self.net[0].in_features))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, h, attn_mask=None):
        z = self.ln1(h)
        a, _ = self.attn(z, z, z, attn_mask=attn_mask, need_weights=False)
        h = h + a
        h = h + self.mlp(self.ln2(h))
        return h


class TinyTransformer(nn.Module):
    """A small pre-norm transformer over a sequence of continuous tokens."""

    def __init__(
        self,
        dim_in: int,
        seq_len: int,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        two_time: bool = False,
        dropout: float = 0.0,
        causal: bool = False,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.seq_len = seq_len
        self.two_time = two_time
        self.causal = bool(causal)
        self.inp = nn.Linear(dim_in, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.t_emb = TimeMLP(d_model)
        self.s_emb = TimeMLP(d_model) if two_time else None
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, dim_in)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t, s=None):
        h = self.inp(x) + self.pos[:, : x.shape[1]]
        h = h + self.t_emb(t)[:, None, :]
        if self.two_time:
            if s is None:
                raise ValueError("two_time model requires s")
            h = h + self.s_emb(s)[:, None, :]
        mask = None
        if self.causal and x.shape[1] > 1:
            # True means "not allowed": position i may only attend to j <= i.
            mask = torch.triu(
                torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device),
                diagonal=1,
            )
        for blk in self.blocks:
            h = blk(h, mask)
        return self.out(self.ln_f(h))


class ToyMLP(nn.Module):
    """MLP velocity field for 2-D toy distributions."""

    def __init__(self, dim_in: int = 2, hidden: int = 128, n_layers: int = 3,
                 two_time: bool = False):
        super().__init__()
        self.dim_in = dim_in
        self.two_time = two_time
        self.t_emb = TimeMLP(hidden)
        self.s_emb = TimeMLP(hidden) if two_time else None
        layers = [nn.Linear(dim_in + hidden, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, dim_in)]
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, t, s=None):
        # x: (B, L, D); toy data always has L == 1
        if x.dim() == 3:
            x = x[:, 0]
        h = self.t_emb(t)
        if self.two_time:
            if s is None:
                raise ValueError("two_time model requires s")
            h = h + self.s_emb(s)
        y = self.net(torch.cat([x, h], dim=-1))
        return y[:, None, :]


def make_model(cfg, dim_in: int, seq_len: int) -> nn.Module:
    """Build a model from the ``model`` section of a config."""
    m = cfg.get("model", {}) or {}
    arch = m.get("arch", "auto")
    two_time = bool(m.get("two_time", False))
    if arch == "auto":
        arch = "mlp" if seq_len == 1 and dim_in <= 8 else "transformer"
    if arch == "mlp":
        return ToyMLP(
            dim_in=dim_in,
            hidden=int(m.get("hidden", 128)),
            n_layers=int(m.get("n_layers", 3)),
            two_time=two_time,
        )
    if arch == "transformer":
        return TinyTransformer(
            dim_in=dim_in,
            seq_len=seq_len,
            d_model=int(m.get("d_model", 64)),
            n_layers=int(m.get("n_layers", 2)),
            n_heads=int(m.get("n_heads", 4)),
            two_time=two_time,
            dropout=float(m.get("dropout", 0.0)),
            causal=bool(m.get("causal", False)),
        )
    raise ValueError("unknown model.arch: %r" % arch)


class EMA:
    """Exponential moving average with a warmup so short runs are not stuck at init."""

    def __init__(self, model: nn.Module, decay: float = 0.999, warmup: bool = True):
        self.decay = float(decay)
        self.warmup = bool(warmup)
        self.step = 0
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if torch.is_tensor(v)
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.step += 1
        d = self.decay
        if self.warmup:
            d = min(d, (1.0 + self.step) / (10.0 + self.step))
        for k, v in model.state_dict().items():
            if k not in self.shadow:
                continue
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=False)

    def state_dict(self):
        return self.shadow

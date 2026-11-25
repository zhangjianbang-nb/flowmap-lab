"""Datasets used by the reference experiments.

Two kinds, both dependency-free beyond torch:

``toy``
    A 4x4 checkerboard in [-2, 2]^2. Only the 8 "even" cells carry mass.
    This is the fastest way to verify that the flow-matching machinery is
    correct: a well-trained model transports N(0, I) onto the checkerboard.

``markov``
    A synthetic token language. Each token has exactly two legal successors,
    so a model can be scored with a hard, interpretable metric (valid-bigram
    rate) instead of a surrogate perplexity. Tokens are represented as one-hot
    vectors, matching the FLM paper's continuous-token formulation.
"""

from __future__ import annotations

import math

import torch


class ToyGaussians:
    """Eight Gaussian modes on a ring: the standard 2-D flow-matching benchmark.

    Unlike a checkerboard, the metric is unambiguous: a good sampler lands
    within ``hit_radius`` of a mode centre, while noise is ~``radius`` away.
    """

    kind = "toy"

    def __init__(self, device="cpu", seed=0, modes=8, radius=2.0, std=0.12):
        self.device = device
        self.modes = int(modes)
        self.radius = float(radius)
        self.std = float(std)
        self.dim_in = 2
        self.seq_len = 1
        self.vocab = None
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        angles = torch.arange(self.modes).float() * (2 * math.pi / self.modes)
        self.centers = torch.stack(
            [self.radius * torch.cos(angles), self.radius * torch.sin(angles)], dim=-1
        )

    def sample(self, n):
        idx = torch.randint(0, self.modes, (n,), generator=self.gen)
        noise = torch.randn(n, 2, generator=self.gen) * self.std
        x = self.centers[idx] + noise
        return x.to(self.device)[:, None, :]

    def reference(self, n=4096):
        return self.sample(n).cpu()

    def decode(self, x):
        return x.squeeze(1).cpu()


class ToyCheckerboard:
    kind = "toy"

    def __init__(self, device="cpu", seed=0, grid=4, bound=2.0):
        self.device = device
        self.grid = grid
        self.bound = bound
        self.dim_in = 2
        self.seq_len = 1
        self.vocab = None
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

    def sample(self, n):
        g = self.grid
        even = [c for c in range(g * g) if ((c // g) + (c % g)) % 2 == 0]
        idx = torch.randint(0, len(even), (n,), generator=self.gen)
        cell = torch.tensor(even, dtype=torch.long)[idx]
        ix = (cell % g).float()
        iy = (cell // g).float()
        step = 2.0 * self.bound / g
        x = (ix + torch.rand(n, generator=self.gen)) * step - self.bound
        y = (iy + torch.rand(n, generator=self.gen)) * step - self.bound
        return torch.stack([x, y], dim=-1).to(self.device)[:, None, :]

    def reference(self, n=4096):
        return self.sample(n).cpu()

    def decode(self, x):
        return x.squeeze(1).cpu()


class MarkovCorpus:
    kind = "markov"

    def __init__(self, vocab=16, seq_len=16, device="cpu", seed=0):
        self.vocab = int(vocab)
        self.seq_len = int(seq_len)
        self.device = device
        self.dim_in = self.vocab
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        # deterministic successor function: token a -> {a+1, 3a+1} mod V
        self.allowed = set()
        for a in range(self.vocab):
            for b in ((a + 1) % self.vocab, (3 * a + 1) % self.vocab):
                self.allowed.add((a, b))

    def sample_tokens(self, n):
        toks = torch.empty((n, self.seq_len), dtype=torch.long)
        cur = torch.randint(0, self.vocab, (n,), generator=self.gen)
        toks[:, 0] = cur
        for i in range(1, self.seq_len):
            nxt = (cur + 1) % self.vocab
            alt = (3 * cur + 1) % self.vocab
            pick = torch.rand(n, generator=self.gen) < 0.5
            cur = torch.where(pick, nxt, alt)
            toks[:, i] = cur
        return toks.to(self.device)

    def sample(self, n):
        toks = self.sample_tokens(n)
        return self.encode(toks)

    def encode(self, toks):
        return torch.nn.functional.one_hot(toks, self.vocab).float()

    def decode(self, x):
        """x: (B, L, V) continuous -> (B, L) token ids."""
        return x.argmax(dim=-1)

    def valid_bigram_rate(self, toks):
        a = toks[:, :-1].reshape(-1).tolist()
        b = toks[:, 1:].reshape(-1).tolist()
        if not a:
            return 1.0
        ok = sum(1 for x, y in zip(a, b) if (x, y) in self.allowed)
        return ok / len(a)

    def token_histogram(self, toks, vocab=None):
        v = vocab or self.vocab
        h = torch.bincount(toks.reshape(-1), minlength=v).float()
        return h / h.sum().clamp_min(1.0)


def make_data(cfg, device="cpu", seed=0):
    d = cfg.get("data", {}) or {}
    kind = d.get("kind", "toy")
    if kind == "toy":
        return ToyGaussians(
            device=device,
            seed=seed,
            modes=int(d.get("modes", 8)),
            radius=float(d.get("radius", 2.0)),
            std=float(d.get("std", 0.12)),
        )
    if kind == "toy_checkerboard":
        return ToyCheckerboard(device=device, seed=seed)
    if kind == "markov":
        return MarkovCorpus(
            vocab=int(d.get("vocab", 16)),
            seq_len=int(d.get("seq_len", 16)),
            device=device,
            seed=seed,
        )
    raise ValueError("unknown data.kind: %r" % kind)

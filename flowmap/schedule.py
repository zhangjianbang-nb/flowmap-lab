"""Log-linear interpolation schedule for flow matching on continuous tokens.

Convention used everywhere in this repository:

    t = 0  ->  pure noise
    t = 1  ->  clean data
    x_t = alpha(t) * x_1 + sigma(t) * eps,   eps ~ N(0, I)
    alpha(t) = exp(-k * (1 - t))
    sigma(t) = 1 - exp(-k * (1 - t))
    v_t = dx_t/dt = alpha'(t) * x_1 + sigma'(t) * eps

The schedule is called "log-linear" because log(alpha/sigma) is linear in t:

    gamma(t) = log(alpha(t) / sigma(t)) = -k * (1 - t)

This makes the inverse analytic: with a = sigmoid(gamma), t = 1 + log(a) / k.
`ScheduleLUT` reproduces the cubic-spline lookup tables used by the reference
FLM implementation, but the analytic form is the source of truth here.
"""

from __future__ import annotations

import math

import torch


def _as_tensor(t, device=None):
    if torch.is_tensor(t):
        return t if device is None else t.to(device)
    return torch.tensor(float(t), device=device)


class LinearSchedule:
    """Rectified-flow path: alpha(t)=t, sigma(t)=1-t (straight trajectories).

    Useful as a control in ablations: it is the standard rectified-flow
    baseline, while the log-linear path is the FLM paper's language schedule.
    """

    def alpha(self, t):
        return _as_tensor(t)

    def sigma(self, t):
        return 1.0 - _as_tensor(t)

    def dalpha(self, t):
        return torch.ones_like(_as_tensor(t))

    def dsigma(self, t):
        return -torch.ones_like(_as_tensor(t))

    def gamma(self, t):
        t = _as_tensor(t).clamp(1e-6, 1 - 1e-6)
        return torch.log(t) - torch.log1p(-t)

    def t_from_gamma(self, gamma):
        return torch.sigmoid(_as_tensor(gamma))

    def interpolate(self, x1, eps, t):
        t = _as_tensor(t)
        while t.dim() < x1.dim():
            t = t.unsqueeze(-1)
        return self.alpha(t) * x1 + self.sigma(t) * eps

    def velocity(self, x1, eps, t):
        t = _as_tensor(t)
        while t.dim() < x1.dim():
            t = t.unsqueeze(-1)
        return self.dalpha(t) * x1 + self.dsigma(t) * eps

    def __repr__(self):
        return "LinearSchedule()"


class LogLinearSchedule:
    """Analytic log-linear path. Stateless and differentiable in t."""

    def __init__(self, k: float = 4.0):
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = float(k)

    def alpha(self, t):
        t = _as_tensor(t)
        return torch.exp(-self.k * (1.0 - t))

    def sigma(self, t):
        t = _as_tensor(t)
        return 1.0 - torch.exp(-self.k * (1.0 - t))

    def dalpha(self, t):
        t = _as_tensor(t)
        return self.k * torch.exp(-self.k * (1.0 - t))

    def dsigma(self, t):
        t = _as_tensor(t)
        return -self.k * torch.exp(-self.k * (1.0 - t))

    def gamma(self, t):
        """log(alpha / sigma); -inf at t=0 is avoided by clamping sigma."""
        t = _as_tensor(t)
        a = self.alpha(t).clamp_min(1e-12)
        s = self.sigma(t).clamp_min(1e-12)
        return torch.log(a) - torch.log(s)

    def t_from_gamma(self, gamma):
        """Exact inverse of gamma(t): t = 1 + log(sigmoid(gamma)) / k."""
        gamma = _as_tensor(gamma)
        a = torch.sigmoid(gamma)
        return 1.0 + torch.log(a) / self.k

    def interpolate(self, x1, eps, t):
        """x_t = alpha(t) x1 + sigma(t) eps. t broadcasts over leading dims."""
        t = _as_tensor(t)
        while t.dim() < x1.dim():
            t = t.unsqueeze(-1)
        return self.alpha(t) * x1 + self.sigma(t) * eps

    def velocity(self, x1, eps, t):
        """Target velocity v = alpha'(t) x1 + sigma'(t) eps."""
        t = _as_tensor(t)
        while t.dim() < x1.dim():
            t = t.unsqueeze(-1)
        return self.dalpha(t) * x1 + self.dsigma(t) * eps

    def __repr__(self):
        return "LogLinearSchedule(k=%g)" % self.k


class ScheduleLUT:
    """Monotone cubic lookup table in gamma-space (paper-style reparameterisation).

    The reference implementation builds cubic splines for alpha/gamma so that
    sampling can be parameterised uniformly in gamma. We keep a dependency-free
    monotone cubic Hermite interpolant (Fritsch-Carlson) and expose the same
    four functions: alpha_to_gamma, gamma_to_alpha, t_to_gamma, gamma_to_t.
    """

    def __init__(self, k: float = 4.0, n_points: int = 4096):
        self.schedule = LogLinearSchedule(k)
        self.k = float(k)
        self.n_points = int(n_points)
        g = torch.linspace(-12.0, 12.0, self.n_points, dtype=torch.float64)
        t = self.schedule.t_from_gamma(g)
        a = self.schedule.alpha(t)
        # gamma -> t and gamma -> alpha are both smooth; interpolate t(g).
        self._g = g
        self._t = t.clamp(0.0, 1.0)
        self._a = a
        self._slopes = self._pchip_slopes(self._g, self._t)

    @staticmethod
    def _pchip_slopes(x, y):
        h = x[1:] - x[:-1]
        delta = (y[1:] - y[:-1]) / h
        m = torch.zeros_like(y)
        m[1:-1] = (delta[:-1] + delta[1:]) / 2.0
        m[0] = delta[0]
        m[-1] = delta[-1]
        # Fritsch-Carlson monotonicity filter
        for i in range(len(delta)):
            if delta[i] == 0:
                m[i] = 0.0
                m[i + 1] = 0.0
            else:
                a = m[i] / delta[i]
                b = m[i + 1] / delta[i]
                s = a * a + b * b
                if s > 9.0:
                    tau = 3.0 / math.sqrt(s)
                    m[i] = tau * a * delta[i]
                    m[i + 1] = tau * b * delta[i]
        return m

    def _eval(self, gq):
        gq = gq.clamp(self._g[0], self._g[-1])
        idx = torch.searchsorted(self._g, gq) - 1
        idx = idx.clamp(0, len(self._g) - 2)
        x0, x1 = self._g[idx], self._g[idx + 1]
        y0, y1 = self._t[idx], self._t[idx + 1]
        m0, m1 = self._slopes[idx], self._slopes[idx + 1]
        h = x1 - x0
        u = (gq - x0) / h
        h00 = 2 * u**3 - 3 * u**2 + 1
        h10 = u**3 - 2 * u**2 + u
        h01 = -2 * u**3 + 3 * u**2
        h11 = u**3 - u**2
        return h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1

    def gamma_to_t(self, gamma):
        gamma = _as_tensor(gamma)
        return self._eval(gamma.to(torch.float64)).to(gamma.dtype)

    def t_to_gamma(self, t):
        t = _as_tensor(t)
        g = torch.log(self.schedule.alpha(t)) - torch.log(
            self.schedule.sigma(t).clamp_min(1e-12)
        )
        return g

    def alpha_to_gamma(self, alpha):
        alpha = _as_tensor(alpha).clamp(1e-9, 1 - 1e-9)
        # alpha = exp(-k(1-t)) => gamma = log(alpha/(1-alpha))
        return torch.log(alpha) - torch.log1p(-alpha)

    def gamma_to_alpha(self, gamma):
        gamma = _as_tensor(gamma)
        return torch.sigmoid(gamma)


def make_schedule(cfg):
    """Build a schedule from a config dict: {"name": "loglinear", "k": 4.0}."""
    cfg = cfg or {}
    name = str(cfg.get("name", "loglinear")).lower()
    if name in ("linear", "rectified", "rectified_flow"):
        return LinearSchedule()
    return LogLinearSchedule(float(cfg.get("k", 4.0)))

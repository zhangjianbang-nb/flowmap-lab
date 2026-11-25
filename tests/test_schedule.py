import math
import unittest

import torch

from flowmap.schedule import LogLinearSchedule, ScheduleLUT


class TestLogLinearSchedule(unittest.TestCase):
    def setUp(self):
        self.k = 4.0
        self.s = LogLinearSchedule(self.k)

    def test_endpoints(self):
        self.assertAlmostEqual(self.s.alpha(torch.tensor(1.0)).item(), 1.0, places=6)
        self.assertAlmostEqual(self.s.sigma(torch.tensor(1.0)).item(), 0.0, places=6)
        self.assertAlmostEqual(
            self.s.alpha(torch.tensor(0.0)).item(), math.exp(-self.k), places=6
        )
        self.assertAlmostEqual(
            self.s.sigma(torch.tensor(0.0)).item(), 1 - math.exp(-self.k), places=6
        )

    def test_alpha_plus_sigma_is_one(self):
        t = torch.linspace(0, 1, 33)
        total = self.s.alpha(t) + self.s.sigma(t)
        self.assertTrue(torch.allclose(total, torch.ones_like(total), atol=1e-6))

    def test_gamma_inverse_roundtrip(self):
        t = torch.linspace(0.05, 0.95, 19)
        g = self.s.gamma(t)
        back = self.s.t_from_gamma(g)
        self.assertTrue(torch.allclose(t, back, atol=1e-5))

    def test_velocity_matches_finite_difference(self):
        torch.manual_seed(0)
        x1 = torch.randn(4, 1, 2)
        eps = torch.randn(4, 1, 2)
        t = torch.tensor([0.2, 0.4, 0.6, 0.8])
        h = 1e-4
        fd = (self.s.interpolate(x1, eps, t + h) - self.s.interpolate(x1, eps, t)) / h
        v = self.s.velocity(x1, eps, t)
        self.assertTrue(torch.allclose(fd, v, atol=2e-2))


class TestScheduleLUT(unittest.TestCase):
    def test_gamma_to_t_matches_analytic(self):
        lut = ScheduleLUT(k=4.0, n_points=2048)
        t = torch.linspace(0.1, 0.9, 17)
        g = lut.t_to_gamma(t)
        back = lut.gamma_to_t(g)
        self.assertTrue(torch.allclose(t, back, atol=2e-3))

    def test_alpha_gamma_roundtrip(self):
        lut = ScheduleLUT(k=4.0, n_points=1024)
        a = torch.linspace(0.05, 0.95, 11)
        g = lut.alpha_to_gamma(a)
        self.assertTrue(torch.allclose(a, lut.gamma_to_alpha(g), atol=1e-6))

    def test_monotone(self):
        lut = ScheduleLUT(k=6.0, n_points=512)
        g = torch.linspace(-8, 8, 101)
        t = lut.gamma_to_t(g)
        self.assertTrue(torch.all(t[1:] >= t[:-1] - 1e-6))


if __name__ == "__main__":
    unittest.main()

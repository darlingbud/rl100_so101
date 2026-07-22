import math
import unittest

import torch
from torch import nn

from rl_100.common.gradient_stats import loss_gradient_stats, parameter_gradient_norm


class GradientStatsTest(unittest.TestCase):
    def test_loss_stats_do_not_populate_parameter_grad(self):
        module = nn.Linear(2, 1, bias=False)
        parameter = next(module.parameters())
        with torch.no_grad():
            parameter.copy_(torch.tensor([[1.0, 2.0]]))

        bc_loss = parameter.square().sum()
        kl_loss = parameter.sum()
        stats = loss_gradient_stats({"bc": bc_loss, "kl": kl_loss}, module)

        self.assertAlmostEqual(stats["grad_encoder_bc_norm"], math.sqrt(20.0), places=6)
        self.assertAlmostEqual(stats["grad_encoder_kl_norm"], math.sqrt(2.0), places=6)
        self.assertAlmostEqual(
            stats["grad_encoder_bc_kl_cosine"],
            6.0 / math.sqrt(40.0),
            places=6,
        )
        self.assertIsNone(parameter.grad)

    def test_parameter_gradient_norm_uses_accumulated_grad(self):
        module = nn.Linear(2, 1, bias=False)
        parameter = next(module.parameters())
        with torch.no_grad():
            parameter.copy_(torch.tensor([[1.0, 2.0]]))

        parameter.square().sum().backward()

        self.assertAlmostEqual(parameter_gradient_norm(module), math.sqrt(20.0), places=6)


if __name__ == "__main__":
    unittest.main()

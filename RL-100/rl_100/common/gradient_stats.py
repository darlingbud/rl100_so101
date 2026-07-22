"""Gradient diagnostics for multi-objective training."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


def _trainable_parameters(
    module: nn.Module,
    exclude_prefixes: tuple[str, ...] = (),
) -> tuple[nn.Parameter, ...]:
    return tuple(
        parameter
        for name, parameter in module.named_parameters()
        if parameter.requires_grad and not name.startswith(exclude_prefixes)
    )


def _gradient_norm(gradients: tuple[torch.Tensor | None, ...]) -> torch.Tensor:
    terms = [gradient.detach().float().square().sum() for gradient in gradients if gradient is not None]
    if not terms:
        return torch.tensor(0.0)
    return torch.stack(terms).sum().sqrt()


def _gradient_cosine(
    left: tuple[torch.Tensor | None, ...],
    right: tuple[torch.Tensor | None, ...],
) -> torch.Tensor:
    products = []
    left_squares = []
    right_squares = []
    for left_gradient, right_gradient in zip(left, right):
        if left_gradient is not None:
            left_gradient = left_gradient.detach().float()
            left_squares.append(left_gradient.square().sum())
        if right_gradient is not None:
            right_gradient = right_gradient.detach().float()
            right_squares.append(right_gradient.square().sum())
        if left_gradient is not None and right_gradient is not None:
            products.append((left_gradient * right_gradient).sum())
    if not left_squares or not right_squares:
        return torch.tensor(0.0)
    denominator = torch.stack(left_squares).sum().sqrt() * torch.stack(right_squares).sum().sqrt()
    if denominator.item() == 0.0:
        return denominator.new_tensor(0.0)
    numerator = torch.stack(products).sum() if products else denominator.new_tensor(0.0)
    return numerator / denominator


def loss_gradient_stats(
    losses: Mapping[str, torch.Tensor],
    module: nn.Module,
    exclude_parameter_prefixes: tuple[str, ...] = (),
) -> dict[str, float]:
    """Measure each loss gradient on ``module`` without modifying ``.grad``."""
    parameters = _trainable_parameters(module, exclude_parameter_prefixes)
    gradients: dict[str, tuple[torch.Tensor | None, ...]] = {}
    for name, loss in losses.items():
        if not torch.is_tensor(loss) or not loss.requires_grad:
            continue
        gradients[name] = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )

    stats = {
        f"grad_encoder_{name}_norm": float(_gradient_norm(values).cpu())
        for name, values in gradients.items()
    }
    for left_name, right_name in (("bc", "kl"), ("bc", "recon")):
        if left_name in gradients and right_name in gradients:
            stats[f"grad_encoder_{left_name}_{right_name}_cosine"] = float(
                _gradient_cosine(gradients[left_name], gradients[right_name]).cpu()
            )
    return stats


def parameter_gradient_norm(
    module: nn.Module,
    exclude_parameter_prefixes: tuple[str, ...] = (),
) -> float:
    """Return the L2 norm of gradients currently accumulated on ``module``."""
    gradients = tuple(
        parameter.grad
        for parameter in _trainable_parameters(module, exclude_parameter_prefixes)
        if parameter.grad is not None
    )
    return float(_gradient_norm(gradients).cpu())

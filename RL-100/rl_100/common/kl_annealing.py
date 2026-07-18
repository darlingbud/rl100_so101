"""Shared helpers for epoch-based KL-weight annealing."""

from __future__ import annotations


def kl_annealing_progress(
    epoch: int,
    num_epochs: int,
    kl_annealing_epoch: int | None = None,
) -> float:
    """Return linear KL annealing progress, clamped to ``[0, 1]``.

    ``None`` preserves the legacy behavior by annealing across the complete
    training run. A configured value specifies how many epochs the ramp lasts;
    the target KL weight is retained after the ramp finishes.
    """
    epoch = int(epoch)
    annealing_epochs = num_epochs if kl_annealing_epoch is None else kl_annealing_epoch
    annealing_epochs = int(annealing_epochs)
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    if annealing_epochs < 1:
        raise ValueError("kl_annealing_epoch must be a positive integer or null")
    if annealing_epochs == 1:
        return 1.0
    return min(epoch / (annealing_epochs - 1), 1.0)

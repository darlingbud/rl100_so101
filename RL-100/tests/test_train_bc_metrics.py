import pytest
import torch

from train_bc import _action_error_metrics


class FakePolicy:
    n_obs_steps = 2
    no_pre_action = True

    def predict_action(self, obs, **_kwargs):
        return {"action": obs["prediction"]}


def test_action_error_metrics_uses_executed_action_slice():
    prediction = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    batch = {
        "obs": {"prediction": prediction},
        "action": torch.tensor(
            [[[9.0, 9.0], [2.0, 0.0], [1.0, 8.0]]]
        ),
    }

    metrics = _action_error_metrics(
        FakePolicy(), [batch], torch.device("cpu")
    )

    assert metrics["action_mae"] == pytest.approx(2.25)
    assert metrics["action_rmse"] == pytest.approx(2.5)


def test_action_error_metrics_honors_max_steps():
    batch = {
        "obs": {"prediction": torch.zeros(1, 1, 1)},
        "action": torch.tensor([[[0.0], [1.0]]]),
    }
    second_batch = {
        "obs": {"prediction": torch.zeros(1, 1, 1)},
        "action": torch.tensor([[[0.0], [9.0]]]),
    }

    metrics = _action_error_metrics(
        FakePolicy(), [batch, second_batch], torch.device("cpu"), max_steps=1
    )

    assert metrics["action_mae"] == pytest.approx(1.0)

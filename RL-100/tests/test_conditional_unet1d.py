import pytest
import torch

from rl_100.model.diffusion.conditional_unet1d import ConditionalUnet1D


def make_unet(project_dim=None):
    return ConditionalUnet1D(
        input_dim=6,
        global_cond_dim=10,
        global_cond_project_dim=project_dim,
        diffusion_step_embed_dim=8,
        down_dims=[8, 16],
        kernel_size=3,
        n_groups=4,
        condition_type="film",
    )


def test_shared_global_condition_projection_changes_film_input_dim():
    model = make_unet(project_dim=4)

    assert model.global_cond_projector[0].in_features == 10
    assert model.global_cond_projector[0].out_features == 4
    first_film = model.down_modules[0][0].cond_encoder[1]
    assert first_film.in_features == 8 + 4

    output = model(
        sample=torch.randn(2, 4, 6),
        timestep=torch.tensor([1, 2]),
        global_cond=torch.randn(2, 10),
    )
    assert output.shape == (2, 4, 6)


def test_default_keeps_unprojected_condition_path():
    model = make_unet()

    assert model.global_cond_projector is None
    first_film = model.down_modules[0][0].cond_encoder[1]
    assert first_film.in_features == 8 + 10


def test_projection_requires_global_condition():
    with pytest.raises(ValueError, match="global_cond_dim is required"):
        ConditionalUnet1D(
            input_dim=6,
            global_cond_dim=None,
            global_cond_project_dim=4,
            diffusion_step_embed_dim=8,
            down_dims=[8, 16],
            n_groups=4,
        )

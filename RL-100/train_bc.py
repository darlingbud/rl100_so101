"""Train an RL100 2D/3D policy with imitation learning only."""

from __future__ import annotations

import copy
import pathlib
import random

import dill
import hydra
import numpy as np
import torch
import tqdm
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from rl_100.common.kl_annealing import kl_annealing_progress
from rl_100.common.pytorch_util import dict_apply, optimizer_to
from rl_100.dataset.base_dataset import BaseDataset
from rl_100.model.common.lr_scheduler import get_scheduler


OmegaConf.register_new_resolver("eval", eval, replace=True)


def _to_device(batch, device):
    return dict_apply(batch, lambda value: value.to(device, non_blocking=True))


def _checkpoint_payload(cfg, model, ema_model, optimizer, lr_scheduler, epoch, global_step):
    state_dicts = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
    }
    if ema_model is not None:
        state_dicts["ema_model"] = ema_model.state_dict()
    return {
        "cfg": cfg,
        "state_dicts": state_dicts,
        "pickles": {
            "epoch": dill.dumps(epoch),
            "global_step": dill.dumps(global_step),
        },
    }


def _save_checkpoint(path, cfg, model, ema_model, optimizer, lr_scheduler, epoch, global_step):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _checkpoint_payload(
        cfg, model, ema_model, optimizer, lr_scheduler, epoch, global_step
    )
    torch.save(payload, path.open("wb"), pickle_module=dill)
    print(f"Checkpoint saved to {path}")


def _load_checkpoint(
    path,
    model,
    ema_model,
    optimizer,
    lr_scheduler,
    load_lr_scheduler=True,
):
    with pathlib.Path(path).open("rb") as checkpoint_file:
        payload = torch.load(checkpoint_file, pickle_module=dill, map_location="cpu")
    states = payload["state_dicts"]
    model.load_state_dict(states["model"])
    if ema_model is not None and "ema_model" in states:
        ema_model.load_state_dict(states["ema_model"])
    if "optimizer" in states:
        optimizer.load_state_dict(states["optimizer"])
    if load_lr_scheduler and "lr_scheduler" in states:
        lr_scheduler.load_state_dict(states["lr_scheduler"])
    epoch = dill.loads(payload.get("pickles", {}).get("epoch", dill.dumps(0)))
    global_step = dill.loads(
        payload.get("pickles", {}).get("global_step", dill.dumps(0))
    )
    return int(epoch), int(global_step)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("rl_100", "config")),
)
def main(cfg: OmegaConf) -> None:
    seed = int(cfg.training.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(cfg.training.device)
    output_dir = pathlib.Path(HydraConfig.get().runtime.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml", resolve=True)

    dataset = hydra.utils.instantiate(cfg.task.dataset)
    if not isinstance(dataset, BaseDataset):
        raise TypeError(f"Expected BaseDataset, received {type(dataset).__name__}")
    val_dataset = dataset.get_validation_dataset()
    train_loader = DataLoader(dataset, **OmegaConf.to_container(cfg.dataloader))
    val_loader = DataLoader(val_dataset, **OmegaConf.to_container(cfg.val_dataloader))
    if len(train_loader) == 0:
        raise RuntimeError("Training dataset produced no batches")

    model = hydra.utils.instantiate(cfg.policy)
    normalizer = dataset.get_normalizer()
    model.set_normalizer(normalizer)
    model.to(device)

    ema_model = None
    ema = None
    if cfg.training.use_ema:
        ema_model = copy.deepcopy(model).to(device)
        ema_model.eval()
        ema = hydra.utils.instantiate(cfg.ema, model=ema_model)

    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    accumulate = int(cfg.training.gradient_accumulate_every)
    updates_per_epoch = max(1, (len(train_loader) + accumulate - 1) // accumulate)
    lr_scheduler = get_scheduler(
        cfg.training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=int(cfg.training.lr_warmup_steps),
        num_training_steps=updates_per_epoch * int(cfg.training.num_epochs),
    )

    start_epoch = 0
    global_step = 0
    latest_path = checkpoint_dir / "latest.ckpt"
    if cfg.training.resume and latest_path.is_file():
        reset_lr_scheduler = bool(
            cfg.training.get("reset_lr_scheduler_on_resume", False)
        )
        start_epoch, global_step = _load_checkpoint(
            latest_path,
            model,
            ema_model,
            optimizer,
            lr_scheduler,
            load_lr_scheduler=not reset_lr_scheduler,
        )
        # LinearNormalizer rebuilds its ParameterDict while loading a state
        # dict, so those dynamically-created tensors start on CPU even when
        # the policy was already moved to CUDA before loading.
        model.to(device)
        if ema_model is not None:
            ema_model.to(device)
        optimizer_to(optimizer, device)
        if reset_lr_scheduler:
            resume_lr = float(
                cfg.training.get("resume_lr", None) or cfg.optimizer.lr
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = resume_lr
                param_group["initial_lr"] = resume_lr
            remaining_updates = max(
                1,
                updates_per_epoch
                * max(int(cfg.training.num_epochs) - start_epoch, 1),
            )
            lr_scheduler = get_scheduler(
                cfg.training.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=int(
                    cfg.training.get("resume_lr_warmup_steps", 0)
                ),
                num_training_steps=remaining_updates,
            )
            print(
                f"Reset LR scheduler: lr={resume_lr:g}, "
                f"remaining_updates={remaining_updates}"
            )
        print(f"Resuming from epoch {start_epoch}, step {global_step}")

    wandb_run = None
    if cfg.use_wandb:
        import wandb

        wandb_run = wandb.init(
            dir=str(output_dir),
            project=str(cfg.logging.project),
            group=str(cfg.logging.group),
            name=str(cfg.logging.name),
            mode=str(cfg.logging.mode),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    best_val_action_rmse = float("inf")
    target_beta_kl = None
    if hasattr(model.obs_encoder, "beta_kl"):
        target_beta_kl = float(model.obs_encoder.beta_kl)

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, int(cfg.training.num_epochs)):
        if cfg.kl_annealing and target_beta_kl is not None:
            progress = kl_annealing_progress(
                epoch,
                cfg.training.num_epochs,
                cfg.get("kl_annealing_epoch"),
            )
            current_beta_kl = target_beta_kl * progress
            model.obs_encoder.beta_kl = current_beta_kl
            if ema_model is not None:
                ema_model.obs_encoder.beta_kl = current_beta_kl

        model.train()
        train_losses = []
        component_losses = {"bc_loss": [], "kl_loss": [], "recon_loss": []}
        progress = tqdm.tqdm(train_loader, desc=f"BC epoch {epoch}", leave=False)
        for batch_idx, batch in enumerate(progress):
            batch = _to_device(batch, device)
            raw_loss, loss_items = model.compute_loss(batch)
            (raw_loss / accumulate).backward()

            should_step = (batch_idx + 1) % accumulate == 0 or (
                batch_idx + 1 == len(train_loader)
            )
            if should_step:
                max_grad_norm = cfg.training.get("max_grad_norm", None)
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                if ema is not None:
                    ema.step(model)
                global_step += 1

            loss_value = float(raw_loss.detach().cpu())
            train_losses.append(loss_value)
            for key in component_losses:
                component_losses[key].append(float(loss_items.get(key, 0.0)))
            progress.set_postfix(loss=f"{loss_value:.5f}")

            max_steps = cfg.training.get("max_train_steps", None)
            if max_steps is not None and batch_idx + 1 >= int(max_steps):
                break

        metrics = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": float(np.mean(train_losses)),
            "lr": float(lr_scheduler.get_last_lr()[0]),
        }
        if target_beta_kl is not None:
            metrics["beta_kl"] = float(model.obs_encoder.beta_kl)
        for key, values in component_losses.items():
            metrics[f"train_{key}"] = float(np.mean(values))

        should_validate = (epoch + 1) % int(cfg.training.val_every) == 0
        if should_validate and len(val_loader) > 0:
            eval_model = ema_model if ema_model is not None else model
            eval_model.eval()
            val_losses = []
            val_bc_losses = []
            action_abs_error = 0.0
            action_squared_error = 0.0
            action_element_count = 0
            with torch.inference_mode():
                for batch_idx, batch in enumerate(val_loader):
                    batch = _to_device(batch, device)
                    val_loss, val_items = eval_model.compute_loss(batch)
                    val_losses.append(float(val_loss.cpu()))
                    val_bc_losses.append(float(val_items["bc_loss"]))

                    prediction = eval_model.predict_action(
                        batch["obs"], deterministic=True, use_cm=False
                    )["action"]
                    target_start = (
                        eval_model.n_obs_steps - 1 if eval_model.no_pre_action else 0
                    )
                    target = batch["action"][
                        :, target_start : target_start + prediction.shape[1]
                    ]
                    error = prediction - target
                    action_abs_error += float(error.abs().sum().cpu())
                    action_squared_error += float(error.square().sum().cpu())
                    action_element_count += error.numel()
                    max_steps = cfg.training.get("max_val_steps", None)
                    if max_steps is not None and batch_idx + 1 >= int(max_steps):
                        break
            metrics["val_loss"] = float(np.mean(val_losses))
            metrics["val_bc_loss"] = float(np.mean(val_bc_losses))
            metrics["val_action_mae"] = action_abs_error / action_element_count
            metrics["val_action_rmse"] = np.sqrt(
                action_squared_error / action_element_count
            )
            if metrics["val_action_rmse"] < best_val_action_rmse:
                best_val_action_rmse = metrics["val_action_rmse"]
                _save_checkpoint(
                    checkpoint_dir / "best.ckpt",
                    cfg,
                    model,
                    ema_model,
                    optimizer,
                    lr_scheduler,
                    epoch + 1,
                    global_step,
                )

        print(" ".join(f"{key}={value:.6g}" for key, value in metrics.items()))
        if wandb_run is not None:
            wandb_run.log(metrics, step=global_step)

        should_save = (epoch + 1) % int(cfg.training.checkpoint_every) == 0
        if cfg.checkpoint.save_ckpt and should_save:
            _save_checkpoint(
                latest_path,
                cfg,
                model,
                ema_model,
                optimizer,
                lr_scheduler,
                epoch + 1,
                global_step,
            )

    _save_checkpoint(
        latest_path,
        cfg,
        model,
        ema_model,
        optimizer,
        lr_scheduler,
        int(cfg.training.num_epochs),
        global_step,
    )
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()

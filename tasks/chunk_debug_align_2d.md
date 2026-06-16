# Port chunk-debug interfaces from 3D path to 2D path

## Context

The chunk_vdelta_gae work (and the broader chunk-debug branch) added a set of new
config keys, constructor args, and code branches across the 3D policy path
(`dp3_cm.py` + `train_ddp.py` + `dp3_cm_epsilon.yaml`). The 2D policy path
(`dp_image_unet.py` + `train_cm_mid.py` + 2D yamls) plus the 3D-flow yaml were
never updated. Today, running 2D experiments or 3D-flow experiments breaks any
attempt to use:

- `chunk_as_single_action` + IQL critic with conv action embedder, action recon
  loss, or any of the new layer-norm / scale-norm switches
- `chunk_vdelta_gae` / `chunk_vdelta_scalar` / `per_step_vdelta` advantage modes
- `predict_r=True` dynamics path with reward prediction
- `offline_chunk_ratio_mode=scalar` chunk-level PPO ratio

Goal: bring the 2D + 3D-flow surfaces to feature parity with the 3D-CM branch
so the same offline BPPO chunk experiments can be launched there. **No new
algorithms**; this is a pure interface-completion task.

## Working directory

All paths below are relative to `/cephfs/lk/check_rl100_eval/chunk_loss_debug`.
Run all commands from that directory.

## Scope

**In scope** (exact list):
1. `RL-100/rl_100/policy/dp_image_unet.py` — extend `initialize_critic` signature + IQL kwargs to match `dp3_cm.py`
2. `RL-100/rl_100/config/dp_image_unet_epsilon.yaml` — add the 14 missing keys
3. `RL-100/rl_100/config/dp_image_unet_flow.yaml` — add the 7 chunk-vdelta keys + 3 critic layer-norm keys (conv keys already present)
4. `RL-100/rl_100/config/dp3_flow.yaml` — add the 14 missing keys
5. `RL-100/train_cm_mid.py` — extend the **offline** `iql` instantiation (line 615-630) to pass the 6 conv-action-embed args (the online `iql_online` block at line 835+ already has them)

**Out of scope** (already complete or unrelated):
- uni_ppo.py chunk_vdelta_gae implementation (done, tested)
- train_ddp.py and dp3_cm.py (already complete on chunk_loss_debug branch)
- launcher scripts (no 2D launcher needed yet)
- **dynamics `with_reward=cfg.predict_r`** — already plumbed through the
  `train_dynamics(...)` factory at `dynamics_eval_batch.py:117,133` which
  `train_cm_mid.py:678` calls. No gap on the 2D path. (`train_ddp.py:1165,1180`
  uses inline construction instead, but that's a separate code path.)

## File-by-file changes

### 1. `RL-100/rl_100/policy/dp_image_unet.py`

`initialize_critic` (line 323) currently has 18 kwargs. The dp3_cm.py reference (line 360) has 27. Add these 9 kwargs with the same defaults:

```python
use_conv_action_embed=False,
conv_hidden_dims=None,
conv_latent_cz=32,
conv_kernel_size=5,
conv_n_groups=8,
action_recon_beta=0.5,
q_layer_norm=False,
action_embed_layer_norm=False,
action_scale_norm: bool = False,
```

Forward them into the `IQL_Q_V_no(...)` constructor (line 349-372), mirroring lines 418-428 of dp3_cm.py. Use `conv_hidden_dims if conv_hidden_dims is not None else [128, 256]` for the same default-list pattern.

Reuse: copy the literal 11-line block from `dp3_cm.py:418-428`. No new helpers needed.

### 2. `RL-100/rl_100/config/dp_image_unet_epsilon.yaml`

Currently missing (14 keys). Add 6 conv-action-embed keys near `use_action_embed`, then 7 chunk keys near the existing `chunk_as_single_action`/`bppo_chunk_level_ratio` block, then 3 critic layer-norm keys inside the `critic:` section (mirror dp3_cm_epsilon.yaml positions):

```yaml
# top-level (after use_action_embed, ~line 17)
use_conv_action_embed: False
conv_hidden_dims: [128, 256]
conv_latent_cz: 32
conv_kernel_size: 5
conv_n_groups: 8
action_recon_beta: 0.5

# top-level (near chunk_as_single_action, ~line 55)
chunk_adv_clip: null
offline_chunk_ratio_mode: per_step
offline_chunk_adv_mode: scalar_iql
chunk_vdelta_gae_n_rollout: 3
chunk_vdelta_gae_lambda: 0.95
chunk_vdelta_gae_chunk_source: repeat_first

# under critic: (mirror dp3_cm_epsilon.yaml lines 321-323)
  q_layer_norm: false
  action_embed_layer_norm: false
  action_scale_norm: false
```

Verified missing via grep — none of these keys appear in the file today.

### 3. `RL-100/rl_100/config/dp_image_unet_flow.yaml`

Already has `use_conv_action_embed` block (L19-24) and `bppo_chunk_level_ratio`/`kl_annealing` (L77-78). **Missing**:

```yaml
# top-level
chunk_adv_clip: null
offline_chunk_ratio_mode: per_step
offline_chunk_adv_mode: scalar_iql
chunk_vdelta_gae_n_rollout: 3
chunk_vdelta_gae_lambda: 0.95
chunk_vdelta_gae_chunk_source: repeat_first

# under critic:
  q_layer_norm: false
  action_embed_layer_norm: false
  action_scale_norm: false
```

### 4. `RL-100/rl_100/config/dp3_flow.yaml`

Same 14 keys missing as `dp_image_unet_epsilon.yaml`. Add identically (6 conv + 6 chunk + 3 critic norm). `chunk_as_single_action` and `predict_r` already present (L59, L56).

### 5. `RL-100/train_cm_mid.py`

The **offline** IQL instantiation at line 615-630 currently passes only `q_layer_norm`, `action_embed_layer_norm`, `action_scale_norm` (3 of the 9 new args). The **online** block at line 835-854 already has all 9. To make them symmetric, add the 6 conv-action-embed args to the offline block:

```python
use_conv_action_embed=getattr(self.cfg, 'use_conv_action_embed', False),
conv_hidden_dims=getattr(self.cfg, 'conv_hidden_dims', [128, 256]),
conv_latent_cz=getattr(self.cfg, 'conv_latent_cz', 32),
conv_kernel_size=getattr(self.cfg, 'conv_kernel_size', 5),
conv_n_groups=getattr(self.cfg, 'conv_n_groups', 8),
action_recon_beta=getattr(self.cfg, 'action_recon_beta', 0.5),
```

Insert between `n_action_steps=...` and `q_layer_norm=...`, matching the order in the online block.

---

# Phase 2 — Stride-experiment parity (chunk_as_single_action dataset routing)

## Phase 2 context

The Phase 1 changes wired the new constructor kwargs / yaml keys but left
`train_cm_mid.py` using the same `train_dataloader` for BC, critic, dynamics,
and BPPO finetune. `train_ddp.py` separates these so the launcher can set
**different `sequence_stride` for critic vs finetune** (the
`CRITIC_STRIDE` / `FINETUNE_STRIDE` env vars in `train_policy_chunk_two_stage.sh`).
This phase ports the dataset role split + stride-aware critic-artifact dir to
`train_cm_mid.py` so 2D / 3D-flow stride sweeps run end-to-end.

After this phase, `train_cm_mid.py` will have full chunk_as_single_action
parity with `train_ddp.py` for the chunk-debug feature set. The DDP-only
`load_state_dict_with_fallback` helper is intentionally NOT ported (single-GPU
path doesn't need it). The `action_dim *= n_action_steps` scaling is also NOT
needed inline — the `train_dynamics(...)` factory at
`RL-100/rl_100/unidpg/dynamics_eval_batch.py:95` already handles it.

## Phase 2 scope

**In scope** (3 changes, all in `RL-100/train_cm_mid.py`):
1. Add `get_critic_artifact_dir()` method
2. `critic_dataset` role split — separate dataloader for IQL critic training
3. `finetune_dataset` role split — separate dataloader for BPPO actor update

**Out of scope**:
- `load_state_dict_with_fallback` (DDP-only)
- Inline `action_dim *= n_action_steps` (handled by factory)
- Task yaml edits (Adroit/Metaworld task yamls already have `critic_dataset`/`finetune_dataset` blocks)

## Phase 2 file-by-file changes

### Change 1: Add `get_critic_artifact_dir()` method

**Location**: `RL-100/train_cm_mid.py`, insert after `get_stage1_artifact_dir()` at L155-157, before `get_stage1_checkpoint_path()` at L158.

**Reference**: `RL-100/train_ddp.py:244-254`. Copy verbatim.

```python
def get_critic_artifact_dir(self):
    """Return the directory for critic/value/encoder artifacts.
    When offline + chunk_as_single_action and a stride-specific
    critic_artifact_dir is provided, use it. Otherwise fall back to the
    standard stage1 artifact dir."""
    if (self.cfg.offline and self.cfg.chunk_as_single_action
            and self.cfg.unio4.get('critic_artifact_dir', None)):
        return self.cfg.unio4.critic_artifact_dir
    return self.get_stage1_artifact_dir()
```

Then update **both** existing usages of `self.get_stage1_artifact_dir()` that
locate critic artifacts (`Q_bc_path`, `value_path`, `encoder_path`) to use
`self.get_critic_artifact_dir()` instead. Search for `Q_bc_20.pt` and
`value_20.pt` in `train_cm_mid.py` — wherever those filenames are joined onto
`stage1_artifact_dir`, replace the dir resolver with
`self.get_critic_artifact_dir()`. Do NOT change BC checkpoint or dynamics path
resolution (those stay on `get_stage1_artifact_dir()`).

Verify by grep — line 633-634 currently uses `stage1_artifact_dir` for both
`Q_bc_path` and `value_path`. After the change those two lines should resolve
via `critic_artifact_dir = self.get_critic_artifact_dir()`. Mirror the dual-dir
pattern in `train_ddp.py` lines around 974-1099.

### Change 2: `critic_dataset` role split

**Location**: `RL-100/train_cm_mid.py`, IQL critic training loop entered at L657.

**Reference**: `train_ddp.py:948-954` for the dataset instantiation and
`train_ddp.py:1009-1055` for the dataloader-selection wiring.

Insert before the IQL `for local_epoch_idx in range(cfg.training.num_critic_epochs):`
at L653 (i.e., once per stage, not inside the loop):

```python
# --- Offline dataset role split for critic ---
if self.cfg.offline and self.cfg.chunk_as_single_action and hasattr(self.cfg.task, 'critic_dataset'):
    critic_dataset = hydra.utils.instantiate(self.cfg.task.critic_dataset)
    cprint(f'Critic dataset: {len(critic_dataset)} samples '
           f'(stride={getattr(critic_dataset, "sequence_stride", 1)})', 'cyan')
    critic_dataloader = DataLoader(critic_dataset, **cfg.dataloader)
else:
    critic_dataloader = train_dataloader
```

Then in the IQL training loop at L657, replace `train_dataloader` with
`critic_dataloader` in:
- `tqdm.tqdm(train_dataloader, ...)` at L657 — change to `critic_dataloader`
- The same `train_dataloader` reference inside that for-loop body (only the IQL
  critic loop block; do NOT touch the dynamics loop at L702 — that uses the
  shared offline dataset).

**Critical**: The dynamics training loop (L702) **stays on `train_dataloader`**.
Only the IQL critic loop (L657) switches to `critic_dataloader`. This matches
train_ddp where dynamics trains on the shared dataset and critic trains on the
stride-specific one.

### Change 3: `finetune_dataset` role split

**Location**: `RL-100/train_cm_mid.py:898-901`.

**Reference**: `train_ddp.py:1483-1491` (DDP-aware version; we just need the
single-GPU subset).

Replace lines 898-901:

```python
# Current (L898-901):
finetune_batch_size = getattr(self.cfg.unio4, 'finetune_batch_size', self.cfg.dataloader.batch_size)
finetune_dataloader_cfg = OmegaConf.to_container(self.cfg.dataloader)
finetune_dataloader_cfg['batch_size'] = finetune_batch_size
self.finetune_dataloader = DataLoader(self.dataset, **finetune_dataloader_cfg)
```

with:

```python
if self.cfg.offline and self.cfg.chunk_as_single_action and hasattr(self.cfg.task, 'finetune_dataset'):
    finetune_dataset = hydra.utils.instantiate(self.cfg.task.finetune_dataset)
    cprint(f'Finetune dataset: {len(finetune_dataset)} samples '
           f'(stride={getattr(finetune_dataset, "sequence_stride", 1)})', 'cyan')
else:
    finetune_dataset = self.dataset

finetune_batch_size = getattr(self.cfg.unio4, 'finetune_batch_size', self.cfg.dataloader.batch_size)
finetune_dataloader_cfg = OmegaConf.to_container(self.cfg.dataloader)
finetune_dataloader_cfg['batch_size'] = finetune_batch_size
# Mirror train_ddp.py: pinned-memory copies on nested batches are brittle on single-GPU sweeps.
finetune_dataloader_cfg['pin_memory'] = False
finetune_dataloader_cfg['persistent_workers'] = False
finetune_dataloader_cfg['num_workers'] = min(finetune_dataloader_cfg.get('num_workers', 8), 2)
self.finetune_dataloader = DataLoader(finetune_dataset, **finetune_dataloader_cfg)
```

Keep `self.finetune_dataloader_iter = iter(self.finetune_dataloader)` (L901)
unchanged after this block.

## Phase 2 verification

1. **Syntax**:
   ```bash
   cd /cephfs/lk/check_rl100_eval/chunk_loss_debug
   python -c "import ast; ast.parse(open('RL-100/train_cm_mid.py').read())"
   ```

2. **Import-and-instantiate smoke test** — verify `get_critic_artifact_dir()` exists and the `cfg.task.critic_dataset` / `cfg.task.finetune_dataset` references don't crash on a vanilla load. Confirm via grep that:
   - exactly **one** new method `def get_critic_artifact_dir(self):` was added
   - `critic_dataloader` is referenced inside the IQL training loop
   - `finetune_dataset` (not `self.dataset`) is what `self.finetune_dataloader` wraps

3. **Functional check (dry run)**:
   ```bash
   cd RL-100 && /root/miniconda3/envs/dp3/bin/python -c "
   from omegaconf import OmegaConf
   cfg = OmegaConf.load('rl_100/config/dp_image_unet_epsilon.yaml')
   task = OmegaConf.load('rl_100/config/task/adroit_door_medium.yaml')
   merged = OmegaConf.merge(cfg, {'task': task})
   assert hasattr(merged.task, 'critic_dataset'), 'critic_dataset missing'
   assert hasattr(merged.task, 'finetune_dataset'), 'finetune_dataset missing'
   print('OK')
   "
   ```

4. **chunk_vdelta_gae unit tests still pass**:
   ```bash
   cd RL-100 && /root/miniconda3/envs/dp3/bin/python test_chunk_vdelta_gae.py
   ```

## Phase 2 critical files

- `/cephfs/lk/check_rl100_eval/chunk_loss_debug/RL-100/train_cm_mid.py` — L155 (insert method), L633-634 (use new method for critic paths), L653-657 (IQL critic loop dataset split), L898-901 (finetune dataset split)

## Phase 2 reference (do-not-modify)

- `RL-100/train_ddp.py:244-254` — get_critic_artifact_dir source
- `RL-100/train_ddp.py:948-954` — critic_dataset instantiation pattern
- `RL-100/train_ddp.py:1483-1491` — finetune_dataset instantiation pattern
- `RL-100/rl_100/config/task/adroit_door_medium.yaml` — confirms target task yamls already have both keys

## Verification

1. **Syntax / parseability**:
   ```bash
   cd /cephfs/lk/check_rl100_eval/chunk_loss_debug
   python -c "import ast; ast.parse(open('RL-100/rl_100/policy/dp_image_unet.py').read())"
   python -c "import ast; ast.parse(open('RL-100/train_cm_mid.py').read())"
   python -c "import yaml; [yaml.safe_load(open(f)) for f in ['RL-100/rl_100/config/dp_image_unet_epsilon.yaml','RL-100/rl_100/config/dp_image_unet_flow.yaml','RL-100/rl_100/config/dp3_flow.yaml']]"
   ```

2. **Hydra config resolution** — run a dry hydra import for each yaml:
   ```bash
   cd RL-100 && /root/miniconda3/envs/dp3/bin/python -c "from omegaconf import OmegaConf; print(OmegaConf.load('rl_100/config/dp_image_unet_epsilon.yaml').offline_chunk_adv_mode)"
   ```
   Expect `scalar_iql`. Repeat for the other two yamls.

3. **Existing chunk_vdelta_gae unit tests still pass** — they exercise uni_ppo.py only:
   ```bash
   cd RL-100 && /root/miniconda3/envs/dp3/bin/python test_chunk_vdelta_gae.py
   ```
   All 5 tests should still pass (no uni_ppo changes in this porting task).

4. **Smoke test (optional, only if a 2D dataset is available)** — start a tiny 2D run with `chunk_as_single_action=True offline_chunk_adv_mode=chunk_vdelta_gae predict_r=True` to confirm the new args are wired end-to-end. Kill after `initialize_critic` completes successfully.

## Critical files (paths)

- `/cephfs/lk/check_rl100_eval/chunk_loss_debug/RL-100/rl_100/policy/dp_image_unet.py` — line 323 (signature) + 349-372 (IQL kwargs)
- `/cephfs/lk/check_rl100_eval/chunk_loss_debug/RL-100/rl_100/config/dp_image_unet_epsilon.yaml`
- `/cephfs/lk/check_rl100_eval/chunk_loss_debug/RL-100/rl_100/config/dp_image_unet_flow.yaml`
- `/cephfs/lk/check_rl100_eval/chunk_loss_debug/RL-100/rl_100/config/dp3_flow.yaml`
- `/cephfs/lk/check_rl100_eval/chunk_loss_debug/RL-100/train_cm_mid.py` — line 615-630

## Reference (do-not-modify)

Use as ground truth for kwarg ordering / defaults:
- `RL-100/rl_100/policy/dp3_cm.py:360-448` — initialize_critic signature
- `RL-100/rl_100/config/dp3_cm_epsilon.yaml` — yaml key positions and values
- `RL-100/train_cm_mid.py:835-854` — online iql_online block already has the 6 conv args
- `RL-100/train_ddp.py:961-993` — same pattern in DDP entry

---

# Phase 3 — `chunk_as_single_action` policy-side branching parity

## Phase 3 context

Phase 1 ported `initialize_critic` kwargs and yaml keys; Phase 2 ported the
training-loop dataset/artifact routing. But the user found that the **policy
class itself** (`dp_image_unet.py`) still doesn't store
`self.chunk_as_single_action` and never branches on it inside `sample_action`.
By contrast `dp3_cm.py` does both:

```
$ grep self.chunk_as_single_action dp3_cm.py
97:        self.chunk_as_single_action = chunk_as_single_action
1714:            if self.chunk_as_single_action:
1720:            if use_gae and not self.chunk_as_single_action:

$ grep self.chunk_as_single_action dp_image_unet.py
(no matches)
```

Effect: when running 2D with `chunk_as_single_action=True`, BPPO sampling
silently goes through `dynamics.multi_step_evaluation()` instead of
`dynamics.chunk_evaluation()`. The Q-value used to rank candidate trajectories
is computed under the wrong rollout semantics. This is a correctness bug for
2D chunk experiments, not just an interface gap.

`sample_action_with_logprob` does NOT need a branch (3D version doesn't have
one either — it always uses `multi_step_evaluation`). `predict_action` doesn't
need a branch (3D doesn't either). The only policy method that branches on
`chunk_as_single_action` in 3D is `sample_action`.

## Phase 3 scope

**In scope** (2 changes, both in `RL-100/rl_100/policy/dp_image_unet.py`):
1. Add `chunk_as_single_action: bool = False` to `__init__` signature and
   store as `self.chunk_as_single_action`
2. Branch on `self.chunk_as_single_action` inside `sample_action` to select
   between `dynamics.chunk_evaluation()` and `dynamics.multi_step_evaluation()`,
   matching dp3_cm.py:1714-1729

**Out of scope**:
- `sample_action_with_logprob` — 3D has no chunk branch here, so 2D is already
  consistent
- `predict_action` — same reason
- Any new attribute beyond `chunk_as_single_action`

## Phase 3 file-by-file changes

### Change 1: Store `chunk_as_single_action` on the 2D policy

**Location**: `RL-100/rl_100/policy/dp_image_unet.py` `__init__`.

The exact insertion lands in two places:

(a) Add to the kwarg signature near other top-level flags. Reference:
`dp3_cm.py:75` has `chunk_as_single_action: bool = False,` in its `__init__`.
Pick a position adjacent to other policy-level flags in the 2D `__init__`
(e.g. near `flow_*` kwargs around L93-98 or near `action_norm` / config
flags). Default `False`.

(b) Store as attribute right after `self.action_norm = action_norm` at
`dp_image_unet.py:108`:

```python
self.chunk_as_single_action = chunk_as_single_action
```

(c) Default OmegaConf wiring: confirmed by grep that **neither** 2D yaml
forwards `chunk_as_single_action` into the `policy:` block today. dp3_flow.yaml
does this at line 92 (`  chunk_as_single_action: ${chunk_as_single_action}`).

**Required edits**:
- `RL-100/rl_100/config/dp_image_unet_epsilon.yaml` — inside the `policy:`
  block, add `chunk_as_single_action: ${chunk_as_single_action}` near other
  policy-level flags (e.g. next to `n_action_steps: ${n_action_steps}`).
- `RL-100/rl_100/config/dp_image_unet_flow.yaml` — same edit inside its
  `policy:` block.

Without these yaml lines the new `__init__` kwarg defaults to `False` even
when the top-level cfg sets it `True`, silently bypassing the chunk branch.

### Change 2: Add chunk branch in `sample_action`

**Location**: `RL-100/rl_100/policy/dp_image_unet.py:1550` (the
`multi_step_evaluation` call inside `sample_action`).

**Reference**: `dp3_cm.py:1713-1729`. Mirror the exact branch structure.

Replace:
```python
            # dynamics rollout
            _, _, _, _, G, gae_advantages = dynamics.multi_step_evaluation(nobs_features, trajectory, Q, state_dict=state_dict, use_gae=use_gae)
            if use_gae:
                if first_action:
                    q_value = gae_advantages[0]
                else:
                    if self.n_action_steps > 1:
                        q_value = torch.mean(gae_advantages[: self.n_action_steps], dim=0)
                    else:
                        q_value = gae_advantages
            else:
                q_value = G
```

with:

```python
            # dynamics rollout
            if self.chunk_as_single_action:
                G = dynamics.chunk_evaluation(nobs_features, trajectory, Q, state_dict=state_dict, use_gae=use_gae)
            else:
                _, _, _, _, G, gae_advantages = dynamics.multi_step_evaluation(nobs_features, trajectory, Q, state_dict=state_dict, use_gae=use_gae)
            if use_gae and not self.chunk_as_single_action:
                if first_action:
                    q_value = gae_advantages[0]
                else:
                    if self.n_action_steps > 1:
                        q_value = torch.mean(gae_advantages[: self.n_action_steps], dim=0)
                    else:
                        q_value = gae_advantages
            else:
                q_value = G
```

Same diff pattern as 3D: when `chunk_as_single_action=True`, `G` carries the
chunk-level Q (no GAE smoothing needed because each chunk is one decision)
and falls through to `q_value = G`.

## Phase 3 verification

1. **Syntax check**:
   ```bash
   cd /cephfs/lk/check_rl100_eval/chunk_loss_debug
   /root/miniconda3/envs/dp3/bin/python -c "import ast; ast.parse(open('RL-100/rl_100/policy/dp_image_unet.py').read())"
   ```

2. **Attribute storage**:
   ```bash
   grep -n "self.chunk_as_single_action" RL-100/rl_100/policy/dp_image_unet.py
   ```
   Expect 3 hits: 1 storage + 2 branch checks.

3. **Branch parity with 3D**:
   ```bash
   grep -c "self.chunk_as_single_action" RL-100/rl_100/policy/dp3_cm.py
   grep -c "self.chunk_as_single_action" RL-100/rl_100/policy/dp_image_unet.py
   ```
   Both should be 3.

4. **Hydra construction smoke test** — confirm chunk_as_single_action is forwarded into the 2D policy via Hydra:
   ```bash
   cd RL-100 && /root/miniconda3/envs/dp3/bin/python -c "
   from omegaconf import OmegaConf
   import hydra
   cfg = OmegaConf.load('rl_100/config/dp_image_unet_epsilon.yaml')
   task = OmegaConf.load('rl_100/config/task/adroit_door_medium.yaml')
   merged = OmegaConf.merge(cfg, {'task': task})
   merged.chunk_as_single_action = True
   # The policy yaml block must reference \${chunk_as_single_action}
   resolved = OmegaConf.to_container(merged.policy, resolve=True)
   assert resolved.get('chunk_as_single_action') == True, f'policy.chunk_as_single_action not wired (got {resolved.get(\"chunk_as_single_action\")})'
   print('Hydra wiring OK')
   "
   ```

5. **chunk_vdelta_gae unit tests still pass**:
   ```bash
   cd RL-100 && /root/miniconda3/envs/dp3/bin/python test_chunk_vdelta_gae.py
   ```

## Phase 3 critical files

- `/cephfs/lk/check_rl100_eval/chunk_loss_debug/RL-100/rl_100/policy/dp_image_unet.py` — `__init__` (around L95 signature, L108 attribute), `sample_action` (L1550-1560)
- `/cephfs/lk/check_rl100_eval/chunk_loss_debug/RL-100/rl_100/config/dp_image_unet_epsilon.yaml` — confirm `policy.chunk_as_single_action: ${chunk_as_single_action}` is present; add if missing
- `/cephfs/lk/check_rl100_eval/chunk_loss_debug/RL-100/rl_100/config/dp_image_unet_flow.yaml` — same check

## Phase 3 reference (do-not-modify)

- `RL-100/rl_100/policy/dp3_cm.py:75` — `__init__` signature with chunk_as_single_action
- `RL-100/rl_100/policy/dp3_cm.py:97` — `self.chunk_as_single_action` storage
- `RL-100/rl_100/policy/dp3_cm.py:1713-1729` — chunk_evaluation vs multi_step_evaluation branch in `sample_action`
- `RL-100/rl_100/config/dp3_flow.yaml:80, 196` — example of `chunk_as_single_action: ${chunk_as_single_action}` forwarding pattern in policy/lddm yaml block

---

# Implementation & Review Results

This section records what was actually applied and verified for each phase.

## Phase 1 — Constructor / yaml interface alignment ✅

**Status**: Applied and verified.

**File-level changes**:

| File | Change | Lines |
|---|---|---|
| `RL-100/rl_100/policy/dp_image_unet.py` | `initialize_critic` signature +9 kwargs (`use_conv_action_embed`, `conv_hidden_dims`, `conv_latent_cz`, `conv_kernel_size`, `conv_n_groups`, `action_recon_beta`, `q_layer_norm`, `action_embed_layer_norm`, `action_scale_norm`); 11 forwarded args into `IQL_Q_V_no(...)` | L341-350 (sig) + L380-391 (forward) |
| `RL-100/rl_100/config/dp_image_unet_epsilon.yaml` | +14 keys: 6 conv-action-embed near `use_action_embed`, 6 chunk-vdelta near `chunk_as_single_action`, 3 critic layer-norm under `critic:` | top-level + `critic:` block |
| `RL-100/rl_100/config/dp_image_unet_flow.yaml` | +10 keys: 6 chunk-vdelta + 3 critic layer-norm + 1 chunk_as_single_action policy forward (rest of conv keys already existed) | top-level + `critic:` + `policy:` |
| `RL-100/rl_100/config/dp3_flow.yaml` | +14 keys (same set as dp_image_unet_epsilon) | top-level + `critic:` block |
| `RL-100/train_cm_mid.py` | offline `iql` instantiation +6 conv-action-embed kwargs (online block already had them) | between `n_action_steps` and `q_layer_norm` |

**Verifications**:
- AST parse OK on `dp_image_unet.py` and `train_cm_mid.py`
- `OmegaConf.load(...)` on all 3 yamls; `offline_chunk_adv_mode == 'scalar_iql'`, `chunk_vdelta_gae_lambda == 0.95`, `q_layer_norm == False` all confirmed
- `test_chunk_vdelta_gae.py` 5/5 passed

## Phase 2 — Stride-experiment dataset/artifact routing ✅

**Status**: Applied and verified.

**File-level changes** (all in `RL-100/train_cm_mid.py`):

| Change | Where | What |
|---|---|---|
| 1. Add `get_critic_artifact_dir()` method | L158-166 | Mirror of `train_ddp.py:244-254`. Returns `cfg.unio4.critic_artifact_dir` when offline + chunk_as_single_action is on, else falls through to `get_stage1_artifact_dir()` |
| 2. Redirect critic artifact paths | L648-656 | `Q_bc_path` / `value_path` / `encoder_path` use new `critic_artifact_dir = self.get_critic_artifact_dir()`. `dynamics_path` (L692/L694) intentionally stays on `stage1_artifact_dir` |
| 3. `critic_dataset` role split | L666-672 + L677 | Before IQL training loop, instantiate `cfg.task.critic_dataset` when offline + chunk_as_single_action + key present, else fall back to `train_dataloader`. The IQL loop's `tqdm.tqdm(...)` switches to `critic_dataloader`. **Dynamics loop at L702 stays on `train_dataloader`** (load-bearing). |
| 4. `finetune_dataset` role split | L917-931 | Replaces `DataLoader(self.dataset, ...)` with conditional `finetune_dataset` instantiation. Adds `pin_memory=False`, `persistent_workers=False`, `num_workers=min(N, 2)` to mirror train_ddp. |

**Verifications**:
- AST parse OK
- Grep shows expected tokens: `def get_critic_artifact_dir`, `critic_dataloader` inside IQL loop, `finetune_dataset` wrapped by `self.finetune_dataloader`
- OmegaConf merge dry-run with `task=adroit_door_medium`: `critic_dataset` and `finetune_dataset` keys present
- `test_chunk_vdelta_gae.py` 5/5 passed

**Out-of-scope skipped**:
- `load_state_dict_with_fallback` (DDP-only helper, not needed on single-GPU path)
- inline `action_dim *= n_action_steps` (handled inside `train_dynamics(...)` factory at `dynamics_eval_batch.py:95`)
- `with_reward=cfg.predict_r` (already routed through `train_dynamics(...)` factory at `dynamics_eval_batch.py:117,133`)

## Phase 3 — `chunk_as_single_action` policy-side branching parity ✅

**Status**: Applied and verified.

**File-level changes**:

| File | Change | Lines |
|---|---|---|
| `RL-100/rl_100/policy/dp_image_unet.py` | `__init__` adds `chunk_as_single_action: bool = False` between `action_norm` and `mlp_policy_depth` | L77 |
| `RL-100/rl_100/policy/dp_image_unet.py` | `self.chunk_as_single_action = chunk_as_single_action` after `self.action_norm = action_norm` | L110 |
| `RL-100/rl_100/policy/dp_image_unet.py` | `sample_action` rollout block: `chunk_evaluation` vs `multi_step_evaluation` branch + `use_gae and not self.chunk_as_single_action` guard for q_value selection | L1552-1565 |
| `RL-100/rl_100/config/dp_image_unet_epsilon.yaml` | `policy:` block adds `chunk_as_single_action: ${chunk_as_single_action}` adjacent to `n_action_steps` | L162 |
| `RL-100/rl_100/config/dp_image_unet_flow.yaml` | same insertion | L174 |

**Out-of-scope (3D doesn't have these branches either)**:
- `sample_action_with_logprob` — no chunk branch in dp3_cm.py
- `predict_action` — no chunk branch in dp3_cm.py

**Verifications**:
- AST parse OK
- `grep "self.chunk_as_single_action"` → 3 hits in 2D (L110 storage, L1552 branch, L1556 guard); matches 3 hits in 3D — exact parity
- `**kwargs` does NOT swallow the new explicit kwarg (verified by signature inspection: explicit `chunk_as_single_action` precedes `**kwargs`)
- Hydra interpolation smoke test on both yamls: when top-level `chunk_as_single_action=True`, `cfg.policy.chunk_as_single_action == True`; when `=False`, gets `False`. Bidirectional wiring verified.
- `test_chunk_vdelta_gae.py` 5/5 still passed

**Code-level review (no bugs found)**:
- 2D `sample_action` block at L1552-1565 character-aligned with 3D `sample_action` at L1714-1729 — same control flow, same kwargs, same fall-through for `q_value = G` when chunk_as_single_action is True
- Constructor kwarg position is sound (between two existing kwargs, default `False` preserves backward compat)
- yaml insertion preserves 2-space indentation under `policy:`
- Did not touch `sample_action_with_logprob` or `predict_action` (correctly out of scope per 3D reference)

## Cumulative diff statistics

```
RL-100/rl_100/unidpg/uni_ppo.py              +163        (chunk_vdelta_gae implementation, pre-this-task)
RL-100/rl_100/policy/dp_image_unet.py      +51 -9      (Phase 1 + 3)
RL-100/train_cm_mid.py                     +47 -10     (Phase 1 + 2)
RL-100/rl_100/config/dp_image_unet_epsilon.yaml  +17 -1   (Phase 1 + 3)
RL-100/rl_100/config/dp_image_unet_flow.yaml     +17 -1   (Phase 1 + 3)
RL-100/rl_100/config/dp3_flow.yaml         +16 -0      (Phase 1)
RL-100/test_chunk_vdelta_gae.py            +new file   (5 tests, all green)
```

## Final state

The 2D policy path (`dp_image_unet.py` + `train_cm_mid.py` + 2D yamls) and the
3D-flow yaml are now feature-complete with respect to chunk_as_single_action +
chunk_vdelta_gae. Specifically:

- **Constructor parity**: 2D `__init__` stores `self.chunk_as_single_action`,
  matching 3D
- **Critic init parity**: 2D `initialize_critic` accepts and forwards all 9 new
  kwargs to `IQL_Q_V_no`, matching 3D
- **`sample_action` parity**: 2D switches between `dynamics.chunk_evaluation()`
  and `dynamics.multi_step_evaluation()` based on
  `self.chunk_as_single_action`, matching 3D
- **Training-loop parity**: `train_cm_mid.py` has stride-aware critic artifact
  dir + critic_dataset / finetune_dataset role split, matching `train_ddp.py`
  (modulo DDP-only helpers)
- **Yaml parity**: 2D yamls + dp3_flow.yaml have all 14 new keys (or 10/16
  depending on what was already present) with identical defaults to
  dp3_cm_epsilon.yaml; policy blocks forward `chunk_as_single_action` correctly

## Reproduction commands

```bash
cd /cephfs/lk/check_rl100_eval/chunk_loss_debug

# Syntax checks
/root/miniconda3/envs/dp3/bin/python -c "import ast; ast.parse(open('RL-100/rl_100/policy/dp_image_unet.py').read()); ast.parse(open('RL-100/train_cm_mid.py').read()); print('OK')"

# Yaml checks
/root/miniconda3/envs/dp3/bin/python -c "
from omegaconf import OmegaConf
for ycfg in ['RL-100/rl_100/config/dp_image_unet_epsilon.yaml',
             'RL-100/rl_100/config/dp_image_unet_flow.yaml',
             'RL-100/rl_100/config/dp3_flow.yaml']:
    cfg = OmegaConf.load(ycfg)
    assert cfg.offline_chunk_adv_mode == 'scalar_iql'
    assert cfg.chunk_vdelta_gae_lambda == 0.95
    print(f'{ycfg}: OK')
"

# Hydra wiring (chunk_as_single_action policy forwarding)
/root/miniconda3/envs/dp3/bin/python -c "
from omegaconf import OmegaConf
for ycfg in ['RL-100/rl_100/config/dp_image_unet_epsilon.yaml',
             'RL-100/rl_100/config/dp_image_unet_flow.yaml']:
    cfg = OmegaConf.load(ycfg)
    cfg.chunk_as_single_action = True
    assert cfg.policy.chunk_as_single_action is True
    cfg.chunk_as_single_action = False
    assert cfg.policy.chunk_as_single_action is False
    print(f'{ycfg}: policy interpolation OK')
"

# Parity grep — both should print 3
grep -c "self\.chunk_as_single_action" RL-100/rl_100/policy/dp3_cm.py
grep -c "self\.chunk_as_single_action" RL-100/rl_100/policy/dp_image_unet.py

# Unit tests
cd RL-100 && /root/miniconda3/envs/dp3/bin/python test_chunk_vdelta_gae.py
```

All commands above are expected to pass / print OK.


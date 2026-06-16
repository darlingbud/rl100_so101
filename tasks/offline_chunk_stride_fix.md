# Offline Chunk Stride Fix — Execution Plan

## Summary

This plan treats offline `chunk_as_single_action=True` as a **chunk-boundary SMDP problem** and fixes the current stride mismatch without affecting:

- non-chunk paths
- normal single-action mode
- online PPO buffer/rollout semantics
- BC pretraining and dynamics training in the first stage

The implementation goal is to make **offline IQL/critic training** and **offline BPPO actor updates** consume **boundary-only chunk samples**, while preserving the current sliding-window pipeline as the default baseline.

The first validation pass should answer one narrow question:

- does `scalar_iql` stop failing once offline critic + actor-update datasets are aligned to chunk boundaries?

---

## 1. Key Constraints

### Must not affect

- non-chunk paths
- normal single-action mode
- online PPO rollout / replay buffer logic
- BC pretrain in stage 1
- dynamics training in stage 1

### Must only affect

- offline `chunk_as_single_action=True`
- offline IQL / chunk critic training
- offline BPPO actor update sampling

### Important default rule

- `sequence_stride=1` remains the default everywhere
- boundary-only behavior must be opt-in through config/script overrides
- do **not** hard-bind boundary stride to `chunk_as_single_action=True` globally

---

## 2. Sampler Change

### File

- `RL-100/rl_100/common/sampler.py`

### Required change

Add `sequence_stride: int = 1` to the generic sequence sampling path.

#### `create_indices(...)`

Change signature to:

```python
def create_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    episode_mask: np.ndarray,
    pad_before: int = 0,
    pad_after: int = 0,
    sequence_stride: int = 1,
    debug: bool = True,
) -> np.ndarray:
```

Validation:

- assert `sequence_stride >= 1`

Core loop change:

```python
for idx in range(min_start, max_start + 1, sequence_stride):
```

instead of step size `1`.

### `SequenceSampler.__init__(...)`

Add:

```python
sequence_stride: int = 1
```

and pass it into `create_indices(...)`.

### Important semantic note

Keep:

- `min_start = -pad_before`
- `max_start = episode_length - sequence_length + pad_after`

unchanged.

This is important because with:

- `pad_before = n_obs_steps - 1`
- `sequence_stride = n_action_steps`

the effective action anchor stays aligned to chunk boundaries.

---

## 3. Dataset Change

### Files

All dataset classes under:

- `RL-100/rl_100/dataset/*`

that instantiate `SequenceSampler`.

Priority includes at least:

- `adroit_dataset.py`
- `d4rl_dataset.py`
- `metaworld_dataset.py`
- `dmc_dataset.py` if present in this tree
- any other SequenceSampler-backed dataset reachable from current chunk tasks

### Required change

Each dataset constructor gains:

```python
sequence_stride: int = 1
```

Behavior:

- store `self.sequence_stride`
- pass it to the training sampler
- preserve it in `get_validation_dataset()`

Example pattern:

```python
self.sampler = SequenceSampler(
    replay_buffer=self.replay_buffer,
    sequence_length=horizon,
    pad_before=pad_before,
    pad_after=pad_after,
    episode_mask=train_mask,
    sequence_stride=sequence_stride,
)
```

and in validation:

```python
val_set.sampler = SequenceSampler(
    ...,
    sequence_stride=self.sequence_stride,
)
```

### Must not change

- output shapes
- padding behavior
- dataset defaults
- non-chunk task behavior

---

## 4. Offline Dataset Role Split in `train_ddp.py`

### File

- `RL-100/train_ddp.py`

### Goal

Do **not** overload one extra dataset for both critic and actor update. Use explicit roles.

### Required dataset roles

#### `task.dataset`

Used for:

- BC pretrain
- dynamics training

#### `task.critic_dataset`

Used for:

- offline IQL / Q-V training

#### `task.finetune_dataset` (new optional hook)

Used for:

- offline BPPO actor updates only

Fallback:

- if `task.finetune_dataset` is absent, fallback to `task.dataset`

### Required implementation behavior

#### BC

Keep current behavior:

- `train_dataloader` comes from `task.dataset`

#### Dynamics

Keep current behavior:

- dynamics training continues to use `task.dataset`

#### Offline IQL

Do **not** continue reusing `train_dataloader`.

Instead:

- instantiate `critic_dataset = hydra.utils.instantiate(cfg.task.critic_dataset)`
- build `critic_dataloader` from it
- offline IQL loop uses `critic_dataloader`

#### Offline BPPO

Build a dedicated `finetune_dataloader`:

- if `task.finetune_dataset` exists, instantiate it
- else use `task.dataset`

`sample_finetune_batch()` must pull from this dedicated `finetune_dataloader`.

---

## 5. First-Stage Experiment Policy

### What changes in stage 1

Only:

- `task.critic_dataset.sequence_stride`
- `task.finetune_dataset.sequence_stride`

### What stays unchanged in stage 1

- `task.dataset.sequence_stride=1`
- BC pretrain
- dynamics training
- online paths
- non-chunk paths

### Why

The immediate hypothesis is:

- offline chunk critic learns on the wrong decision distribution
- offline BPPO actor updates also optimize on the wrong decision distribution

`scalar_iql` depends directly on those two.

BC and dynamics are second-stage cleanup, not first-stage root-cause validation.

---

## 6. Config / Override Strategy

### Do not make boundary stride the default

Do **not** encode:

- “if `chunk_as_single_action=True`, automatically use boundary stride”

into core config defaults.

Instead:

- keep all dataset defaults at `sequence_stride=1`
- use explicit overrides in chunk debug scripts / commands

### Supported config fields

#### Dataset constructors

Each SequenceSampler-backed dataset now accepts:

```yaml
sequence_stride: 1
```

#### New optional task field

Support:

```yaml
task:
  finetune_dataset:
    ...
```

If absent, offline BPPO falls back to `task.dataset`.

### Recommended experiment overrides

#### Sliding baseline

- `task.dataset.sequence_stride=1`
- `task.critic_dataset.sequence_stride=1`
- `task.finetune_dataset.sequence_stride=1`

#### Actor-boundary only

- `task.dataset.sequence_stride=1`
- `task.critic_dataset.sequence_stride=1`
- `task.finetune_dataset.sequence_stride=${n_action_steps}`

#### Critic+actor boundary

- `task.dataset.sequence_stride=1`
- `task.critic_dataset.sequence_stride=${n_action_steps}`
- `task.finetune_dataset.sequence_stride=${n_action_steps}`

---

## 7. Chunk Debug Launcher Changes

### Target

- chunk debug launcher only

Do not alter generic training behavior for other branches/policy variants.

### Add explicit controls for

- BC/dynamics dataset stride
- critic dataset stride
- finetune dataset stride

### Launcher defaults

Keep launcher default behavior as current sliding baseline:

- all stride values default to `1`

### Add named experiment presets

- `sliding_scalar_iql`
- `actor_boundary_scalar_iql`
- `critic_actor_boundary_scalar_iql`

### Do not include yet

- `per_step_vdelta`
- plan A single-step dynamics
- chunk-vdelta surrogate
- BC boundary alignment
- dynamics boundary alignment

First isolate the stride hypothesis with `scalar_iql` only.

---

## 8. Validation Plan

### Static checks

1. `sequence_stride=1` reproduces current sample count and index layout exactly
2. `sequence_stride=n_action_steps` reduces sample count and advances in chunk-boundary steps
3. validation dataset preserves `sequence_stride`
4. offline IQL uses `critic_dataset`, not `train_dataloader`
5. offline BPPO uses `finetune_dataset` when present
6. non-chunk and normal single-action paths are unchanged when no stride override is set

### Smoke tests

1. BC still trains with `task.dataset.sequence_stride=1`
2. dynamics still trains with `task.dataset.sequence_stride=1`
3. offline IQL runs with `task.critic_dataset.sequence_stride=1`
4. offline IQL runs with `task.critic_dataset.sequence_stride=n_action_steps`
5. offline BPPO runs with `task.finetune_dataset` absent
6. offline BPPO runs with `task.finetune_dataset.sequence_stride=n_action_steps`

### Required experiments

#### Exp 1: Sliding baseline

- `chunk_as_single_action=True`
- critic stride `1`
- finetune stride `1`
- `offline_chunk_adv_mode=scalar_iql`

Expected:

- current failing baseline

#### Exp 2: Actor-boundary only

- `chunk_as_single_action=True`
- critic stride `1`
- finetune stride `n_action_steps`
- `offline_chunk_adv_mode=scalar_iql`

Purpose:

- isolate actor update distribution effect

#### Exp 3: Critic+actor boundary

- `chunk_as_single_action=True`
- critic stride `n_action_steps`
- finetune stride `n_action_steps`
- `offline_chunk_adv_mode=scalar_iql`

Purpose:

- main root-cause test

### Interpretation

- if Exp 3 clearly outperforms Exp 1, stride mismatch is strongly supported
- if Exp 2 helps but Exp 3 helps more, both actor update distribution and critic semantics matter
- if neither helps, the root cause is elsewhere or requires later full BC/dynamics alignment

---

## 9. Data-Volume Guardrail

Boundary stride will reduce sample count by about `n_action_steps`.

So the implementation and experiment plan must also include:

- logging dataset lengths for baseline vs boundary runs
- at least one boundary critic run with increased critic budget

For example:

- higher `num_critic_epochs`
- or repeated boundary run with a larger critic budget than baseline

This prevents false negatives caused purely by smaller sample counts.

---

## 10. Second-Stage Follow-Up (Only If Stage 1 Succeeds)

If stage 1 confirms the hypothesis, then add a second pass:

- `task.dataset.sequence_stride=n_action_steps` for BC
- `task.dataset.sequence_stride=n_action_steps` for dynamics

This produces a fully boundary-aligned offline chunk pipeline.

This second stage is explicitly **not required** for the first root-cause verdict.

---

## 11. One-Line Conclusion

The first fix should be:

**keep BC and dynamics unchanged, but make offline chunk critic/IQL and offline BPPO actor update read boundary-only chunk samples via explicit `critic_dataset` and `finetune_dataset` stride control, while preserving `sequence_stride=1` as the default everywhere else.**

---

## 12. First Round Review

The current first-pass implementation is directionally correct, but it is not ready to serve as the definitive stride experiment yet.

### Issue 1: stage1 artifact reuse is currently unsafe across different stride settings

This is the most important issue.

Current launcher behavior:

- `root_run_dir` / `stage1_run_dir` are keyed only by `exp_name` and `seed`
- stage1 now receives:
  - `task.critic_dataset.sequence_stride`
  - `task.finetune_dataset.sequence_stride`
- but `stage1_complete()` only checks whether artifacts exist, not whether they were trained with the intended stride configuration

That means:

- a run with `CRITIC_STRIDE=1 FINETUNE_STRIDE=1`
- and a later run with `CRITIC_STRIDE=n_action_steps FINETUNE_STRIDE=n_action_steps`

can silently share the same stage1 run directory and the same `Q_bc_20.pt` / `value_20.pt` / dynamics artifacts.

This is unacceptable for the stride root-cause experiment because the boundary run may actually be reusing sliding critic artifacts.

### Issue 2: the launcher still mixes stride debugging with `per_step_vdelta`

The current first-pass script still defaults to:

- `per_step:scalar_iql`
- `per_step:per_step_vdelta`

But this stride-fix round is supposed to isolate exactly one hypothesis:

- whether boundary-aligned critic + actor-update datasets fix offline chunk under `scalar_iql`

So the first-round launcher must not reintroduce:

- `per_step_vdelta`
- single-step dynamics
- chunk-vdelta surrogate

Otherwise the experiment becomes unclean again.

### Issue 3: dataset coverage is currently only partial

The generic sampler capability was added correctly, but only `AdroitDataset` was updated to accept and preserve `sequence_stride`.

This is sufficient for the current `adroit_door_medium` experiment, but it does not yet satisfy the broader plan statement that SequenceSampler-backed datasets should support the same field.

This is not a blocker for the current task if the work remains explicitly scoped to Adroit chunk debugging.

But it must be called out clearly:

- current implementation scope = Adroit chunk tasks only
- broader dataset rollout is future work

### What is already correct

The following parts of the implementation are aligned with the intended plan:

- `SequenceSampler` now supports `sequence_stride`
- `sequence_stride=1` preserves existing behavior
- `critic_dataset` is used for offline IQL only when:
  - `offline=True`
  - `chunk_as_single_action=True`
- `finetune_dataset` is used for offline BPPO actor update only when:
  - `offline=True`
  - `chunk_as_single_action=True`
- BC and dynamics still use `task.dataset`

So the core code-path scoping is good.

The remaining work is mainly about:

- experiment isolation
- launcher safety
- truthful scope declaration

---

## 13. Revise Steps

The next work-agent round should implement the following revisions.

### Revise Step 1: make stage1 artifact keys stride-aware

The launcher must not allow sliding and boundary runs to share the same stage1 artifact directory.

Acceptable fixes include either:

- encoding stride settings into `root_run_dir` / `stage1_run_dir`
- or adding an explicit stride-specific artifact subdirectory for stage1

Minimum requirement:

- changing `CRITIC_STRIDE` and/or `FINETUNE_STRIDE` must change the stage1 artifact identity

Examples of acceptable naming schemes:

- append `_cstride_${CRITIC_STRIDE}_fstride_${FINETUNE_STRIDE}`
- or create a stage1 suffix like `stage1_c${CRITIC_STRIDE}_f${FINETUNE_STRIDE}`

Also update `stage1_complete()` so it only validates the artifact set corresponding to the active stride configuration.

This revision is mandatory before trusting any boundary-vs-sliding comparison.

### Revise Step 2: reduce the default launcher to stride-only `scalar_iql` experiments

For this stride-fix round, the chunk launcher must default to stride-only experiments.

That means:

- remove `per_step_vdelta` from the default `CHUNK_LOSS_MODE_COMBOS`
- default to:
  - `per_step:scalar_iql`

If you want multiple presets, make them stride presets, not signal presets.

Recommended first-round presets:

- `sliding_scalar_iql`
- `actor_boundary_scalar_iql`
- `critic_actor_boundary_scalar_iql`

But even if you do not add preset names yet, the default launcher behavior must no longer mix in `per_step_vdelta`.

### Revise Step 3: explicitly document current scope as Adroit-only

Since only `AdroitDataset` currently accepts `sequence_stride`, the implementation scope must be stated accurately.

Do one of the following:

1. Either keep the code scoped to Adroit and document clearly that:
   - this stride-fix implementation currently supports Adroit chunk tasks only

2. Or expand the same `sequence_stride` threading to the other SequenceSampler-backed datasets that are intended to participate in chunk stride experiments

For the current round, option 1 is acceptable and lower risk.

Do not claim repo-wide dataset support if the code only updates Adroit.

### Revise Step 4: preserve the red-line guard explicitly

During the revision, do not relax the core guardrails.

The revised implementation must still ensure:

- only `offline=True` and `chunk_as_single_action=True` use the split `critic_dataset` / `finetune_dataset` behavior
- non-chunk paths are unchanged
- normal single-action mode is unchanged
- online rollout / buffer code is unchanged
- BC and dynamics still read `task.dataset` in this round

If any revision risks touching other paths, stop and report it instead of expanding scope.

### Revise Step 5: add one minimal verification for stride-specific artifact separation

Besides syntax checks, add one minimal verification that the launcher now distinguishes stride settings correctly.

Examples:

- print the resolved `stage1_run_dir` for:
  - `CRITIC_STRIDE=1 FINETUNE_STRIDE=1`
  - `CRITIC_STRIDE=16 FINETUNE_STRIDE=16`
- confirm they differ

This can be a script-level dry verification; it does not require running full training.

---

## 14. Work-Agent Instruction Summary

For the next round, the work agent should:

1. Make stage1 artifact directories stride-aware so sliding and boundary runs cannot share critic artifacts.
2. Remove `per_step_vdelta` from the default stride-fix launcher path and keep the default experiment focused on `scalar_iql`.
3. Either document the current implementation as Adroit-only, or expand dataset support consistently. Prefer documenting Adroit-only for this round.
4. Preserve the red-line scope:
   - only `offline + chunk_as_single_action=True`
   - no effect on non-chunk, normal single-action, online, BC-default, or dynamics-default paths.
5. Add a minimal check proving that different stride settings resolve to different stage1 artifact identities.

# 2D Flow Policy - Implementation TODO

## Goal
Add native flow-matching support to `DP3_2D` in `RL-100/rl_100/policy/dp_image_unet.py` while keeping the existing 2D diffusion / CM behavior intact and making the 2D policy compatible with the current `BC -> offline BPPO -> online PPO/BPPO` training pipeline.

The 2D flow path must follow the same design principles as the 3D flow path in `dp3_cm.py`, and the distillation path must use the current **Plan B** design:
- `flow_sde_type='cps'`
- `flow_cps_logprob_mode='gaussian'`
- native flow teacher/student distillation, not DDIM->CM fallback

## Scope
- `RL-100/rl_100/policy/dp_image_unet.py`
- `RL-100/rl_100/config/dp_image_unet_flow.yaml` (new)
- 2D flow training scripts
- minimal compatibility wiring only if required in:
  - `RL-100/train_cm_mid.py`
  - `RL-100/train_ddp.py`
  - `RL-100/rl_100/unidpg/uni_ppo.py`

## Out Of Scope (v1)
- 3D flow main path (`dp3_cm.py`)
- low-step / random-mid / Plan C
- changing existing 2D diffusion defaults
- redesigning the 2D obs encoder architecture

## Red Line

This task must **not** break or silently alter the original 2D diffusion DDIM/CM pipeline.

In particular:
- do not regress existing `scheduler_type='ddim'` behavior
- do not regress existing `scheduler_type='cm'` behavior
- do not change the default behavior of existing 2D diffusion configs or launchers unless a flow-specific mode is explicitly enabled
- do not "simplify" implementation by rewriting shared DDIM/CM logic in a way that changes old training, inference, or distillation semantics

If a code path cannot support both old diffusion behavior and new flow behavior safely, preserve the old diffusion path and add a flow-specific branch instead.

---

## High-Level Principle

Do **not** implement 2D flow by routing 2D scripts into `DP3CM` as a compatibility workaround.

The correct target is:
- native 2D flow support in `rl_100.policy.dp_image_unet.DP3_2D`
- a dedicated 2D flow config
- 2D flow scripts pointing to `DP3_2D`

This keeps:
- 2D diffusion on the existing 2D policy path
- 3D flow on the existing 3D policy path
- training entrypoints generic instead of filled with 2D special cases

---

## Task List

### Step 0: Freeze the interface target

Before implementation, treat `DP3_2D` as needing feature parity with the 3D flow policy for the interfaces used by training code.

At minimum, the 2D policy must expose:
- `self.is_flow`
- `set_target()`
- `promote_distilled_model()`
- `conditional_sample(...)`
- `compute_loss(...)`
- `all_step_logprob(...)`
- `sample_action(...)`
- `sample_action_with_logprob(...)`
- `compute_flow_distill_loss(...)`
- `get_unet_timesteps(...)`

**Done when**:
- `train_cm_mid.py`, `train_ddp.py`, and `uni_ppo.py` can treat 2D flow policy like 3D flow policy at the interface level.

### Step 1: Add flow scheduler/config parameters to `DP3_2D.__init__`

**MODIFY**: `RL-100/rl_100/policy/dp_image_unet.py`

Add constructor parameters mirroring the 3D flow policy:
- `flow_noise_scheduler=None`
- `flow_inference_steps=10`
- `flow_sde_type='sde'`
- `flow_noise_level=0.7`
- `flow_sde_window_size=0`
- `flow_sigma_safe_max=0.9`
- `flow_logit_normal_sampling=False`
- `flow_noise_on_final_step=False`
- `flow_cps_logprob_mode='pseudo'`
- `flow_distill_inference_steps=1`
- `flow_distill_teacher_steps=10`

Add internal state:
- `self.is_flow = (scheduler_type == 'flow')`
- maintain `self.ddim_scheduler`, `self.cm_scheduler`, `self.flow_scheduler` explicitly
- if `self.is_flow`:
  - `self.noise_scheduler = self.flow_scheduler`
  - `self.ddim_scheduler = self.flow_scheduler` only for shared BC forward-process style logic where needed
  - set `self.flow_inference_steps`, `self.flow_logit_normal_sampling`, `self.flow_distill_inference_steps`, `self.flow_distill_teacher_steps`
- if `not self.is_flow`:
  - keep existing DDIM/CM behavior exactly

Guard DDIM-only state:
- create `DDIMSolver`
- `self.alpha_schedule`
- `self.sigma_schedule`

only when `not self.is_flow`.

**Done when**:
- `DP3_2D` instantiates successfully with `scheduler_type='flow'`
- existing `scheduler_type='ddim'` / `cm` behavior remains unchanged

### Step 2: Port scheduler selection and timestep handling

**MODIFY**: `RL-100/rl_100/policy/dp_image_unet.py`

Add `get_unet_timesteps()` from `dp3_cm.py`:
- flow path preserves float scheduler timesteps until the final integer conversion for the UNet
- DDIM/CM path remains unchanged

Update `conditional_sample(...)`:
- `use_cm=False` + `self.is_flow=False` -> current DDIM path
- `use_cm=True` + `self.is_flow=False` -> current CM distilled path
- `use_cm=False` + `self.is_flow=True` -> flow teacher/default model path
- `use_cm=True` + `self.is_flow=True` -> flow student/distilled model path

In flow mode:
- use `flow_scheduler` or `flow_student_scheduler`
- set inference steps from `flow_inference_steps` or `flow_distill_inference_steps`
- support `deterministic=True` via `step_mean`
- support stochastic sampling via `step`

**Done when**:
- `predict_action()` can call `conditional_sample()` in flow mode without falling through to DDIM logic

### Step 3: Implement 2D flow BC loss

**MODIFY**: `compute_loss()` in `dp_image_unet.py`

Add a flow branch modeled after `dp3_cm.py`:
- use scheduler-derived noisy sample construction
- use scheduler-derived training target
- do not reuse DDIM epsilon/sample/v-pred loss
- support `flow_logit_normal_sampling`
- return `(loss, loss_dict)` with the existing logging structure

Requirements:
- image augmentation / encoder call pattern remains the same as current 2D policy
- no changes to action normalization behavior
- no shape changes to the returned action head

**Done when**:
- `compute_loss(batch)` returns finite values in flow mode
- DDIM 2D BC path still returns the same outputs as before

### Step 4: Make 2D flow compatible with PPO/BPPO rollout/logprob APIs

**MODIFY**: `dp_image_unet.py`

Port the flow-capable variants of:
- `all_step_logprob(...)`
- `sample_action(...)`

Add the currently missing:
- `sample_action_with_logprob(...)`

Requirements:
- 2D flow must provide the exact policy-side interface expected by:
  - `train_cm_mid.py`
  - `train_ddp.py`
  - `uni_ppo.py`
- flow log-prob path must use the scheduler methods, not DDIM-only assumptions
- return tensor shapes must remain compatible with the current PPO/BPPO slicing behavior

**Critical**:
Right now 2D policy has no `sample_action_with_logprob()`, so online flow rollout would fail immediately. This must be implemented, not deferred.

**Done when**:
- 2D flow policy can be used for rollout collection and PPO/BPPO updates through the existing training entrypoints

### Step 5: Add native Plan B distillation to 2D policy

**MODIFY**: `dp_image_unet.py`

#### 5.1 `set_target()` flow branch
Mirror `dp3_cm.py`:
- if `self.is_flow`:
  - create `self.teacher` from `self.model`
  - create `self.distilled_model`
  - do **not** create DDIM-style `target_model`
  - create independent `self.flow_teacher_scheduler`
  - create independent `self.flow_student_scheduler`

#### 5.2 `promote_distilled_model()`
Add the flow-only promote step:
- copy `distilled_model` weights into `self.model`
- switch default active inference steps to `flow_distill_inference_steps`
- keep teacher/student semantics aligned with Plan B

#### 5.3 `compute_flow_distill_loss()`
Implement the same-noise teacher/student distillation loss:
- teacher rollout uses `self.model`
- teacher scheduler uses `self.flow_teacher_scheduler`
- student rollout uses `self.distilled_model`
- student scheduler uses `self.flow_student_scheduler`
- `distill2mean=True` -> teacher uses `step_mean`
- `distill2mean=False` -> teacher uses `step`

Plan B constraints:
- no DDIM->CM fallback for flow
- no teacher/student scheduler sharing
- no unnecessary `target_model` for flow

**Done when**:
- `uni_ppo.distill_update()` can call `compute_flow_distill_loss()` on the 2D policy
- `promote_distilled_model()` makes the student the default active policy for evaluation / rollout

### Step 6: Add a dedicated 2D flow config

**NEW**: `RL-100/rl_100/config/dp_image_unet_flow.yaml`

Base it on `dp_image_unet_epsilon.yaml`, not `dp3_flow.yaml`.

Keep the 2D-specific structure:
- top-level 2D fields:
  - `feature_type`
  - `use_agent_pos`
  - `use_pretrained_2DEncoder`
  - `encoder_type`
  - `use_recon`
  - `use_vib`
  - `kl_annealing`
- `policy._target_ = rl_100.policy.dp_image_unet.DP3_2D`
- `policy.obs_encoder = ${encoders.${encoder_type}}`
- existing `encoders.*` subtree

Replace scheduler path with flow:
- `policy.scheduler_type: flow`
- `policy.flow_noise_scheduler._target_: ...FlowMatchSchedulerExtended`
- `flow_inference_steps: 10`
- `flow_distill_inference_steps: 1`
- `flow_distill_teacher_steps: 10`
- `flow_sde_type: 'cps'`
- `flow_cps_logprob_mode: 'gaussian'`
- `flow_noise_level: 0.7`
- `flow_sde_window_size: 0`
- `flow_sigma_safe_max: 0.9`
- `flow_logit_normal_sampling: false`
- `flow_noise_on_final_step: false`

Do **not** delete the existing scheduler constructor blocks:
- keep `policy.cm_noise_scheduler`
- keep `policy.ddim_noise_scheduler`

Reason:
- `DP3_2D.__init__()` still requires `cm_noise_scheduler` and `ddim_noise_scheduler`
- the new 2D flow config must stay constructor-compatible during the migration

Defaults:
- `distill_phase: null`
- `distill_loss_type` can remain compatible with existing config, because `uni_ppo.py` already branches on `is_flow`

**Done when**:
- 2D flow config instantiates directly into `DP3_2D`
- existing 2D diffusion configs remain untouched

### Step 7: Rewire 2D flow scripts to native 2D policy

**MODIFY**:
- `scripts/train_policy_image_unet_flow.sh`
- `scripts/train_policy_image_unet_flow_two_stage.sh`

They must use:
- `config_name='dp_image_unet_flow'`
- `policy._target_=rl_100.policy.dp_image_unet.DP3_2D`

They must **not** use:
- `DP3CM` as a 2D flow workaround
- `policy.backbone=resnet18` as the primary 2D flow implementation route

3D scripts remain on:
- `dp3_flow`
- `DP3CM`

#### 7.1 2D flow DDP launcher must support Plan B distill

The 2D flow DDP / two-stage launcher must support both:
- normal flow training (`distill_phase=null`)
- flow distillation training (Plan B)

At minimum the DDP launcher must be able to pass through:
- `distill_phase='after_dp'` or `distill_phase='after_offline'`
- `distill2mean=True`
- `distill_loss_type='action_same_noise'`
- `flow_distill_inference_steps=1`
- `flow_distill_teacher_steps=10`
- `flow_sde_type='cps'`
- `flow_cps_logprob_mode='gaussian'`

Decision:
- do **not** add a second DDP launcher just for distill
- use the same `scripts/train_policy_image_unet_flow_two_stage.sh`
- add an explicit distill mode switch to that script

Required script interface:
- `ENABLE_DISTILL=${ENABLE_DISTILL:-false}`
- `DISTILL_PHASE=${DISTILL_PHASE:-after_dp}`
- `DISTILL2MEAN=${DISTILL2MEAN:-true}`
- `DISTILL_LOSS_TYPE=${DISTILL_LOSS_TYPE:-action_same_noise}`
- `FLOW_DISTILL_INFERENCE_STEPS=${FLOW_DISTILL_INFERENCE_STEPS:-1}`
- `FLOW_DISTILL_TEACHER_STEPS=${FLOW_DISTILL_TEACHER_STEPS:-10}`
- `FLOW_SDE_TYPE=${FLOW_SDE_TYPE:-cps}`
- `FLOW_CPS_LOGPROB_MODE=${FLOW_CPS_LOGPROB_MODE:-gaussian}`

Required behavior:
- default mode is non-distill:
  - `ENABLE_DISTILL=false`
  - launcher passes `distill_phase=null`
- distill mode:
  - `ENABLE_DISTILL=true`
  - launcher passes the full Plan B parameter set above into both stage-1 and stage-2 command construction
  - `DISTILL_PHASE` must accept at least `after_dp` and `after_offline`
- script examples must include:
  - one normal 2D flow two-stage command
  - one 2D flow two-stage distill command using `ENABLE_DISTILL=true`

Do not leave this as a "manual edit the script body before running" workflow.

**Done when**:
- 2D flow launchers point to the new native 2D flow config
- 2D flow DDP launcher can start both non-distill and distill runs
- 3D flow launchers are unchanged

### Step 8: Only add training-entrypoint glue if interface parity is insufficient

Check:
- `train_cm_mid.py`
- `train_ddp.py`
- `uni_ppo.py`

Current repo already routes many flow decisions through:
- `getattr(model, 'is_flow', False)`
- `compute_flow_distill_loss()`
- `promote_distilled_model()`
- `sample_action_with_logprob()`

So the preferred implementation is:
- fix the 2D policy to satisfy the existing generic flow interface
- do not add 2D-specific special cases unless forced

Only modify training entrypoints if:
- they assume a method signature that 2D policy still cannot match
- they make a hardcoded DDIM-only assumption that blocks 2D flow specifically

**Important current repo fact**:
- `train_cm_mid.py` already has a flow-aware distill branch
- `train_ddp.py` does **not** yet have the matching flow-aware distill branch

So for DDP support, this is not optional cleanup; it is a required implementation item.

Required DDP change:
- port the same `getattr(model_to_optimize, 'is_flow', False)` distill dispatch used in `train_cm_mid.py`
- when `is_flow`, call `compute_flow_distill_loss(...)`
- skip DDIM-style EMA target update for flow, consistent with the current `train_cm_mid.py` behavior
- keep DDIM/CM distill behavior unchanged for non-flow policies

**Done when**:
- 2D flow works through the existing generic flow code paths

---

## Acceptance Checklist

### Configuration
- [ ] `dp_image_unet_flow.yaml` instantiates successfully
- [ ] old 2D diffusion configs still instantiate
- [ ] 3D flow config remains unchanged

### BC
- [ ] 2D flow `compute_loss(batch)` returns finite values
- [ ] `predict_action(obs)` works in flow mode

### Offline RL
- [ ] `all_step_logprob()` returns tensors compatible with the current PPO/BPPO code
- [ ] one offline update path can run without scheduler/interface errors

### Online RL
- [ ] `sample_action_with_logprob()` exists and is callable from `train_cm_mid.py` / `train_ddp.py`
- [ ] online rollout collection does not fail because of missing policy methods
- [ ] one PPO mini-batch can run without scheduler attribute errors

### Distillation (Plan B)
- [ ] `set_target()` flow branch creates teacher + student schedulers correctly
- [ ] `compute_flow_distill_loss()` returns finite values
- [ ] `promote_distilled_model()` makes student the default active model
- [ ] `cfg.ppo.num_inference_steps` stays aligned with promoted student steps

### Regression
- [ ] existing 2D DDIM/CM path still works
- [ ] 3D flow path still works

---

## Review Standard

Implementation is incomplete if any of these is true:
- only `scheduler_type='flow'` is added, but 2D policy still lacks `sample_action_with_logprob()`
- flow BC loss reuses DDIM epsilon/sample loss
- 2D flow distill silently falls back to DDIM->CM loss
- 2D flow scripts still route into `DP3CM` instead of `DP3_2D`
- training entrypoints need new 2D special cases because the 2D policy interface was left incomplete
- the implementation changes existing 2D diffusion DDIM/CM training, inference, or distillation behavior when flow mode is not enabled
- flow distill checkpoint load crashes because `promote_distilled_model()` still touches non-flow state
- old DDIM/CM deterministic inference changes because the flow refactor removed fixed-noise / `generator` behavior
- `train_ddp.py` still assumes `target_model` exists during flow distill
- modified shell launchers do not pass `bash -n`

---

## Round 1 Review Results

The first implementation pass already exposed these blockers. They are mandatory follow-up items, not optional cleanup.

### Blocker 1: `promote_distilled_model()` flow path is broken

Current problem:
- flow mode `set_target()` intentionally does not create `self.target_model`
- current `promote_distilled_model()` still contains leftover non-flow logic
- current method still touches `self.target_model`
- current method still references `model_for_copy` from the non-flow branch

Required fix:
- make `promote_distilled_model()` a clean flow-only method
- only copy `distilled_model` weights into `self.model`
- only update the active inference-step count to `flow_distill_inference_steps`
- do not touch `self.target_model` in flow mode
- do not reference `model_for_copy` inside the flow-only promote path

Required outcome:
- loading a distilled 2D-flow checkpoint through `load()` must not crash
- promoting the distilled student must succeed both after training and after reload

### Blocker 2: `train_ddp.py` still does not support flow distill

Current problem:
- `train_ddp.py` still dispatches only to `compute_ddim2cm_loss*`
- it still performs DDIM-style EMA updates on `target_model`
- this is incompatible with native flow teacher/student distill

Required fix:
- port the same `getattr(model_to_optimize, 'is_flow', False)` branch already used in `train_cm_mid.py`
- when `is_flow`, call `compute_flow_distill_loss(...)`
- when `not is_flow`, keep the current DDIM/CM distill behavior unchanged
- skip DDIM-style EMA target update for flow
- after flow distill finishes, call `promote_distilled_model()` and save the promoted student checkpoint, matching `train_cm_mid.py`

Required outcome:
- DDP distill can run with native 2D flow without touching `target_model`
- non-flow DDP distill semantics remain unchanged

### Blocker 3: `conditional_sample(..., deterministic=True)` regressed old DDIM/CM behavior

Current problem:
- the current flow refactor removed the fixed-noise deterministic initialization path
- the current code no longer preserves `generator`-driven deterministic sampling semantics
- this changes old DDIM/CM inference behavior, which violates the red line

Required fix:
- restore the original deterministic fixed-noise path for DDIM/CM
- restore `generator` support
- keep the old DDIM/CM inference semantics intact
- add a separate flow branch instead of collapsing DDIM/CM and flow into one shared randomized initialization path

Required outcome:
- `conditional_sample(..., deterministic=True)` remains deterministic for existing 2D DDIM/CM
- flow support does not alter old DDIM/CM inference semantics

### Blocker 4: modified flow launchers are syntax-broken

Current problem:
- `scripts/train_policy_flow.sh` currently fails `bash -n`
- `scripts/train_policy_flow_two_stage.sh` currently fails `bash -n`
- the current errors are from quoting / trailing backslash issues

Required fix:
- repair command continuation and quoting in both scripts
- keep the scripts executable and parser-clean
- if the added KL overrides are intentional, keep them with correct shell syntax
- if the KL overrides were accidental, remove them while fixing syntax

Required outcome:
- both modified shell launchers pass `bash -n`

### Blocker 5: `dp_image_unet_flow.yaml` must match the documented Plan B defaults

Current problem:
- the current todo defines Plan B defaults as:
  - `flow_sde_type='cps'`
  - `flow_cps_logprob_mode='gaussian'`
- the first implementation used a config value that is inconsistent with that default plan

Required fix:
- align `dp_image_unet_flow.yaml` with the documented Plan B defaults
- use `flow_sde_type='cps'`
- use `flow_cps_logprob_mode='gaussian'`
- if a launcher intentionally overrides these values, document that explicitly in the script section instead of leaving config and todo inconsistent

Required outcome:
- the 2D flow config and this todo document describe the same default Plan B path

### Blocker 6: `dp_image_unet_flow.yaml` must be part of the actual deliverable

Current problem:
- the config currently exists as a new file but is not yet part of a complete tracked end-to-end launch path

Required fix:
- ensure `dp_image_unet_flow.yaml` is git-tracked
- ensure the 2D flow launchers actually reference it
- do not claim end-to-end 2D flow support while the config remains unwired or untracked

Required outcome:
- the config is part of the committed deliverable and is used by the intended 2D flow scripts

---

## Round 1 Fix Order For Work Agent

The next patch should be a correction pass on top of the first implementation, not a fresh redesign.

Implement in this order:

1. Fix `DP3_2D` regression blockers first.
   - Repair `promote_distilled_model()` so it is flow-only and only updates `self.model` plus the active inference-step count.
   - Remove leftover non-flow `target_model` / `model_for_copy` logic from the flow promote path.
   - Restore deterministic and `generator` behavior in `conditional_sample()` for DDIM/CM exactly as before.
   - Keep flow sampling in a separate branch so old DDIM/CM behavior is unchanged.

2. Finish DDP flow-distill support.
   - Update `RL-100/train_ddp.py` to mirror `train_cm_mid.py` for distill dispatch.
   - `is_flow -> compute_flow_distill_loss(...)`.
   - `not is_flow -> existing compute_ddim2cm_loss*`.
   - Skip `target_model` EMA updates for flow.
   - After flow distill, call `promote_distilled_model()` and save the promoted checkpoint.

3. Repair launcher integrity.
   - Fix `scripts/train_policy_flow.sh`.
   - Fix `scripts/train_policy_flow_two_stage.sh`.
   - Ensure both parse under `bash -n`.
   - Keep the KL-related overrides only if they are intentional.

4. Align config and launcher wiring.
   - Update `dp_image_unet_flow.yaml` to match the documented Plan B defaults.
   - Ensure 2D flow launchers point to `dp_image_unet_flow` and native `DP3_2D`.
   - Do not route 2D flow through `DP3CM`.

5. Re-run the validation checklist and record the result.
   - `python -m py_compile RL-100/rl_100/policy/dp_image_unet.py`
   - `bash -n` for all touched shell scripts
   - instantiate `dp_image_unet_flow.yaml`
   - one minimal smoke path for 2D flow BC loss
   - one minimal smoke path for `sample_action_with_logprob()`
   - one regression check for old 2D DDIM/CM deterministic inference

---

## Round 1 Fix Acceptance

- [ ] flow distill checkpoint load no longer crashes because of `promote_distilled_model()`
- [ ] `train_ddp.py` can enter flow distill without calling DDIM distill loss or touching `target_model`
- [ ] `conditional_sample(..., deterministic=True)` remains deterministic for old 2D DDIM/CM
- [ ] `bash -n scripts/train_policy_flow.sh` passes
- [ ] `bash -n scripts/train_policy_flow_two_stage.sh` passes
- [ ] `dp_image_unet_flow.yaml` is tracked and instantiates
- [ ] old 2D diffusion DDIM/CM path still works after the flow patch

---

## Round 2 Review Results

The second review pass found that the native 2D flow implementation is still not wired into the actual launch path.

### Blocker 7: current training scripts still route to `DP3CM` / `dp3_flow`

Current problem:
- the currently modified flow launchers are still:
  - `scripts/train_policy_flow.sh`
  - `scripts/train_policy_flow_two_stage.sh`
- they still point to the 3D flow route:
  - `config_name='dp3_flow'`
  - `policy._target_=rl_100.policy.dp3_cm.DP3CM`
- this means the runtime launch path still instantiates the 3D policy/config stack, not native 2D flow

Required fix:
- do not use the 3D flow launchers as the 2D flow entrypoint
- add or restore the intended 2D launchers:
  - `scripts/train_policy_image_unet_flow.sh`
  - `scripts/train_policy_image_unet_flow_two_stage.sh`
- those 2D launchers must use:
  - `config_name='dp_image_unet_flow'`
  - native `DP3_2D`
- preferred implementation:
  - rely on `dp_image_unet_flow.yaml` to provide the correct `_target_`
  - do not redundantly override `_target_` unless there is a concrete reason
- if `_target_` is overridden in script, it must be:
  - `policy._target_=rl_100.policy.dp_image_unet.DP3_2D`

Required outcome:
- 2D flow training can actually be launched through dedicated 2D scripts
- no 2D flow launcher instantiates `DP3CM`
- 3D flow launchers remain 3D-only

### Minor 1: constructor default for `flow_cps_logprob_mode` is inconsistent with Plan B

Current problem:
- the todo defines Plan B default as:
  - `flow_cps_logprob_mode='gaussian'`
- `dp_image_unet_flow.yaml` already uses `gaussian`
- but `DP3_2D.__init__` still defaults `flow_cps_logprob_mode='pseudo'`

Required fix:
- align the constructor default with the documented Plan B default
- use `flow_cps_logprob_mode='gaussian'` in `DP3_2D.__init__`

Required outcome:
- constructor defaults, config defaults, and todo requirements all describe the same Plan B behavior
- direct policy instantiation does not silently diverge from the documented default

---

## Round 2 Fix Order For Work Agent

Apply after the Round 1 blockers are fixed:

1. Complete the 2D launcher wiring.
   - Add or restore `scripts/train_policy_image_unet_flow.sh`.
   - Add or restore `scripts/train_policy_image_unet_flow_two_stage.sh`.
   - Point both scripts to `config_name='dp_image_unet_flow'`.
   - Ensure they instantiate native `DP3_2D`, not `DP3CM`.
   - Keep `scripts/train_policy_flow.sh` and `scripts/train_policy_flow_two_stage.sh` as 3D-only launchers.

2. Align constructor defaults with the Plan B spec.
   - Change the `DP3_2D.__init__` default for `flow_cps_logprob_mode` from `pseudo` to `gaussian`.

3. Re-run launch-path validation.
   - Confirm 2D flow scripts reference `dp_image_unet_flow`.
   - Confirm no 2D flow script references `dp3_flow`.
   - Confirm no 2D flow script overrides `_target_` to `DP3CM`.

---

## Round 2 Fix Acceptance

- [ ] `scripts/train_policy_image_unet_flow.sh` exists and targets `dp_image_unet_flow`
- [ ] `scripts/train_policy_image_unet_flow_two_stage.sh` exists and targets `dp_image_unet_flow`
- [ ] no 2D flow launcher instantiates `rl_100.policy.dp3_cm.DP3CM`
- [ ] `DP3_2D.__init__` default `flow_cps_logprob_mode` is `gaussian`
- [ ] config default, constructor default, and todo spec agree on Plan B defaults

---

## References
- 3D flow policy reference: `RL-100/rl_100/policy/dp3_cm.py`
- 2D diffusion policy base: `RL-100/rl_100/policy/dp_image_unet.py`
- 3D flow master todo: `tasks/todo.md`
- current Plan B implementation / fixes documented in `tasks/todo.md`

---

## Plan B-Specific Notes To Preserve

The 2D flow implementation must preserve these already-validated Plan B rules:
- flow distillation is native teacher/student flow distillation
- teacher rollout uses `self.model`, not frozen `self.teacher` weights as the source of truth for online-distill semantics
- teacher and student must not share the same scheduler instance
- promoted student must become the default active policy with the correct inference-step count
- PPO/BPPO buffer shapes must be synchronized with the active policy inference-step count after promote

If any of these need to differ in 2D, that difference must be justified explicitly before implementation.

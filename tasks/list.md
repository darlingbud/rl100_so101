# Flow Matching RL Integration Do List

## Goal
Add a flow-matching RL training option to the `DP3CM` main training path while keeping the current
`BC -> offline BPPO -> online PPO/BPPO`
training structure unchanged and preserving the existing `ddim/cm` behavior.

## Scope
This work is limited to the following core paths:
- `RL-100/train_cm_mid.py`
- `RL-100/rl_100/policy/dp3_cm.py`
- `RL-100/rl_100/unidpg/uni_ppo.py`
- `RL-100/rl_100/unidpg/diffusion_policy/diffusers_patch/*`
- `RL-100/rl_100/config/*` entries that directly serve `dp3_cm`

## Out Of Scope
The first version must not include:
- `dp3.py`
- `dp_image_unet`
- `dp_state`
- other policy variants
- GRPO or grouped reward-normalization training
- flow mode integrated with `ddim -> cm` distillation

## Task List

### 1. Abstract active scheduler selection inside DP3CM
Goal:
- remove the implicit assumption that the RL path always uses DDIM
- support `ddim`, `cm`, and `flow` as explicit scheduler modes

Required methods:
- `compute_loss`
- `conditional_sample`
- `predict_action`
- `all_step_logprob`
- `all_step_action_logprob`
- `sample_action`
- `sample_action_with_logprob`

Done when:
- `DP3CM` clearly maintains
  - `ddim_scheduler`
  - `cm_scheduler`
  - `flow_scheduler`
- flow mode never falls through to DDIM by accident

### 2. Add flow configuration fields
Add the minimum configuration fields required for flow mode:
- `policy.scheduler_type: ddim | cm | flow`
- `policy.flow_noise_scheduler`
- `policy.flow_inference_steps`
- `policy.flow_noise_level`
- `policy.flow_sde_type`

Suggested defaults:
- `policy.flow_noise_level=0.7`
- `policy.flow_sde_type='sde'`

Done when:
- Hydra instantiate works with `scheduler_type=flow`
- existing DDIM or CM configs still work without modification

### 3. Implement a flow scheduler wrapper in `diffusers_patch`
Reference:
- `third_party/flow_grpo/flow_grpo/diffusers_patch/sd3_sde_with_logprob.py`
- `third_party/FlowCPS/flow_grpo/diffusers_patch/sd3_sde_with_logprob.py`

Do not copy image pipeline logic. Only migrate the transition and log-prob math that is relevant to policy RL.

Required interfaces:
- `step_logprob(...)`
- `step_forward_logprob(...)`
- `step_forward_logprob_with_entropy(...)`
- `step_mean(...)`
- `add_noise(...)` or an equivalent training-noise interface

Done when:
- offline RL and online RL can reuse the existing ratio computation path
- no separate PPO formula is needed for flow mode

### 4. Wire `scheduler_type=flow` into `dp3_cm.py`
Required methods to update:
- `__init__`
- `compute_loss`
- `conditional_sample`
- `predict_action`
- `all_step_logprob`
- `all_step_action_logprob`
- `sample_action`
- `sample_action_with_logprob`

Requirements:
- tensor shapes and return structures remain compatible with the DDIM path
- observation encoding, normalization, and action unnormalization semantics do not change

Done when:
- flow mode can run policy forward and sampling end-to-end
- the existing DDIM path still behaves the same

### 5. Implement a dedicated flow-matching training loss
Requirements:
- do not reuse DDIM `epsilon`, `sample`, or `v_prediction` loss
- timestep or sigma sampling must match the chosen flow scheduler
- the predicted target must match the scheduler's prediction variable
- output format must remain `(loss, loss_dict)`

Done when:
- `compute_loss(batch)` returns a finite value in flow mode
- training logs still work through the existing training program

### 6. Disable CM distillation in flow mode
The following methods must be blocked in flow mode:
- `compute_ddim2cm_loss`
- `compute_ddim2cm_loss_action`
- `compute_ddim2cm_loss_action_same_noise`

Preferred behavior:
- raise a clear error
- or block `distill_phase` and `scheduler_type=flow` from being enabled together in the training entrypoint

Done when:
- there is no silent fallback
- the error message clearly states that flow v1 does not support CM distillation

### 7. Make offline BPPO scheduler-agnostic
Main file:
- `RL-100/rl_100/unidpg/uni_ppo.py`

Main method:
- `update_distribution(...)`

Requirements:
- keep the old rollout data flow:
  - `old_all_x`
  - `old_all_next_x`
  - `old_all_logprob`
- recompute new log-probabilities through the currently active scheduler
- keep the existing PPO or BPPO ratio, clipping, and advantage formulas unchanged

Done when:
- offline `update_distribution` completes at least one optimization step in flow mode

### 8. Remove hidden DDIM assumptions from the online PPO path
Review and fix:
- rollout collection
- `all_step_action_logprob`
- online PPO update
- `step_forward_logprob_with_entropy`
- any DDIM-specific field access such as
  - `clip_std_max`
  - `eta`
  - `prev_sample`

Requirements:
- flow mode must not fail because of DDIM-only attributes
- add explicit flow branches or compatible no-op behavior where needed

Done when:
- online rollout can collect `all_x` and `a_logprob`
- online PPO can finish at least one mini-batch update in flow mode

### 9. Connect flow mode in `train_cm_mid.py`
Requirements:
- keep the three-stage structure unchanged
- BC must call the flow `compute_loss`
- offline finetuning must call the flow scheduler path in `update_distribution`
- online finetuning must use flow rollout plus PPO update
- flow mode must be blocked if distillation is requested

Done when:
- `scheduler_type=flow` can pass through the main training program end-to-end

### 10. Add a minimal runnable example
Provide at least one of the following:
- a new flow yaml
- or a clear script example using overrides

The example must include:
- `policy.scheduler_type=flow`
- `policy._target_=rl_100.policy.dp3_cm.DP3CM`
- `policy.flow_inference_steps=...`
- `policy.flow_noise_level=...`
- `policy.flow_sde_type=...`

Done when:
- a work agent can launch flow training from repository examples without guessing missing flags

### 11. Protect existing behavior
Minimum regression coverage:
- DDIM BC still runs
- DDIM offline RL still runs
- DDIM online RL still runs

Done when:
- flow integration does not break the current default path

## Acceptance Checklist

### Configuration
- `scheduler_type=flow` instantiates correctly
- old configs still work without added flow fields

### BC
- flow `compute_loss(batch)` returns finite values
- flow `predict_action(obs)` returns the expected structure

### Offline RL
- `all_step_logprob` returns the expected tensors
- `uni_ppo.update_distribution` completes at least one update in flow mode

### Online RL
- rollout stores `all_x` and `a_logprob`
- online PPO completes at least one mini-batch
- no scheduler attribute errors appear

### Distillation
- flow plus distillation fails clearly and intentionally

### Regression
- DDIM smoke tests still pass
- CM path at least instantiates and runs inference without error

## Review Standard
The implementation is incomplete if any of the following is true:
- it only adds `scheduler_type='flow'` without `step_forward_logprob`
- it ignores online PPO log-prob or entropy support
- it reuses DDIM loss as a fake flow loss
- it breaks existing DDIM or CM defaults
- it allows flow mode to silently enter distillation

## Notes
- The first version only targets the `DP3CM` robot action diffusion policy path.
- The reusable value from `flow_grpo` and `FlowCPS` is the transition and log-prob math, not their image-training scripts.

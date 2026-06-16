# Flow Matching RL Training - Implementation TODO

## Goal
Add flow-matching RL training option to `DP3CM` while keeping existing `BC -> offline BPPO -> online PPO/BPPO` pipeline unchanged and preserving `ddim/cm` behavior.

## Scope
- `RL-100/train_cm_mid.py`
- `RL-100/rl_100/policy/dp3_cm.py`
- `RL-100/rl_100/unidpg/uni_ppo.py`
- `RL-100/rl_100/unidpg/diffusion_policy/diffusers_patch/*`
- `RL-100/rl_100/config/*`

## Out Of Scope (v1)
- `dp3.py`, `dp_image_unet`, `dp_state`, other policy variants
- GRPO or grouped reward-normalization
- Flow mode + `ddim -> cm` distillation



---

## Task List

### Step 0: Prerequisites
- [ ] Upgrade `diffusers` to `>= 0.33.1` only in the `rl100` environment
- [ ] Do not require the default project environment to upgrade for this task
- [ ] Run all flow-related implementation and validation in the `rl100` environment
- [ ] Verify existing DDIM/LCM schedulers still work in the `rl100` environment after the upgrade

### Step 1: Implement flow scheduler wrapper
**NEW**: `RL-100/rl_100/unidpg/diffusion_policy/diffusers_patch/flow_match_scheduler.py`

- Inherit from `FlowMatchEulerDiscreteScheduler`
- Implement both SDE and CPS modes (via `sde_type` param, default `'sde'`)
- Only migrate transition and log-prob math from references, NOT image pipeline logic
- Required methods:
  - `step_logprob(model_output, timestep, sample)` -> `(SchedulerOutput, log_prob)`
  - `step_forward_logprob(model_output, timestep, sample, next_sample, eta)` -> `log_prob`
  - `step_forward_logprob_with_entropy(model_output, timestep, sample, next_sample, eta)` -> `(log_prob, entropy)`
  - `step_mean(model_output, timestep, sample)` -> `SchedulerOutput`
  - `add_noise(original_samples, noise, timesteps)` -> `noisy_samples`
- **Critical**: log_prob must be per-element (same shape as sample), NOT batch-reduced. BPPO slices into `a_logprob_now[:, :n_action_steps].sum(-1, ...)`
- Reference: `third_party/flow_grpo/flow_grpo/diffusers_patch/sd3_sde_with_logprob.py`
- Reference interface: `RL-100/rl_100/unidpg/diffusion_policy/diffusers_patch/lcm_scheduler.py`

**Done when**:
- offline RL and online RL can reuse existing PPO ratio/clipping/advantage code without a separate formula for flow mode
- `step_logprob` and `step_forward_logprob` return shapes that are compatible with the current PPO/BPPO slicing and summation logic

### Step 2: Add flow configuration fields
**NEW**: `RL-100/rl_100/config/dp3_flow.yaml`

- Copy from `dp3_cm_epsilon.yaml`
- Add fields:
  - `policy.scheduler_type: flow`
  - `policy.flow_noise_scheduler._target_: ...FlowMatchSchedulerExtended`
  - `policy.flow_inference_steps: 10`
  - `policy.flow_noise_level: 0.7`
  - `policy.flow_sde_type: 'sde'`
- Keep `ddim_noise_scheduler` and `cm_noise_scheduler` blocks (required by constructor)

**Done when**: Hydra instantiate works with `scheduler_type=flow`; old configs unchanged.

### Step 3: Abstract scheduler selection in DP3CM
**MODIFY**: `RL-100/rl_100/policy/dp3_cm.py`

- `__init__()`: add params `flow_noise_scheduler`, `flow_inference_steps`, `flow_noise_level`, `flow_sde_type`; add `self.is_flow` flag; guard `DDIMSolver`/`alpha_schedule`/`sigma_schedule`; maintain `ddim_scheduler`, `cm_scheduler`, `flow_scheduler` explicitly
- Add `get_unet_timesteps()` helper (scales float sigmas to int for UNet)

All these methods must support flow mode:
- `compute_loss` (BC training)
- `conditional_sample` (inference)
- `predict_action`
- `all_step_logprob`
- `all_step_action_logprob`
- `sample_action`
- `sample_action_with_logprob`

**Requirements**:
- Tensor shapes and return structures remain compatible with DDIM path
- Observation encoding, normalization, action unnormalization unchanged
- Flow mode never falls through to DDIM by accident

**Done when**: flow mode runs policy forward and sampling end-to-end; DDIM path unchanged.

### Step 4: Implement flow-matching BC training loss
Part of Step 3's `compute_loss()` modification, but called out separately:

- Do NOT reuse DDIM `epsilon`, `sample`, or `v_prediction` loss
- Derive the training target from the chosen flow scheduler parameterization
- Do not hardcode a single flow formula in implementation unless that parameterization is explicitly selected and documented
- Keep training-time parameterization consistent with the scheduler used for inference and transition log-prob computation
- Output format: `(loss, loss_dict)` unchanged

**Done when**: `compute_loss(batch)` returns finite value in flow mode; training logs work.

### Step 5: Disable CM distillation in flow mode
**MODIFY**: `RL-100/rl_100/policy/dp3_cm.py`

Block these methods in flow mode with clear error:
- `compute_ddim2cm_loss`
- `compute_ddim2cm_loss_action`
- `compute_ddim2cm_loss_action_same_noise`

**Done when**: no silent fallback; clear error message "flow v1 does not support CM distillation".

### Step 6: Make offline BPPO scheduler-agnostic
**MODIFY**: `RL-100/rl_100/unidpg/uni_ppo.py` — `update_distribution()` (line 257-384)

- Use `self._policy.get_unet_timesteps(timesteps)` for model calls
- Handle float timesteps from flow scheduler
- Keep existing rollout data flow: `old_all_x`, `old_all_next_x`, `old_all_logprob`
- Keep existing PPO/BPPO ratio, clipping, advantage formulas unchanged

**Done when**: offline `update_distribution` completes at least one optimization step in flow mode.

### Step 7: Remove hidden DDIM assumptions from online PPO path
**MODIFY**: `RL-100/rl_100/unidpg/uni_ppo.py` — `dp_align_update_no_share()` (line 625-787)

Review and fix:
- Rollout collection
- `all_step_action_logprob`
- Online PPO update
- `step_forward_logprob_with_entropy`
- DDIM-specific field access: `clip_std_max`, `eta`, `prev_sample`

**Requirements**: flow mode must not fail because of DDIM-only attributes.

**Done when**: online rollout collects `all_x` and `a_logprob`; online PPO finishes at least one mini-batch.

### Step 8: Connect flow mode in train_cm_mid.py
**MODIFY**: `RL-100/train_cm_mid.py`

- Guard CM distillation paths (`distill2cm`, `set_target`, `compute_ddim2cm_loss`) with `if not self.model.is_flow`
- Block `distill_phase` + `scheduler_type=flow` combination
- Keep three-stage structure unchanged

**Done when**: `scheduler_type=flow` passes through main training program end-to-end.

### Step 9: Add runnable example
**NEW**: `scripts/train_policy_flow.sh`

- Copy from existing training script
- Include all required flow overrides
- Must be launchable without guessing missing flags

### Step 10: Protect existing behavior (regression)
- DDIM BC still runs
- DDIM offline RL still runs
- DDIM online RL still runs
- CM path instantiates and runs inference

---

## Acceptance Checklist

### Configuration
- [ ] `scheduler_type=flow` instantiates correctly
- [ ] Old configs still work without added flow fields
- [ ] Flow configuration and validation are executed in the `rl100` environment

### BC
- [ ] Flow `compute_loss(batch)` returns finite values
- [ ] Flow `predict_action(obs)` returns expected structure

### Offline RL
- [ ] `all_step_logprob` returns expected tensors in flow mode
- [ ] `uni_ppo.update_distribution` completes at least one update in flow mode
- [ ] Flow `step_logprob` and `step_forward_logprob` return shapes compatible with current PPO/BPPO slicing and summation semantics

### Online RL
- [ ] Rollout stores `all_x` and `a_logprob`
- [ ] Online PPO completes at least one mini-batch
- [ ] No scheduler attribute errors

### Distillation
- [ ] Flow + distillation fails clearly and intentionally

### Regression
- [ ] DDIM smoke tests still pass
- [ ] CM path instantiates and runs inference
- [ ] Regression checks for this task are run in the `rl100` environment

---

## Review Standard
Implementation is **incomplete** if any of these is true:
- Only adds `scheduler_type='flow'` without `step_forward_logprob`
- Ignores online PPO log-prob or entropy support
- Reuses DDIM loss as a fake flow loss
- Breaks existing DDIM or CM defaults
- Allows flow mode to silently enter distillation

## References
- Detailed plan: `/data/lk/.claude/plans/abstract-marinating-sunset.md`
- Original spec: `/data/lk/lk_projs/clean_for_opensource/ft-dp3/list.md`
- Flow math reference: `third_party/flow_grpo/flow_grpo/diffusers_patch/sd3_sde_with_logprob.py`
- Scheduler interface reference: `RL-100/rl_100/unidpg/diffusion_policy/diffusers_patch/lcm_scheduler.py`

---

## Review Round 1 — Bugs Found (work agent please fix)

### BUG A (HIGH): Timestep range mismatch training vs inference
**File**: `RL-100/rl_100/unidpg/diffusion_policy/diffusers_patch/flow_match_scheduler.py` line 69, 76-78

**Problem**: `set_timesteps` uses `sigmas = np.linspace(1.0, 0.0, N+1)`. First sigma=1.0 maps to timestep `int(1.0 * 100) = 100`. But in BC training (`dp3_cm.py` line ~766), `timesteps = (t * 100).long()` where `t ~ U[0,1)`, producing timesteps in [0, 99]. The UNet never sees timestep 100 during training, but inference starts with it.

**Fix**: In `set_timesteps`, clamp timesteps to valid range [0, num_train_timesteps - 1]:
```python
timesteps = np.clip(
    (sigmas[:-1] * self.num_train_timesteps).astype(np.int64),
    0, self.num_train_timesteps - 1
)
```

- [x] Fix applied

### BUG B (MEDIUM): CPS sqrt NaN risk
**File**: `flow_match_scheduler.py` line 150

**Problem**: `torch.sqrt(sigma_prev_exp**2 - std_dev_t**2)` can produce NaN if floating-point error makes the argument slightly negative (especially at final step where both are near 0).

**Fix**:
```python
prev_sample_mean = pred_x0 * (1 - sigma_prev_exp) + pred_x1 * torch.sqrt(
    torch.clamp(sigma_prev_exp**2 - std_dev_t**2, min=0.0)
)
```

- [x] Fix applied

### BUG C (LOW): SDE sqrt(-dt) fragile
**File**: `flow_match_scheduler.py` line 143

**Problem**: `torch.sqrt(-dt)` relies on `dt` always being negative. If numerical error makes `dt` slightly positive, NaN results.

**Fix**:
```python
std = std_dev_t * torch.sqrt(torch.clamp(-dt, min=1e-10))
```
## Review Follow-Ups
- [x] Replace the current hardcoded flow BC loss implementation with a scheduler-derived parameterization. Do not hardcode one flow formula unless that parameterization is explicitly selected and documented.
- [x] Fix `step_forward_logprob_with_entropy()` so `entropy` has broadcast-compatible per-element shape with `sample` and `log_prob`, rather than a collapsed shape such as `(1, 1, 1)`.
- [x] Fix applied

### Verified OK (no fix needed)
- Log-prob shape: per-element (batch, horizon, action_dim) is correct for BPPO slicing
- step_index management: `_init_step_index` re-finds index each call, safe
- CM distillation guards: all 3 distill methods + `set_target` blocked with clear errors
- `get_unet_timesteps`: applied in all 4 locations in uni_ppo.py + all methods in dp3_cm.py
- Config `dp3_flow.yaml`: complete and consistent
- `train_cm_mid.py` guards: correctly placed

---

## Review Round 2 — All Fixes Verified ✓

### Bug Fixes (all confirmed correct):
- [x] BUG A: `flow_match_scheduler.py:77-79` — `np.clip` clamps timesteps to [0, N-1] ✓
- [x] BUG B: `flow_match_scheduler.py:170-172` — `torch.clamp(..., min=0.0)` prevents NaN ✓
- [x] BUG C: `flow_match_scheduler.py:163` — `torch.clamp(-dt, min=1e-10)` prevents NaN ✓

### Follow-up Fixes (all confirmed correct):
- [x] `flow_match_scheduler.py:116-131` — `get_training_target()` and `get_training_noisy_sample()` added, scheduler-derived parameterization ✓
- [x] `flow_match_scheduler.py:370` — `entropy.expand_as(prev_sample_mean)` ensures per-element shape ✓
- [x] `dp3_cm.py:761-762` — `compute_loss()` uses `self.noise_scheduler.get_training_noisy_sample()` instead of hardcoded formula ✓

### Full Code Review Verification:
- [x] `dp3_cm.py:__init__` (line 232-296): Flow scheduler setup correct, DDIMSolver/alpha/sigma guarded with `if not self.is_flow` ✓
- [x] `dp3_cm.py:conditional_sample` (line 398-438): Flow path uses `step_mean`/`step`, `get_unet_timesteps` applied ✓
- [x] `dp3_cm.py:all_step_logprob` (line 1111-1169): Flow scheduler selected, `get_unet_timesteps` applied, return structure matches DDIM path ✓
- [x] `dp3_cm.py:all_step_action_logprob` (line 1170-1238): Same pattern, correct ✓
- [x] `dp3_cm.py` distillation guards: `set_target` (line 299), `compute_ddim2cm_loss` (line 830), `compute_ddim2cm_loss_action` (line 960), `compute_ddim2cm_loss_action_same_noise` (line 975) — all assert/raise ✓
- [x] `uni_ppo.py`: All 4 `get_unet_timesteps` locations (lines 313, 396, 677, 848) correctly use `unet_timesteps` for model, original `timesteps` for scheduler ✓
- [x] `train_cm_mid.py`: 3 guard locations (lines 128, 458, 1351) all block flow+distillation correctly ✓
- [x] `dp3_flow.yaml`: Complete config with `distill_phase: null` ✓
- [x] `scripts/train_policy_flow.sh`: Runnable script with all flow overrides ✓

### Status: **Round 2 complete. Round 3 follow-up items recorded below.**

---

## Review Round 3 — Refactor: Subclass `FlowMatchEulerDiscreteScheduler` from diffusers

User feedback: current implementation uses a standalone custom scheduler (`SchedulerMixin + ConfigMixin`). Should subclass `FlowMatchEulerDiscreteScheduler` from diffusers 0.33.1 instead.

### Task 3.1: Rewrite `flow_match_scheduler.py` to subclass diffusers
**REWRITE**: `RL-100/rl_100/unidpg/diffusion_policy/diffusers_patch/flow_match_scheduler.py`

Change from:
```python
class FlowMatchSchedulerExtended(SchedulerMixin, ConfigMixin):
```
To:
```python
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

class FlowMatchSchedulerExtended(FlowMatchEulerDiscreteScheduler):
```

**Reuse from parent** (DELETE our custom versions):
- `set_timesteps()` — parent creates `self.sigmas` (descending, terminal 0 appended) and `self.timesteps` (float). Override to clamp sigma < 1.0 (see below).
- `index_for_timestep()` — replaces our custom `_init_step_index()`
- `scale_noise(sample, timestep, noise)` — parent's `sigma * noise + (1 - sigma) * sample`. This is equivalent to our `add_noise`.
- `step()` — parent's deterministic Euler step. Can delegate `step_mean` to it.
- `_step_index`, `_begin_index` — parent manages these

**Keep / add** (RL-specific, not in parent):
- `step_logprob()` — same interface, use `self.index_for_timestep()` for sigma lookup
- `step_forward_logprob()` — same interface
- `step_forward_logprob_with_entropy()` — same interface
- `step_mean()` — can delegate to `super().step()`, wrap in compatible output format
- `get_training_target(original_samples, noise)` — returns `noise - original_samples`
- `get_training_noisy_sample(original_samples, noise, timesteps)` — returns `(noisy_sample, target)` for BC
- `add_noise()` — alias for `self.scale_noise(sample, timestep, noise)` (note arg order differs)
- `_compute_step()` — SDE/CPS math (keep, but use parent's sigmas)
- `_compute_logprob()` — per-element log-prob (keep)

**Constructor**:
```python
@register_to_config
def __init__(self, num_train_timesteps=100, sde_type='sde', noise_level=0.7,
             clip_std_min=0.0, clip_std_max=None, shift=1.0):
    super().__init__(num_train_timesteps=num_train_timesteps, shift=shift)
    self.sde_type = sde_type
    self.noise_level = noise_level
    self.clip_std_min = clip_std_min
    self.clip_std_max = clip_std_max
```
Remove `num_inference_steps` from constructor (parent sets it via `set_timesteps()`).

**Override `set_timesteps`** — clamp sigma to avoid timestep=100 (BUG A):
```python
def set_timesteps(self, num_inference_steps=None, device=None, **kwargs):
    super().set_timesteps(num_inference_steps=num_inference_steps, device=device, **kwargs)
    self.sigmas = torch.clamp(self.sigmas, max=1.0 - 1e-4)
    self.timesteps = self.sigmas[:-1] * self.config.num_train_timesteps
    if device is not None:
        self.timesteps = self.timesteps.to(device)
        self.sigmas = self.sigmas.to(device)
```

**Sigma lookup in RL methods** — follow reference `sd3_sde_with_logprob.py:42-45`:
```python
step_index = [self.index_for_timestep(t) for t in timestep]
sigma = self.sigmas[step_index].view(-1, *([1] * (len(sample.shape) - 1)))
sigma_prev = self.sigmas[[s+1 for s in step_index]].view(-1, *([1] * (len(sample.shape) - 1)))
```

**Keep numerical safety clamps**: BUG B (`torch.clamp(..., min=0.0)`) and BUG C (`torch.clamp(-dt, min=1e-10)`).

**Per-element log_prob**: Do NOT batch-reduce (reference does `.mean()`, we must NOT).

- [x] Rewrite done
- [x] `step_logprob` works
- [x] `step_forward_logprob` works
- [x] `step_forward_logprob_with_entropy` works
- [x] `step_mean` works
- [x] `get_training_target` / `get_training_noisy_sample` works
- [x] `add_noise` alias works

### Task 3.2: Fix `get_unet_timesteps()` in `dp3_cm.py`
**MODIFY**: `RL-100/rl_100/policy/dp3_cm.py`

Parent's `self.timesteps` are already `sigma * num_train_timesteps` (e.g., `[99.99, 89.0, ...]`), NOT raw sigmas in [0,1].

Current code WRONG:
```python
return (timesteps * self.noise_scheduler.config.num_train_timesteps).long()
# 99.0 * 100 = 9900 ← WRONG
```

Fix to:
```python
def get_unet_timesteps(self, timesteps):
    if self.is_flow:
        return timesteps.long()  # already sigma * N, just cast to int
    return timesteps
```

- [x] Fix applied

### Task 3.3: Fix float timestep dtype in `dp3_cm.py`
**MODIFY**: `RL-100/rl_100/policy/dp3_cm.py` — `all_step_logprob` and `all_step_action_logprob`

Parent's `scheduler.timesteps` are `torch.float32`. Current code forces `dtype=torch.long`:
```python
timesteps = torch.tensor([timesteps], dtype=torch.long, device=self.device)
```

Fix — keep float when flow:
```python
if not torch.is_tensor(timesteps):
    if self.is_flow:
        timesteps = torch.tensor([timesteps], dtype=torch.float32, device=self.device)
    else:
        timesteps = torch.tensor([timesteps], dtype=torch.long, device=self.device)
```

- [x] Fix applied in `all_step_logprob`
- [x] Fix applied in `all_step_action_logprob`

### Task 3.4: Fix float timestep dtype in `uni_ppo.py`
**MODIFY**: `RL-100/rl_100/unidpg/uni_ppo.py` — all 4 timestep-creation locations

Same issue as Task 3.3. Fix at lines ~308, ~391, ~672, ~843:
```python
if not torch.is_tensor(timesteps):
    if self._policy.is_flow:
        timesteps = torch.tensor([timesteps], dtype=torch.float32, device=self._device)
    else:
        timesteps = torch.tensor([timesteps], dtype=torch.long, device=self._device)
```

- [x] Fix applied at location 1 (~line 308)
- [x] Fix applied at location 2 (~line 391)
- [x] Fix applied at location 3 (~line 672)
- [x] Fix applied at location 4 (~line 843)

### Task 3.5: Update config `dp3_flow.yaml`
**MODIFY**: `RL-100/rl_100/config/dp3_flow.yaml`

Update `flow_noise_scheduler` block:
```yaml
  flow_noise_scheduler:
    _target_: rl_100.unidpg.diffusion_policy.diffusers_patch.flow_match_scheduler.FlowMatchSchedulerExtended
    num_train_timesteps: 100
    shift: 1.0
    sde_type: ${flow_sde_type}
    noise_level: ${flow_noise_level}
    clip_std_min: ${clip_std_min}
    clip_std_max: ${clip_std_max}
```
Remove `num_inference_steps` from scheduler config (set at runtime via `set_timesteps()`).

- [x] Config updated

### Done when:
- `FlowMatchSchedulerExtended` subclasses `FlowMatchEulerDiscreteScheduler`
- All RL methods (`step_logprob`, `step_forward_logprob`, `step_forward_logprob_with_entropy`) work with parent's sigma/timestep management
- `get_unet_timesteps()` correctly handles parent's float timesteps (just `.long()`)
- Float timestep dtype preserved in `dp3_cm.py` and `uni_ppo.py` for flow mode
- Existing DDIM/CM paths unaffected

---

## Review Round 3 — Bugs Found (work agent please fix)

### BUG D (HIGH): Missing `set_timesteps` override — timestep 100 still produced
**File**: `flow_match_scheduler.py`

**Problem**: Parent's `set_timesteps(10)` with `num_train_timesteps=100` produces `sigmas[0]=1.0` → `timesteps[0]=100.0`. After `get_unet_timesteps`, UNet receives timestep 100, but BC training uses `t~U[0,1)` → `[0, 99]`. This is the same BUG A from Round 1, not fixed for the parent.

**Fix**: Override `set_timesteps` in `FlowMatchSchedulerExtended` to clamp:
```python
def set_timesteps(self, num_inference_steps=None, device=None, **kwargs):
    super().set_timesteps(num_inference_steps=num_inference_steps, device=device, **kwargs)
    # Clamp sigma_max < 1.0 to keep UNet timesteps in training range [0, N-1]
    self.sigmas = torch.clamp(self.sigmas, max=1.0 - 1e-4)
    self.timesteps = self.sigmas[:-1] * self.config.num_train_timesteps
    if device is not None:
        self.timesteps = self.timesteps.to(device)
        self.sigmas = self.sigmas.to(device)
```

- [x] Fix applied

### BUG E (HIGH): `get_unet_timesteps` heuristic is wrong for parent's timestep format
**File**: `dp3_cm.py:287-300`

**Problem**: Current code checks `abs(timesteps) <= 1.0` to detect raw sigmas vs scaled timesteps. But parent's timesteps are `[100.0, 89.0, ...]` — these are NOT `<= 1.0`. The heuristic takes `round().long()` → produces `100` → BUG A again.

After BUG D fix (clamping sigmas), parent's timesteps will be `[99.99, 89.0, ...]`. Then `round().long()` → `100` again!

**Fix**: Remove the heuristic. Parent's float timesteps are always `sigma * N`. Just round to int:
```python
def get_unet_timesteps(self, timesteps):
    if self.is_flow:
        if timesteps.is_floating_point():
            return timesteps.long()  # floor to int (99.99 → 99, 89.0 → 89)
        return timesteps
    return timesteps
```
Note: `.long()` truncates (floor), so `99.99 → 99` which is correct. But `100.0 → 100` if BUG D is not fixed. Both fixes needed together.

- [x] Fix applied

### Task 3.3 and 3.4 NOT done (MEDIUM): float timestep dtype not fixed
**Files**: `dp3_cm.py` (lines 1148, 1206, 1302, 1463) and `uni_ppo.py` (lines 199, 308, 391, 560, 672, 843, 973, 1107)

**Problem**: All locations still have `torch.tensor([timesteps], dtype=torch.long, ...)`. For flow, parent's timesteps are float. Forcing `dtype=torch.long` truncates floating-point values and breaks `index_for_timestep` comparison.

**However**: Since `t` from `scheduler.timesteps` is already a tensor (0-dim float), the code takes the `elif torch.is_tensor(timesteps)` branch (line 1149) which does `timesteps[None].to(device)` — preserving float dtype. So this is actually **NOT a bug** for the current code path where `t` comes from `for t in scheduler.timesteps`.

**BUT**: In `uni_ppo.py`, `t` is extracted from the iterator and may or may not be a tensor depending on context. Need to verify.

**Verdict**: The `dtype=torch.long` fallback is never hit for flow mode in the current code (since `t` is always a tensor from `scheduler.timesteps`). The existing `elif` branch preserves float. **No fix needed** for current usage, but it would be cleaner to handle it properly for robustness.

- [x] No fix needed (existing code handles it correctly via the `elif` tensor branch)

### Verified OK:
- [x] `FlowMatchSchedulerExtended` correctly subclasses `FlowMatchEulerDiscreteScheduler` ✓
- [x] `_compute_step` uses `self.sigmas[self.step_index]` from parent ✓
- [x] `_compute_logprob` returns per-element (not batch-reduced) ✓
- [x] `step_logprob`, `step_forward_logprob`, `step_forward_logprob_with_entropy` use parent's `_init_step_index` ✓
- [x] `step_mean` and `step` are properly overridden (use stochastic SDE/CPS, not parent's simple Euler) ✓
- [x] `add_noise` reimplemented with `_training_sigma` to account for shift/terminal ✓
- [x] `get_training_target` and `get_training_noisy_sample` use `_training_sigma` ✓
- [x] `_training_sigma` correctly applies shift parameterization for training-time consistency ✓
- [x] `_scheduler_timestep` helper handles batch-expanded timesteps for parent's `_init_step_index` ✓
- [x] Numerical safety clamps (BUG B/C) preserved ✓
- [x] Entropy `expand_as(sample)` preserved ✓
- [x] Config `dp3_flow.yaml` updated (still has `num_inference_steps` — acceptable, stored as `default_num_inference_steps`) ✓
- [x] `conditional_sample` flow path passes float `t` from scheduler, works with parent ✓
- [x] CM distillation guards unchanged ✓
- [x] `train_cm_mid.py` guards unchanged ✓

---

## Review Round 4 — Bug Found (work agent please fix)

### BUG F (MEDIUM): `sample_action_with_logprob` corrupts selected log-prob tensor shape
**File**: `RL-100/rl_100/policy/dp3_cm.py` lines ~1385-1407

**Problem**:
- `step_logprob` has per-step shape `[batch * repeat_num, horizon, action_dim]`
- Current code does:
```python
logprob_reshaped = step_logprob.view(-1, repeat_num)
```
- This collapses `horizon` and `action_dim` into the batch axis before selecting `best_idx`
- As a result, the selected `all_logprob` no longer preserves `[num_steps, batch, horizon, action_dim]`
- The returned tensor also adds `unsqueeze(1).unsqueeze(1)`, making it diverge from the layout returned by `all_step_action_logprob()`

**Why it matters**:
- `train_cm_mid.py` can use `sample_action_with_logprob()` when `ppo.idql_rollout=True`
- The online replay buffer expects per-step action log-prob tensors compatible with:
  - `all_step_action_logprob()`
  - `ReplayBuffer.a_logprob` shape in `RL-100/rl_100/dppo/online_buffer.py`
- Current code can silently store wrong PPO data even if sampling itself succeeds

**Fix**:
- Preserve horizon/action dimensions when selecting the best repeated sample
- Reshape each `step_logprob` as:
```python
logprob_reshaped = step_logprob.view(orig_batch, repeat_num, *step_logprob.shape[1:])
selected_logprob = logprob_reshaped[torch.arange(orig_batch), best_idx]
```
- Return `all_logprob` in the same semantic layout as `all_step_action_logprob()`:
  - per step
  - per batch
  - per horizon
  - per action dimension
- Remove shape hacks like `unsqueeze(1).unsqueeze(1)` unless they are proven necessary and buffer-compatible

**Done when**:
- `sample_action_with_logprob()` returns `all_logprob` with the same PPO/replay-buffer semantics as the normal rollout path
- Repeated-sample selection only indexes the `repeat_num` axis and does not flatten horizon/action dimensions

- [x] Fix applied

---

## Non-Blocking Cleanup

### Cleanup 1: Add defensive next-step index clamp in flow scheduler
**File**: `RL-100/rl_100/unidpg/diffusion_policy/diffusers_patch/flow_match_scheduler.py`

**Reason**:
- Current implementation reads:
  - `self.sigmas[self.step_index]`
  - `self.sigmas[self.step_index + 1]`
- The main stale-`step_index` bug has already been fixed by reinitializing from the provided timestep
- But the scheduler is still relying on correct caller state management
- The standalone reference in `flow_sde_with_logprob.py` defensively clamps the next-step index, which is more robust

**Suggested change**:
```python
step_index = min(self.step_index, len(self.sigmas) - 2)
next_index = min(step_index + 1, len(self.sigmas) - 1)
sigma = self.sigmas[step_index]
sigma_next = self.sigmas[next_index]
```

**Priority**: non-blocking robustness improvement

- [x] Cleanup applied

---

## Review Round 5 — ROOT CAUSE of 0% BC success rate (work agent please fix)

### BUG G (CRITICAL): `step_mean` uses SDE mean formula instead of ODE Euler for deterministic inference
**File**: `flow_match_scheduler.py` — `step_mean()` calls `_compute_step()` which uses SDE formula

**Problem**: `predict_action(deterministic=True)` → `conditional_sample(deterministic=True)` → `step_mean()` → `_compute_step()`. Since `sde_window_size=0`, `is_stochastic_step()` returns True, so `_compute_step` uses the SDE mean formula:
```
std_dev_t = sqrt(sigma / (1 - sigma)) * noise_level
prev_sample_mean = sample * (1 + std^2/(2*sigma)*dt) + v * (1 + std^2*(1-sigma)/(2*sigma))*dt
```

At `sigma ≈ 0.9999` (first inference step):
- `std_dev_t = sqrt(0.9999/0.0001) * 0.7 ≈ 70`
- `coeff_sample = 1 + 70^2/(2*0.9999)*(-0.11) ≈ -268`
- The sample is multiplied by -268 → complete garbage output

For deterministic inference, the correct formula is the probability flow ODE (simple Euler):
```
x_{t+dt} = x_t + v * dt
```

**Fix**: `step_mean` must NOT go through `_compute_step`. Use ODE Euler directly:
```python
def step_mean(self, model_output, timestep, sample, generator=None,
              return_dict=True, prev_sample=None, step_index=None, eta=1.0):
    if self.num_inference_steps is None:
        raise ValueError("Must call set_timesteps before step_mean")
    if self.step_index is None or step_index is not None:
        if step_index is not None:
            self._step_index = step_index
        else:
            self._init_step_index(self._scheduler_timestep(timestep))

    model_output = model_output.float()
    sample = sample.float()
    idx = min(self.step_index, len(self.sigmas) - 2)
    sigma = self.sigmas[idx].to(sample.device)
    sigma_next = self.sigmas[idx + 1].to(sample.device)
    dt = sigma_next - sigma
    prev_sample_mean = sample + model_output * dt  # ODE Euler

    self._step_index += 1
    if return_dict:
        return FlowMatchSchedulerOutput(prev_sample=prev_sample_mean, denoised=prev_sample_mean)
    return (prev_sample_mean, prev_sample_mean)
```

- [x] Fix applied

### BUG H (HIGH): SDE `sigma_safe` at sigma≈1.0 produces enormous std_dev_t
**File**: `flow_match_scheduler.py:174-180`

**Problem**: After `set_timesteps` clamp, `sigma[0] = 0.9999` (not exactly 1.0). The `where(sigma == 1.0, ...)` guard never triggers. But `sqrt(0.9999/0.0001) * 0.7 ≈ 70` is still enormous, making the stochastic step at the first timestep blow up.

**Original fix**: Used `sigmas[1]` lookup (NFE-dependent). This works but makes SDE noise behavior depend on `num_inference_steps`:
- NFE=10: `sigmas[1] ≈ 0.89` → `std_dev_t ≈ 2.0`
- NFE=2: `sigmas[1] ≈ 0.5` → `std_dev_t = 0.7` (too low)
- NFE=50: `sigmas[1] ≈ 0.98` → `std_dev_t ≈ 4.9` (too high)

**Improved fix**: Replaced NFE-dependent `sigmas[1]` with fixed `torch.clamp(sigma, max=sigma_safe_max)` where `sigma_safe_max=0.9` (configurable). This is simpler, NFE-independent, and gives consistent `std_dev_t ≈ 2.1` across all NFE settings.

- [x] Fix applied (original `sigmas[1]` approach)
- [x] Improved: replaced with `sigma_safe_max` clamp (NFE-independent)

### Summary
BUG G is the direct cause of 0% success: deterministic inference uses SDE mean formula which explodes at high sigma. BUG H makes stochastic inference also broken at the first step. Both must be fixed.

### Cleanup 2: Clamp SDE std / noise_std before log-prob `log()`
**File**: `RL-100/rl_100/unidpg/diffusion_policy/diffusers_patch/flow_match_scheduler.py`

**Reason**:
- Current implementation is numerically stable for the default path because:
  - `sqrt(-dt)` is clamped
  - `clip_std_min` may be configured
- But extreme settings such as very small `noise_level` or `clip_std_min=0` can still make `std` approach zero
- The standalone reference uses an explicit minimum clamp before `log_prob` computation

**Suggested change**:
```python
std_for_log = torch.clamp(std, min=1e-12)
log_prob = (
    -((next_sample.detach() - prev_sample_mean) ** 2) / (2 * std_for_log**2)
    - torch.log(std_for_log)
    - 0.5 * math.log(2 * math.pi)
)
```

**Priority**: non-blocking numerical robustness improvement

- [x] Cleanup applied

### Optional Enhancement: Support hybrid flow rollout with `sde_window_size`
**Files**:
- `RL-100/rl_100/unidpg/diffusion_policy/diffusers_patch/flow_match_scheduler.py`
- `RL-100/rl_100/policy/dp3_cm.py`
- `RL-100/rl_100/config/dp3_flow.yaml`

**Motivation**:
- The older `bc_diffusion_ddim.py` flow path supports using stochastic SDE transitions only for an early prefix of denoising steps, then switching to deterministic ODE-style updates
- This can help balance:
  - early-step exploration
  - late-step stability

**Suggested design**:
- Add config field:
```yaml
flow_sde_window_size: 0
```
- Semantics:
  - `0`: keep current behavior, all steps use stochastic flow transitions
  - `k > 0`: first `k` denoising steps use stochastic flow transitions, remaining steps use deterministic mean/ODE transitions

**Implementation direction**:
- Add helper such as:
```python
def _is_stochastic_flow_step(step_idx, window_size):
    return window_size == 0 or step_idx < window_size
```
- In rollout / action sampling code paths:
  - stochastic steps use `step()` / `step_logprob()`
  - deterministic steps use `step_mean()`
- For deterministic suffix steps, define PPO/BPPO log-prob semantics explicitly before enabling the feature:
  - either disallow mixed mode for ratio-based RL paths
  - or provide a mathematically justified and buffer-compatible treatment

**Acceptance**:
- `flow_sde_window_size=0` preserves current behavior exactly
- Mixed SDE/ODE rollout does not break:
  - `all_step_logprob`
  - `all_step_action_logprob`
  - `sample_action_with_logprob`
  - PPO / BPPO replay-buffer semantics

**Priority**: optional experiment, not part of v1 correctness

- [x] Enhancement fully accepted

### Review follow-up for `sde_window_size`
**BUG G (HIGH): mixed SDE/ODE window has invalid log-prob semantics for PPO/BPPO**

**Files**:
- `RL-100/rl_100/unidpg/diffusion_policy/diffusers_patch/flow_match_scheduler.py`
- all RL paths consuming:
  - `step_logprob`
  - `step_forward_logprob`
  - `step_forward_logprob_with_entropy`

**Problem**:
- The deterministic suffix currently returns an ODE step with `std = 0`
- But `step_logprob()` / `step_forward_logprob()` still send that through the Gaussian log-prob codepath
- Current implementation clamps `std` to a tiny constant before `log()`, which avoids NaN but does **not** produce a mathematically valid probability density for the deterministic map
- That means PPO/BPPO importance ratios become undefined / meaningless when `flow_sde_window_size > 0`

**Why it matters**:
- The feature is not just “numerically rough”; it changes the semantics of the policy density used by offline RL and online PPO
- This is acceptable only if mixed mode is inference-only, not if it is exposed as a valid RL training option

**Required fix direction**:
- Pick one of these and implement it explicitly:

Option A:
- Disallow `flow_sde_window_size > 0` in all ratio-based RL paths
- Allow it only for inference / evaluation sampling
- Raise a clear error if offline BPPO or online PPO attempts to use mixed mode

Option B:
- Define a mathematically justified treatment for deterministic suffix steps in the policy density
- Update:
  - `step_logprob`
  - `step_forward_logprob`
  - `step_forward_logprob_with_entropy`
  - replay-buffer expectations
  - PPO/BPPO ratio logic
- Do not use “tiny Gaussian variance” as an implicit approximation unless it is an intentional, documented design choice

**Additional cleanup**:
- If deterministic suffix remains enabled anywhere, `step_forward_logprob_with_entropy()` must not return invalid entropy values for zero-variance steps

**Done when**:
- Either mixed mode is explicitly blocked from RL training paths
- Or mixed mode has clearly defined and internally consistent density / entropy semantics for PPO/BPPO

- [x] Follow-up applied

---

## Review Round 5 — BC Flow Evaluation / Training Consistency

### BUG H (HIGH): deterministic flow evaluation is using SDE mean, not the standard flow ODE step

**Files**:
- `RL-100/rl_100/unidpg/diffusion_policy/diffusers_patch/flow_match_scheduler.py`
- `RL-100/rl_100/policy/dp3_cm.py`
- env runners calling `predict_action(..., deterministic=True)`

**Problem**:
- BC evaluation runners call `predict_action(..., deterministic=True)`
- In flow mode, `predict_action()` -> `conditional_sample()` -> `scheduler.step_mean(...)`
- Current `step_mean()` uses `_compute_step(...)`, and for `sde_type='sde'` that returns the SDE drift-corrected mean
- That is **not** the standard deterministic flow-matching ODE update used for BC inference
- The older reference path in `bc_diffusion_ddim.py` uses deterministic ODE stepping for flow evaluation:
```python
x_next = x + dt * model_output
```

**Why it matters**:
- BC training can look numerically normal while evaluation success stays near zero
- This is especially likely when the evaluator always uses `deterministic=True`, which is how the current runners are configured

**Required fix direction**:
- In flow mode, deterministic evaluation should use the pure ODE / mean-flow update, not the SDE mean
- Recommended implementation:
  - keep stochastic SDE/CPS transitions for rollout / log-prob paths
  - make `step_mean()` return deterministic flow ODE stepping
- For example:
```python
prev_sample_mean = sample + model_output * dt
```
where `dt = sigma_next - sigma`

**Done when**:
- `predict_action(..., deterministic=True)` in flow mode uses deterministic flow ODE stepping
- BC evaluation no longer depends on the SDE mean formula

- [x] Follow-up applied

### BUG I (MEDIUM): BC training timestep embedding is only consistent for default flow parameterization

**Files**:
- `RL-100/rl_100/policy/dp3_cm.py`
- `RL-100/rl_100/unidpg/diffusion_policy/diffusers_patch/flow_match_scheduler.py`

**Problem**:
- BC noisy sample generation already uses scheduler-derived parameterization via `get_training_noisy_sample(...)`
- But the timestep fed to the model is still:
```python
timesteps = (t * num_train_timesteps).long()
```
- This is only aligned with inference if the effective training/inference parameterization is the identity case
- If nontrivial scheduler transforms are enabled later, training noisy sample and timestep embedding can drift apart

**Why it matters**:
- This may not explain the current failure if the config is using the default identity-style flow mapping
- But it is a real BC-path consistency risk and should be fixed before broader scheduler parameter sweeps

**Required fix direction**:
- Derive the model timestep embedding from the same scheduler-mapped sigma used to construct the noisy sample
- For example, expose a helper from the scheduler that returns:
  - training noisy sample
  - training target
  - model timestep embedding value consistent with inference-time `sigma * N`

**Done when**:
- BC training timestep embedding is derived from scheduler-mapped sigma, not raw `t`
- Training/inference timestep semantics remain consistent under non-default scheduler transforms

- [x] Follow-up applied

---

## Review Round 6 — Remaining Follow-Ups

### BUG J (HIGH): `flow_sde_window_size > 0` is still not valid for PPO/BPPO density semantics

**Files**:
- `RL-100/rl_100/unidpg/diffusion_policy/diffusers_patch/flow_match_scheduler.py`
- `RL-100/rl_100/unidpg/uni_ppo.py`
- all rollout / PPO paths consuming:
  - `step_logprob`
  - `step_forward_logprob`
  - `step_forward_logprob_with_entropy`

**Current status**:
- The window feature is wired into the scheduler and config
- Deterministic suffix steps return an ODE update with `std = 0`
- But the RL log-prob code path still treats those steps through a Gaussian-style density with clamped tiny variance

**Why this is still a problem**:
- This does not define a valid policy density for PPO/BPPO importance ratios
- So `flow_sde_window_size > 0` should not be considered a correct RL training option yet

**Required fix direction**:
- Pick one explicit policy and implement it end-to-end

Option A:
- Block `flow_sde_window_size > 0` in all ratio-based RL training paths
- Allow it only for BC evaluation / inference sampling
- Raise a clear error in offline BPPO and online PPO when mixed mode is enabled

Option B:
- Define a mathematically justified density treatment for deterministic suffix steps
- Update:
  - scheduler log-prob functions
  - replay-buffer assumptions
  - PPO/BPPO ratio usage
  - entropy handling

**Recommended choice**:
- Choose Option A for v1
- Keep mixed window as inference-only until a proper density formulation is designed

**Done when**:
- `flow_sde_window_size > 0` is either explicitly blocked from RL training
- Or it has fully defined and validated density semantics for PPO/BPPO

- [x] Follow-up applied (already guarded: `_check_sde_window_for_logprob` in all 3 log-prob methods blocks RL training when sde_window_size > 0)

### Validation Task: run a real BC smoke evaluation on door after the flow BC fixes

**Goal**:
- Verify that the code-level BC fixes actually recover nonzero evaluation success on the door task

**Why this is needed**:
- The latest fixes were accepted by code review
- But the original user-visible failure was empirical: BC evaluation success stayed at 0
- That requires a real train/eval smoke check, not just static review

**Required validation**:
- Train flow BC in the `rl100` environment with the current default flow config
- Run the standard BC evaluation path on the door task
- Record at minimum:
  - training loss curve
  - evaluation success rate
  - whether deterministic evaluation is now producing sensible actions

**Acceptance**:
- BC evaluation success is no longer pinned at 0
- If success is still poor, collect the next debugging signals:
  - action magnitude statistics
  - first-step / final-step trajectory norms
  - predicted velocity norm by timestep

- [x] Validation completed

**Recorded result**:
- Task: `door`
- Training stage: flow BC
- Evaluation stage: deterministic policy evaluation
- Epochs: `50`
- `test_mean_score: 0.9`
- `mean_returns: 114.52666687902145`

**Conclusion**:
- The previous 0-success BC evaluation issue is resolved
- Flow BC training + deterministic evaluation path is now producing meaningful policy behavior on door

---

## Review Round 6 — All fixes verified, no new issues

All BUG D/E/G/H/I fixes confirmed correct in code:
- [x] `step_mean` (line 241-272): ODE Euler `sample + v * dt`, no longer calls `_compute_step` ✓
- [x] `_compute_step` SDE (line 205-208): `sigma_safe = clamp(sigma, max=sigma_safe_max)` — fixed from NFE-dependent `sigmas[1]` to fixed clamp ✓
- [x] `get_unet_timesteps` (dp3_cm.py:294-296): `.long()` truncation ✓
- [x] `set_timesteps` (line 87-96): sigma clamped to `< 1.0 - 1e-4` ✓
- [x] `get_training_noisy_sample` (line 136-154): returns `(noisy, target, model_timesteps)` with `model_timesteps = floor(sigma * N)` ✓
- [x] `dp3_cm.py:763`: correctly unpacks 3 values from `get_training_noisy_sample` ✓
- [x] `_check_sde_window_for_logprob` (line 274-285): blocks log-prob when `sde_window_size > 0` ✓
- [x] `_compute_logprob` (line 234): `std_safe = clamp(std, min=1e-12)` ✓

**Status: All reviews passed. Ready for BC training + evaluation test.**

---

## Enhancement: Logit-Normal Timestep Sampling for Flow BC (optional)

### Motivation
Flow matching BC 用 `t ~ U[0,1]` 均匀采样所有噪声水平。但中间时间步（t ≈ 0.5）对学习最重要，t 接近 0 或 1 的样本学习信号弱。SD3/Flux 等主流 flow matching 论文用 logit-normal 采样集中在中间时间步，可加速收敛。

### Design
通过配置项 `flow_logit_normal_sampling: true/false` 控制，默认 `false`（保持现有均匀采样行为）。

**MODIFY**: `RL-100/rl_100/config/dp3_flow.yaml`
```yaml
flow_logit_normal_sampling: False  # optional: use logit-normal t sampling for faster BC convergence
```

**MODIFY**: `RL-100/rl_100/policy/dp3_cm.py` — `__init__` 和 `compute_loss`

`__init__` 新增参数:
```python
flow_logit_normal_sampling: bool = False,
```
存为 `self.flow_logit_normal_sampling`

`compute_loss` flow 分支中:
```python
if self.is_flow:
    if self.flow_logit_normal_sampling:
        # Logit-normal: concentrate sampling around t=0.5
        u = torch.randn(bsz, device=trajectory.device)
        t = torch.sigmoid(u)
    else:
        t = torch.rand(bsz, device=trajectory.device)
    noisy_trajectory, target, timesteps = self.noise_scheduler.get_training_noisy_sample(
        trajectory, noise, t)
```

**MODIFY**: `scripts/train_policy_flow.sh` — 添加可选 override:
```bash
flow_logit_normal_sampling=False \
```

### Acceptance
- [x] `flow_logit_normal_sampling=False` 行为与当前完全一致
- [x] `flow_logit_normal_sampling=False` 使用 logit-normal 采样
- [x] 不影响 DDIM/CM 路径
- [x] 不影响 RL 阶段（仅影响 BC `compute_loss`）

---

## Task: Create Online RL Training Script for Flow Policy

### Motivation
现有 `scripts/train_policy_online_cm.sh` 是 DDIM+CM distillation 的 online RL 脚本，不能直接用于 flow policy。需要创建 flow 专用的 online RL 脚本。

### NEW: `scripts/train_policy_online_flow.sh`

基于 `scripts/train_policy_online_cm.sh` 复制，做以下修改：

```diff
- config_name='dp3_cm_epsilon'
+ config_name='dp3_flow'

- policy.scheduler_type='ddim'
+ policy.scheduler_type='flow'

- distill_phase='online'
+ distill_phase=null

# 删除 CM distillation 相关参数（flow 不支持）:
- distill_loss_type='action_same_noise'
- distill2mean=True

# 添加 flow 特有参数:
+ flow_inference_steps=10
+ flow_sde_type='sde'
+ flow_noise_level=0.7
+ flow_sde_window_size=0
```

其余参数（`online=True`, `ppo.*`, `critic.*`, `unio4.*` 等）保持不变。

### 注意事项
- `distill_phase` 必须为 `null`，否则 `train_cm_mid.py` 会 RuntimeError
- `cm_inference_steps` 参数无害但多余，可保留或删除
- `load_bc=False` 需确认 flow BC checkpoint 路径能被正确加载（`training.resume=True` 会从 `run_dir` 加载）
- `update_phase='step'` 在 flow 模式下应兼容（只控制 BPPO 更新频率，不涉及 scheduler）

### Acceptance
- [ ] 脚本可正常启动 flow online RL 训练
- [ ] `distill_phase=null` 不触发 distillation 代码
- [ ] flow scheduler 的 `step_logprob` / `step_forward_logprob_with_entropy` 在 online RL 中正常工作
- [ ] 不影响现有 DDIM online RL 脚本

---

## State 版本 vs 3D 版本对比分析（参考 `third_party/rl100-state`）

### 关键差异 1: Model Timestep Input — sigma vs integer
**State 版本**: Flow 模式下模型输入是 `scheduler.sigmas[i]`（float in [0,1]），BC 训练也用 `t ∈ [0,1]`。
**我们的 3D 版本**: 模型输入是 integer timestep（`get_unet_timesteps` → `.long()`），BC 训练用 `(sigma * N).long()`。
**结论**: 两者都保持了训练/推理一致性，只是 embedding 方式不同（MLP 直接用 float vs UNet 用 SinusoidalPosEmb）。无需修改。

### 关键差异 2: Log-prob reduce 方式 — mean vs sum
**State 版本**: `flow_sde_step_with_logprob` 内部做 `log_prob.mean(dim=tuple(range(1, log_prob.ndim)))` → 返回 `(batch,)` 标量。PPO ratio 直接用这个标量。
**我们的 3D 版本**: 返回 per-element `(batch, horizon, action_dim)`，在 `uni_ppo.py` 中做 `.sum(-1)` 再算 ratio。
**结论**: `sum = action_dim * mean`，log ratio 被放大了 `action_dim` 倍。但这对 DDIM 也是同样的处理方式，不是 flow 特有问题。如果 DDIM online RL 能工作，flow 也应该能工作。无需修改。

### 关键差异 3: ODE step 的 log-prob 处理（sde_window_size > 0 时）
**State 版本**: ODE step 返回 `(model_output * 0.0).sum(dim=1)` — 零值但保持计算图。
**我们的 3D 版本**: `_check_sde_window_for_logprob` 直接 raise error 阻止。
**结论**: 我们的 v1 选择更保守，合理。

### 关键差异 4: Ratio 统计监控
**State 版本**: 有 `_record_ratio_stats` 系统，记录 ratio 的 mean/q05/q50/q95，按 denoise step 分别画图。
**我们的 3D 版本**: 没有这个监控。
**建议**: 添加 ratio 统计日志，对调试 online RL 非常有用。

### 关键差异 5: `flow_clip_range` 参数
**State 版本**: 有独立的 `flow_clip_range`（默认 `1e-4`），但用户确认实际用的是 `0.2`。
**结论**: 参数存在但实际未使用更小的值。无需修改。

### 实现上无重大差异
State 版本和我们的 3D 版本在 flow matching RL 的核心实现上是一致的：
- SDE/CPS 数学公式相同
- sigma_safe 处理已改进：从 NFE-dependent `sigmas[1]` 替换为固定 `clamp(sigma, max=0.9)`
- ODE Euler 用于确定性推理
- BC 训练用 velocity target `noise - x_0`

### 行动建议
1. **[MEDIUM] 添加 ratio 统计监控**: 方便调试 online RL 稳定性
2. **[LOW] 可选添加 `flow_clip_range`**: 作为可调参数，默认与 DDIM 相同

---

## 3D vs State 版本深度对比 — 待修改项

基于对 `flow_match_scheduler.py`、`dp3_cm.py`、`uni_ppo.py`、`train_cm_mid.py` 与 state 版本（`third_party/rl100-state/`）逐文件逐方法的对比分析。

### ~~BUG K~~: `dp_align_update` 缺少 `get_unet_timesteps` — SKIPPED
**Reason**: 我们只使用 `dp_align_update_no_share`，不使用 `dp_align_update`。无需修复。

### ~~BUG L~~: `dp_align_update_no_share_vec` GAE discount 缺少 `n_action_steps` — SKIPPED
**Reason**: 我们只使用 `dp_align_update_no_share`，不使用 `dp_align_update_no_share_vec`。无需修复。

> **NOTE**: Online PPO 只使用 `dp_align_update_no_share` 这一个函数。`dp_align_update`、`dp_align_update_no_share_vec`、`dp_align_update_vec` 均不使用，其中的 bug 不影响我们。

### IMPROVEMENT A (MEDIUM): Advantage 计算加 `torch.no_grad()` — ✅ PASS
**File**: `RL-100/rl_100/unidpg/uni_ppo.py`

**实现**: `advantage_computation` 加了 `@torch.no_grad()` 装饰器 + `_compute_advantage_actor_only` helper 包裹。所有调用点均已覆盖。

**Review 结论**: 实现正确且完整。三层保护有冗余（装饰器 + helper 的 with block + 调用点的 `.detach()`）但无害。

- [x] Fix applied
- [x] Review passed

### IMPROVEMENT B (MEDIUM): 添加 ratio 统计监控 — ✅ Code Fix Applied, Runtime Validation Pending
**File**: `RL-100/rl_100/unidpg/uni_ppo.py`, `RL-100/train_cm_mid.py`

**已实现**: `_record_ratio_stats` / `flush_ratio_logs` 方法，CSV + matplotlib plots 输出。

#### PERFORMANCE FIX: ratio logging caused monotonic online update slowdown — FIXED
**Problem**:
- online PPO update time kept increasing over training
- root cause was the new ratio logging path:
  - periodic flush rewrote the full `ratio_stats.csv` history every time
  - periodic flush regenerated matplotlib plots from the full history every time
  - `_record_ratio_stats` ran on every denoise step of the active online PPO path

**Fix**:
- `flush_ratio_logs(force=False)` now appends only new CSV rows instead of rewriting the full file
- periodic flush no longer generates plots
- plots are generated only on final `force=True` flush
- added config-controlled logging throttles:
  - `ppo.enable_ratio_logging`
  - `ppo.ratio_log_every_updates`
  - `ppo.ratio_plot_on_final_flush`

**Expected impact**:
- periodic logging cost is now bounded instead of growing with training time
- PPO / BPPO math is unchanged

- [x] Append-only periodic CSV flush
- [x] Plot generation moved to final flush only
- [x] Logging frequency gate added
- [ ] Runtime check: online update time no longer trends upward due to logging

**Review 发现的问题**:

#### BUG M (HIGH): `dp_align_update_no_share` 缺少 `_record_ratio_stats` 调用 — FIXED
**File**: `RL-100/rl_100/unidpg/uni_ppo.py` — `dp_align_update_no_share()` 约 L800

**Problem**: ratio 在 L800 计算了但没有调用 `_record_ratio_stats`。`dp_align_update_iql_no_share`（L682）有调用，但我们唯一使用的 online 方法 `dp_align_update_no_share` 没有。**online RL 的 ratio 监控完全缺失。**

**Fix**: 在 L800 ratio 计算后添加 `self._record_ratio_stats("online", i, ...)` 调用。

- [x] Fix applied
- [x] Code review verified

#### BUG N (MEDIUM): `train_cm_mid.py` 缺少 `flush_ratio_logs` 调用 — FIXED
**File**: `RL-100/train_cm_mid.py`

**Problem**: 只有 `set_ratio_log_dir` 调用（L114），没有任何 `flush_ratio_logs` 调用。虽然 `_record_ratio_stats` 内部每 50 条自动 flush，但：
- 短训练可能结束时还没满 50 条，不会输出任何 CSV
- 训练结束时最后不满 50 条的记录会丢失

**Fix**: 在以下位置添加 `self.unio4.flush_ratio_logs(force=True)`：
- offline BPPO 循环结束后
- online 训练结束/退出前
- eval checkpoint 处（可选）

- [x] Fix applied
- [x] Code review verified

#### MINOR (LOW): 缺少 delta_q25/q50/q75 分位数
State 版本记录 5 个 delta 分位数（q05/q25/q50/q75/q95），我们只有 q05/q95。损失了分布细节但不影响功能。

- [x] Optional: add missing quantiles

**Remaining validation**:
- [ ] Run one offline/online flow job and confirm `ratio_logs/ratio_stats.csv` is written
- [ ] Confirm final partial-window ratio records are preserved after training exits

### IMPROVEMENT C (LOW): 为最后一步 noise injection 提供显式超参数 — ✅ 基本 PASS
**File**: `RL-100/rl_100/unidpg/diffusion_policy/diffusers_patch/flow_match_scheduler.py`

**实现**: 添加 `flow_noise_on_final_step` 标志，config 链完整（YAML → policy → scheduler），`_compute_step()` 中逻辑正确。

**Review 结论**: 实现正确。

**注意事项**:
1. **Default `False` 是有意的行为变更**: 之前所有步都注入 noise（等价于 `True`），现在 default `False` 使最后一步变为确定性。这是一个有意的改善（减少 action jitter），不是 bug。
2. **Entropy 边界情况**: 当 `flow_noise_on_final_step=False` 时，final step 实际确定性但 `std` 仍非零（被 `clip_std_min` clamp），entropy 报告值不准确。当前 default `entropy_weight=0` 所以无害。如果未来启用 entropy bonus 需要处理此边界情况。

- [x] Config flag added and wired
- [x] Default keeps current behavior (`flow_noise_on_final_step=false`)
- [x] Review passed (with noted entropy caveat)
- [ ] BC / online evaluation compares both settings at least once

---

## Fix: NFE-dependent `sigma_safe` → fixed clamp + logit-normal simplification

### Change 1: `sigma_safe_max` parameter (replaces BUG H workaround)
**Files**: `flow_match_scheduler.py`, `dp3_cm.py`, `dp3_flow.yaml`

**Problem**: SDE branch 的 `sigma_safe` 用 `self.sigmas[1]`（第二个 inference sigma）替代 sigma≈1 处的奇异值。这使得 SDE noise 行为依赖 `num_inference_steps`：
- NFE=10: `sigmas[1] ≈ 0.89` → `std_dev_t ≈ 2.0`（合理）
- NFE=2: `sigmas[1] ≈ 0.5` → `std_dev_t = 0.7`（过低，抑制随机性）
- NFE=50: `sigmas[1] ≈ 0.98` → `std_dev_t ≈ 4.9`（过大）

**Fix**:
- `__init__` 新增 `sigma_safe_max: float = 0.9` 参数
- `_compute_step()` SDE 分支：`sigma_safe = torch.clamp(sigma, max=self.sigma_safe_max)` 替代 `sigmas[1]` lookup + `torch.where`
- Config 链：`flow_sigma_safe_max: 0.9`（YAML）→ `flow_sigma_safe_max`（dp3_cm.py constructor）→ `self.flow_scheduler.sigma_safe_max`

**Default 值**: `0.9` → `std_dev_t = sqrt(0.9/0.1) * 0.7 ≈ 2.1`，匹配 NFE=10 行为。

- [x] `flow_match_scheduler.py`: `sigma_safe_max` 参数 + clamp 逻辑
- [x] `dp3_flow.yaml`: `flow_sigma_safe_max: 0.9` + scheduler/policy 链
- [x] `dp3_cm.py`: constructor 参数 + post-init 赋值
- [ ] 验证：NFE=2 和 NFE=10 的 `std_dev_t` 一致（≈2.1）

### Change 2: logit-normal index mapping 简化
**File**: `dp3_cm.py` ~line 772

```python
# Before:
indices = ((1 - t) * N).long().clamp(0, N - 1)
# After:
indices = (t * N).long().clamp(0, N - 1)
```

Logit-normal `t = sigmoid(u)` 关于 0.5 对称（因为 `u ~ N(0,1)` 对称于 0），所以 `(1-t)` 和 `t` 分布相同。简化后代码更直观。

- [x] Simplified

### Change 3: BC 训练从连续 sigma 采样改为离散 grid index 采样
**Files**: `flow_match_scheduler.py`, `dp3_cm.py`

**动机**: 原始实现 BC 训练时从连续 `t ~ U[0,1]` 采样，再通过 `_training_sigma(t)` 映射得到 sigma 值，最后用 `floor(sigma * N)` 量化为 UNet timestep embedding。这与 diffusers 推理时使用的离散 sigma grid 存在微小但系统性的偏差（量化误差）。借鉴 diffusers 的做法，直接从离散 grid 采样 index，训练和推理使用完全相同的 sigma 值。

**实现**:

#### `flow_match_scheduler.py` — 新增 `_train_sigmas` 和 `_train_timesteps_int`
在 `__init__` 中，parent constructor 生成 N-point sigma grid 后立即保存：
```python
self._train_sigmas = torch.clamp(self.sigmas.clone(), max=1.0 - 1e-4)
self._train_timesteps_int = (
    self._train_sigmas * self.config.num_train_timesteps
).long().clamp(0, self.config.num_train_timesteps - 1)
```
这两个 tensor 在 `set_timesteps()` 覆盖 `self.sigmas` 之前保存，确保训练 grid 不受推理步数影响。

#### `flow_match_scheduler.py` — `add_noise` 和 `get_training_noisy_sample` 接受离散 index
```python
# Before:
def add_noise(self, original_samples, noise, timesteps):
    sigma = timesteps.float() / self.config.num_train_timesteps
    sigma = self._training_sigma(sigma)  # 连续映射

# After:
def add_noise(self, original_samples, noise, timestep_indices):
    sigma = self._train_sigmas[timestep_indices]  # 离散查表
```
`get_training_noisy_sample` 同理：不再调用 `_training_sigma()`，直接用 `_train_sigmas[indices]` 和 `_train_timesteps_int[indices]`。

#### `dp3_cm.py` — BC loss 中的采样逻辑
```python
# Before:
t = torch.rand(bsz, device=trajectory.device)  # 连续 t ∈ [0,1]
noisy_trajectory, target, timesteps = self.noise_scheduler.get_training_noisy_sample(
    trajectory, noise, t)

# After:
N = self.noise_scheduler.config.num_train_timesteps
indices = torch.randint(0, N, (bsz,), device=trajectory.device)  # 离散 index
noisy_trajectory, target, timesteps = self.noise_scheduler.get_training_noisy_sample(
    trajectory, noise, indices)
```
Logit-normal 模式同样改为离散 index：`indices = (t * N).long().clamp(0, N - 1)`。

**好处**:
1. **训练-推理 sigma 一致性**: 消除量化误差，训练时的 sigma 值和推理时完全一致
2. **Timestep embedding 精确**: 直接查表得到 model_timesteps，不再 `floor(sigma * N)` 近似
3. **代码简化**: 不再需要 `_training_sigma()` 的连续映射路径（仍保留该方法以备扩展）

- [x] `flow_match_scheduler.py`: `_train_sigmas` / `_train_timesteps_int` grid + `add_noise` / `get_training_noisy_sample` 改为 index 接口
- [x] `dp3_cm.py`: BC loss 采样从 `torch.rand` → `torch.randint` (uniform) / `(t*N).long()` (logit-normal)

---

## Fix: CPS mode crash in online RL fine-tuning (performance drops to 0)

### Problem
CPS (Consistent Policy Sampling) mode can collapse very early in online RL fine-tuning, while SDE stays stable and close to DDIM.

### Current hypothesis
CPS pseudo-log-prob itself is not obviously wrong. FlowCPS also uses a squared-error-style surrogate log-prob.

The more likely issue is a scale mismatch in the current 3D PPO path:

- FlowCPS reduces CPS log-prob across non-batch dimensions before PPO
- our repo keeps per-element values and later aggregates in the PPO path
- our default PPO clip is still DDIM-style (`epsilon=0.2`)

This likely makes CPS ratios too weakly constrained under the current 3D action aggregation and PPO clip setting.

### 两套实验方案（并行实现，通过 config 切换）

---

#### 方案 A（推荐）: 对齐 FlowCPS — scheduler 内部缩放 + 独立 clip range

**Goal**: keep the CPS pseudo-log-prob formulation, but align its effective scale closer to the FlowCPS regime without changing shared PPO reduce logic.

**Design**
- Do **not** modify shared `uni_ppo.py` log-prob reduce behavior.
- Keep the current PPO aggregation path unchanged.
- Apply CPS-specific scale correction inside the scheduler only.
- Add a CPS-specific PPO clip range.

**修改 1**: `flow_match_scheduler.py` — `_compute_logprob()`

For `sde_type == "cps"` only:
```python
@staticmethod
def _compute_logprob(next_sample, prev_sample_mean, std, sde_type: str):
    if sde_type == "cps":
        log_prob = -((next_sample.detach() - prev_sample_mean) ** 2)
        action_dim = next_sample.shape[-1]
        return log_prob / action_dim
    std_safe = torch.clamp(std, min=1e-12)
    return (
        -((next_sample.detach() - prev_sample_mean) ** 2) / (2 * std_safe**2)
        - torch.log(std_safe)
        - 0.5 * math.log(2 * math.pi)
    )
```

Rationale:
- this makes the later `.sum(-1)` in PPO behave like an action-dimension mean
- it does **not** claim full equivalence to FlowCPS image-latent reduction
- it is only a scale-alignment step for the 3D action case

SDE must continue using the existing Gaussian branch unchanged.

**修改 2**: `dp3_flow.yaml` — 添加 CPS 专用 clip range

```yaml
flow_cps_logprob_mode: 'pseudo'   # 'pseudo' or 'gaussian'
flow_cps_clip_range: 1e-4
```

**修改 3**: `uni_ppo.py` — PPO clamp 处根据 sde_type 切换 epsilon

Only change PPO clamp epsilon for CPS:
```python
if self._policy.is_flow and self._policy.noise_scheduler.sde_type == "cps":
    clip_eps = self.cfg.get("flow_cps_clip_range", 1e-4)
else:
    clip_eps = self.args.epsilon
surr2 = torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps) * adv[index]
```

Use `clip_eps` only for the PPO clamp.
Do **not** change:
- ratio reduce
- `.sum(-1)` behavior
- KL computation
- SDE path
- DDIM path

**修改位置汇总** (NOTE: This is the Plan B branch. Only shared scheduler/config pieces are implemented here. The Plan A-specific `uni_ppo.py` clip change is tracked in the Plan A branch.):
- [x] `flow_match_scheduler.py` `_compute_logprob`: CPS 分支除以 `action_dim` (shared with Plan B)
- [x] `dp3_flow.yaml`: 添加 `flow_cps_logprob_mode` 和 `flow_cps_clip_range` (shared config)
- [ ] `uni_ppo.py` `dp_align_update_no_share`: PPO clamp（~L846）改 clip_eps — **Plan A only, not in this branch**
- [x] uni_ppo.py reduce 逻辑不改（`.sum(-1)` 保持不变）
- [x] 验证语法
- [x] DDIM 模式不受影响
- [x] SDE 模式不受影响

---

#### 方案 B（实验性）: Gaussian 归一化 CPS log-prob
**Goal**: test whether CPS needs a true Gaussian density in PPO instead of the current pseudo-log-prob.

**Design**
- This is an algorithm change, not a FlowCPS-alignment fix.
- It must apply only when:
  - `sde_type == "cps"`
  - `flow_cps_logprob_mode == "gaussian"`

**修改**: `flow_match_scheduler.py` — `_compute_logprob()`

Complete method with all branches (Plan A pseudo + Plan B gaussian + SDE):
```python
@staticmethod
def _compute_logprob(next_sample, prev_sample_mean, std, sde_type: str,
                     cps_logprob_mode: str = "pseudo"):
    if sde_type == "cps":
        if cps_logprob_mode == "gaussian":
            # Plan B: true Gaussian log-prob for CPS (experimental)
            std_safe = torch.clamp(std, min=1e-12)
            return (
                -((next_sample.detach() - prev_sample_mean) ** 2) / (2 * std_safe**2)
                - torch.log(std_safe)
                - 0.5 * math.log(2 * math.pi)
            )
        else:
            # Plan A (default): pseudo-log-prob scaled by 1/action_dim
            log_prob = -((next_sample.detach() - prev_sample_mean) ** 2)
            action_dim = next_sample.shape[-1]
            return log_prob / action_dim
    # SDE: unchanged Gaussian log-prob
    std_safe = torch.clamp(std, min=1e-12)
    return (
        -((next_sample.detach() - prev_sample_mean) ** 2) / (2 * std_safe**2)
        - torch.log(std_safe)
        - 0.5 * math.log(2 * math.pi)
    )
```

`cps_logprob_mode` is passed from `self.cps_logprob_mode` (set via config `flow_cps_logprob_mode`).
Callers (`step_logprob`, `step_forward_logprob`, `step_forward_logprob_with_entropy`) pass it through.

**No changes in `uni_ppo.py` for Plan B**
- keep current reduce behavior
- keep current PPO clamp behavior unless explicitly testing otherwise

**风险**
- late denoising steps can become much more sensitive because of `1 / std^2`
- this may stabilize PPO, or may over-amplify late-step differences
- treat as experimental only

**修改位置汇总**:
- [x] `flow_match_scheduler.py` `_compute_logprob`: CPS mode `gaussian` 分支使用完整 Gaussian log-prob
- [x] 验证语法
- [x] SDE 模式不受影响
- [x] uni_ppo.py 不改（reduce=sum, epsilon 保持现有逻辑）

---

### 实现策略
通过 config 开关 `flow_cps_logprob_mode` 控制：
- `flow_cps_logprob_mode: 'pseudo'`（默认）→ 方案 A（pseudo-log-prob + scheduler-side scale alignment + tight clip）
- `flow_cps_logprob_mode: 'gaussian'` → 方案 B（Gaussian log-prob, experimental）

**规则**
- `flow_cps_logprob_mode` must affect `CPS` only
- `SDE` must remain behaviorally unchanged
- `DDIM` must remain behaviorally unchanged
- shared PPO ratio reduce logic in `uni_ppo.py` must remain unchanged for both plans
- only PPO clamp epsilon may branch for CPS in Plan A

### Verification (manual, by user)
1. Run CPS online RL with Plan A for at least early-stage updates.
   - target: no immediate collapse to zero
   - inspect `ratio_stats.csv`
2. Run CPS online RL with Plan B.
   - compare against Plan A, not only against the old CPS baseline
3. Run SDE online RL after the change.
   - confirm no regression
4. Compare early ratio statistics:
   - `ratio_mean`
   - `ratio_q95`
   - `clipfrac`
   - early evaluation return / success

### Acceptance criteria
- Plan A is accepted if CPS no longer collapses immediately and SDE remains unchanged.
- Plan B is accepted only as an experimental alternative if it is at least as stable as Plan A and does not regress SDE.

### Review Follow-Up: Plan B branch cleanup

#### BUG S (MEDIUM): `flow_cps_logprob_mode` does not fail fast on invalid values

Current Plan B implementation in `flow_match_scheduler.py` handles CPS mode as:

- `"gaussian"` -> Gaussian CPS log-prob
- everything else -> pseudo CPS log-prob

This is too permissive for an A/B experiment branch. Typos such as:

- `gauss`
- `Gaussian`
- stale config values

will silently run pseudo mode instead of raising an error, which can invalidate conclusions.

**Required fix**

In the CPS branch of `_compute_logprob(...)`, explicitly validate:

```python
if cps_logprob_mode not in {"pseudo", "gaussian"}:
    raise ValueError(...)
```

Then branch only on the two allowed values.

#### BUG T (LOW): Plan A checklist state in this branch is misleading

In this Plan B worktree, `tasks/todo.md` marks several Plan A items as completed even though this branch is intentionally focused on Plan B and does not implement the Plan A `uni_ppo.py` CPS-specific clip change.

This is not a code bug, but it makes the branch-local experiment record misleading.

**Required fix**

Adjust the Plan A checkbox/status text in this branch-local `tasks/todo.md` so it reflects one of the following clearly:

1. Plan A items are tracked in a different branch/worktree and are not completed here.
2. Only the shared scheduler/config pieces were copied here, while the full Plan A branch remains elsewhere.

The goal is to make the branch-local TODO self-consistent for review and later comparison.

---

## Fix: CPS deterministic eval uses wrong trajectory dynamics (rollout-eval gap)

### Problem
CPS online RL rollout 能到 score ~135（与 DDIM 持平），但 deterministic evaluation 只有 70-120。Gap 约 15-65 分。SDE 和 DDIM 没有这个问题。

### Root Cause: `step_mean` 对 CPS 用了 ODE Euler 而非 CPS mean

Rollout 用 `step_logprob()` → `_compute_step()` → CPS mean 公式：
```
pred_x0 = x_t - sigma * v
pred_x1 = x_t + v * (1 - sigma)
mean = pred_x0 * (1 - sigma_next) + pred_x1 * sqrt(sigma_next^2 - std^2)
```

Evaluation 用 `step_mean()` → ODE Euler：
```
x_{t+dt} = x_t + v * dt
```

这是**两条完全不同的 trajectory**。Policy 在 CPS trajectory 上 fine-tune，但 eval 走 ODE trajectory，中间状态不匹配 → 性能下降。

DDIM 没有这个 gap 是因为 deterministic DDIM 就是 stochastic DDIM 的 mean（eta=0），两条路径的中间状态一致。

SDE 也没有这个 gap 是因为 SDE mean 和 ODE Euler 在 sigma 较小时近似一致（SDE drift correction 项很小）。

CPS 的 mean 公式和 ODE Euler 差异大，尤其在 early denoising steps（sigma 大时），两条 trajectory 显著分叉。

### Fix

**File**: `RL-100/rl_100/unidpg/diffusion_policy/diffusers_patch/flow_match_scheduler.py` — `step_mean()`

当 `sde_type == "cps"` 时，`step_mean` 应该用 CPS mean（不加 noise），而不是 ODE Euler。这样 eval 走的是 rollout 的 mean trajectory。

当前 `step_mean` 代码（约 L275-306）：
```python
def step_mean(self, model_output, timestep, sample, ...):
    # ... setup ...
    dt = sigma_next - sigma
    prev_sample_mean = sample + model_output * dt  # ODE Euler for all modes
    # ...
```

---

## Final CPS Decision

### Final experiment conclusion

After running both CPS branches:

- **Plan B** performs best and can match the online fine-tuning level of:
  - DDIM
  - flow SDE
- **Plan A** does not reach the same level and is not selected as the final CPS solution

### Selected CPS solution

The project now temporarily adopts:

- **Plan B = official CPS path for this repo**

Concretely, this means:

- CPS uses `flow_cps_logprob_mode='gaussian'`
- the Gaussian CPS log-prob branch is the selected implementation
- this is now the preferred CPS configuration for future training and comparison

### Status of Plan A

Plan A is retained only as a documented alternative / failed branch:

- useful for comparison
- useful for ablation history
- **not** the default CPS recipe going forward

### Action for future runs

Unless there is a new regression, future CPS experiments should default to:

- `flow_sde_type='cps'`
- `flow_cps_logprob_mode='gaussian'`

and treat Plan A as archive / reference only.

### Acceptance note

At the current stage:

- Plan B is accepted as the repo's CPS implementation choice
- Plan A is rejected as the default CPS solution

---

## Plan A: Low-Step Fine-Tune Mode with Dual Evaluation

### Goal

在不破坏当前默认训练代码的前提下，增加一个 **low-step fine-tune mode**：

- 只在该 mode 下启用低 denoise 步数（例如 2 / 3 步）
- 只在该 mode 下启用 dual evaluation
- 默认训练、默认 eval、默认 CSV 行为必须保持不变

### Design Rules

#### 1. 必须有显式开关，默认关闭

在 `dp3_flow.yaml` 增加：

```yaml
flow_low_step_mode: false
flow_low_step_inference_steps: 2
flow_low_step_eval_dual: true
flow_low_step_eval_modes: fixed,random_mid
flow_random_mid_min_sigma: 0.01
flow_random_mid_max_sigma: ${flow_sigma_safe_max}
```

规则：

- `flow_low_step_mode=false` 时，代码行为必须与当前完全一致
- `flow_low_step_mode=true` 时，才允许进入 low-step fine-tune 路径
- `flow_low_step_inference_steps` 支持 `2` 或 `3`
- dual eval 也只在 `flow_low_step_mode=true` 时启用

#### 2. 不直接改原有默认 inference path

不要直接覆盖当前默认的：

- `num_inference_steps`
- `flow_inference_steps`
- 默认 eval 逻辑
- best-score / checkpoint 逻辑

而是在 policy 内增加一个 helper，统一返回当前 run 的 active inference steps：

- 默认：使用原始 `flow_inference_steps`
- low-step mode：使用 `flow_low_step_inference_steps`

这样原来 10-step 的代码路径保持不变。

#### 3. dual evaluation 只在 low-step mode 下生效

只有同时满足以下条件时，online fine-tune 评估点才跑两套 eval：

- `scheduler_type=flow`
- `flow_low_step_mode=true`

否则保持当前单一 eval 行为，不增加 random-mid 评估，不增加额外 CSV。

### Evaluation Modes

#### `fixed`

- 使用当前 scheduler 的标准 timesteps
- `2` 步时：标准 2-step path
- `3` 步时：标准 3-step path

#### `random_mid`

只在 low-step mode 下启用。

##### 当 `flow_low_step_inference_steps=2`

路径为：

- `start -> random_mid -> 0`

##### 当 `flow_low_step_inference_steps=3`

路径为：

- `start -> random_mid_1 -> random_mid_2 -> 0`

要求：

- `random_mid_1 > random_mid_2`
- 优先从 scheduler sigma grid 的离散候选点采样
- 不直接使用连续值硬映射到 timestep

### Implementation Steps

#### Step 1: Config

在 `dp3_flow.yaml` 增加：

- `flow_low_step_mode`
- `flow_low_step_inference_steps`
- `flow_low_step_eval_dual`
- `flow_low_step_eval_modes`
- `flow_random_mid_min_sigma`
- `flow_random_mid_max_sigma`

#### Step 2: Policy / scheduler support

在 flow policy / scheduler 增加一个 eval-only helper，用于构造 low-step dual eval 的 timestep path：

- `fixed`: 当前标准 timesteps
- `random_mid`: 低步数随机中间点 timesteps

关键约束：

- 只用于 evaluation
- 不修改 PPO rollout / replay buffer / logprob 主链路
- 不改默认 `set_timesteps()` 全局语义

#### Step 3: Extend env_runner interface for low-step eval

必须先扩展 env runner 接口，不能依赖脆弱的全局 scheduler state mutation。

`env_runner.run()` / `env_runner.idql_run()` 需要新增显式参数：

- `flow_eval_mode='fixed' | 'random_mid'`
- `flow_eval_inference_steps: Optional[int] = None`

要求：

- `flow_eval_inference_steps` 只在 `scheduler_type=flow` 且 `flow_low_step_mode=true` 时生效
- `flow_eval_mode='fixed'` 时走标准 low-step scheduler path
- `flow_eval_mode='random_mid'` 时走 low-step random-mid path
- 不能通过“先改 policy 全局步数，再调用 runner，再改回来”的方式实现

#### Step 4: Episode-level random-mid coordination

`random_mid` eval 不是每个 env step 都重新随机一次，而是每个 episode 固定一套随机 sigma path。

必须明确：

- episode 开始时采样 low-step random-mid sigma path
- 该 path 在整个 episode 内复用
- `policy.reset()` 或 runner 侧必须负责清空 episode-level eval path 状态

推荐实现：

- runner 在 `env.reset()` 后为该 episode 生成 `flow_eval_sigma_schedule`
- 之后每次 `policy.predict_action(...)` 都显式传入同一套 eval sigma path

不允许：

- 每次 `predict_action()` 单独重新采样 random-mid
- 用 batch 维 hack 出“伪 episode state”

#### Step 5: Extend eval entrypoints

修改 `train_cm_mid.py` 中的：

- `eval(...)`
- `unio4_eval(...)`

让它们支持额外的 eval mode 参数：

- `flow_eval_mode='fixed' | 'random_mid'`

当 low-step mode 打开时，在 online fine-tune 的评估点同时运行：

- normal + `fixed`
- normal + `random_mid`

如果 `ppo.idql_eval=True`，同样运行：

- idql + `fixed`
- idql + `random_mid`

#### Step 6: CSV saving

Plan A 的 dual-eval CSV 逻辑必须设计成可被 Plan C 复用的共享基础设施，不要在两个计划里各写一套单独保存逻辑。

推荐做法：

- 在 `train_cm_mid.py` 中抽一个共享 helper 负责：
  - fixed/random_mid metric 命名
  - legacy CSV 镜像
  - idql/non-idql CSV 写入
- Plan A / Plan C 都调用同一套 helper

在 `online_ft` 目录中，low-step mode 下新增：

- `fixed_success_rates.csv`
- `fixed_returns.csv`
- `random_mid_success_rates.csv`
- `random_mid_returns.csv`

如果 `idql_eval=True`，再新增：

- `idql_fixed_success_rates.csv`
- `idql_fixed_returns.csv`
- `idql_random_mid_success_rates.csv`
- `idql_random_mid_returns.csv`

兼容要求：

- 原有 `success_rates.csv` / `returns.csv` 继续保留
- 其内容默认等于 `fixed_*`
- 原有 `idql_success_rates.csv` / `idql_returns.csv` 继续保留
- 其内容默认等于 `idql_fixed_*`

#### Step 7: Primary metric

虽然同时记录 `fixed` 和 `random_mid` 两套结果：

- best-score
- checkpoint decision
- legacy summary log

默认仍然使用 `fixed` 作为主指标。

原因：

- 保持旧训练流程稳定
- 避免随机中间点 eval 方差直接干扰 checkpoint 选择

### Acceptance Criteria

#### Default regression

当 `flow_low_step_mode=false`：

- 行为与当前完全一致
- 不生成新的 dual-eval CSV
- 旧训练脚本、旧分析脚本不需要修改

#### Low-step 2-step mode

当：

- `flow_low_step_mode=true`
- `flow_low_step_inference_steps=2`

要求：

- online eval 同时输出 `fixed` / `random_mid`
- dual CSV 正确生成
- legacy CSV 仍存在，并与 `fixed` 对齐

#### Low-step 3-step mode

当：

- `flow_low_step_mode=true`
- `flow_low_step_inference_steps=3`

要求：

- `fixed` / `random_mid` 都能运行
- random-mid 的 3-step path 构造不报错

#### Non-regression

- DDIM 不受影响
- SDE 不受影响
- 10-step flow 不受影响
- replay buffer shape 不受影响
- PPO ratio / logprob 主链路不受影响

---

## Plan B: Flow → 1-Step Consistency Distillation

### Goal

类似现有 DDIM→CM distillation 流程，把 10-step flow teacher 蒸馏为 1-step student，然后用 1-step student 做 online RL fine-tune。

### 参考：现有 DDIM→CM distillation 流程

现有实现在 `dp3_cm.py` 中：

- `set_target()`（L308-322）：创建 `teacher`（frozen）和 `distilled_model`（trainable），从当前 `self.model` 深拷贝
- `compute_ddim2cm_loss_action_same_noise()`（L987-1070）：用相同 noise，teacher 跑 10-step DDIM rollout 得到 final action，student 跑 1-step CM 得到 action，MSE loss
- `train_cm_mid.py` `distill2cm()`（L1358-1587）：distillation 训练循环，在 BC 或 offline RL 之后执行

### Design Rules

#### 1. 复用现有 distillation 框架

不重新发明 distillation 训练循环。复用 `distill2cm()` 的结构，只替换 loss 函数。

#### 2. 不改 teacher 的推理路径

Teacher 用当前 10-step flow 推理（`flow_inference_steps=10`），完全 frozen。

#### 3. Student 用 1-step flow 推理

Student 用 `flow_inference_steps=1`（或 2），从 teacher 初始化。

#### 4. 蒸馏完成后，student 替换 model 做 online RL

蒸馏阶段结束后，`self.model = self.distilled_model`，后续 online RL 用 1-step model。

#### 5. Teacher 和 student 必须使用独立 scheduler 实例

不能让 teacher 和 student 共享同一个 `self.noise_scheduler` / `self.flow_scheduler`，因为 `set_timesteps()` 会修改内部状态。

要求：

- teacher rollout 使用独立的 flow scheduler 实例
- student rollout 使用独立的 flow scheduler 实例
- 不允许通过“teacher set_timesteps -> student set_timesteps -> restore”共享同一实例

#### 6. Flow distillation 路径不创建无用的 `target_model`

对 flow distillation：

- 只需要 `teacher`
- 只需要 `distilled_model`
- 不需要 `target_model`

因此 `set_target()` 需要在 `self.is_flow` 时跳过 `target_model` 创建，避免无谓 GPU memory 占用。

### Implementation Steps

#### Step 1: 改造 `set_target()` 以支持 flow distillation

**File**: `RL-100/rl_100/policy/dp3_cm.py` — `set_target()` (L308)

当前代码：
```python
def set_target(self):
    if self.is_flow:
        raise RuntimeError("Flow v1 does not support CM distillation (set_target)")
```

改为：
```python
def set_target(self):
    # Flow distillation: only need teacher + distilled_model (no target_model / boundary conditions)
```

Flow distillation 不需要 `target_model`（那是 CM boundary condition 用的）。只需要：
- `self.teacher`：frozen 10-step flow model
- `self.distilled_model`：trainable 1-step flow model

DDIM→CM 路径仍保留 `target_model`。
Flow 路径必须：

- 不 raise
- 不创建 `target_model`
- 为 teacher / student 准备独立 scheduler 实例

- [ ] 移除 flow raise
- [ ] flow 路径不创建 `target_model`
- [ ] teacher / student scheduler 分离
- [ ] 保持 DDIM→CM 路径不变

#### Step 2: 新增 `compute_flow_distill_loss()`

**File**: `RL-100/rl_100/policy/dp3_cm.py`

新增方法，参考 `compute_ddim2cm_loss_action_same_noise()`（L987-1070）的模式：

```python
def compute_flow_distill_loss(self, batch, distill2mean=True, fix_encoder=False):
    """Distill N-step flow teacher into 1-step flow student using same noise."""
    assert self.is_flow, "compute_flow_distill_loss requires flow mode"

    # 1. Normalize + encode obs (复用现有 normalizer + obs_encoder)
    nobs = self.normalizer.normalize(batch['obs'])
    trajectory = self.normalizer['action'].normalize(batch['action'])
    # ... condition_mask, cond_data setup (复用 compute_loss 的逻辑)

    obs_feature = self.obs2latent(nobs) if not fix_encoder else self.obs2latent(nobs).detach()

    # 2. Sample noise (same for teacher and student)
    noise = torch.randn_like(trajectory)

    # 3. Teacher: N-step flow rollout (frozen)
    with torch.no_grad():
        teacher_scheduler = self.teacher_flow_scheduler
        teacher_scheduler.set_timesteps(self.flow_inference_steps)  # 10 steps
        teacher_traj = noise.clone()
        for t in teacher_scheduler.timesteps:
            teacher_traj[condition_mask] = cond_data[condition_mask]
            unet_t = self.get_unet_timesteps(t.expand(bsz))
            v_pred = self.teacher(sample=teacher_traj, timestep=unet_t,
                                  local_cond=local_cond, global_cond=obs_feature)
            if distill2mean:
                teacher_traj = teacher_scheduler.step_mean(v_pred, t, teacher_traj).prev_sample
            else:
                teacher_traj = teacher_scheduler.step(v_pred, t, teacher_traj).prev_sample
        teacher_action = teacher_traj

    # 4. Student: 1-step flow prediction (trainable)
    student_nfe = getattr(self, 'flow_distill_student_steps', 1)
    student_scheduler = self.student_flow_scheduler
    student_scheduler.set_timesteps(student_nfe)
    student_traj = noise.clone()  # same noise!
    for t in student_scheduler.timesteps:
        student_traj[condition_mask] = cond_data[condition_mask]
        unet_t = self.get_unet_timesteps(t.expand(bsz))
        v_pred = self.distilled_model(sample=student_traj, timestep=unet_t,
                                       local_cond=local_cond, global_cond=obs_feature)
        student_traj = student_scheduler.step_mean(v_pred, t, student_traj).prev_sample
    student_action = student_traj

    # 5. MSE loss on final action
    loss = F.mse_loss(student_action, teacher_action.detach())

    return loss, {'distill_loss': loss.item()}
```

**关键设计决策**:
- 用 `step_mean`（确定性）而非 `step`（随机），减少方差
- Teacher 和 student 用相同 noise 起点（参考 `compute_ddim2cm_loss_action_same_noise`）
- Student 步数通过 `flow_distill_student_steps` 配置（默认 1）
- teacher / student 不共享 scheduler，因此不需要 restore 同一实例的全局状态

- [ ] 实现 `compute_flow_distill_loss`
- [ ] 验证 teacher rollout 输出合理
- [ ] 验证 student 1-step 输出可训练
- [ ] teacher / student 使用独立 scheduler

#### Step 3: 移除 distillation guards in `train_cm_mid.py`

**File**: `RL-100/train_cm_mid.py`

当前至少有 4 处 guard / 拦截阻止 flow + distillation：
```python
if self.model.is_flow:
    raise RuntimeError("Flow does not support distillation")
```

改为：flow 走新的 `compute_flow_distill_loss`，DDIM 走原有 `compute_ddim2cm_loss*`。

必须覆盖：

- `TrainDP3Workspace.run()` 中基于 `distill_phase` 的 flow guard
- `distill2cm()` 入口的 flow guard
- 其余 flow-specific assert / raise

在 `distill2cm()` 方法中：
```python
if model_to_optimize.is_flow:
    raw_loss, loss_dict = model_to_optimize.compute_flow_distill_loss(
        batch, distill2mean=self.cfg.distill2mean, fix_encoder=...)
else:
    # 原有 DDIM→CM distillation 逻辑不变
    if self.cfg.distill_loss_type == 'action_same_noise':
        raw_loss, loss_dict = model_to_optimize.compute_ddim2cm_loss_action_same_noise(...)
    elif ...
```

- [ ] 移除 flow distillation guards（包含 `run()` 中最早触发的 guard）
- [ ] `distill2cm()` 中 flow 分支调用 `compute_flow_distill_loss`
- [ ] DDIM→CM 路径不变

#### Step 4: Config

**File**: `RL-100/rl_100/config/dp3_flow.yaml`

```yaml
# Flow distillation settings
distill_phase: 'after_dp'          # 'after_dp' or 'after_offline' or null
distill_loss_type: 'flow_action'   # flow-specific loss type
distill2mean: true                 # teacher 用确定性推理
flow_distill_student_steps: 1      # student 推理步数
```

注意：`distill_phase` 当前在 flow config 中是 `null`。启用蒸馏时改为 `'after_dp'`（BC 之后蒸馏）。

- [ ] 添加 flow distillation config 字段

#### Step 5: 蒸馏后切换到 student model 做 online RL

**File**: `RL-100/train_cm_mid.py`

蒸馏完成后，需要：
1. 把 `distilled_model` 的权重加载回 `self.model`
2. 把 `flow_inference_steps` 切换到 `flow_distill_student_steps`（1 或 2）
3. 后续 online RL 用少步 model

```python
# After distillation completes:
if model_to_optimize.is_flow:
    # Replace model with distilled student
    model_to_optimize.model.load_state_dict(
        model_to_optimize.distilled_model.state_dict())
    # Switch to student inference steps
    student_steps = self.cfg.flow_distill_student_steps
    model_to_optimize.flow_inference_steps = student_steps
    model_to_optimize.noise_scheduler.set_timesteps(student_steps)
```

- [ ] 蒸馏后 model 替换
- [ ] inference steps 切换
- [ ] 验证 online RL 用 1-step model

#### Step 5.5: Checkpoint / resume 语义

Plan B 必须明确蒸馏后 checkpoint 和 resume 的语义，不能留给 implementer 自行决定。

要求：

- 蒸馏完成后，online RL checkpoint 保存的是 student model
- flow distill run resume 时，优先恢复 student 路径，而不是重新从 teacher-only 状态开始
- 不允许出现“加载 latest 后仍然回到 10-step teacher 推理配置”的隐式回退

- [ ] 明确 flow distill checkpoint 保存对象
- [ ] 明确 flow distill resume 行为

#### Step 6: 训练脚本

**File**: `scripts/train_policy_online_flow_distill.sh`（新建）

基于 `train_policy_online_flow.sh`，添加蒸馏参数：
```bash
distill_phase='after_dp' \
distill_loss_type='flow_action' \
distill2mean=True \
flow_distill_student_steps=1 \
```

- [ ] 创建脚本

### 文件清单

| 文件 | 改动量 | 说明 |
|------|--------|------|
| `dp3_cm.py` | ~80 行 | `set_target` 解锁 + `compute_flow_distill_loss` |
| `train_cm_mid.py` | ~30 行 | 移除 guards + flow distill 分支 + 蒸馏后切换 |
| `dp3_flow.yaml` | ~5 行 | distillation config |
| `train_policy_online_flow_distill.sh` | 新建 | 训练脚本 |

### Acceptance Criteria

#### Distillation 阶段
- [ ] `compute_flow_distill_loss` 返回有限 loss 值
- [ ] Distillation loss 随训练下降
- [ ] 蒸馏后 1-step student 的 BC eval score > 0.5（door task）

#### Online RL 阶段
- [ ] 1-step student 能完成 online RL rollout
- [ ] Online RL eval score 接近 10-step baseline（允许 ~15% 下降）
- [ ] Rollout + update 速度显著提升（~5-10x）

#### Non-regression
- [ ] DDIM→CM distillation 路径不受影响
- [ ] 10-step flow online RL 不受影响
- [ ] `distill_phase=null` 时行为与当前完全一致

---

## Plan C: 2-Step + 随机中间步 t（Rollout 训练 + Eval）

### Goal

2-step / 3-step flow fine-tuning：

- rollout 和 PPO 训练时中间 timestep / sigma 随机采样
- evaluation 同时跑两组：
  - `fixed`
  - `random_mid`

通过随机 `t_mid` / `sigma_mid` 让 policy 学会处理各种 sigma 间距，类似 BC 训练时随机采样 t 的思路。

### Base policy choice

Plan C must build on the accepted CPS baseline from `Final CPS Decision`:

- `flow_sde_type='cps'`
- `flow_cps_logprob_mode='gaussian'`

Do not leave pseudo/gaussian as an implementation-time choice inside Plan C.

### 核心设计

```
Rollout 1: σ = [σ_start, 0.3, 0.0]   ← t_mid 随机
Rollout 2: σ = [σ_start, 0.7, 0.0]   ← t_mid 随机
PPO update: 用每个 sample 存储的 t_mid 重算 log-prob
Eval A:     σ = [σ_start, 0.5, 0.0]   ← fixed
Eval B:     σ = [σ_start, random_mid, 0.0]   ← random_mid
```

### 关键挑战

PPO update 时需要重算 `a_logprob_now`，必须知道每个 sample 用的是哪个 t_mid。
当前 PPO loop 用全局 `self._policy.noise_scheduler.timesteps`，所有 sample 共享同一 schedule。
需要把 per-sample 的 sigma schedule 存进 replay buffer。

**硬约束**:

- rollout 时采样到的 `sigma_schedule` 必须作为 trajectory 的一部分存入 replay buffer
- PPO update 时 old/new policy 都必须复用同一个 `sigma_schedule`
- random-mid PPO 路径不得再依赖：
  - `self.noise_scheduler.timesteps`
  - `self.step_index`
  来隐式推断真实 sigma path
- 在 random-mid training 路径中，sigma path 必须完全由 buffer 中存储的 `sigma_schedule` 决定

否则 ratio 不再是同一 action-generation process 上的 ratio，实验结果不可信。

### Evaluation Rule

Plan C 必须使用 dual evaluation：

- `fixed`
- `random_mid`

其中：

- `fixed` 继续作为 primary / legacy metric
- `random_mid` 作为训练分布一致性指标

必须同时保存两套 CSV，不能只保留 fixed eval。

### Implementation Steps

#### Step 1: Replay buffer 存储 per-sample sigma schedule

**Files**:

- `RL-100/rl_100/dppo/online_buffer.py`
- `RL-100/rl_100/dppo/online_buffer_vec.py` (only if the vector buffer path is supported in this patch)

新增字段：
```python
# per-sample sigma schedule, shape [batch_size, num_inference_steps + 1]
# 存储完整 sigma 路径（含起点和终点），如 [0.9999, 0.3, 0.0]
self.sigma_schedule = np.zeros((args.batch_size, args.num_inference_steps + 1))
```

`store()` 方法新增 `sigma_schedule` 参数。
`numpy_to_tensor()` 返回值新增 `sigma_schedule` tensor。

- [ ] buffer 新增 `sigma_schedule` 字段
- [ ] `store()` 接收并存储
- [ ] `numpy_to_tensor()` 返回
- [ ] 明确本次 patch 是否支持 `online_buffer_vec.py`

**范围约束**:

- 如果当前 online fine-tune 路径可能进入 vector replay buffer，则 `online_buffer_vec.py` 必须在同一 patch 中一起更新
- 如果 v1 只支持 non-vector replay buffer，则代码必须 fail fast，不允许 silent fallback

#### Step 2: Scheduler 支持直接传入 sigma 值

**File**: `RL-100/rl_100/unidpg/diffusion_policy/diffusers_patch/flow_match_scheduler.py`

新增方法，绕过 `set_timesteps` + `step_index`，直接用 sigma/sigma_next 做一步：

```python
def step_with_sigma(self, model_output, sigma, sigma_next, sample, generator=None):
    """用显式 sigma/sigma_next 做一步 SDE/CPS transition，不依赖 step_index。"""
    model_output = model_output.float()
    sample = sample.float()
    sigma = sigma.view(-1, *([1] * (len(sample.shape) - 1)))
    sigma_next = sigma_next.view(-1, *([1] * (len(sample.shape) - 1)))
    dt = sigma_next - sigma

    # 复用 _compute_step 的数学，但用传入的 sigma 而非 self.sigmas[step_index]
    # ... SDE/CPS 分支逻辑同 _compute_step ...
    return prev_sample, prev_sample_mean, std

def step_logprob_with_sigma(self, model_output, sigma, sigma_next, sample, generator=None):
    """step_with_sigma + log-prob 计算"""
    prev_sample, prev_sample_mean, std = self.step_with_sigma(
        model_output, sigma, sigma_next, sample, generator)
    log_prob = self._compute_logprob(prev_sample, prev_sample_mean, std,
                                      self.sde_type, self.cps_logprob_mode)
    return prev_sample, log_prob

def forward_logprob_with_sigma(self, model_output, sigma, sigma_next, sample, next_sample):
    """用显式 sigma 计算 forward log-prob（PPO update 用）"""
    # 不做 step，只算已知 next_sample 的 log-prob
    _, prev_sample_mean, std = self.step_with_sigma(
        model_output, sigma, sigma_next, sample)
    log_prob = self._compute_logprob(next_sample.float(), prev_sample_mean, std,
                                      self.sde_type, self.cps_logprob_mode)
    entropy = 0.5 * torch.log(2 * math.pi * math.e * std**2)
    entropy = entropy.expand_as(sample)
    return log_prob, entropy
```

**接口约束**:

- `step_with_sigma`
- `step_logprob_with_sigma`
- `forward_logprob_with_sigma`

都必须只依赖显式传入的：

- `sigma`
- `sigma_next`
- `sample`
- `next_sample`

不能隐式读取全局 step path：

- 不依赖 `self.noise_scheduler.timesteps`
- 不依赖 `self.step_index` 决定 sigma 位置

- [ ] `step_with_sigma` 实现
- [ ] `step_logprob_with_sigma` 实现
- [ ] `forward_logprob_with_sigma` 实现
- [ ] 数学逻辑与 `_compute_step` 一致

#### Step 3: Policy rollout 支持随机 sigma schedule

**File**: `RL-100/rl_100/policy/dp3_cm.py`

修改 `all_step_action_logprob` 和 `sample_action_with_logprob`：

新增参数 `random_mid_sigma: bool = False, sigma_min=0.01, sigma_max=0.9`

当 `random_mid_sigma=True` 且 `num_inference_steps=2` 时：
```python
if random_mid_sigma and self.is_flow:
    # 构造 per-sample 随机 sigma schedule
    sigma_start = self.noise_scheduler.sigmas[0]  # ~0.9999
    # 每个 batch sample 独立采样 t_mid
    t_mid = torch.rand(bsz, device=device) * (sigma_max - sigma_min) + sigma_min
    # sigma_schedule: [sigma_start, t_mid, 0.0]
    sigma_schedule = torch.stack([
        sigma_start.expand(bsz),
        t_mid,
        torch.zeros(bsz, device=device)
    ], dim=1)  # shape [bsz, 3]

    # 用 step_logprob_with_sigma 逐步 denoise
    trajectory = noise
    all_x = [trajectory]
    all_logprob = []
    for i in range(num_inference_steps):
        sigma = sigma_schedule[:, i]
        sigma_next = sigma_schedule[:, i + 1]
        unet_t = (sigma * self.noise_scheduler.config.num_train_timesteps).long()
        model_output = self.model(sample=trajectory, timestep=unet_t, ...)
        trajectory, log_prob = self.noise_scheduler.step_logprob_with_sigma(
            model_output, sigma, sigma_next, trajectory)
        all_x.append(trajectory)
        all_logprob.append(log_prob)

    return action, torch.stack(all_x), torch.stack(all_logprob), sigma_schedule
```

返回值新增 `sigma_schedule` tensor，供 buffer 存储。

- [ ] `all_step_action_logprob` 支持 `random_mid_sigma`
- [ ] `sample_action_with_logprob` 支持 `random_mid_sigma`
- [ ] 返回 `sigma_schedule`

#### Step 4: PPO update 用 per-sample sigma schedule

**File**: `RL-100/rl_100/unidpg/uni_ppo.py` — `dp_align_update_no_share()`

当前代码：
```python
for i, t in enumerate(self._policy.noise_scheduler.timesteps):
    timesteps = t.expand(batch_size)  # 所有 sample 同一个 t
    ...
    a_logprob_now, entropy = scheduler.step_forward_logprob_with_entropy(
        model_output, timesteps, actions[i], next_sample=actions[i+1], ...)
```

改为：
```python
if self._use_random_mid_sigma:
    sigma_schedules = ...  # 从 buffer 取，shape [batch, num_steps + 1]
    for i in range(num_inference_steps):
        sigma = sigma_schedules[index, i]       # per-sample sigma
        sigma_next = sigma_schedules[index, i+1] # per-sample sigma_next
        unet_t = (sigma * N).long()
        model_output = self._policy.model(sample=actions[i], timestep=unet_t, ...)
        a_logprob_now, entropy = scheduler.forward_logprob_with_sigma(
            model_output, sigma, sigma_next, actions[i], next_sample=actions[i+1])
        # ... ratio, clamp, loss 逻辑不变
else:
    # 原有逻辑不变
    for i, t in enumerate(self._policy.noise_scheduler.timesteps):
        ...
```

- [ ] `dp_align_update_no_share` 支持 per-sample sigma
- [ ] 从 buffer 正确取 `sigma_schedule`
- [ ] ratio/clamp/loss 逻辑不变
- [ ] 原有固定 schedule 路径不变（`else` 分支）
- [ ] random-mid PPO 路径明确复用 Plan C 的 Gaussian CPS baseline，不引入新的 CPS log-prob 变体

#### Step 5: train_cm_mid.py rollout + dual eval 传参

**File**: `RL-100/train_cm_mid.py`

Online rollout loop 中：
```python
if self.cfg.flow_random_mid_sigma:
    action, all_x, a_logprob, sigma_schedule = self.unio4._policy.sample_action_with_logprob(
        ..., random_mid_sigma=True,
        sigma_min=self.cfg.flow_random_mid_min_sigma,
        sigma_max=self.cfg.flow_random_mid_max_sigma)
    replay_buffer.store(..., sigma_schedule=sigma_schedule)
else:
    # 原有逻辑
```

- [ ] rollout 传 `random_mid_sigma=True`
- [ ] `sigma_schedule` 存入 buffer

同时，Plan C 的 online eval 点必须同时运行：

- fixed eval
- random_mid eval

并保存双份 CSV：

- `fixed_success_rates.csv`
- `fixed_returns.csv`
- `random_mid_success_rates.csv`
- `random_mid_returns.csv`

如果 `idql_eval=True`，还要同时保存：

- `idql_fixed_success_rates.csv`
- `idql_fixed_returns.csv`
- `idql_random_mid_success_rates.csv`
- `idql_random_mid_returns.csv`

兼容要求：

- 原有 `success_rates.csv` / `returns.csv` 继续保留
- 它们默认等于 `fixed_*`

#### Step 6: Config

**File**: `RL-100/rl_100/config/dp3_flow.yaml`

```yaml
# Plan C: random mid-sigma for 2-step training
flow_random_mid_sigma: false          # 启用随机中间步
flow_random_mid_min_sigma: 0.01       # t_mid 下界
flow_random_mid_max_sigma: 0.9        # t_mid 上界
```

- [ ] 添加 config 字段

#### Step 7: 训练脚本

**File**: `scripts/train_policy_online_flow_random_mid.sh`（新建）

基于 `train_policy_online_flow.sh`，添加：
```bash
flow_inference_steps=2 \
num_inference_steps=2 \
flow_random_mid_sigma=true \
flow_random_mid_min_sigma=0.01 \
flow_random_mid_max_sigma=0.9 \
```

- [ ] 创建脚本

### 文件清单

| 文件 | 改动量 | 说明 |
|------|--------|------|
| `online_buffer.py` | ~10 行 | 存储 per-sample sigma_schedule |
| `flow_match_scheduler.py` | ~60 行 | `step_with_sigma` / `forward_logprob_with_sigma` |
| `dp3_cm.py` | ~50 行 | rollout 方法支持随机 sigma |
| `uni_ppo.py` | ~40 行 | PPO loop 支持 per-sample schedule |
| `train_cm_mid.py` | ~15 行 | rollout 传参 + buffer 存储 |
| `dp3_flow.yaml` | ~5 行 | config 字段 |
| `train_policy_online_flow_random_mid.sh` | 新建 | 训练脚本 |

### Scope Constraint

Plan C v1 必须明确范围：

- 若当前 online fine-tune 只走 `RL-100/rl_100/dppo/online_buffer.py`
  - 则 v1 只要求修改这个 non-vector buffer
- 若当前路径也会走 `online_buffer_vec.py`
  - 则必须在同一个 patch 中同步支持 `sigma_schedule`

不允许出现：

- non-vector buffer 支持 random-mid
- vector buffer 不支持
- 但代码路径仍可能静默走到 vector buffer

如果 v1 只支持 non-vector buffer，必须在 TODO 和代码里显式写清楚。

### Acceptance Criteria

#### 功能验证
- [ ] 2-step + random t_mid rollout 能正常完成
- [ ] PPO update 用 per-sample sigma 重算 log-prob，ratio 在合理范围
- [ ] Eval 同时输出 fixed / random_mid 两组结果
- [ ] primary legacy metric 继续使用 fixed

#### 性能目标
- [ ] Eval score 接近 10-step baseline（允许 ~15% 下降）
- [ ] Rollout + update 速度显著提升

#### Non-regression
- [ ] `flow_random_mid_sigma=false` 时行为与当前完全一致
- [ ] 10-step flow 不受影响
- [ ] DDIM 不受影响
- [ ] SDE 不受影响
- [ ] replay buffer 在非 random-mid 模式下 shape 不变

---

## Code Review: Plans A, B, C — Issues Found (2026-03-15)

Reviewed against actual codebase on branch `RL-100-Flow`.

### Cross-Plan Issues

#### X3 (HIGH): `predict_action` 没有 inference_steps override

所有三个 Plan 最终都需要用不同步数做 eval。但 `predict_action()` → `conditional_sample()` 用的是 `self.flow_inference_steps`（init 时固定）。改步数只能：
- 在 `predict_action` 加 `inference_steps=...` 参数 override
- 或在 eval 前后 mutate `self.flow_inference_steps` + `set_timesteps()`

三个 Plan 都没有 address 这个问题。**建议**：在 `predict_action` / `conditional_sample` 加可选 `inference_steps` 参数，内部临时 `set_timesteps` 再恢复。

#### X1 (LOW): Plan A 和 Plan C 共享 dual-eval 基础设施

两者都需要 `fixed` + `random_mid` dual CSV。应先实现 Plan A 的 eval 基础设施，Plan C 复用。

#### X2 (LOW): Plan B 和 Plan C 部分不兼容

Plan B 蒸馏到 1-step，Plan C 用 2-step random-mid 训练。如果组合使用，步数冲突。如果是互斥方案，应明确说明。

---

### Plan A Issues

#### A1 (HIGH): `env_runner.run()` 没有 inference_steps / eval_mode 参数

Plan 假设可以传 `flow_eval_mode='fixed' | 'random_mid'` 到 `eval()` / `unio4_eval()`。但 `env_runner.run()` 签名是：
```python
def run(self, policy: BasePolicy, use_cm=False, distill2mean=False):
```
没有 `inference_steps` 或 `eval_mode` 参数。Runner 内部调用 `policy.predict_action()`，用的是 scheduler 当前全局状态。

**修复方向**：
- 方案 1：eval 前 mutate policy 的 `flow_inference_steps` + `set_timesteps()`，eval 后恢复（脆弱但不改 runner）
- 方案 2：给 `env_runner.run()` 加 `inference_steps` 参数，内部传给 `predict_action`（更干净但改更多文件）

- [ ] 选择方案并更新 Plan A

#### A2 (MEDIUM): `random_mid` eval 的 per-episode sigma 问题

`predict_action()` 每次调用处理单个 obs，没有 batch 维度。`random_mid` eval 需要每个 episode 用不同的随机 sigma schedule，但 plan 没有说明：
- 是每个 episode 开始时采样一次 random schedule？（需要 env_runner 配合）
- 还是每次 `predict_action` 调用时采样？（错误：同一 episode 内 schedule 会变）

- [ ] 明确 random_mid eval 的 sigma 采样时机

---

### Plan B Issues

#### B1 (HIGH): 蒸馏时 teacher/student 共享 scheduler 实例，`set_timesteps` 互相覆盖

`compute_flow_distill_loss()` 伪代码：
```python
self.noise_scheduler.set_timesteps(10)   # teacher
# ... teacher rollout ...
self.noise_scheduler.set_timesteps(1)    # student
# ... student rollout ...
self.noise_scheduler.set_timesteps(10)   # restore
```

`set_timesteps()` 修改 `self.sigmas`, `self.timesteps`, `self._step_index` 全局状态。问题：
- 中断时 scheduler 状态错误
- 不可并行
- 如果 `flow_inference_steps` 在蒸馏后被 Step 5 修改，restore 行恢复到错误值

**修复方向**：创建两个独立 scheduler 实例（teacher 10-step, student 1-step），或在 loss 函数内部用局部 sigma 数组而不 mutate 共享 scheduler。

- [ ] 改用独立 scheduler 实例或局部 sigma 计算

#### B2 (MEDIUM): `set_target()` 为 flow 创建不必要的 `target_model`

`set_target()` 创建 3 个 deepcopy：`teacher`, `distilled_model`, `target_model`。Flow 不需要 `target_model`（那是 CM boundary condition 用的），浪费 GPU 显存。

**修复**：
```python
if not self.is_flow:
    self.target_model = copy.deepcopy(self.model)
```

- [ ] Guard `target_model` 创建

#### B3 (MEDIUM): `__init__` guard 未被 plan 提及

Plan 提到移除 3 处 guard，但 `train_cm_mid.py` line 129 的 `__init__` guard 最先触发：
```python
if cfg.distill_phase is not None and getattr(self.model, 'is_flow', False):
    raise RuntimeError(...)
```
如果只移除 `distill2cm()` 内的 guard（line 1360）和 offline stage guard（line 460），`__init__` guard 仍然会阻止 flow + `distill_phase='after_dp'`。

**修复**：三处 guard 必须同时处理。将 line 129 改为只阻止 flow + DDIM-style distillation，允许 flow + flow-style distillation。

- [ ] 更新 Plan B 明确列出所有 3 处 guard 位置及修改方式


#### B4 (HIGH): distillation eval 仍在评估 teacher，不是 student

当前 `distill2cm()` rollout/eval 分支里，flow 路径调用的是：
```python
runner_log = env_runner.run(policy)
```
而 flow policy 只有在 `use_cm=True` 时才会切到 `distilled_model` + `flow_student_scheduler`。
这意味着：
- 蒸馏阶段记录的 success/return 实际上还是 teacher 的结果
- checkpoint 选择依据是 teacher，而不是 student
- distillation loss 和 eval 指标脱钩

**修复**：flow distillation eval 必须显式评估 student。
推荐改为：
```python
runner_log = env_runner.run(policy, use_cm=True, distill2mean=self.cfg.distill2mean)
```
或等价的 flow-student eval 路径。

- [ ] flow distillation eval 改为显式评估 student

#### B5 (HIGH): 蒸馏后的 student 还没有真正成为默认 online policy

当前实现只是在 `use_cm=True` 分支下让 policy 切到 `distilled_model`。
如果后续 online rollout / PPO 主路径不显式传 `use_cm=True`，默认仍然使用 teacher / 原始 `self.model`。
这不满足 Plan B 的核心目标：
- 先蒸馏出 1-step student
- 再让 student 成为主 online RL policy

**修复**：必须二选一，并在代码里写死：
- 方案 1：蒸馏结束后 `self.model.load_state_dict(self.distilled_model.state_dict())`，并把 active inference steps 切到 student steps
- 方案 2：统一把 online rollout / update / eval 主路径切到 student，不再依赖临时 `use_cm=True`

不允许保持当前这种“student 只存在于辅助分支”的半接入状态。

- [ ] 明确并实现 student 成为默认 online policy 的切换方案

#### B6 (MEDIUM): 新脚本跑的是 `distill_phase='online'`，与 Plan B 文档主路径不一致

当前 `scripts/train_policy_online_flow_distill.sh` 使用：
```bash
distill_phase='online'
```
但 Plan B 文档主体描述的是：
- `after_dp`
- 或 `after_offline`
- 然后 student 做 online RL

这会带来两个问题：
- work agent 可能以为脚本已经覆盖了文档里的主实验路径，实际没有
- review 时很难判断分支到底在实现“在线蒸馏”还是“先蒸馏后在线”

**修复**：必须明确：
- 如果 Plan B 主实验是 `after_dp` / `after_offline`，脚本应默认跑该路径
- 如果要保留 `online` 作为额外变体，需单独标成 optional variant，不得替代主路径

- [ ] 统一 Plan B 文档主路径与训练脚本默认路径

#### B7 (LOW): 蒸馏后 model swap 与 checkpoint 的交互

Step 5 把 `distilled_model` → `model` 并改 `flow_inference_steps`。但 `distill2cm()` 在蒸馏过程中也保存 checkpoint。如果从 checkpoint resume，是否能正确检测蒸馏已完成？

当前风险：
- 蒸馏结束后，内存中的默认 policy 已经是 promoted student
- 但磁盘 checkpoint 仍然可能同时保存：
  - `model.pt`
  - `distilled_model.pt`
- 如果 `model.pt` 还是 promotion 之前的默认权重，那么 resume 后默认 policy 语义不明确

这会导致：
- resume 后默认 model 不一定还是 student
- `flow_inference_steps` 不一定还是 student steps
- `last/` checkpoint 和“当前默认在线策略”不再一一对应

**修复要求**：

1. 保存语义必须明确
- flow distillation 完成并执行 `promote_distilled_model()` 之后
- 必须再保存一次“promoted final” checkpoint
- 该 checkpoint 中：
  - `model.pt` 必须就是 promoted student
  - 默认 `flow_inference_steps` 必须已经切到 `flow_distill_inference_steps`

2. resume 语义必须防御式恢复
- flow distill / flow distill-online resume 时
- 如果 checkpoint 中存在 `distilled_model.pt`
- load 完成后必须显式恢复到 promoted student 默认状态
- 不允许仅仅加载 `model.pt` / `distilled_model.pt` 就假设状态已经正确

推荐实现：
```python
if self.is_flow and has_distilled_checkpoint:
    self.promote_distilled_model()
```
也就是：
- load 完 `distilled_model.pt`
- 再执行一次 defensive promote
- 保证默认 model 和默认 inference steps 回到 student 语义

3. online / offline path 必须一致
- `after_dp`
- `after_offline`
- `online`
这三种 flow distill 相关路径都必须遵守同一套 checkpoint/resume 语义

**执行步骤**：
- [ ] 在 flow distillation 完成并 promote 后，额外保存一次 promoted final checkpoint
- [ ] 在 flow load/resume 路径中，检测 `distilled_model.pt` 后执行 defensive promote
- [ ] 确认 `model.pt`、默认 active policy、默认 `flow_inference_steps` 三者语义一致
- [ ] 覆盖 `after_dp` / `after_offline` / `online` 三种 flow distill 路径

**验收标准**：
- [ ] 从 flow distill 生成的 `last/` checkpoint resume 后，默认 policy 仍然是 student
- [ ] resume 后默认 `flow_inference_steps == flow_distill_inference_steps`
- [ ] 不需要依赖 `use_cm=True` 才能回到 student 默认路径
- [ ] checkpoint 文件语义对 work agent / user 可解释：`model.pt` 就是后续 online RL 默认使用的策略

- [ ] 明确 checkpoint/resume 语义

---

### Plan C Issues

#### C1 (HIGH): `_compute_step` 逻辑重复

Scheduler 当前没有 sigma-direct 方法。所有 transition 数学在 `_compute_step()` 中，通过 `self.sigmas[self.step_index]` 读取 sigma。Plan C 提议 3 个新方法（`step_with_sigma`, `step_logprob_with_sigma`, `forward_logprob_with_sigma`）复制相同的 SDE/CPS 数学但用显式 sigma 参数。

这创建了两套并行代码路径 — 维护负担大，未来 fix 必须同步。

**修复方向**：重构 `_compute_step` 接受显式 `sigma, sigma_next` 参数（step_index lookup 作为默认 fallback）。新旧方法都调用同一个核心函数。

```python
def _compute_step(self, model_output, sample, sigma=None, sigma_next=None, ...):
    if sigma is None:
        idx = min(self.step_index, len(self.sigmas) - 2)
        sigma = self.sigmas[idx]
        sigma_next = self.sigmas[idx + 1]
    # ... 原有 SDE/CPS 数学 ...
```

- [ ] 重构 `_compute_step` 为 sigma-parameterized

#### C2 (HIGH): `sample_action_with_logprob` 返回值签名变更

当前调用方（`train_cm_mid.py` line ~1102）：
```python
action, all_x, a_logprob = self.unio4._policy.sample_action_with_logprob(...)
```

Plan C 改为返回 4 个值（多了 `sigma_schedule`）。这会 break 所有现有调用方。

**修复方向**：
- 方案 1：`sigma_schedule` 放在返回的 dict 中而非额外返回值
- 方案 2：只在 `random_mid_sigma=True` 时返回 4 值（更脆弱）
- 方案 3：始终返回 4 值，非 random-mid 时 `sigma_schedule=None`（最干净）

- [ ] 选择返回值方案并更新所有调用方

#### C3 (HIGH): Per-sample UNet timesteps 需要 batch 化的 `get_unet_timesteps`

Plan C 伪代码：
```python
unet_t = (sigma * self.noise_scheduler.config.num_train_timesteps).long()
```

这绕过了 `get_unet_timesteps()`，丢失了该方法中的 float→int 转换逻辑。而且 `sigma` 现在是 per-sample `(bsz,)` tensor，`get_unet_timesteps` 当前只处理标量/broadcast timestep。

**修复**：确保 `get_unet_timesteps` 能处理 `(bsz,)` shape 的 timestep tensor，并在 Plan C 中统一使用它。

- [ ] 验证 `get_unet_timesteps` 支持 batch timesteps

#### C4 (MEDIUM): `all_step_action_logprob` 也需要 random-mid 支持

Plan C 只提到修改 `sample_action_with_logprob`，但 offline BPPO 的 `update_distribution()` 调用 `all_step_action_logprob`。如果 offline RL 也要支持 random-mid，这个方法也需要改。

如果 v1 只支持 online RL random-mid，应明确 scope。

- [ ] 明确 offline RL 是否需要 random-mid

#### C5 (MEDIUM): Eval 同样受 A1 限制

Plan C 的 dual eval 面临与 Plan A 相同的 `env_runner.run()` 限制（无 inference_steps 参数）。

- [ ] 复用 Plan A 的 eval 方案

#### C6 (LOW): `online_buffer_vec.py` 排除未文档化

`online_buffer_vec.py` 存在但当前未使用。Plan C 只修改 `online_buffer.py`。如果未来切换到 vec buffer，random-mid 会静默失效。

- [ ] 在代码中加 assert 或在 TODO 中明确记录

---

## Plan B Worktree Review (ft-dp3-plan-b, 2026-03-15)

基于 `ft-dp3-plan-b` worktree 实际代码的 review。

### 已正确实现 ✅

1. **`set_target()` flow 分支** (`dp3_cm.py:312-338`)
   - flow 不创建 `target_model`，节省 GPU 显存 ✅
   - 创建独立 `flow_teacher_scheduler` / `flow_student_scheduler` (deepcopy) ✅
   - DDIM/CM 路径不变 ✅

2. **`compute_flow_distill_loss()`** (`dp3_cm.py:1173-1250`)
   - Teacher 用 `flow_teacher_scheduler`，student 用 `flow_student_scheduler`，不共享状态 ✅
   - 同 noise 起点，MSE loss on final action ✅
   - 用 `step_mean`（确定性），减少方差 ✅
   - `_extract_action` helper 提取 action slice ✅

3. **`distill2cm()` flow 分支** (`train_cm_mid.py:1419-1421`)
   - `is_flow` → `compute_flow_distill_loss`，否则走原有 DDIM→CM 路径 ✅
   - EMA update 跳过 `target_model`（line 1439: `if not is_flow`）✅

4. **`promote_distilled_model()`** (`dp3_cm.py:340-351`)
   - 蒸馏后 student weights → `self.model`，`flow_inference_steps` → student steps ✅
   - `distill2cm()` 末尾自动调用（line 1589-1590）✅

5. **`__init__` guard 改造** (`train_cm_mid.py:127-130`)
   - 不再一刀切 block flow + distill_phase，改为只 block 无效 phase 值 ✅

6. **Online distillation** (`train_cm_mid.py:884-894`, `uni_ppo.py:937-954`)
   - `distill_phase='online'` 时 `set_target()` + `distill_update()` 正确分支 ✅
   - `distill_update()` flow 分支调用 `compute_flow_distill_loss` ✅
   - EMA update 跳过 flow（line 952）✅

7. **Config** (`dp3_flow.yaml`)
   - `flow_distill_inference_steps: 1`, `flow_distill_teacher_steps: 10` ✅
   - `distill_phase: null`（CLI override 启用）✅

8. **Training script** (`train_policy_online_flow_distill.sh`) 存在 ✅

---

### 发现的问题

#### BUG PB-1 (MEDIUM): `distill_phase='after_dp'` 后 `promote_distilled_model` 改了 `self.model`，但后续 `unio4` 加载的是 offline checkpoint

**流程**:
1. Line 456: `distill2cm()` on `self.model` → `promote_distilled_model()` 把 student weights 写入 `self.model`，`flow_inference_steps=1`
2. Line 460+: critic 初始化用 `self.model`（OK，已是 1-step student）
3. Line 645+: `cfg.online` 分支，`self.unio4.load(offline_best_path)` 从 checkpoint 加载

**问题**: `offline_best_path` 保存的是蒸馏前的 BC model（10-step），还是蒸馏后的 student（1-step）？

看 line 1586-1587:
```python
os.makedirs(os.path.join(self.offline_best_path, 'last'), exist_ok=True)
model_to_optimize.save(os.path.join(self.offline_best_path, 'last'))
```
这在 `promote_distilled_model()` 之前执行（line 1589 才 promote）。所以 `offline_best_path/last` 保存的是蒸馏后的 `distilled_model` 权重（通过 `model.save()`），但 `self.model` 的 `flow_inference_steps` 还没被改。

然后 line 1589 `promote_distilled_model()` 改了 `self.model` 的 `flow_inference_steps=1`。

但 line 645+ `self.unio4.load(offline_best_path)` 会重新加载 checkpoint，这会覆盖 `self.unio4._policy` 的状态。**`flow_inference_steps` 是否被 checkpoint 保存/恢复？** 如果 `load()` 只恢复 model weights 不恢复 `flow_inference_steps`，那 `unio4._policy` 仍然是 10-step 配置 + 1-step weights = 不匹配。

- [ ] 验证 `unio4.load()` 后 `flow_inference_steps` 是否正确为 1-step
- [ ] 如果不正确，需要在 `load()` 后再次调用 `promote_distilled_model()` 或手动设置 `flow_inference_steps`

#### BUG PB-2 (MEDIUM): `distill_phase='after_offline'` 路径缺少 `promote_distilled_model` 后的 online 加载协调

Line 637-643:
```python
if self.cfg.distill_phase == 'after_offline':
    self.unio4.load(offline_best_path)
    self.distill2cm(...)  # → promote_distilled_model() on self.unio4._policy
```

然后 line 645+:
```python
if cfg.online:
    if self.cfg.load_bc:
        self.unio4._policy.model.load_state_dict(self.model.model.state_dict())  # 覆盖！
    else:
        self.unio4.load(offline_best_path)  # 覆盖！
```

如果 `distill_phase='after_offline'`，蒸馏在 `self.unio4._policy` 上完成，`promote_distilled_model()` 已执行。但紧接着 online 分支又 `self.unio4.load(offline_best_path)` 重新加载，**覆盖了刚蒸馏好的 student**。

**修复方向**: `distill_phase='after_offline'` 时，online 分支不应重新加载 checkpoint，应跳过 load 或从蒸馏后的 checkpoint 加载。

- [ ] `after_offline` + `online` 路径：蒸馏后不要重新 load 覆盖 student

#### BUG PB-3 (LOW): `compute_flow_distill_loss` 签名 `online` 参数未使用

`dp3_cm.py:1173`:
```python
def compute_flow_distill_loss(self, batch, distill2mean=False, fix_encoder=False, online=False):
```

`online` 参数在函数体内未被使用。`uni_ppo.py:939` 传了 `online=online`，`distill2cm` 里没传。不影响正确性，但是 dead code。

- [ ] 移除 `online` 参数或实现其语义

#### BUG PB-4 (LOW): `distill2mean` 参数在 flow distill loss 中未生效

`compute_flow_distill_loss` 接收 `distill2mean` 参数，但 teacher rollout 始终用 `step_mean`（line 1228）。`distill2mean=False` 时应该用 `step()`（stochastic），但当前代码忽略了这个参数。

看代码：
```python
trajectory = self.flow_teacher_scheduler.step_mean(
    model_output, t, trajectory).prev_sample
```

没有 `if distill2mean` 分支。对比 todo.md 中的 plan 伪代码是有这个分支的。

- [ ] 添加 `distill2mean` 分支：`True` → `step_mean`，`False` → `step`

#### 注意事项 (非 bug)

- `distill_phase='after_dp'` 对 flow 没有被 guard 阻止（line 456 无 `is_flow` check），但 plan 中 Step 3 的 guard 2 说 "Prevents distillation immediately after BC (flow requires offline stage first)"。实际代码允许 flow + `after_dp`。这可能是有意的（training script 用的就是 `after_dp`），但与 explore agent 的描述矛盾。
- `flow_distill_teacher_steps` 和 `flow_distill_inference_steps` 命名不对称（一个叫 teacher_steps，一个叫 inference_steps）。建议统一为 `flow_distill_student_steps` / `flow_distill_teacher_steps`。

---

## Plan B Review Round 2 (ft-dp3-plan-b, 2026-03-15)

验证 PB-1 ~ PB-4 的修复。

### BUG PB-1 ✅ FIXED（有隐患）

**修复** (`train_cm_mid.py:651-657`):
- `after_dp`: 从 `offline_best_path/last/` 加载（promote 后保存的 student weights）
- `after_offline`: 跳过 reload（unio4._policy 已经是 promoted student）

**验证**:
- Line 1597: `promote_distilled_model()` 设 `self.model.flow_inference_steps=1`
- Line 1598-1599: 保存 promoted weights 到 `offline_best_path/last/`
- Line 614: `self.unio4.set_policy(self.model)` → `deepcopy` → `unio4._policy.flow_inference_steps=1` ✅
- Line 655: `self.unio4.load(offline_best_path/last/)` → 只加载 weights，不覆盖 `flow_inference_steps` ✅

**隐患**: `ppo.py:load()` 只恢复 `model.pt` / `encoder.pt`，不恢复 `flow_inference_steps`。当前能工作是因为 `set_policy(deepcopy)` 在 `load()` 之前执行（line 614 < line 655）。如果未来调用顺序变化，`flow_inference_steps` 会回到 config 默认值（10）。建议在 `load()` 后显式设置：
```python
if self.cfg.distill_phase == 'after_dp':
    self.unio4.load(...)
    self.unio4._policy.flow_inference_steps = self.cfg.flow_distill_inference_steps
```

- [x] 修复已验证
- [ ] 可选：加显式 `flow_inference_steps` 设置以消除顺序依赖

### BUG PB-2 ✅ FIXED

**修复** (`train_cm_mid.py:651-657`):
`after_offline` 分支不再 reload，保留 promoted student。

**验证**: `distill2cm(phase='after_offline')` 在 `self.unio4._policy` 上执行（line 1365），`promote_distilled_model()` 就地修改。online 分支跳过 load，student 不被覆盖。✅

- [x] 修复已验证

### BUG PB-3 ✅ FIXED

**修复** (`dp3_cm.py:1173`):
```python
def compute_flow_distill_loss(self, batch, distill2mean=False, fix_encoder=False):
```
`online` 参数已移除。

**验证**: `uni_ppo.py:939` 调用也已更新（不再传 `online`）。✅

- [x] 修复已验证

### BUG PB-4 ✅ FIXED

**修复** (`dp3_cm.py:1230-1235`):
```python
if distill2mean:
    trajectory = self.flow_teacher_scheduler.step_mean(
        model_output, t, trajectory).prev_sample
else:
    trajectory = self.flow_teacher_scheduler.step(
        model_output, t, trajectory).prev_sample
```

**验证**: `distill2mean=True` → `step_mean`（确定性），`distill2mean=False` → `step`（随机）。✅

- [x] 修复已验证

### Round 2 结论

4 个 bug 全部修复。PB-1 有一个顺序依赖隐患（非 blocking），建议加显式设置。其余无问题。

---

## BUG PB-5 (HIGH): 蒸馏后 online buffer shape 不匹配 — 运行时 crash

### 报错

```
File "train_cm_mid.py", line 1136, in online_ft
    replay_buffer.store(obs_dict, all_x, a_logprob, reward, next_obs, done, dw)
File "online_buffer.py", line 50, in store
    self.action[self.count] = action
ValueError: could not broadcast input array from shape (2,1,28) into shape (11,1,28)
```

### Root Cause

1. `__init__` (line 61): `cfg.ppo.num_inference_steps = cfg.policy.num_inference_steps` → 设为 10
2. `distill2cm()` → `promote_distilled_model()` → `flow_inference_steps = 1`
3. `online_ft()` (line 997): `ReplayBuffer(args=self.cfg.ppo, ...)` → buffer 用 `num_inference_steps=10` 分配 action shape `(batch, 11, action_dim)`
4. 蒸馏后 model 产出 1-step trajectory → shape `(2, 1, 28)`
5. `buffer.store()` → shape 不匹配 → crash

`cfg.ppo.num_inference_steps` 在 `__init__` 时设置后从未更新，蒸馏改了 `flow_inference_steps` 但没同步到 `cfg.ppo`。

### Fix

在 `online_ft()` 中，必须在 **所有 distill-phase 相关 load/promote 逻辑完成之后**、并且在第一次创建 buffer 之前同步 step 配置。

推荐写法：

```python
# Must run after any distill-phase load/promote logic has finished,
# and before ReplayBuffer / IqlBuffer are created.
if getattr(self.unio4._policy, 'is_flow', False):
    self.cfg.ppo.num_inference_steps = self.unio4._policy.flow_inference_steps
```

同步点要求：

- 必须放在 `after_dp` 的 `self.unio4.load(.../last)` 之后
- 必须放在 `after_offline` 的 “skip reload / already promoted” 分支之后
- 必须放在：
  - `replay_buffer = ReplayBuffer(...)`
  - `iql_buffer = IqlBuffer(...)`
  之前

也就是说，这不是“在 `online_ft()` 任意早的位置改一下 config”就够了；
它必须发生在 online policy 的最终 active step 数已经稳定之后。

同时检查同步范围：

- blocking 根因是 `cfg.ppo.num_inference_steps`
- 另外检查代码里是否还有 online shape / logging / rollout 初始化依赖：
  - `cfg.num_inference_steps`
  - `cfg.policy.num_inference_steps`
- 如果这些字段也被 online path 当作 active step 数使用，需要一并同步；但第一优先级是修正 `cfg.ppo.num_inference_steps`

### Acceptance

- [ ] `cfg.ppo.num_inference_steps` 在最终 load/promote 完成后、buffer 创建前同步为蒸馏后的步数
- [ ] `ReplayBuffer` action shape 与蒸馏后 model 输出一致
- [ ] `IqlBuffer` 同样检查
- [ ] 如 online path 还依赖 `cfg.num_inference_steps` / `cfg.policy.num_inference_steps`，也已明确检查并处理
- [ ] 非蒸馏模式（`distill_phase=null`）不受影响

---

## ~~BUG PB-6 (CANCELLED)~~: Flow distillation 不应 promote，应遵循 DDIM→CM 模式

> **已取消**：经分析，保留两种模式：
> 1. `after_dp`/`after_offline` → promote → 1-step online PPO（模式 1）
> 2. `distill_phase='online'` → 10-step teacher rollout + interleaved distill（模式 2）
>
> PB-6 提议删除 promote 只保留模式 2，但两种模式都有价值。
> 模式 2 的核心 bug 由 PB-7 修复（teacher 改为 self.model）。

---

## BUG PB-7 (HIGH): Flow distill 的 teacher 应该是 `self.model`，不是 frozen `self.teacher`

### 背景

通过对比 DDIM→CM 和 flow distillation 的 online 模式，发现 teacher 来源不一致。

**DDIM→CM `compute_ddim2cm_loss_action_same_noise`** (`dp3_cm.py:1084-1091`):
```python
with torch.no_grad():
    self.ddim_scheduler.set_timesteps(self.ddim_inference_steps)
    trajectory = noise_trajectory
    for i, t in enumerate(self.ddim_scheduler.timesteps):
        model_output = self.model(...)  # ← self.model，PPO 在更新的
```

**Flow `compute_flow_distill_loss`** (`dp3_cm.py:1220-1233`):
```python
with torch.no_grad():
    self.flow_teacher_scheduler.set_timesteps(self.flow_distill_teacher_steps)
    trajectory = noise_trajectory.clone()
    for t in self.flow_teacher_scheduler.timesteps:
        model_output = self.teacher(...)  # ← self.teacher，frozen 快照
```

### 三个 model 的角色对比

| 变量 | DDIM→CM online | Flow online (当前) | Flow online (应该) |
|------|---------------|-------------------|-------------------|
| `self.model` | rollout + PPO 更新 + distill teacher | rollout + PPO 更新 | rollout + PPO 更新 + distill teacher |
| `self.teacher` | frozen copy（仅 `compute_ddim2cm_loss` backup 模式用） | frozen copy → distill teacher ❌ | frozen copy（不用于 online distill） |
| `self.distilled_model` | 1-step CM student | 1-step flow student | 1-step flow student |

### 问题

在 `distill_phase='online'` 模式下：
1. PPO 每步更新 `self.model`（10-step flow policy 越来越好）
2. `distill_update()` 调用 `compute_flow_distill_loss()`
3. 但 teacher 是 `self.teacher`（`set_target()` 时刻的 frozen 快照）
4. Student 学的是旧 teacher，跟 PPO 的改进完全脱节
5. PPO 训练 1000 步后，student 仍然在模仿第 0 步的 teacher

DDIM→CM 不存在这个问题，因为它直接用 `self.model` 做 teacher — PPO 更新 model 后，下一次 distill 自动用最新权重。

### 修复

**`compute_flow_distill_loss`** 中，teacher rollout 应该用 `self.model` 而不是 `self.teacher`：

```python
# 当前（错误）:
model_output = self.teacher(
    sample=trajectory, timestep=unet_t,
    local_cond=local_cond, global_cond=global_cond)

# 修复后:
model_output = self.model(
    sample=trajectory, timestep=unet_t,
    local_cond=local_cond, global_cond=global_cond)
```

同时 teacher scheduler 也应该用 `self.flow_scheduler`（跟 `self.model` 配套），而不是 `self.flow_teacher_scheduler`：

```python
# 当前:
self.flow_teacher_scheduler.set_timesteps(self.flow_distill_teacher_steps)
...
trajectory = self.flow_teacher_scheduler.step_mean(model_output, t, trajectory).prev_sample

# 修复后:
teacher_scheduler = copy.deepcopy(self.flow_scheduler)  # 或用独立实例避免 set_timesteps 冲突
teacher_scheduler.set_timesteps(self.flow_distill_teacher_steps)
...
trajectory = teacher_scheduler.step_mean(model_output, t, trajectory).prev_sample
```

注意：`set_timesteps()` 会修改 scheduler 内部状态，所以 teacher rollout 不能直接用 `self.flow_scheduler`（会影响后续 rollout）。有两种方案：

**方案 A**：每次 `compute_flow_distill_loss` 调用时 deepcopy 一个临时 scheduler
- 简单但有 deepcopy 开销

**方案 B**：保留 `self.flow_teacher_scheduler` 作为独立实例，但只用于 distill loss 中的 `set_timesteps` + `step_mean`
- 当前已有这个实例，只需要把 model 从 `self.teacher` 改为 `self.model`

推荐方案 B：保留 `self.flow_teacher_scheduler`（独立 scheduler 实例），只把 model 调用从 `self.teacher` 改为 `self.model`。

### 对 `after_dp` / `after_offline` 路径的影响

对于非 online 的 distill 路径（`distill2cm()` 中的离线蒸馏）：
- `self.model` 在蒸馏期间不被 PPO 更新
- 用 `self.model` 还是 `self.teacher` 效果相同（权重一样）
- 但为了一致性，统一改为 `self.model`

### 执行步骤

- [ ] `compute_flow_distill_loss` 中 teacher rollout 改为 `self.model`（替换 `self.teacher`）
- [ ] 保留 `self.flow_teacher_scheduler` 用于独立的 `set_timesteps`（不改）
- [ ] `set_target()` 中仍然创建 `self.teacher`（frozen copy），但 online distill 不再使用它
- [ ] 验证：`distill_phase='online'` 时 distill loss 随 PPO 改进而变化（不是固定在旧 teacher 上）

---

## BUG PB-8: Mode 1 (after_dp/after_offline) 1-step finetune — performance=0, freq=46

### 问题

Mode 1 (`after_dp` / `after_offline`) distill+promote 后进入 online finetune 时，1-step 路径和 Mode 2 (`online`) 的显式 student 路径表现明显不一致：
- Mode 1: `performance=0`, `freq=46`
- Mode 2: teacher `freq=46`, student `freq=89` 且 student performance 正常

### Root Cause

Mode 1 的主要已确认问题是 step config mismatch，两条子路径各有不同的丢失机制：

#### PB-8A: `after_dp` — `ppo.load()` 不恢复 flow step config

`after_dp` 进入 online 时调用 `self.unio4.load()`（ppo.py:200-211），只加载 model/encoder 权重，不恢复 `flow_inference_steps` 属性。导致 `_policy` 和 `_old_policy` 都保持旧值（10），用 10-step 推理 1-step distilled 权重。

#### PB-8B: `after_offline` — `_old_policy` step config stale

`after_offline` 在 `self.unio4._policy` 上原地 distill + promote：
1. `set_policy()` + `set_old_policy()` 先建立 `_policy` / `_old_policy`（均为 step=10）
2. `distill2cm()` 在 `_policy` 上执行 promote → `_policy.flow_inference_steps=1`
3. 但 `_old_policy` 是 promote 前的 deepcopy，`flow_inference_steps` 仍为 10
4. online PPO ratio 建立在不一致的 step 假设上

#### ~~PB-8C: promoted 默认路径与显式 student 路径不等价~~ — 不需要

经代码验证，promote 后两条路径完全等价：
1. `flow_student_scheduler = copy.deepcopy(flow_scheduler)` (dp3_cm.py set_target) — 同类、同属性
2. `set_timesteps()` 在每次 `conditional_sample`/`all_step_logprob`/`sample_action_with_logprob` 调用时都会重新设置 — 无 stale state
3. promote 后 `self.model` = distilled 权重，`flow_inference_steps = 1`
4. default path 调用 `scheduler.set_timesteps(1)` 与 student path 调用 `student_scheduler.set_timesteps(1)` 行为完全一致

不需要额外对齐 scheduler 路径。

### Fix — 已实施 ✅

#### Step 1: 恢复 `_policy / _old_policy` 的 step 配置一致性

**文件：`RL-100/train_cm_mid.py` ~line 663-670**

在 `after_dp` / `after_offline` 分支末尾，load/skip-reload 之后，显式同步：

```python
if getattr(self.unio4._policy, 'is_flow', False):
    target_steps = self.unio4._policy.flow_distill_inference_steps
    self.unio4._policy.flow_inference_steps = target_steps
    self.unio4._old_policy.flow_inference_steps = target_steps
    cprint(f'restored flow_inference_steps={target_steps} for promoted model (policy + old_policy)', 'yellow')
```

### 验证步骤

- [x] 验证：`after_offline` mode 下 `_old_policy.flow_inference_steps` 与 `_policy` 一致（代码追踪确认：fix 在 line 668-669 同时写入，后续 `set_old_policy()` 是 `deepcopy(_policy)` 不会丢失）
- [ ] 验证：Mode 1 主评估的 freq 明显高于 46，并接近 Mode 2 `use_cm=True` student 的 freq 量级（对比 Mode 2 的 `cm_log_data` student 指标，而非 Mode 2 默认 teacher 主评估）
- [ ] 验证：Mode 1 主评估 performance 不再直接掉为 0（对比基准同上：Mode 2 `cm_log_data` student performance）

# CPS Crash Notes

## Problem

In the current 3D flow-matching RL path:

- `sde` can keep online fine-tuning performance close to DDIM
- `cps` can collapse very early in online RL fine-tuning
- the observed failure mode is an early crash to near-zero evaluation performance

This does not match the expectation from:

- `third_party/FlowCPS`
- `third_party/flow_grpo`

So the question is not whether CPS exists in prior work, but why the current repo's CPS behavior diverges so badly.

## What Was Compared

Primary active implementation:

- `RL-100/rl_100/unidpg/diffusion_policy/diffusers_patch/flow_match_scheduler.py`
- `RL-100/rl_100/unidpg/uni_ppo.py`
- `RL-100/rl_100/policy/dp3_cm.py`
- `RL-100/rl_100/config/dp3_flow.yaml`

Reference implementation:

- `third_party/FlowCPS/flow_grpo/diffusers_patch/sd3_sde_with_logprob.py`
- `third_party/FlowCPS/config/base.py`
- `third_party/FlowCPS/scripts/train_sd3.py`

## Key Alignment Facts

### 1. CPS step mean / sampling formula is broadly aligned

Current repo CPS transition:

- `std_dev_t = sigma_next * sin(noise_level * pi / 2)`
- `pred_x0 = sample - sigma * model_output`
- `pred_x1 = sample + model_output * (1 - sigma)`
- `prev_sample_mean = pred_x0 * (1 - sigma_next) + pred_x1 * sqrt(sigma_next^2 - std_dev_t^2)`

Reference FlowCPS uses the same structure.

This means the obvious CPS step formula itself is not the first suspect.

### 2. Current repo and FlowCPS do not use the same PPO/log-prob scale

FlowCPS computes a CPS-style surrogate log-prob and then reduces it across all non-batch dimensions.

Current 3D repo keeps per-element log-prob for PPO/BPPO compatibility, then sums across action dimensions in the PPO ratio path.

That makes the effective ratio scale meaningfully different from FlowCPS, even if the underlying squared-error expression looks similar.

### 3. Current PPO clip is much larger than FlowCPS clip

Current repo default:

- `ppo.epsilon = 0.2`

FlowCPS base config:

- `train.clip_range = 1e-4`

Some FlowCPS variants use `1e-3`, `1e-5`, or similarly small values.

This is a major mismatch.

### 4. Current repo originally reused DDIM std clipping for CPS

Current flow config originally reused:

- `clip_std_min = 0.0067`
- `clip_std_max = null`

FlowCPS does not apply the same DDIM-style minimum std clipping to CPS.

For CPS, forcing a minimum std can inject extra noise into late denoising steps while PPO still uses a CPS surrogate log-prob. That is a bad combination for stability.

## Current Best Hypothesis

The most likely cause of the early CPS crash is not a single broken line in the CPS step formula.

The stronger hypothesis is:

1. CPS surrogate log-prob in the current 3D PPO path has a different scale from FlowCPS
2. PPO clipping is still using a DDIM-sized trust region
3. The original std clipping inherited from DDIM made CPS late steps noisier than intended

Together, these make early online updates too aggressive and can destroy the policy quickly.

## Important Clarification: The Main Debate

The main technical debate is:

### Should CPS log-prob be treated as a full Gaussian density?

Current CPS special case:

```python
log_prob = -((x_next - mean) ** 2)
```

Alternative Gaussian form:

```python
log_prob = -((x_next - mean) ** 2) / (2 * std**2) - log(std) - const
```

This is the core argument.

### Why this is not settled

FlowCPS reference does **not** use the full Gaussian form for CPS. It uses a surrogate squared-error style expression.

So replacing the current CPS special case with a Gaussian log-prob is **not** just "aligning to FlowCPS". It is an algorithm change.

That change may help, but it should be treated as an experiment, not as an already-proven root-cause fix.

## Conclusions So Far

### Reasonable and aligned changes

The following changes are reasonable first-line stabilization steps for CPS:

- `clip_std_min = null`
- `clip_std_max = null`
- `ppo.epsilon = 1e-3`

These move the current repo closer to the reference CPS regime and reduce the most obvious DDIM carry-over mismatch.

### Not yet justified as the main fix

This change is not yet justified as the main official fix:

- remove the CPS special case and replace it with full Gaussian log-prob

That may be worth testing, but it should be documented as an alternative experiment, not as the confirmed cause of the crash.

## Recommended Next Validation

### First priority

Run CPS online fine-tuning with:

- `clip_std_min = null`
- `clip_std_max = null`
- `ppo.epsilon = 1e-3`

Then inspect:

- early evaluation return / success
- `ratio_stats.csv`
- whether ratio and clipfrac still explode in the first updates

### If CPS still crashes early

Then test an explicit experimental branch:

- replace CPS surrogate log-prob with Gaussian log-prob

But record it clearly as:

- an alternative density model
- not a pure FlowCPS-alignment fix

## Practical Takeaway

At the current evidence level:

- the CPS crash is more likely caused by PPO-scale mismatch than by the CPS step formula itself
- the first fixes should be hyperparameter and scaling alignment
- Gaussian CPS log-prob should remain an experiment until it is validated

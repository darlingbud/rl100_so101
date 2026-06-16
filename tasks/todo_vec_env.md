# VecEnv Online FT TODO

## Current Status
- `online_ft()` now supports a vec rollout branch guarded by `ppo.use_vec_env_online`.
- The new vec PPO update path is `dp_align_update_no_share_vec_batched()` in `RL-100/rl_100/unidpg/uni_ppo.py`.
- The old `dp_align_update_no_share_vec()` is intentionally left unused.
- Vec replay buffer has been rewritten in `RL-100/rl_100/dppo/online_buffer_vec.py`.
- 3D runners now expose `vec_env` / `env_num` in:
  - `RL-100/rl_100/env_runner/adroit_runner.py`
  - `RL-100/rl_100/env_runner/dmc_runner.py`
- `SubprocVecEnv` auto-reset is handled by writing `info['terminal_observation']` into the rollout buffer for done transitions.

## Supported In Vec v1
- 3D online finetune PPO main path.
- `all_step_action_logprob()` rollout path.
- Standard reward path and `scale_strategy == 'number'`.
- PPO loss / ratio / clipping unchanged from single-env logic.

## Explicitly Not Supported In Vec v1
- `ppo.iql_ft`
- `update_phase == 'outloop'`
- `ppo.iql_adv`
- `ppo.idql_rollout`
- `ppo.scale_strategy == 'dynamic'`

These combinations are blocked in `online_ft()` and should remain blocked until a dedicated vec implementation is added.

## Remaining Work

### 1. Extend vec_env exposure to other 3D runners
- Add `vec_env`, `env_num`, and per-env factory functions to the remaining 3D runners used by `train_cm_mid.py`, especially:
  - `RL-100/rl_100/env_runner/dexart_runner.py`
  - `RL-100/rl_100/env_runner/metaworld_runner.py`
- Keep existing eval semantics intact.
- Use `SubprocVecEnv('spawn')` consistently.

### 2. Add config entries
- Add these fields to the active training configs used for 3D online FT:
  - `ppo.use_vec_env_online: false`
  - `ppo.train_env_num: ${env_num}`
- Keep defaults backward compatible.

### 3. End-to-end smoke test
- Run one minimal online finetune smoke test with:
  - `ppo.use_vec_env_online=true`
  - `ppo.train_env_num=2` or `4`
  - `ppo.iql_ft=false`
  - `ppo.idql_rollout=false`
  - non-dynamic reward scaling
- Verify:
  - vec rollout fills one batch
  - one call to `dp_align_update_no_share_vec_batched()` completes
  - `old_logprob`, `new_logprob`, and `ratio` shapes match
  - `adv` is finite
  - no scheduler attribute errors

### 4. Runtime validation for terminal handling
- Validate that `terminal_observation` exists for all done transitions in the chosen vec envs.
- If a specific env wrapper does not provide the expected terminal obs structure, add env-specific handling before rollout is considered stable.

### 5. Consider vec support for blocked branches
- If needed later, implement vec-aware versions of:
  - IQL online finetune buffer/store path
  - `idql_rollout` path
  - `dynamic` reward scaling per-env state handling
- Do not enable these without explicit testing.

## Notes For Work Agent
- Treat `dp_align_update_no_share()` as the single source of truth for PPO semantics.
- Any future vec updates should be cloned from the single-env implementation first, then minimally adapted.
- Do not revive the old `dp_align_update_no_share_vec()` path.
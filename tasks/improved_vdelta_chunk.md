# TODO: Implement `chunk_vdelta_gae` (clean v1)

## Goal

Add a new offline chunk advantage mode:

- `offline_chunk_adv_mode = chunk_vdelta_gae`

This mode should implement **multi-chunk boundary GAE smoothing** on top of the current
chunk-level vdelta signal, while keeping the implementation as controlled and interpretable as possible.

This v1 is **not** allowed to introduce imagined policy sampling.
It must only test whether **cross-chunk smoothing itself** helps.

---

## Design summary

Current best scalar chunk signal:

```text
chunk_vdelta_scalar:
A = R_chunk + gamma^K * V(s_next) - V(s)
```

New target signal:

```text
chunk_vdelta_gae:
delta_k = R_k + gamma^K * V(s_{k+1}) - V(s_k)
A_0     = delta_0 + (gamma^K * lambda) delta_1 + (gamma^K * lambda)^2 delta_2 + ...
```

where:
- `K = n_action_steps`
- rollout happens across **chunk boundaries**
- v1 uses **repeat_first** chunk source:
  - first chunk = current dataset/policy chunk
  - later imagined chunks = repeat the same normalized chunk action

---

## Hard requirements

### 1. v1 must stay controlled
Do **not** add in v1:
- policy-sampled imagined chunks
- latent policy rollout
- best-of-N action search
- new candidate distributions
- new ratio formulas
- per-step intra-chunk rollout

### 2. chunk-level only
Because `use_action_embed=True` and chunk dynamics already consume the full flatten chunk action,
this mode must operate at the **chunk boundary level** only.

### 3. Strict regression guard
When:

```yaml
chunk_vdelta_gae_n_rollout: 1
chunk_vdelta_gae_chunk_source: repeat_first
```

the new mode must be **numerically identical** to `chunk_vdelta_scalar`:

- compare raw pre-normalization scalar values
- compare post-normalization returned advantages

This is mandatory.

### 4. Terminal handling must be explicit
Need `alive_mask`:
- terminal samples must stop contributing future deltas
- future imagined state updates must be frozen for dead samples
- do not only mask the one-step delta

### 5. `chunk_vdelta_gae` must be scalar-ratio only
Add an explicit safety check:
- `chunk_vdelta_gae` only allowed with `offline_chunk_ratio_mode='scalar'`

### 6. `chunk_vdelta_gae` v1 only supports `final_reward=True`
If selected with `final_reward=False`, raise a clear error.

---

## Files to change

### File A
`RL-100/rl_100/config/dp3_cm_epsilon.yaml`

### File B
`RL-100/rl_100/unidpg/uni_ppo.py`

### File C
`scripts/train_policy_chunk_two_stage.sh`

Do **not** modify policy sampling code in v1.
No `predict_action_from_latent` in this version.

---

## File A: config changes

Add these config keys after `offline_chunk_adv_mode`:

```yaml
chunk_vdelta_gae_n_rollout: 3
chunk_vdelta_gae_lambda: 0.95
chunk_vdelta_gae_chunk_source: repeat_first
```

Meaning:
- `chunk_vdelta_gae_n_rollout`: number of chunk boundaries to evaluate, including the first/current one
- `chunk_vdelta_gae_lambda`: GAE lambda across chunk boundaries
- `chunk_vdelta_gae_chunk_source`: v1 must only support `repeat_first`

---

## File B: `uni_ppo.py` changes

## Step 1 — extend `_get_offline_chunk_modes()`

Add new valid mode:

```python
'chunk_vdelta_gae'
```

Required allowlist behavior:
- `chunk_vdelta_gae` must be allowed for scalar ratio branch
- but explicitly reject:

```python
if adv_mode == 'chunk_vdelta_gae' and ratio_mode != 'scalar':
    raise ValueError("chunk_vdelta_gae requires offline_chunk_ratio_mode='scalar'")
```

---

## Step 2 — factor out shared one-chunk scalar delta helper

Before implementing `chunk_vdelta_gae`, first factor the existing single-chunk scalar-vdelta math into a reusable helper.

Create something like:

```python
@torch.no_grad()
def _compute_single_chunk_boundary_delta(
    self,
    policy_features,        # (B, n_obs_steps, feat)
    chunk_actions,          # normalized chunk action
    dynamics,
    value,
    iql,
    gamma,
):
    ...
    return {
        "delta": delta,                     # (B,)
        "reward": reward_t,                 # (B,)
        "terminal": terminal_t,             # (B,)
        "value_now": value_now,             # (B,)
        "value_next": value_next,           # (B,)
        "next_policy_features": next_policy_features,
        "next_single_nob": next_single_nob,
    }
```

This helper must be the **shared source of truth** for:
- `chunk_vdelta_scalar`
- `chunk_vdelta_gae`

Reason:
- guarantees `n_rollout=1` regression
- avoids drift between two separate implementations

---

## Step 3 — implement `_compute_chunk_gae_advantage_vdelta()`

Add new helper after scalar chunk helper.

Suggested signature:

```python
@torch.no_grad()
def _compute_chunk_gae_advantage_vdelta(
    self,
    nobs_features,      # (B, n_obs_steps * feat_dim)
    chunk_actions,      # normalized chunk action for the first/current chunk
    dynamics,
    value,
    iql,
    gamma,
    lamda,
    n_rollout_chunks,
    chunk_source,
):
    ...
```

### Required behavior

1. Decode:
   - `policy_features = nobs_features.reshape(B, n_obs_steps, feat_dim)`
   - `current_policy_features = policy_features`
   - `current_chunk = chunk_actions`

2. Initialize:
   - `gamma_T = gamma ** chunk_actions.shape[1]`
   - `alive_mask = ones(B,)`
   - `deltas = []`

3. For each rollout chunk `k`:
   - call `_compute_single_chunk_boundary_delta(...)`
   - get:
     - `delta`
     - `terminal`
     - `next_policy_features`
   - multiply current delta by `alive_mask`
   - append masked delta
   - update alive mask:
     - `alive_mask = alive_mask * (1 - terminal_t)`
   - freeze dead-sample state updates:
     - dead samples must keep previous state
     - do **not** advance imagined state for dead samples
   - choose next chunk according to `chunk_source`

### v1 chunk source behavior

Only support:

```python
chunk_source == "repeat_first"
```

Meaning:
- all imagined future chunks reuse the same normalized `chunk_actions`

If any other source is requested, raise:
```python
raise ValueError(...)
```

### Freeze dead-sample state update

Do not only zero future deltas.
Also freeze state updates for dead samples, e.g. conceptually:

```python
next_policy_features = alive_mask * next_policy_features + (1 - alive_mask) * current_policy_features
```

with proper broadcasting.

### GAE step

After collecting chunk-boundary deltas:

```python
deltas = torch.stack(deltas)
gae_adv = self.GAE_withQ(deltas, gamma_T, lamda)
advantages = gae_adv[0]
```

Then normalize exactly as current scalar chunk modes do.

### Output shape

Returned `advantages` must be shape:

```python
(B,)
```

or shape fully compatible with current scalar-ratio branch.

---

## Step 4 — wire into `update_distribution()`

Only wire `chunk_vdelta_gae` into the **chunk + final_reward=True** branch.

Required behavior:
- `chunk_adv_mode == 'chunk_vdelta_gae'` is supported only when `final_reward=True`
- if `final_reward=False`, raise a clear error

Use config values:

```python
n_rollout = int(getattr(self.cfg, 'chunk_vdelta_gae_n_rollout', 3))
gae_lamda = float(getattr(self.cfg, 'chunk_vdelta_gae_lambda', 0.95))
chunk_source = str(getattr(self.cfg, 'chunk_vdelta_gae_chunk_source', 'repeat_first'))
```

---

## Step 5 — keep launcher defaults unchanged

In `train_policy_chunk_two_stage.sh`:

### Add pass-through env vars
Add:

```bash
CHUNK_VDELTA_GAE_N_ROLLOUT=${CHUNK_VDELTA_GAE_N_ROLLOUT:-3}
CHUNK_VDELTA_GAE_LAMBDA=${CHUNK_VDELTA_GAE_LAMBDA:-0.95}
CHUNK_VDELTA_GAE_CHUNK_SOURCE=${CHUNK_VDELTA_GAE_CHUNK_SOURCE:-repeat_first}
```

### Pass them to hydra/common params
Add:
```bash
chunk_vdelta_gae_n_rollout=${CHUNK_VDELTA_GAE_N_ROLLOUT} \
chunk_vdelta_gae_lambda=${CHUNK_VDELTA_GAE_LAMBDA} \
chunk_vdelta_gae_chunk_source=${CHUNK_VDELTA_GAE_CHUNK_SOURCE} \
```

### Do NOT change default sweep
Keep existing launcher default behavior.
Do not silently change:
- default `CHUNK_LOSS_MODE_COMBOS`
- baseline sweep settings

New mode should be opt-in only.

---

## Verification / acceptance checklist

### Check 1 — regression
Run on the same batch:

- `chunk_vdelta_scalar`
- `chunk_vdelta_gae` with:
  - `n_rollout_chunks=1`
  - `chunk_source=repeat_first`

Verify identical:
- raw scalar delta before normalization
- normalized final returned advantage

### Check 2 — shape sanity
Verify:
- no ratio / advantage broadcast mismatch
- returned scalar advantage is compatible with scalar ratio branch
- no NaNs

### Check 3 — terminal behavior
Construct a batch with terminal samples and verify:
- future deltas after terminal are zero
- dead samples do not continue to update imagined states

### Check 4 — smoke training
Run tiny offline chunk training with:
- `offline_chunk_ratio_mode=scalar`
- `offline_chunk_adv_mode=chunk_vdelta_gae`
- `chunk_vdelta_gae_n_rollout=2`
- `chunk_vdelta_gae_chunk_source=repeat_first`

Need:
- no crash
- no shape mismatch
- no exploding advantages

### Check 5 — minimal ablation
Run:
1. `scalar_iql`
2. `chunk_vdelta_scalar`
3. `chunk_vdelta_gae` with rollout=2
4. `chunk_vdelta_gae` with rollout=3

Keep everything else fixed.

---

## Explicit non-goals for this task

Do **not** in this task:
- add imagined policy-sampled chunks
- modify `predict_action` / policy sampling code
- change online PPO
- change dataset stride construction
- change ratio formulas
- add intra-chunk per-step rollout

Those are separate experiments.

---

## One-line intent

This task is a **controlled signal-improvement experiment**:
test whether cross-chunk boundary GAE smoothing improves the current chunk scalar-vdelta signal, without introducing policy-rollout distribution shift or new sampling semantics.

---

## Implementation guard (read before touching any code)

### Functions that must NOT be modified

The following functions and code paths must remain completely untouched — no refactoring, no cleanup, no "while I'm here" edits:

```
_compute_advantage_actor_only()            # scalar_iql path
_compute_chunk_step_advantages_vdelta()    # per_step_vdelta path
NStepValueEstimation()                     # non-chunk multi-step path
update_distribution_single_step()          # single-action mode
dp_align_update_no_share()                 # online PPO
update_distribution() — opt_steps==1 branch
update_distribution() — chunk_as_single_action=False branch
```

### Refactoring constraint on `_compute_chunk_scalar_advantage_vdelta`

When factoring out `_compute_single_chunk_boundary_delta`, the external behavior of `_compute_chunk_scalar_advantage_vdelta` must be completely unchanged.
Same input → bitwise-identical output tensor before and after the refactor.
No "opportunistic" changes to logic, naming, or normalization.

### The `else`-branch trap in `update_distribution`

`chunk_vdelta_scalar` currently falls into a catch-all `else` branch.
When adding `chunk_vdelta_gae`, the agent **must** convert the structure to:

```python
if chunk_adv_mode == 'scalar_iql':
    ...
elif chunk_adv_mode == 'per_step_vdelta':
    ...
elif chunk_adv_mode == 'chunk_vdelta_scalar':
    ...
elif chunk_adv_mode == 'chunk_vdelta_gae':
    ...
else:
    raise ValueError(f"Unsupported chunk_adv_mode: {chunk_adv_mode}")
```

Do **not** put the new mode into the `else` slot.

### `predict_r` guard must live inside the shared helper

`_compute_single_chunk_boundary_delta` must include:

```python
if not getattr(dynamics, 'predict_r', False):
    raise ValueError("chunk_vdelta_gae requires predict_r=True ...")
```

This guard must not be removed from `_compute_chunk_scalar_advantage_vdelta` and must not only live in the outer calling function.

### `alive_mask` broadcasting — explicit unsqueeze required

`alive_mask` is shape `(B,)`. `policy_features` is `(B, n_obs_steps, feat_dim)`.
Do not rely on implicit broadcasting. Use explicit reshape:

```python
mask = alive_mask[:, None, None]  # (B, 1, 1)
current_policy_features = mask * next_policy_features + (1 - mask) * current_policy_features
```

### Use `CONST_EPS`, not a hardcoded value

Advantage normalization must use the existing module-level constant `CONST_EPS`, not a magic number like `1e-8`.

### `final_reward=False` ValueError placement

The `ValueError` for unsupported `final_reward=False` must be placed **inside** the `final_reward=False` chunk dispatch block — not at the top of `update_distribution`. Otherwise it would fire even for `final_reward=True`.

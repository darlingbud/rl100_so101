# Dynamics Chunk Bug Review

## Goal

This document focuses only on the likely bugs / semantic issues in `ensemble_dynamics_for_batch.py`, especially around:

- `prediction_mode="full"` vs `prediction_mode="last"`
- how `single obs feature` is used
- chunk evaluation and rollout-time Q evaluation

This is intentionally separate from BPPO / actor-loss discussion.

---

## File in scope

- `ensemble_dynamics_for_batch.py`

Relevant methods:

- `step(...)`
- `multi_step(...)`
- `multi_step_evaluation(...)`
- `chunk_evaluation(...)`

---

## High-level conclusion

## What does NOT look like the main bug
The core `step(...)` interface itself is mostly consistent:

- in `prediction_mode="last"`, the main state input is the single-frame feature
- in `prediction_mode="full"`, the main state input is the full observation window (flattened), when `policy_features` is provided

So the `step(...)` method itself does **not** currently look like the main bug.

## What DOES look buggy / semantically wrong
There are two high-priority issues:

1. **`chunk_evaluation()` appears to use the wrong state representation for chunk critic evaluation**
2. **`multi_step_evaluation()` does not update the state input used for `Q(...)` across rollout steps**

There is also one lower-priority semantic check:

3. **`terminal_fn(...)` in `prediction_mode="full"` may receive flattened full-window features instead of single-step features**

---

## 1. `step(...)` review

Current logic:

```python
if self.prediction_mode == "full" and policy_features is not None:
    input_features = policy_features.reshape(batch_size, -1)
else:
    input_features = nobs_features
```

### Interpretation
- `last` mode:
  - use single-frame feature as the dynamics input
- `full` mode:
  - use full observation window feature as the dynamics input
  - `single_nob_features` is still passed in, but is not the main predictor input

### Verdict
This part is **internally consistent**.

So if the concern is specifically:
> “Is `single obs feature` incorrectly used inside `step(...)` for full vs last?”

Then the answer is:

- **No obvious bug found in `step(...)` itself**
- the full/last handling there is reasonable

---

## 2. High-priority bug: `chunk_evaluation()`

Current logic is effectively:

```python
policy_features = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
single_nob_features = policy_features[:, -1, :]
Q_value = Q(single_nob_features, nactions)
return Q_value
```

### Why this is likely wrong
For chunk critic / IQL with `chunk_as_single_action=True`, the expected state input for `Q(s, a_chunk)` is the **full flattened state feature**, not only the last-frame feature.

In other words, the chunk critic state should be something like:

```python
state_features = nobs_features.reshape(batch_size, -1)
```

not:

```python
single_nob_features = policy_features[:, -1, :]
```

### Why this matters
If `chunk_evaluation()` is used for:
- chunk ranking diagnostics
- chunk Q evaluation
- chunk candidate comparison

then using only the last-frame feature will mismatch the critic's actual training/input semantics.

### Required patch
Replace the current state input with:

```python
batch_size = nactions.shape[0]
state_features = nobs_features.reshape(batch_size, -1)
Q_value = Q(state_features, nactions)
return Q_value
```

### Priority
**Must fix**

---

## 3. High-priority bug: `multi_step_evaluation()` does not update Q state input across rollout

Current pattern is effectively:

```python
if self.cfg.online:
    if self.cfg.ppo.iql_ft:
        iql_input = state_dict
    else:
        iql_input = nobs_features
else:
    iql_input = nobs_features

for i in range(n_step_actions):
    Qs.append(Q(iql_input, nactions[:, i, :self.action_dim]))
    next_obs, reward, terminal, info = self.step(...)
    ...
```

### Problem
`iql_input` is initialized once before the loop and is **not updated** inside the rollout loop.

That means:
- the dynamics rollout state evolves
- rewards / next states evolve
- but `Q(...)` is still evaluated on the **initial state input** for all rollout steps

This is semantically wrong if the intention is:
- evaluate Q along the rollout trajectory
- estimate multi-step returns / GAE-like quantities
- compare chunk candidate quality over imagined rollout

### Why this matters
This can poison:
- multi-step evaluation
- GAE-style diagnostics
- ranking diagnostics based on rollout-time Q estimates

### Required patch
Inside the rollout loop, after state update, refresh the `Q(...)` input using the **current** rollout state.

At minimum:

- if using feature tensors:
  - update `iql_input` each step from current `policy_features`
- if using state_dict in online mode:
  - ensure there is an equivalent updated representation, or do not use stale `state_dict`

### Priority
**Must fix**

---

## 4. Lower-priority semantic check: `terminal_fn(...)` input in `full` mode

Current code passes:

```python
terminal = self.terminal_fn(input_features_np, action_np, next_obs, self.env)
```

But in `prediction_mode="full"`:

- `input_features_np` is flattened full-window feature
- not a single-step feature

### Why this may matter
If `terminal_fn(...)` expects:
- only the current single-step state
- environment-space single-state representation

then passing flattened full-window feature may be semantically wrong.

### Is this definitely a bug?
Not necessarily.

It depends on how `terminal_fn(...)` is implemented.

### Action item
Check whether `terminal_fn(...)` supports:
- full-window flattened feature inputs

If not, adapt the terminal call for `full` mode.

### Priority
**Check, but not as urgent as the first two**

---

## 5. What does NOT need rewriting right now

The following parts do not currently look like the main issue:

1. `step(...)` full/last input routing
2. `multi_step(...)` full/last rollout update logic
3. passing:
   - `single_nob_features`
   - `policy_features`
   together into `step(...)`

These are broadly consistent with the intended `prediction_mode` semantics.

---

## 6. Minimal patch checklist

### Must-do
- [ ] Fix `chunk_evaluation()` to use full flattened state features for chunk critic input
- [ ] Fix `multi_step_evaluation()` so Q state input is updated along rollout

### Should-check
- [ ] Verify whether `terminal_fn(...)` supports full-window flattened state input in `prediction_mode="full"`

---

## 7. Acceptance criteria

This dynamics-side bugfix work is complete if:

1. `chunk_evaluation()` evaluates `Q(s, a_chunk)` using full chunk-state features
2. `multi_step_evaluation()` no longer uses stale initial state input for all Q evaluations
3. `step(...)` still works unchanged for both:
   - `prediction_mode="last"`
   - `prediction_mode="full"`
4. terminal behavior in `full` mode is either confirmed valid or explicitly fixed

---

## Final one-line summary

**The main dynamics-side issue is not `step(...)` full/last handling itself, but that chunk-Q evaluation currently uses the wrong state representation, and multi-step Q evaluation uses stale state input across rollout.**

# Offline Chunk Dataset Stride Mismatch Hypothesis

## Goal

This document summarizes a highly plausible root-cause hypothesis for why:

- online `chunk_as_single_action` works
- offline `chunk_as_single_action` does not work

The hypothesis is:

> **offline chunk data is constructed with stride = 1, while online chunk rollouts operate with stride = n_action_steps, so they do not represent the same decision process.**

This document is intended for the work agent to verify and patch.

---

## 1. Problem statement

Current suspected mismatch:

### Online chunk data / rollout semantics
Online chunk control uses **non-overlapping chunk decisions**:

- sample 1:
  - `s_0, a_{0:15}, s_16, r`
- sample 2:
  - `s_16, a_{16:31}, s_32, r`

So online decisions happen every `n_action_steps`.

This is a true **macro-action / SMDP style decision process**.

---

### Offline chunk dataset semantics
Offline chunk dataset currently appears to use **sliding windows with stride = 1**:

- sample 1:
  - `s_0, a_{0:15}, s_16, r`
- sample 2:
  - `s_1, a_{1:16}, s_17, r`
- sample 3:
  - `s_2, a_{2:17}, s_18, r`

So offline treats **every single intermediate state** as a valid chunk decision point.

This is **not** the same decision process as online chunk execution.

---

## 2. Why this matters

## 2.1 Online chunk is an SMDP-style decision process
In online PPO / chunk execution:

- the policy chooses one chunk action at a chunk boundary state
- the system commits to that chunk for `n_action_steps`
- then the next decision happens only after the chunk completes

This is aligned with chunk-value targets such as:

- `r + gamma^n * V(s_next)`

So online actor and critic are both solving a **chunk-level decision problem**.

---

## 2.2 Offline stride=1 constructs a different decision problem
If offline uses stride=1 sliding windows, then the critic sees training samples that mean:

> “At *every* environment step, I may re-plan a fresh future chunk of length `n_action_steps`.”

This is not the same as:

> “At chunk boundaries only, I choose one chunk and commit to it.”

So offline `Q(s, a_chunk)` is trained on a different state-action distribution and possibly a different Bellman semantics.

---

## 2.3 This can directly break `scalar_iql = Q(s, a_chunk) - V(s)`
Current evidence already suggests that offline chunk actor fails when using:

- `scalar_iql = Q(s, a_chunk) - V(s)`

If the chunk critic is trained on stride=1 sliding windows, it may be learning:

- chunk quality at arbitrary overlapping intermediate states

while the actor deployed online actually needs:

- chunk quality only at true chunk decision boundary states

This can make the critic signal unreliable for actor updates, even if the critic training loss looks fine.

---

## 3. Why this hypothesis is strong

This hypothesis is consistent with all current observations:

1. **online strict chunk-scalar PPO works**
2. **offline chunk fails when advantage source is `scalar_iql`**
3. changing ratio aggregation does not solve offline drop
4. chunk size reduction alone does not solve offline drop

This suggests the problem is not primarily:
- PPO ratio shape
- chunk PPO semantics
- action dimensionality alone

Instead, the problem may be:
- **offline chunk critic is trained on the wrong decision process**

---

## 4. What the work agent should verify

The work agent should inspect the offline chunk dataset builder / transition constructor and answer:

### Q1
For chunk-as-single-action offline data, is the chunk dataset generated with:

- stride = 1
or
- stride = `n_action_steps`

### Q2
Does offline critic / IQL training see chunk samples starting from:
- every state
or
- only chunk boundary states

### Q3
Is the reward target consistent with the same stride?
For example:
- chunk reward = discounted sum over the next `n_action_steps`
- next state = state after exactly `n_action_steps`

But if the start states use stride=1 while deployment decisions use stride=`n_action_steps`, then semantics are mismatched.

---

## 5. Required experiments

## Experiment A (highest priority)
### Rebuild offline chunk dataset with stride = `n_action_steps`

Only keep true chunk-boundary transitions:

- `s_0, a_{0:15}, s_16, r`
- `s_16, a_{16:31}, s_32, r`
- ...

Then retrain:
- chunk critic / IQL
- offline BPPO

### Expected interpretation
If offline chunk suddenly becomes stable or improves, then this strongly supports:
- **dataset stride mismatch is a key root cause**

---

## Experiment B
### Keep critic as-is, but restrict offline BPPO actor updates to chunk-boundary samples only

Even if the critic was trained on stride=1 data, this experiment can help separate:

- actor update distribution mismatch
vs
- critic training distribution mismatch

### Expected interpretation
If actor update becomes more stable using only chunk-boundary samples, then actor-side state distribution mismatch is part of the issue.

---

## Experiment C
### Full SMDP-aligned pipeline
Train and update only on chunk-boundary samples:

- critic / IQL uses stride = `n_action_steps`
- offline actor update also uses stride = `n_action_steps`

This is the cleanest test of whether offline chunk works when dataset semantics match online chunk execution.

---

## 6. Required code tasks for work agent

### Task 1
Find the offline chunk dataset construction code path.

### Task 2
Add a configuration option such as:

```python
chunk_stride_mode: "sliding" | "boundary"
```

or directly:

```python
chunk_stride = 1 or n_action_steps
```

### Task 3
Implement chunk-boundary dataset construction:
- start index increments by `n_action_steps`
- not by 1

### Task 4
Make it possible to:
- train critic on boundary-only chunk data
- run offline BPPO actor updates on boundary-only chunk data

### Task 5
Preserve the old sliding-window behavior as an ablation baseline.

---

## 7. Acceptance criteria

This work is complete if:

1. The offline chunk dataset builder supports both:
   - stride = 1
   - stride = `n_action_steps`
2. Critic / IQL can be trained on boundary-only chunk data
3. Offline BPPO can be run on boundary-only chunk data
4. There is a clean experiment comparing:
   - sliding-window offline chunk
   - boundary-only offline chunk

---

## 8. Final one-line summary

**A major likely root cause is that online chunk control is a chunk-boundary SMDP decision process, while offline chunk data is currently built with stride=1 sliding windows, so offline critic/actor are learning on a different decision semantics than online execution.**

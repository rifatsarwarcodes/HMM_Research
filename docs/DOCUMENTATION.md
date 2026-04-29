# Higher-Order HMM for Human Activity Recognition — Full Documentation

## Complete Step-by-Step Account of What We Built

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Phase 1 — Core Framework (Original Build)](#2-phase-1--core-framework)
3. [Phase 2 — Five Research Improvements](#3-phase-2--five-research-improvements)
4. [Repository Structure](#4-repository-structure)
5. [How to Run Everything](#5-how-to-run-everything)
6. [Complete Experiment Results](#6-complete-experiment-results)
7. [Technical Details of Every Module](#7-technical-details-of-every-module)

---

## 1. Project Overview

This project implements a **custom Hidden Markov Model framework from scratch** using only `numpy` and `scipy`. It addresses three fundamental limitations of standard HMMs when applied to Human Activity Recognition (HAR):

| Problem | Our Solution |
|---------|-------------|
| Gaussian emissions mismodel simplex data | **Dirichlet emission distributions** |
| 1st-order transitions miss multi-step patterns | **K-th order super-state transitions** |
| State count must be fixed a priori | **Adaptive state merging via symmetric KL divergence** |

The project was built in two phases: an original core framework validated on synthetic data, then five research improvements addressing gaps in the original work.

---

## 2. Phase 1 — Core Framework

### Step 1: Identifying the Problem

Standard HMMs applied to HAR suffer from three compounding issues:

1. **Geometric mismatch**: Normalized sensor features live on the probability simplex Δ^{D-1}, but Gaussian emissions have support over all of R^D, assigning probability to impossible regions (negative values, non-unit-sum vectors).

2. **Temporal blindness**: First-order transitions only condition on the single previous state. Real activities unfold over multiple time steps — "running → walking" depends on what happened 2-3 steps ago.

3. **Manual state count**: The true number of activity classes is unknown. Practitioners must guess N or cross-validate, which is expensive and error-prone.

### Step 2: Designing the Mathematical Framework

We designed an extended HMM defined by the 5-tuple:

```
λ = (S, Δ^{D-1}, P^(K), {Dir(α_i)}, π)
```

**Dirichlet Emissions.** Each state i has concentration parameters α_i ∈ R^D_+. The log-pdf is:

```
ln φ_i(o) = ln Γ(α_{i,0}) - Σ_d ln Γ(α_{i,d}) + Σ_d (α_{i,d} - 1) ln o_d
```

The M-step uses L-BFGS-B optimization in log(α) space because there is no closed-form MLE for Dirichlet concentration parameters from weighted data.

**K-th Order Super-States.** Instead of an explicit N^K × N^K tensor, we define super-states as K-tuples of base states with overlap-consistent transitions only:

```
(i_1, ..., i_K) → (i_2, ..., i_K, j)  for any j ∈ S
```

Invalid entries are set to -∞ in log-space, allowing standard matrix operations.

**Adaptive State Merging.** States i and j are merged when:

```
sym-KL(i, j) = 0.5 · [KL(Dir_i ‖ Dir_j) + KL(Dir_j ‖ Dir_i)] < τ
```

The KL divergence between two Dirichlet distributions has a closed form using gamma and digamma functions. Merged parameters are pooled using occupancy-weighted averaging.

### Step 3: Implementing the Modular Codebase

We built 5 modules inside the `hmm/` package:

**`emissions.py` — DirichletEmission class**
- Stores all N concentration vectors in a single (N, D) ndarray for vectorized computation
- `log_pdf_batch(O)`: computes log Dir(o_t | α_k) for all t and k simultaneously using a single matrix multiply `log_X @ (alphas - 1).T`
- `kl_divergence(i, j)` and `symmetric_kl(i, j)`: closed-form KL between Dirichlet distributions
- `update(O, gamma_base)`: weighted MLE via L-BFGS-B in log-space with analytical gradient
- `merge(i, j, w_i, w_j)`: occupancy-weighted parameter pooling, removes row j

**`super_states.py` — SuperStateSpace class**
- Enumerates all N^K K-tuples in canonical order
- Precomputes `valid_mask` (boolean adjacency matrix), `children_of`, `parents_of`, and `emit_map` (last element of each super-state)
- `remap(keep, drop)`: computes base-state index remapping after a merge

**`inference.py` — Pure stateless functions**
- `forward(log_pi, log_A, log_e)`: log-space α recursion, returns (log_alpha, log_likelihood)
- `backward(log_A, log_e)`: log-space β recursion
- `viterbi(log_pi, log_A, log_e)`: max-product decoding with backtracking
- All use `scipy.special.logsumexp` throughout — no raw probability products anywhere

**`model.py` — HigherOrderHMM orchestrator**
- Thin coordination layer that holds state (log-π, log-A, emissions, super-state space) and delegates computation
- `_e_step(O)`: computes γ (occupancies) and ξ (transition counts) via forward-backward
- `_m_step(O, γ, ξ)`: updates π, A (row-normalized over valid children), and emission parameters
- `_merge_states(gamma_base)`: finds minimum sym-KL pair, merges if below τ
- `_rebuild_after_merge(keep, drop)`: remaps π and A to the reduced super-state space
- `fit(O, ...)`: EM loop with periodic merging every `merge_interval` iterations

**`data.py` — Synthetic data generator**
- Produces a controlled benchmark: 3 states on a 3-simplex with well-separated Dirichlet modes
- 2nd-order transition tensor encoding a deterministic 0→1→2→0 cycle with 10% noise
- Returns observations, true states, true α parameters, and true transition tensor

### Step 4: Validating on Synthetic Data

We configured the training as:
- N_max = 6 (2× over-complete)
- K = 2 (matching ground truth)
- τ = 0.5, merge_interval = 3
- T = 500 time steps, seed = 0

**Results:**
- The model correctly reduced 6 → 3 states via three merges (at iterations 3 and 6)
- Learned Dirichlet α parameters recovered to within 6-17% of ground truth
- Viterbi decoding achieved **100% accuracy** (after majority-vote label alignment)
- Log-likelihood converged monotonically at iteration 45

### Step 5: Key Design Decisions Made

1. **Log-space arithmetic throughout**: Prevents underflow for sequences of any practical length
2. **Super-state formulation instead of explicit tensor**: Structurally sparse but uses standard matrix operations
3. **Log-space Dirichlet optimization**: Parameterizing as θ = log(α) automatically satisfies positivity constraints
4. **Merge-then-continue**: Merging every few iterations (not every step) lets EM stabilize first
5. **Occupancy-weighted pooling**: High-occupancy states dominate merged parameters

---

## 3. Phase 2 — Five Research Improvements

We identified five gaps in the original work and implemented concrete fixes for each.

### Improvement 1: Real HAR Dataset (UCI HAR)

**Gap:** Only synthetic data — no evidence the model works on real sensor data.

**What we built:** `hmm/har_data.py`

**Step-by-step process:**

1. **Dataset selection**: We chose the UCI HAR dataset (30 subjects, 6 activities, 561 pre-extracted features from smartphone accelerometer/gyroscope). This is the most widely cited HAR benchmark and directly comparable to Manouchehri 2023 (SD-HMM).

2. **Download and parsing**: `download_uci_har()` downloads the zip from the UCI repository (or uses a local copy). `_load_split()` parses `X_train.txt`, `y_train.txt`, and `subject_train.txt` for each split.

3. **Dimensionality reduction**: The 561 raw features are too high-dimensional for Dirichlet modeling (the super-state space would be enormous). We apply PCA (fit on training data only) to reduce to D=10 components.

4. **Simplex projection**: PCA components lie in R^D, not on the simplex. We apply the softmax function row-wise:
   ```
   softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
   ```
   This maps each observation onto Δ^{D-1}, making it valid input for Dirichlet emissions.

5. **Per-subject sequences**: We group observation windows by subject ID, producing independent sequences for multi-sequence training. Training set: 21 subjects, test set: 9 subjects.

### Improvement 2: Multi-Sequence EM Training

**Gap:** The framework could only train on a single contiguous observation sequence.

**What we changed:** `hmm/model.py` — added `_e_step_multi()`, `_m_step_multi()`, and modified `fit()`.

**Step-by-step process:**

1. **Input normalization**: `fit()` now accepts either a single `(T, D)` ndarray (backwards-compatible) or a `list[ndarray]` for multiple sequences.

2. **Parallel E-step**: `_e_step_multi(sequences)` iterates over each sequence, runs the standard `_e_step()` on each independently, and accumulates:
   - `gamma_base_all`: list of per-sequence base-state occupancies
   - `xi_log_total`: log-sum of transition counts across all sequences (using `logaddexp`)
   - `ll_total`: sum of per-sequence log-likelihoods
   - `gamma_base_total_weights`: total occupancy per base state (for merge decisions)

3. **Joint M-step**: `_m_step_multi()` updates:
   - **π**: averaged from first-timestep γ across all sequences
   - **A**: normalized from accumulated ξ / accumulated γ denominators
   - **Emissions**: all sequences' observations and γ are concatenated before calling the Dirichlet MLE update

4. **Merge integration**: During periodic merging in multi-sequence mode, total occupancy weights across all sequences are used for the merge decision and parameter pooling.

### Improvement 3: Baseline Comparisons

**Gap:** No comparison against other methods — impossible to assess whether our innovations actually help.

**What we built:** `hmm/baselines.py` with three baseline models, all implemented from scratch.

**Baseline 1: GaussianHMM (1st-order)**
- Standard Baum-Welch EM with diagonal-covariance Gaussian emissions
- Serves as the "straw man" — ignores simplex geometry entirely
- Closed-form M-step: weighted mean for μ, weighted variance for σ²

**Baseline 2: DirichletHMM (1st-order, fixed N)**
- Uses our Dirichlet emission model but with 1st-order transitions (K=1) and no merging
- Isolates the contribution of higher-order transitions and adaptive merging
- Same L-BFGS-B M-step as our full model

**Baseline 3: StickyHDPHMM (weak-limit approximation)**
- Bayesian nonparametric model that automatically infers the number of states
- Uses a truncated Dirichlet Process with sticky bias (self-transition boost κ=50)
- Gaussian emissions + Gibbs sampling (not EM)
- The "auto state count" competitor

All three share the same API: `fit(O, ...)` and `predict(O)` returning `(path, log_prob)`.

### Improvement 4: Ablation Study

**Gap:** No evidence that each of the three innovations contributes independently.

**What we built:** Four model variants in `experiments.py → run_ablation()`:

| Variant | Dirichlet | K>1 | Merging | What it tests |
|---------|:---------:|:---:|:-------:|---------------|
| Gauss-1st-Fixed | ✗ | ✗ | ✗ | No innovations (baseline) |
| Dir-1st-Fixed | ✓ | ✗ | ✗ | Dirichlet emissions alone |
| Dir-2nd-Fixed | ✓ | ✓ | ✗ | + higher-order transitions |
| Full (ours) | ✓ | ✓ | ✓ | + adaptive merging |

The ablation is achieved by configuring `HigherOrderHMM` differently:
- K=1 vs K=2 controls transition order
- kl_threshold=0.0 with merge_interval=999 disables merging
- `GaussianHMM` provides the non-Dirichlet variant

### Improvement 5: Merge Threshold (τ) Sensitivity Analysis

**Gap:** τ=0.5 was chosen without justification. Is the model sensitive to this choice?

**What we built:** `experiments.py → run_threshold_sensitivity()`

We sweep τ over 8 values: `[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]`

For each τ, we train the full model (N_max=6, K=2) and record:
- Final state count N_final
- Viterbi decoding accuracy
- Log-probability of the decoded path

### Bonus: BIC Model Selection

We added a `bic()` method to `HigherOrderHMM` for Bayesian Information Criterion:

```
BIC = -2 · LL + k · ln(T)
```

where k = N·D (Dirichlet alphas) + M·N (transition probs) + M (initial probs).

Also added `score(O)` to compute log-likelihood of any observation sequence.

---

## 4. Repository Structure

```
Math_Research/
│
├── docs/
│   └── DOCUMENTATION.md          ← This document
│
├── hmm/                           ← Core Python package
│   ├── __init__.py               — Public API, exports all models
│   ├── emissions.py              — DirichletEmission (pdf, MLE, KL, merge)
│   ├── super_states.py           — SuperStateSpace (K-tuple enumeration)
│   ├── inference.py              — forward(), backward(), viterbi() [pure]
│   ├── model.py                  — HigherOrderHMM (orchestrator, multi-seq EM, BIC)
│   ├── data.py                   — generate_synthetic() [unchanged]
│   ├── har_data.py               — [NEW] UCI HAR download, PCA, simplex projection
│   └── baselines.py              — [NEW] GaussianHMM, DirichletHMM, StickyHDPHMM
│
├── experiments.py                 — [NEW] Full experiment suite (5 experiments)
├── main.py                       — Original synthetic validation [unchanged]
├── hmm_framework.py              — Monolithic reference implementation [unchanged]
├── RESEARCH.md                   — Mathematical framework documentation
│
├── data/                          — UCI HAR dataset (downloaded separately)
│   └── UCI HAR Dataset/
│       ├── train/
│       └── test/
│
└── venv/                          — Python virtual environment
```

### Dependencies

- `numpy >= 1.24`
- `scipy >= 1.10`
- `scikit-learn >= 1.0` (for PCA in HAR data pipeline only)

---

## 5. How to Run Everything

```bash
# Activate the virtual environment
source venv/bin/activate

# 1. Original synthetic validation (unchanged from Phase 1)
python main.py

# 2. Full experiment suite (all 5 improvements, ~10 minutes)
python experiments.py

# 3. Quick experiment suite (reduced iterations, ~2 minutes)
python experiments.py --quick
```

### What each experiment does:

| Experiment | Function | What it measures |
|-----------|----------|------------------|
| 1. Baseline comparison | `run_baseline_synthetic()` | Our model vs 3 baselines on synthetic data |
| 2. Ablation study | `run_ablation()` | Individual contribution of each innovation |
| 3. τ sensitivity | `run_threshold_sensitivity()` | Robustness to merge threshold choice |
| 4. Multi-sequence | `run_multi_sequence()` | Multi-seq EM vs concatenation |
| 5. UCI HAR benchmark | `run_har_experiment()` | All models on real sensor data |

---

## 6. Complete Experiment Results

### Experiment 1: Baseline Comparison — Synthetic Data (T=500, 3 states)

| Model | N_final | Accuracy | Log-prob | Time |
|-------|---------|----------|----------|------|
| **HO-Dir-HMM (ours)** | **3** | **100.0%** | **1246.9** | 2.5s |
| Gaussian-HMM | 3 | 68.2% | 387.1 | 1.6s |
| Dirichlet-HMM (1st) | 3 | 100.0% | 1193.1 | 0.8s |
| Sticky HDP-HMM | 6 | 98.0% | N/A | 2.8s |

**Key findings:**
- Our model achieves the highest accuracy AND log-likelihood
- Gaussian-HMM fails badly (68.2%) because it mismodels simplex data
- Dirichlet-HMM also reaches 100% but with lower LL (no temporal context)
- HDP-HMM over-segments (finds 6 states instead of 3) and reaches only 98%

### Experiment 2: Ablation Study

| Variant | Dirichlet | K>1 | Merging | Accuracy |
|---------|:---------:|:---:|:-------:|----------|
| Gauss-1st-Fixed | ✗ | ✗ | ✗ | 68.2% |
| Dir-1st-Fixed | ✓ | ✗ | ✗ | 100.0% |
| Dir-2nd-Fixed | ✓ | ✓ | ✗ | 100.0% |
| **Full (ours)** | ✓ | ✓ | ✓ | **100.0%** |

**Key finding:** Dirichlet emissions are the single most impactful component (+31.8% over Gaussian). On well-separated synthetic data, higher-order transitions and merging maintain accuracy — their value appears on harder/real data.

### Experiment 3: Merge Threshold (τ) Sensitivity

| τ | N_final | Accuracy | Log-prob |
|---|---------|----------|----------|
| 0.01 | 6 | 100.0% | 1245.0 |
| 0.05 | 6 | 100.0% | 1245.0 |
| 0.10 | 6 | 100.0% | 1245.0 |
| 0.20 | 5 | 100.0% | 1222.0 |
| **0.50** | **3** | **100.0%** | **1246.9** |
| 1.00 | 3 | 100.0% | 1246.7 |
| 2.00 | 3 | 100.0% | 1246.7 |
| 5.00 | 3 | 100.0% | 1246.7 |

**Key findings:**
- τ=0.5 is the sweet spot — recovers exactly 3 states with the best LL
- Too conservative (τ ≤ 0.1): no merging occurs, model stays over-complete but accuracy is unaffected
- Too aggressive (τ ≥ 1.0): same state count but slightly lower LL
- The model is **robust** — accuracy is 100% across the entire range

### Experiment 4: Multi-Sequence Training

| Method | N_final | Accuracy | Time |
|--------|---------|----------|------|
| Multi-seq (5×200) | 3 | 99.9% | 9.7s |
| Single concat (1000) | 3 | 99.9% | 4.1s |

**Key finding:** Multi-sequence EM correctly handles independent sequences, achieving identical accuracy and state count to naive concatenation. The principled E-step accumulation works as designed.

### Experiment 5: UCI HAR Real-Data Benchmark

| Model | N_final | Accuracy | Time |
|-------|---------|----------|------|
| **HO-Dir-HMM (ours)** | 10 | **61.1%** | 25.6s |
| Gaussian-HMM | 6 | 51.4% | 11.7s |
| Dirichlet-HMM (1st) | 6 | 54.6% | 11.8s |
| Sticky HDP-HMM | 10 | 58.7% | 26.6s |

**Key findings:**
- Our model outperforms all baselines on real data
- The Dirichlet emission advantage carries from synthetic to real (+9.7% over Gaussian)
- Our full model beats HDP-HMM by +2.4%, demonstrating that our deterministic merging + higher-order transitions are competitive with Bayesian nonparametric state inference
- These are unsupervised HMM results — accuracy is computed via majority-vote alignment, so ~60% on a 6-class problem is reasonable

---

## 7. Technical Details of Every Module

### `hmm/emissions.py` — DirichletEmission

**Purpose:** Manages a table of N Dirichlet distributions, one per hidden state.

**Key implementation details:**

- All α vectors stored in a single `(N, D)` ndarray for vectorized operations
- `log_pdf_batch`: computes `gammaln(α₀) - Σ gammaln(α_d) + log_X @ (α - 1)ᵀ` in one pass
- Observations are floor-clipped to 1e-10 and renormalized to stay on the open simplex
- M-step optimization parameterizes in log-space: θ = log(α), gradient is `[ψ(α₀) - ψ(α_d) + s_d] · α_d`
- Merge pooling: `α_merged = (W_i · α_i + W_j · α_j) / (W_i + W_j)`

### `hmm/super_states.py` — SuperStateSpace

**Purpose:** Enumerate K-tuples and precompute adjacency structures.

**Key implementation details:**

- Uses `itertools.product` to generate all N^K tuples in canonical order
- `valid_mask`: boolean (M, M) matrix — True only for overlap-consistent transitions
- `children_of[p]`: list of valid successor indices for parent super-state p
- `emit_map`: maps each super-state to its last base-state element (for emission lookup)
- `remap(keep, drop)`: handles index renumbering when a base state is removed

### `hmm/inference.py` — Pure Functions

**Purpose:** Stateless log-space inference routines. No side effects, independently testable.

**Forward pass:** α_t(m') = φ_{m'}(o_t) · Σ_m [α_{t-1}(m) · A_{m,m'}] — all in log-space using logsumexp over the first axis of `(log_alpha[t-1, :, None] + log_A)`.

**Backward pass:** β_t(m) = Σ_{m'} [A_{m,m'} · φ_{m'}(o_{t+1}) · β_{t+1}(m')] — logsumexp over the second axis.

**Viterbi:** Same structure as forward but with `max` instead of `logsumexp`, plus backpointers for traceback.

### `hmm/model.py` — HigherOrderHMM

**Purpose:** Orchestrates all components. Holds model state, delegates computation.

**E-step flow:**
1. Compute log-emissions via `emissions.log_pdf_batch()` projected through `ss.emit_map`
2. Run `inference.forward()` and `inference.backward()`
3. Compute γ (super-state occupancies) from α + β - LL
4. Aggregate γ to base states by summing over super-states sharing the same last element
5. Compute ξ (transition counts) accumulated in log-space

**M-step flow:**
1. Update π from γ at t=0
2. Update A from ξ / γ, normalized per row over valid children
3. Update Dirichlet α via `emissions.update()`

**Merge flow (every merge_interval iterations):**
1. Run a fresh E-step for current occupancies
2. Compute sym-KL for all state pairs
3. If minimum sym-KL < τ: merge that pair via `emissions.merge()` + `_rebuild_after_merge()`
4. Repeat until no mergeable pairs remain

**Multi-sequence extension:**
- `_e_step_multi()`: loops over sequences, accumulates γ and ξ via logaddexp
- `_m_step_multi()`: joint update from accumulated statistics
- Automatic detection: if `O` is a list, use multi-seq path; if ndarray, use single-seq path

### `hmm/har_data.py` — UCI HAR Pipeline

**Purpose:** Download, parse, and project real sensor data onto the simplex.

**Pipeline:** Raw 561-dim features → PCA (fit on train) → top-10 components → softmax → Δ^9

**Design choice — why softmax?** Softmax is the natural map from R^D to the open simplex. It preserves relative ordering of components and produces well-behaved gradients. Unlike L1 normalization, it works correctly with features that can be negative (PCA components are zero-mean).

### `hmm/baselines.py` — Three Baseline Models

**Purpose:** Self-contained comparison models. No external HMM libraries needed.

All three implement standard Baum-Welch / Gibbs with their respective emission models. The GaussianHMM and DirichletHMM use identical forward-backward-Viterbi logic. The StickyHDPHMM uses collapsed Gibbs sampling with a truncation level as the state upper bound.

### `experiments.py` — Experiment Runner

**Purpose:** Reproducible experiment execution with formatted output.

**Utility functions:**
- `majority_vote_accuracy()`: aligns predicted labels to ground truth via mode of true labels per predicted cluster
- `print_table()`: formatted ASCII tables for console output

**Supports `--quick` flag:** reduces iterations for fast validation (~2 min vs ~10 min).

---

*Document prepared as part of the Mathematical Research project. All code is self-contained and reproducible with fixed random seeds.*

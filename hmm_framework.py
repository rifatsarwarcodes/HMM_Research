"""
Higher-Order Hidden Markov Model with Dirichlet Emissions
and Adaptive State Merging

A custom HMM framework for Human Activity Recognition (HAR) extending
the standard HMM 5-tuple λ = (S, Ω, P, Φ, π) with:
  1. Dirichlet emission distributions for compositional (simplex) data
  2. K-th order temporal dependencies via super-state transition tensors
  3. Adaptive state merging via symmetric KL divergence

All forward/backward/Viterbi computations use log-space arithmetic
to prevent numerical underflow.
"""

import numpy as np
from scipy.special import gammaln, digamma, logsumexp
from scipy.optimize import minimize
from itertools import product as cartesian_product
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

OBS_FLOOR = 1e-10


# ═══════════════════════════════════════════════════════════════════════════
#  Dirichlet Emission Model
# ═══════════════════════════════════════════════════════════════════════════

class DirichletEmission:
    """
    Manages Dirichlet emission distributions for each hidden state.

    Each state i has concentration parameters α_i ∈ R^D_+ defining:
        Dir(x | α) = Γ(Σα_k) / Πk Γ(α_k) · Πk x_k^{α_k - 1}

    Supports:
      - Batch log-pdf evaluation
      - Weighted MLE parameter updates (M-step)
      - KL divergence computation for state merging
    """

    def __init__(self, n_states, n_dims, rng=None):
        self.n_states = n_states
        self.n_dims = n_dims
        self.rng = rng or np.random.default_rng(42)
        self.alphas = self.rng.uniform(1.5, 5.0, size=(n_states, n_dims))

    def _safe_obs(self, O):
        """Clip and renormalize observations to stay on the open simplex."""
        X = np.clip(O, OBS_FLOOR, None)
        return X / X.sum(axis=-1, keepdims=True)

    def log_pdf_batch(self, O):
        """
        Vectorized log Dir(x_t | α_k) for all t and all states k.

        Parameters
        ----------
        O : ndarray, shape (T, D)

        Returns
        -------
        log_b : ndarray, shape (T, n_states)
        """
        X = self._safe_obs(O)
        log_X = np.log(X)                                    # (T, D)
        a0 = self.alphas.sum(axis=1)                          # (N,)
        log_norm = gammaln(a0) - gammaln(self.alphas).sum(1)  # (N,)
        log_kern = log_X @ (self.alphas - 1.0).T              # (T, N)
        return log_norm[None, :] + log_kern

    def kl_divergence(self, i, j):
        """
        KL(Dir(α_i) ‖ Dir(α_j)).

        KL = ln Γ(α0_i) − Σ ln Γ(α_ik) − ln Γ(α0_j) + Σ ln Γ(α_jk)
             + Σ (α_ik − α_jk)(ψ(α_ik) − ψ(α0_i))
        """
        a, b = self.alphas[i], self.alphas[j]
        a0, b0 = a.sum(), b.sum()
        return (gammaln(a0) - gammaln(a).sum()
                - gammaln(b0) + gammaln(b).sum()
                + ((a - b) * (digamma(a) - digamma(a0))).sum())

    def symmetric_kl(self, i, j):
        """0.5 · (KL(i‖j) + KL(j‖i))"""
        return 0.5 * (self.kl_divergence(i, j) + self.kl_divergence(j, i))

    def update(self, O, gamma_base):
        """
        Weighted MLE for Dirichlet parameters (M-step).

        For each state k the objective is:
            max_α  W·[ln Γ(Σα) − Σ ln Γ(α_d) + Σ(α_d−1)·s_d]
        where W = Σ_t γ_t(k), s_d = Σ_t γ_t(k)·ln x_{t,d} / W.

        Solved via L-BFGS-B on log(α) to enforce positivity.
        """
        X = self._safe_obs(O)
        log_X = np.log(X)

        for k in range(self.n_states):
            w = gamma_base[:, k]
            W = w.sum()
            if W < 1e-12:
                continue
            s = (w[:, None] * log_X).sum(0) / W

            a0 = self.alphas[k].copy()

            def _neg_ll(log_a):
                a = np.exp(log_a)
                return -(gammaln(a.sum()) - gammaln(a).sum()
                         + ((a - 1.0) * s).sum())

            def _grad(log_a):
                a = np.exp(log_a)
                g = -(digamma(a.sum()) - digamma(a) + s)
                return g * a

            try:
                res = minimize(_neg_ll, np.log(a0), jac=_grad,
                               method='L-BFGS-B',
                               options={'maxiter': 80, 'ftol': 1e-10})
                a_new = np.exp(res.x)
                if np.all(np.isfinite(a_new)) and np.all(a_new > 0.01):
                    self.alphas[k] = a_new
            except Exception:
                pass

    def merge(self, i, j, w_i, w_j):
        """
        Merge state j into state i using occupancy-weighted parameter pooling.
        Removes row j from self.alphas and decrements n_states.
        """
        total = w_i + w_j
        if total < 1e-12:
            merged = 0.5 * (self.alphas[i] + self.alphas[j])
        else:
            merged = (w_i * self.alphas[i] + w_j * self.alphas[j]) / total
        self.alphas[i] = merged
        self.alphas = np.delete(self.alphas, j, axis=0)
        self.n_states -= 1


# ═══════════════════════════════════════════════════════════════════════════
#  Higher-Order HMM
# ═══════════════════════════════════════════════════════════════════════════

class HigherOrderHMM:
    """
    K-th order HMM over an adaptive state space with Dirichlet emissions.

    The model operates on *super-states*: K-tuples of base states.
    For N base states and order K the super-state space has N^K elements.
    Only "overlap-consistent" transitions are allowed:
        (i1,…,iK) → (i2,…,iK, j)   for every base state j.

    Parameters
    ----------
    N_max : int
        Initial (over-complete) number of base hidden states.
    order_K : int
        Markov order (K = 1 gives a standard HMM).
    kl_threshold : float
        Symmetric KL threshold τ below which two states are merged.
    n_dims : int
        Dimensionality D of observation vectors on the simplex.
    seed : int
        Random seed.
    """

    def __init__(self, N_max, order_K, kl_threshold, n_dims, seed=42):
        self.N = N_max
        self.K = order_K
        self.kl_threshold = kl_threshold
        self.n_dims = n_dims
        self.rng = np.random.default_rng(seed)
        self.emissions = DirichletEmission(N_max, n_dims, rng=self.rng)
        self._build_super_states()
        self._init_params()

    # ── super-state bookkeeping ──────────────────────────────────────────

    def _build_super_states(self):
        """Enumerate all K-tuples and precompute adjacency structures."""
        bases = list(range(self.N))
        self.super_states = list(cartesian_product(bases, repeat=self.K))
        self.n_super = len(self.super_states)
        self.ss_idx = {ss: i for i, ss in enumerate(self.super_states)}
        self.emit_map = np.array([ss[-1] for ss in self.super_states])

        # valid transition mask  (parent, child)
        self.valid_mask = np.zeros((self.n_super, self.n_super), dtype=bool)
        self.children_of = {}
        self.parents_of = {}

        for p, p_ss in enumerate(self.super_states):
            ch = []
            for j in bases:
                c_ss = p_ss[1:] + (j,)
                c = self.ss_idx[c_ss]
                self.valid_mask[p, c] = True
                ch.append(c)
            self.children_of[p] = ch

        for c, c_ss in enumerate(self.super_states):
            pa = []
            for i in bases:
                p_ss = (i,) + c_ss[:-1]
                pa.append(self.ss_idx[p_ss])
            self.parents_of[c] = pa

    def _init_params(self):
        """Uniform initialisation of π and A."""
        self.log_pi = np.full(self.n_super, -np.log(self.n_super))

        self.log_A = np.full((self.n_super, self.n_super), -np.inf)
        log_unif = -np.log(self.N)
        for p in range(self.n_super):
            for c in self.children_of[p]:
                self.log_A[p, c] = log_unif

    # ── emission helpers ─────────────────────────────────────────────────

    def _log_emissions(self, O):
        """
        Returns
        -------
        log_e : ndarray (T, n_super)
            log φ_{last(ss)}(o_t) for every observation / super-state pair.
        """
        log_b = self.emissions.log_pdf_batch(O)      # (T, N)
        return log_b[:, self.emit_map]                # (T, M)

    # ── forward / backward ───────────────────────────────────────────────

    def _forward(self, log_e):
        """
        Log-space forward recursion.

        α_{t+1}(i2…iK,j) = Σ_{i1} α_t(i1…iK) · A(i1…iK → j) · φ_j(o_{t+1})

        Returns  log_alpha (T, M),  log_likelihood  scalar
        """
        T, M = log_e.shape
        la = np.full((T, M), -np.inf)
        la[0] = self.log_pi + log_e[0]
        for t in range(1, T):
            # la[t, j] = log_e[t, j] + logsumexp_i(la[t-1, i] + log_A[i, j])
            la[t] = log_e[t] + logsumexp(
                la[t - 1, :, None] + self.log_A, axis=0)
        return la, logsumexp(la[-1])

    def _backward(self, log_e):
        """Log-space backward recursion.  Returns log_beta (T, M)."""
        T, M = log_e.shape
        lb = np.full((T, M), -np.inf)
        lb[-1] = 0.0
        for t in range(T - 2, -1, -1):
            lb[t] = logsumexp(
                self.log_A + log_e[t + 1][None, :] + lb[t + 1][None, :],
                axis=1)
        return lb

    # ── E-step ───────────────────────────────────────────────────────────

    def _e_step(self, O):
        """
        Compute γ (state occupancies) and accumulated ξ (transition counts).

        Returns
        -------
        gamma_ss  : (T, M)   super-state occupancies
        gamma_base: (T, N)   base-state occupancies  (for emission M-step)
        xi_log    : (M, M)   log Σ_t ξ_t(i,j)
        ll        : float    log P(O | λ)
        """
        log_e = self._log_emissions(O)
        la, ll = self._forward(log_e)
        lb = self._backward(log_e)

        log_gamma = la + lb - ll
        gamma_ss = np.exp(log_gamma)

        gamma_base = np.zeros((O.shape[0], self.N))
        for k in range(self.N):
            gamma_base[:, k] = gamma_ss[:, self.emit_map == k].sum(1)

        # accumulate xi in log-space
        xi_log = np.full((self.n_super, self.n_super), -np.inf)
        for t in range(O.shape[0] - 1):
            lx = (la[t, :, None] + self.log_A
                  + log_e[t + 1][None, :] + lb[t + 1][None, :] - ll)
            xi_log = np.logaddexp(xi_log, lx)

        return gamma_ss, gamma_base, xi_log, ll

    # ── M-step ───────────────────────────────────────────────────────────

    def _m_step(self, O, gamma_ss, gamma_base, xi_log):
        """Update π, A, and Dirichlet emission parameters."""
        M = self.n_super

        # π
        pi_raw = gamma_ss[0] + 1e-300
        self.log_pi = np.log(pi_raw / pi_raw.sum())

        # A  (normalise per parent row over valid children only)
        denom = np.log(gamma_ss[:-1].sum(0) + 1e-300)   # (M,)
        new_A = xi_log - denom[:, None]
        new_A[~self.valid_mask] = -np.inf
        for p in range(M):
            ch = self.children_of[p]
            ln = logsumexp(new_A[p, ch])
            if np.isfinite(ln):
                new_A[p, ch] -= ln
            else:
                new_A[p, ch] = -np.log(len(ch))
        self.log_A = new_A

        # emissions
        self.emissions.update(O, gamma_base)

    # ── state merging ────────────────────────────────────────────────────

    def _merge_states(self, gamma_base):
        """
        Find the closest pair of base-state emissions by symmetric KL.
        If distance < τ, merge them and rebuild the super-state space.

        Returns True if a merge happened.
        """
        if self.N <= 1:
            return False

        best_kl, best_pair = np.inf, None
        for i in range(self.N):
            for j in range(i + 1, self.N):
                kl = self.emissions.symmetric_kl(i, j)
                if kl < best_kl:
                    best_kl, best_pair = kl, (i, j)

        if best_kl >= self.kl_threshold:
            return False

        i, j = best_pair
        w_i = gamma_base[:, i].sum()
        w_j = gamma_base[:, j].sum()
        print(f"  ↳ merge state {j} → {i}  "
              f"(sym-KL = {best_kl:.4f}, γ weights {w_i:.1f} / {w_j:.1f})")

        self.emissions.merge(i, j, w_i, w_j)
        self.N -= 1
        self._rebuild_after_merge(i, j)
        return True

    def _rebuild_after_merge(self, keep, drop):
        """Remap super-states after base state *drop* is absorbed into *keep*."""
        remap = {}
        c = 0
        for old in range(self.N + 1):
            if old == drop:
                remap[old] = remap[keep]
            else:
                remap[old] = c
                c += 1

        old_pi = self.log_pi.copy()
        old_A = self.log_A.copy()
        old_ss = self.super_states[:]
        old_idx = self.ss_idx.copy()
        old_M = self.n_super

        self._build_super_states()

        # map old super-state indices → new
        o2n = {}
        for oi, oss in enumerate(old_ss):
            nss = tuple(remap[s] for s in oss)
            ni = self.ss_idx.get(nss)
            if ni is not None:
                o2n[oi] = ni

        # π
        new_pi = np.full(self.n_super, -np.inf)
        for oi, ni in o2n.items():
            new_pi[ni] = np.logaddexp(new_pi[ni], old_pi[oi])
        z = logsumexp(new_pi)
        self.log_pi = new_pi - z if np.isfinite(z) else np.full(
            self.n_super, -np.log(self.n_super))

        # A
        new_A = np.full((self.n_super, self.n_super), -np.inf)
        for op in range(old_M):
            if op not in o2n:
                continue
            np_ = o2n[op]
            for oc in range(old_M):
                if oc not in o2n:
                    continue
                nc = o2n[oc]
                if np.isfinite(old_A[op, oc]):
                    val = old_pi[op] + old_A[op, oc]
                    new_A[np_, nc] = np.logaddexp(new_A[np_, nc], val)

        new_A[~self.valid_mask] = -np.inf
        for p in range(self.n_super):
            ch = self.children_of[p]
            z = logsumexp(new_A[p, ch])
            if np.isfinite(z):
                new_A[p, ch] -= z
            else:
                new_A[p, ch] = -np.log(len(ch))
        self.log_A = new_A

    # ── fit (Baum-Welch + merging) ───────────────────────────────────────

    def fit(self, O, max_iter=100, tol=1e-4, merge_interval=3,
            verbose=True):
        """
        Extended Baum-Welch with integrated state merging.

        Parameters
        ----------
        O : ndarray (T, D)
        max_iter : int
        tol : float   convergence threshold on |Δ log-likelihood|
        merge_interval : int   attempt merges every k iterations
        verbose : bool
        """
        prev_ll = -np.inf

        for it in range(1, max_iter + 1):
            g_ss, g_base, xi_log, ll = self._e_step(O)
            if verbose:
                print(f"  iter {it:3d} │ N={self.N} │ LL = {ll:+.4f}")

            self._m_step(O, g_ss, g_base, xi_log)

            # periodic merging
            if it % merge_interval == 0:
                merged = True
                while merged and self.N > 1:
                    _, g_base_fresh, _, _ = self._e_step(O)
                    merged = self._merge_states(g_base_fresh)

            if abs(ll - prev_ll) < tol and it > 5:
                if verbose:
                    print(f"  converged (ΔLL = {ll - prev_ll:.2e})")
                break
            prev_ll = ll

        return self

    # ── Viterbi decoding ─────────────────────────────────────────────────

    def predict(self, O):
        """
        Higher-order Viterbi in log-space.

        Returns
        -------
        base_states : ndarray (T,)
        log_prob    : float
        """
        log_e = self._log_emissions(O)
        T, M = log_e.shape
        delta = np.full((T, M), -np.inf)
        psi = np.zeros((T, M), dtype=int)

        delta[0] = self.log_pi + log_e[0]

        for t in range(1, T):
            scores = delta[t - 1, :, None] + self.log_A   # (M, M)
            psi[t] = scores.argmax(axis=0)
            delta[t] = scores.max(axis=0) + log_e[t]

        path_ss = np.empty(T, dtype=int)
        path_ss[-1] = delta[-1].argmax()
        log_prob = delta[-1, path_ss[-1]]
        for t in range(T - 2, -1, -1):
            path_ss[t] = psi[t + 1, path_ss[t + 1]]

        return self.emit_map[path_ss], log_prob


# ═══════════════════════════════════════════════════════════════════════════
#  Synthetic Data & Validation
# ═══════════════════════════════════════════════════════════════════════════

def generate_synthetic(T=500, seed=123):
    """
    3-state, 2nd-order HMM on a 3-simplex.

    Ground-truth emissions (well-separated):
        state 0 → Dir(10, 1, 1)   concentrated on x1
        state 1 → Dir(1, 10, 1)   concentrated on x2
        state 2 → Dir(1, 1, 10)   concentrated on x3

    2nd-order transitions create a strong 0→1→2→0 cycle.
    """
    rng = np.random.default_rng(seed)
    N, D = 3, 3
    alphas = np.array([[10., 1., 1.],
                       [1., 10., 1.],
                       [1., 1., 10.]])

    # trans[i, j, k] = P(q_t = k | q_{t-2} = i, q_{t-1} = j)
    trans = np.full((N, N, N), 0.05)
    trans[:, 0, 1] = 0.9
    trans[0, 1, 2] = 0.9
    trans[1, 2, 0] = 0.9
    trans[2, 0, 1] = 0.9
    for i in range(N):
        for j in range(N):
            trans[i, j] /= trans[i, j].sum()

    states = np.empty(T, dtype=int)
    states[0], states[1] = 0, 1
    for t in range(2, T):
        states[t] = rng.choice(N, p=trans[states[t - 2], states[t - 1]])

    obs = np.zeros((T, D))
    for t in range(T):
        obs[t] = rng.dirichlet(alphas[states[t]])

    return obs, states, alphas, trans


# ─── main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  Higher-Order HMM ─ Synthetic Validation")
    print("=" * 65)

    # 1) generate data
    T = 500
    obs, true_states, true_alphas, true_trans = generate_synthetic(T)
    print(f"\n• Generated T={T} observations on a 3-simplex")
    print(f"• Ground-truth states: {np.unique(true_states)} "
          f"(cycle 0→1→2→0…)")

    # 2) train
    print(f"\n{'─' * 65}")
    print("  Training:  N_max=6,  K=2,  τ=0.5")
    print(f"{'─' * 65}\n")

    model = HigherOrderHMM(N_max=6, order_K=2, kl_threshold=0.5,
                           n_dims=3, seed=0)
    model.fit(obs, max_iter=80, merge_interval=3, verbose=True)

    # 3) results
    print(f"\n{'─' * 65}")
    print("  Results")
    print(f"{'─' * 65}")
    print(f"  Final number of states : {model.N}  (target = 3)")

    decoded, log_p = model.predict(obs)
    print(f"  Viterbi log-prob       : {log_p:.2f}")

    # map decoded labels to ground-truth via majority vote
    from collections import Counter
    label_map = {}
    for pred_label in np.unique(decoded):
        votes = true_states[decoded == pred_label]
        label_map[pred_label] = Counter(votes.tolist()).most_common(1)[0][0]

    mapped = np.array([label_map[d] for d in decoded])
    acc = (mapped == true_states).mean()
    print(f"  Accuracy (mapped)      : {acc * 100:.1f} %")

    print(f"\n  Learned Dirichlet α per state:")
    for k in range(model.N):
        print(f"    state {k}: {model.emissions.alphas[k].round(2)}")

    print(f"\n  Ground-truth α:")
    for k in range(3):
        print(f"    state {k}: {true_alphas[k]}")

    print(f"\n  Decoded path (first 60): {decoded[:60].tolist()}")
    print(f"  True path   (first 60): {true_states[:60].tolist()}")
    print("=" * 65)

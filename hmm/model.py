"""
hmm/model.py
------------
HigherOrderHMM — the core model class.

Orchestrates:
  • SuperStateSpace   (super_states.py)   — K-tuple enumeration & adjacency
  • DirichletEmission (emissions.py)      — pdf, M-step, KL, merge
  • forward / backward / viterbi           (inference.py) — log-space routines

Public API
----------
HigherOrderHMM(N_max, order_K, kl_threshold, n_dims, seed=42)
    fit(O, ...)        → self
    predict(O)         → (base_state_sequence, log_prob)
"""

import numpy as np
from scipy.special import logsumexp

from .emissions import DirichletEmission
from .super_states import SuperStateSpace
from . import inference


class HigherOrderHMM:
    """
    K-th order HMM with Dirichlet emissions and adaptive state merging.

    The model works with *super-states* — K-tuples of base states.  Only
    overlap-consistent transitions  (i_1,…,i_K) → (i_2,…,i_K, j)  are
    permitted, encoding the K-th order Markov structure.

    State merging is integrated into the Baum-Welch loop: every
    ``merge_interval`` iterations the symmetric KL divergence matrix is
    evaluated and the closest pair is merged if their distance is below τ.

    Parameters
    ----------
    N_max : int
        Initial (over-complete) number of base hidden states.
    order_K : int
        Markov order.  K=1 gives a standard first-order HMM.
    kl_threshold : float
        Symmetric KL threshold τ.  State pairs with sym-KL < τ are merged.
    n_dims : int
        Dimensionality D of observation vectors on the simplex.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        N_max: int,
        order_K: int,
        kl_threshold: float,
        n_dims: int,
        seed: int = 42,
    ):
        self.N = N_max
        self.K = order_K
        self.kl_threshold = kl_threshold
        self.n_dims = n_dims
        self.rng = np.random.default_rng(seed)

        self.emissions = DirichletEmission(N_max, n_dims, rng=self.rng)
        self.ss = SuperStateSpace(N_max, order_K)
        self._init_params()

    # ── parameter initialisation ─────────────────────────────────────────

    def _init_params(self) -> None:
        """Uniform log-π and log-A respecting the valid-transition mask."""
        M = self.ss.n_super
        self.log_pi = np.full(M, -np.log(M))

        self.log_A = np.full((M, M), -np.inf)
        log_unif = -np.log(self.N)
        for p, ch in self.ss.children_of.items():
            for c in ch:
                self.log_A[p, c] = log_unif

    # ── emission projection ──────────────────────────────────────────────

    def _log_emissions(self, O: np.ndarray) -> np.ndarray:
        """
        Project base-state log-emissions onto the super-state space.

        Returns
        -------
        log_e : ndarray (T, n_super)
            log φ_{last(ss)}(o_t) for every (t, super-state) pair.
        """
        log_b = self.emissions.log_pdf_batch(O)        # (T, N)
        return log_b[:, self.ss.emit_map]              # (T, M)

    # ── E-step ───────────────────────────────────────────────────────────

    def _e_step(self, O: np.ndarray) -> tuple:
        """
        Run the E-step: compute γ (occupancies) and ξ (transition counts).

        Returns
        -------
        gamma_ss   : ndarray (T, M)   super-state occupancies
        gamma_base : ndarray (T, N)   base-state occupancies γ_t(k)
        xi_log     : ndarray (M, M)   log Σ_t ξ_t(i, j)
        ll         : float            log P(O | λ)
        """
        log_e = self._log_emissions(O)
        M = self.ss.n_super
        T = O.shape[0]

        # Forward & backward
        log_alpha, ll = inference.forward(self.log_pi, self.log_A, log_e)
        log_beta       = inference.backward(self.log_A, log_e)

        # γ for super-states
        log_gamma = log_alpha + log_beta - ll
        gamma_ss  = np.exp(log_gamma)

        # γ aggregated to base states (sum over super-states sharing last elem)
        gamma_base = np.zeros((T, self.N))
        for k in range(self.N):
            gamma_base[:, k] = gamma_ss[:, self.ss.emit_map == k].sum(1)

        # ξ accumulated in log-space
        xi_log = np.full((M, M), -np.inf)
        for t in range(T - 1):
            lx = (
                log_alpha[t, :, None]
                + self.log_A
                + log_e[t + 1][None, :]
                + log_beta[t + 1][None, :]
                - ll
            )
            xi_log = np.logaddexp(xi_log, lx)

        return gamma_ss, gamma_base, xi_log, ll

    # ── M-step ───────────────────────────────────────────────────────────

    def _m_step(
        self,
        O: np.ndarray,
        gamma_ss: np.ndarray,
        gamma_base: np.ndarray,
        xi_log: np.ndarray,
    ) -> None:
        """
        M-step: update log-π, log-A, and Dirichlet emission parameters.

        Parameters
        ----------
        O          : ndarray (T, D)
        gamma_ss   : ndarray (T, M)
        gamma_base : ndarray (T, N)
        xi_log     : ndarray (M, M)   log accumulated ξ
        """
        M = self.ss.n_super

        # ── π  (from first time-step occupancies) ────────────────────────
        pi_raw = gamma_ss[0] + 1e-300
        self.log_pi = np.log(pi_raw / pi_raw.sum())

        # ── A  (normalize each row over its valid children) ───────────────
        denom = np.log(gamma_ss[:-1].sum(0) + 1e-300)   # (M,)
        new_A = xi_log - denom[:, None]
        new_A[~self.ss.valid_mask] = -np.inf

        for p, ch in self.ss.children_of.items():
            z = logsumexp(new_A[p, ch])
            if np.isfinite(z):
                new_A[p, ch] -= z
            else:
                new_A[p, ch] = -np.log(len(ch))

        self.log_A = new_A

        # ── emissions ────────────────────────────────────────────────────
        self.emissions.update(O, gamma_base)

    # ── state merging ────────────────────────────────────────────────────

    def _merge_states(self, gamma_base: np.ndarray) -> bool:
        """
        Evaluate the symmetric KL matrix and merge the closest state pair
        if their distance is below the threshold τ.

        Returns
        -------
        bool — True if a merge was performed.
        """
        if self.N <= 1:
            return False

        # Find the minimum-KL pair
        best_kl, best_pair = np.inf, None
        for i in range(self.N):
            for j in range(i + 1, self.N):
                kl = self.emissions.symmetric_kl(i, j)
                if kl < best_kl:
                    best_kl, best_pair = kl, (i, j)

        if best_kl >= self.kl_threshold:
            return False

        i, j = best_pair
        w_i = float(gamma_base[:, i].sum())
        w_j = float(gamma_base[:, j].sum())
        print(
            f"  ↳ merge state {j} → {i}  "
            f"(sym-KL = {best_kl:.4f}, γ weights {w_i:.1f} / {w_j:.1f})"
        )

        # Merge emission parameters
        self.emissions.merge(i, j, w_i, w_j)

        # Rebuild super-state space and remap π, A
        self._rebuild_after_merge(i, j)
        self.N -= 1
        return True

    def _rebuild_after_merge(self, keep: int, drop: int) -> None:
        """
        Remap log-π and log-A after absorbing base state *drop* into *keep*.

        The old SuperStateSpace (before self.N is decremented) is used to
        compute the remapping, then a fresh SuperStateSpace is built.
        """
        old_ss   = self.ss
        old_pi   = self.log_pi.copy()
        old_A    = self.log_A.copy()
        old_M    = old_ss.n_super

        # Remap using the *old* N (before decrement), so we pass N (not N-1)
        # to remap, which internally iterates range(self.N + 1).
        # At this point self.N has not yet been decremented.
        base_remap = old_ss.remap(keep, drop)

        # Build new super-state space with N-1 states
        self.N -= 1
        self.ss = SuperStateSpace(self.N, self.K)
        self.N += 1   # will be decremented by caller (_merge_states)

        # Build old-index → new-index mapping
        o2n = {}
        for oi, oss in enumerate(old_ss.super_states):
            nss = tuple(base_remap[s] for s in oss)
            ni = self.ss.ss_idx.get(nss)
            if ni is not None:
                o2n[oi] = ni

        # ── π ────────────────────────────────────────────────────────────
        new_pi = np.full(self.ss.n_super, -np.inf)
        for oi, ni in o2n.items():
            new_pi[ni] = np.logaddexp(new_pi[ni], old_pi[oi])
        z = logsumexp(new_pi)
        self.log_pi = (
            new_pi - z if np.isfinite(z)
            else np.full(self.ss.n_super, -np.log(self.ss.n_super))
        )

        # ── A ────────────────────────────────────────────────────────────
        new_A = np.full((self.ss.n_super, self.ss.n_super), -np.inf)
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

        new_A[~self.ss.valid_mask] = -np.inf
        for p, ch in self.ss.children_of.items():
            z = logsumexp(new_A[p, ch])
            if np.isfinite(z):
                new_A[p, ch] -= z
            else:
                new_A[p, ch] = -np.log(len(ch))
        self.log_A = new_A

    # ── fit (Baum-Welch + adaptive merging) ─────────────────────────────

    def fit(
        self,
        O: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-4,
        merge_interval: int = 3,
        verbose: bool = True,
    ) -> "HigherOrderHMM":
        """
        Extended Baum-Welch EM with integrated periodic state merging.

        Parameters
        ----------
        O              : ndarray (T, D)  — observation sequence
        max_iter       : int             — maximum EM iterations
        tol            : float           — |ΔLL| convergence tolerance
        merge_interval : int             — attempt merges every k iters
        verbose        : bool            — print progress

        Returns
        -------
        self
        """
        prev_ll = -np.inf

        for it in range(1, max_iter + 1):
            gamma_ss, gamma_base, xi_log, ll = self._e_step(O)

            if verbose:
                print(f"  iter {it:3d} │ N={self.N} │ LL = {ll:+.4f}")

            self._m_step(O, gamma_ss, gamma_base, xi_log)

            # Periodic state merging
            if it % merge_interval == 0:
                merged = True
                while merged and self.N > 1:
                    _, g_base_fresh, _, _ = self._e_step(O)
                    merged = self._merge_states(g_base_fresh)

            # Convergence check
            if abs(ll - prev_ll) < tol and it > 5:
                if verbose:
                    print(f"  converged (ΔLL = {ll - prev_ll:.2e})")
                break
            prev_ll = ll

        return self

    # ── predict (Viterbi) ────────────────────────────────────────────────

    def predict(self, O: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Decode the most probable base-state sequence via Viterbi.

        Parameters
        ----------
        O : ndarray (T, D)

        Returns
        -------
        base_states : ndarray (T,)  — decoded base hidden states
        log_prob    : float         — log-probability of the decoded path
        """
        log_e = self._log_emissions(O)
        path_ss, log_prob = inference.viterbi(self.log_pi, self.log_A, log_e)
        return self.ss.emit_map[path_ss], log_prob

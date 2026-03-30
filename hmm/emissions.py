"""
hmm/emissions.py
----------------
Dirichlet emission model for compositional (simplex) observations.

Each hidden state i carries concentration parameters α_i ∈ R^D_+ defining:

    Dir(x | α) = Γ(Σ α_k) / Π_k Γ(α_k)  ·  Π_k x_k^{α_k - 1}

Public API
----------
DirichletEmission
    log_pdf_batch(O)         → (T, N) log-probabilities
    kl_divergence(i, j)      → scalar KL(Dir_i ‖ Dir_j)
    symmetric_kl(i, j)       → scalar 0.5·(KL_ij + KL_ji)
    update(O, gamma_base)    → weighted MLE update of alphas (M-step)
    merge(i, j, w_i, w_j)   → pool state j into state i, shrink table
"""

import numpy as np
from scipy.special import gammaln, digamma
from scipy.optimize import minimize

# Smallest value allowed for each observation dimension before renormalizing.
OBS_FLOOR = 1e-10


class DirichletEmission:
    """
    Manages a table of Dirichlet distributions, one per hidden state.

    Parameters
    ----------
    n_states : int
        Number of base hidden states N.
    n_dims : int
        Dimensionality D of observation vectors (simplex dimension).
    rng : numpy.random.Generator, optional
        Random generator used for parameter initialisation.
    """

    def __init__(self, n_states: int, n_dims: int, rng=None):
        self.n_states = n_states
        self.n_dims = n_dims
        self.rng = rng or np.random.default_rng(42)
        # Initialise concentration parameters uniformly in [1.5, 5.0].
        self.alphas = self.rng.uniform(1.5, 5.0, size=(n_states, n_dims))

    # ── helpers ──────────────────────────────────────────────────────────

    def _safe_obs(self, O: np.ndarray) -> np.ndarray:
        """Clip and renormalize *O* so every row lies on the open simplex."""
        X = np.clip(O, OBS_FLOOR, None)
        return X / X.sum(axis=-1, keepdims=True)

    # ── pdf ──────────────────────────────────────────────────────────────

    def log_pdf_batch(self, O: np.ndarray) -> np.ndarray:
        """
        Vectorized log Dir(x_t | α_k) for all time-steps and all states.

        Parameters
        ----------
        O : ndarray, shape (T, D)

        Returns
        -------
        log_b : ndarray, shape (T, N)
        """
        X = self._safe_obs(O)
        log_X = np.log(X)                                     # (T, D)
        a0 = self.alphas.sum(axis=1)                           # (N,)
        log_norm = gammaln(a0) - gammaln(self.alphas).sum(1)   # (N,)
        log_kern = log_X @ (self.alphas - 1.0).T               # (T, N)
        return log_norm[None, :] + log_kern

    # ── KL divergence ────────────────────────────────────────────────────

    def kl_divergence(self, i: int, j: int) -> float:
        """
        KL(Dir(α_i) ‖ Dir(α_j)).

        Closed-form expression:
            KL = ln Γ(α0_i) − Σ ln Γ(α_ik)
               − ln Γ(α0_j) + Σ ln Γ(α_jk)
               + Σ (α_ik − α_jk)(ψ(α_ik) − ψ(α0_i))
        """
        a, b = self.alphas[i], self.alphas[j]
        a0, b0 = a.sum(), b.sum()
        return (
            gammaln(a0) - gammaln(a).sum()
            - gammaln(b0) + gammaln(b).sum()
            + ((a - b) * (digamma(a) - digamma(a0))).sum()
        )

    def symmetric_kl(self, i: int, j: int) -> float:
        """Symmetrised KL: 0.5 · (KL(i ‖ j) + KL(j ‖ i))."""
        return 0.5 * (self.kl_divergence(i, j) + self.kl_divergence(j, i))

    # ── M-step update ────────────────────────────────────────────────────

    def update(self, O: np.ndarray, gamma_base: np.ndarray) -> None:
        """
        Weighted MLE for Dirichlet concentration parameters (M-step).

        For state k the weighted log-likelihood objective is:

            ℓ_k(α) = ln Γ(Σ α_d) − Σ ln Γ(α_d) + Σ (α_d − 1) s_d

        where  W = Σ_t γ_t(k)  and  s_d = Σ_t γ_t(k) ln x_{t,d} / W.

        Optimisation is performed in log-space (log α) with L-BFGS-B to
        enforce strict positivity of all concentration parameters.

        Parameters
        ----------
        O          : ndarray (T, D)  — observation matrix
        gamma_base : ndarray (T, N)  — base-state occupancy weights γ_t(k)
        """
        X = self._safe_obs(O)
        log_X = np.log(X)

        for k in range(self.n_states):
            w = gamma_base[:, k]
            W = w.sum()
            if W < 1e-12:
                continue

            # Weighted mean of log-observations for state k
            s = (w[:, None] * log_X).sum(0) / W
            a0 = self.alphas[k].copy()

            def _neg_ll(log_a):
                a = np.exp(log_a)
                return -(gammaln(a.sum()) - gammaln(a).sum()
                         + ((a - 1.0) * s).sum())

            def _grad(log_a):
                a = np.exp(log_a)
                return -(digamma(a.sum()) - digamma(a) + s) * a

            try:
                res = minimize(
                    _neg_ll, np.log(a0), jac=_grad,
                    method="L-BFGS-B",
                    options={"maxiter": 80, "ftol": 1e-10},
                )
                a_new = np.exp(res.x)
                if np.all(np.isfinite(a_new)) and np.all(a_new > 0.01):
                    self.alphas[k] = a_new
            except Exception:
                pass  # Keep previous alphas if optimisation fails

    # ── state merging ────────────────────────────────────────────────────

    def merge(self, i: int, j: int, w_i: float, w_j: float) -> None:
        """
        Pool state *j* into state *i* using occupancy-weighted averaging,
        then remove row *j* from the parameter table.

        Parameters
        ----------
        i, j   : indices of the two states to merge (i is kept, j removed)
        w_i, w_j : total occupancy weights Σ_t γ_t(·) for each state
        """
        total = w_i + w_j
        if total < 1e-12:
            merged_alpha = 0.5 * (self.alphas[i] + self.alphas[j])
        else:
            merged_alpha = (w_i * self.alphas[i] + w_j * self.alphas[j]) / total

        self.alphas[i] = merged_alpha
        self.alphas = np.delete(self.alphas, j, axis=0)
        self.n_states -= 1

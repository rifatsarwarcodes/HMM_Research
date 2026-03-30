"""
hmm/inference.py
----------------
Pure-function log-space inference routines for a K-th order HMM.

All three algorithms operate entirely in log-space to prevent floating-point
underflow on long observation sequences.

Public API
----------
forward(log_pi, log_A, log_e)
    → (log_alpha, log_likelihood)

backward(log_A, log_e)
    → log_beta

viterbi(log_pi, log_A, log_e)
    → (path_super_states, log_prob)
"""

import numpy as np
from scipy.special import logsumexp


def forward(
    log_pi: np.ndarray,
    log_A: np.ndarray,
    log_e: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Log-space forward (α) recursion.

    Implements the K-th order extension:

        α_{t+1}(i_2,…,i_K, j) =
            [ Σ_{i_1} α_t(i_1,…,i_K) · A(i_1…i_K → j) ] · φ_j(o_{t+1})

    Parameters
    ----------
    log_pi : ndarray (M,)
        Log initial super-state distribution.
    log_A  : ndarray (M, M)
        Log transition matrix (−∞ for invalid transitions).
    log_e  : ndarray (T, M)
        Log emission probabilities per time-step and super-state.

    Returns
    -------
    log_alpha : ndarray (T, M)
    log_likelihood : float   —  log P(O | λ)
    """
    T, M = log_e.shape
    log_alpha = np.full((T, M), -np.inf)

    # Initialise
    log_alpha[0] = log_pi + log_e[0]

    # Recursion
    for t in range(1, T):
        # log_alpha[t, j] = log_e[t, j]
        #                  + logsumexp_i( log_alpha[t-1, i] + log_A[i, j] )
        log_alpha[t] = log_e[t] + logsumexp(
            log_alpha[t - 1, :, None] + log_A, axis=0
        )

    log_likelihood = float(logsumexp(log_alpha[-1]))
    return log_alpha, log_likelihood


def backward(
    log_A: np.ndarray,
    log_e: np.ndarray,
) -> np.ndarray:
    """
    Log-space backward (β) recursion.

    Parameters
    ----------
    log_A : ndarray (M, M)
    log_e : ndarray (T, M)

    Returns
    -------
    log_beta : ndarray (T, M)
    """
    T, M = log_e.shape
    log_beta = np.full((T, M), -np.inf)
    log_beta[-1] = 0.0  # β_T(i) = 1 ∀ i  →  log β_T = 0

    for t in range(T - 2, -1, -1):
        # log_beta[t, i] = logsumexp_j( log_A[i,j] + log_e[t+1,j]
        #                               + log_beta[t+1,j] )
        log_beta[t] = logsumexp(
            log_A + log_e[t + 1][None, :] + log_beta[t + 1][None, :],
            axis=1,
        )

    return log_beta


def viterbi(
    log_pi: np.ndarray,
    log_A: np.ndarray,
    log_e: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Log-space Viterbi decoding for a K-th order HMM.

    Returns the most probable super-state path via max-product recursion
    and backtracking.

    Parameters
    ----------
    log_pi : ndarray (M,)
    log_A  : ndarray (M, M)
    log_e  : ndarray (T, M)

    Returns
    -------
    path : ndarray (T,)   — optimal super-state indices
    log_prob : float      — log-probability of the optimal path
    """
    T, M = log_e.shape
    delta = np.full((T, M), -np.inf)   # best log-prob reaching each state
    psi   = np.zeros((T, M), dtype=int)  # backpointer

    # Initialise
    delta[0] = log_pi + log_e[0]

    # Recursion
    for t in range(1, T):
        scores = delta[t - 1, :, None] + log_A    # (M, M)
        psi[t]   = scores.argmax(axis=0)
        delta[t] = scores.max(axis=0) + log_e[t]

    # Backtrack
    path = np.empty(T, dtype=int)
    path[-1] = int(delta[-1].argmax())
    log_prob = float(delta[-1, path[-1]])
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]

    return path, log_prob

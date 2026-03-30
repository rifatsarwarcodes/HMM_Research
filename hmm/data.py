"""
hmm/data.py
-----------
Synthetic data generation for testing and benchmarking the HigherOrderHMM.

generate_synthetic(T, seed)
    Produces a 3-state, 2nd-order HMM sequence on a 3-simplex.

Ground-truth setup
------------------
    Emissions:
        state 0  →  Dir(10,  1,  1)   concentrated on x_1
        state 1  →  Dir( 1, 10,  1)   concentrated on x_2
        state 2  →  Dir( 1,  1, 10)   concentrated on x_3

    Transitions (2nd order, strong 0→1→2→0 cycle):
        P(q_t = k | q_{t-2}=i, q_{t-1}=j) defined by *trans* below.
"""

import numpy as np


def generate_synthetic(
    T: int = 500,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic observation/state sequence for a 3-state,
    2nd-order HMM on a 3-dimensional simplex.

    Parameters
    ----------
    T    : int   — number of time-steps
    seed : int   — random seed

    Returns
    -------
    obs         : ndarray (T, 3)    — observation vectors on the simplex
    states      : ndarray (T,)      — ground-truth hidden states {0, 1, 2}
    true_alphas : ndarray (3, 3)    — ground-truth Dirichlet parameters
    true_trans  : ndarray (3, 3, 3) — true transition tensor
                                       trans[i,j,k] = P(q_t=k | q_{t-2}=i, q_{t-1}=j)
    """
    rng = np.random.default_rng(seed)
    N, D = 3, 3

    # ── emission parameters ───────────────────────────────────────────────
    true_alphas = np.array([
        [10., 1., 1.],   # state 0 → mode near (1, 0, 0)
        [ 1., 10., 1.],  # state 1 → mode near (0, 1, 0)
        [ 1., 1., 10.],  # state 2 → mode near (0, 0, 1)
    ])

    # ── 2nd-order transition tensor ───────────────────────────────────────
    # Base: small uniform noise; override dominant transitions for the cycle.
    trans = np.full((N, N, N), 0.05)
    trans[:, 0, 1] = 0.9    # after any i, then 0 → go to 1
    trans[0, 1, 2] = 0.9    # after 0, then 1    → go to 2
    trans[1, 2, 0] = 0.9    # after 1, then 2    → go to 0
    trans[2, 0, 1] = 0.9    # after 2, then 0    → go to 1

    # Normalize each row to sum to 1
    for i in range(N):
        for j in range(N):
            trans[i, j] /= trans[i, j].sum()

    # ── simulate state sequence ───────────────────────────────────────────
    states = np.empty(T, dtype=int)
    states[0], states[1] = 0, 1
    for t in range(2, T):
        states[t] = rng.choice(N, p=trans[states[t - 2], states[t - 1]])

    # ── sample observations from the corresponding Dirichlet ──────────────
    obs = np.zeros((T, D))
    for t in range(T):
        obs[t] = rng.dirichlet(true_alphas[states[t]])

    return obs, states, true_alphas, trans

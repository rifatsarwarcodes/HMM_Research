"""
hmm/super_states.py
-------------------
Super-state enumeration and adjacency bookkeeping for a K-th order HMM.

A *super-state* is a K-tuple of base states (i_1, …, i_K).  For N base
states and order K the full super-state space has N^K elements.

Only *overlap-consistent* transitions are structurally valid:

    (i_1, i_2, …, i_K)  →  (i_2, …, i_K, j)   for any base state j.

Public API
----------
SuperStateSpace(N, K)
    .super_states   list[tuple]          — ordered list of all K-tuples
    .n_super        int                  — total number of super-states (N^K)
    .ss_idx         dict[tuple, int]     — K-tuple → super-state index
    .emit_map       ndarray (n_super,)   — last element of each super-state
    .valid_mask     ndarray (n_super, n_super) bool
    .children_of    dict[int, list[int]] — valid successor indices
    .parents_of     dict[int, list[int]] — valid predecessor indices
"""

import numpy as np
from itertools import product as cartesian_product


class SuperStateSpace:
    """
    Enumerate and cache the super-state structure for a K-th order HMM.

    Parameters
    ----------
    N : int   — number of base hidden states
    K : int   — Markov order (K=1 → standard HMM)
    """

    def __init__(self, N: int, K: int):
        self.N = N
        self.K = K
        self._build()

    def _build(self) -> None:
        """Enumerate all K-tuples and precompute adjacency structures."""
        bases = list(range(self.N))

        self.super_states: list = list(cartesian_product(bases, repeat=self.K))
        self.n_super: int = len(self.super_states)
        self.ss_idx: dict = {ss: i for i, ss in enumerate(self.super_states)}

        # Each super-state emits via its *last* base state.
        self.emit_map: np.ndarray = np.array(
            [ss[-1] for ss in self.super_states], dtype=int
        )

        # Transition adjacency
        self.valid_mask: np.ndarray = np.zeros(
            (self.n_super, self.n_super), dtype=bool
        )
        self.children_of: dict = {}
        self.parents_of: dict = {}

        # Children: (i1,…,iK) → (i2,…,iK, j) for each base state j
        for p, p_ss in enumerate(self.super_states):
            ch = []
            for j in bases:
                c_ss = p_ss[1:] + (j,)
                c = self.ss_idx[c_ss]
                self.valid_mask[p, c] = True
                ch.append(c)
            self.children_of[p] = ch

        # Parents: (i1,…,iK) has parent (i0, i1,…,i_{K-1}) for each i0
        for c, c_ss in enumerate(self.super_states):
            pa = []
            for i in bases:
                p_ss = (i,) + c_ss[:-1]
                pa.append(self.ss_idx[p_ss])
            self.parents_of[c] = pa

    def remap(self, keep: int, drop: int) -> dict:
        """
        Build a base-state remapping after merging *drop* into *keep*.

        All base-state indices above *drop* are shifted down by 1;
        *drop* itself maps to the (post-shift) index of *keep*.

        Returns
        -------
        remap : dict[old_base_state → new_base_state]
        """
        result = {}
        c = 0
        for old in range(self.N + 1):          # N+1 because N was decremented
            if old == drop:
                result[old] = result[keep]
            else:
                result[old] = c
                c += 1
        return result

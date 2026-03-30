"""
main.py
-------
Entry-point for the modular Higher-Order HMM package.

Runs the same synthetic validation as hmm_framework.py but imports
everything from the `hmm` package instead of a single-file module.

Usage
-----
    python main.py
"""

from collections import Counter
import numpy as np

from hmm import HigherOrderHMM
from hmm.data import generate_synthetic


def main() -> None:
    print("=" * 65)
    print("  Higher-Order HMM ─ Synthetic Validation  (modular)")
    print("=" * 65)

    # ── 1. Generate data ─────────────────────────────────────────────────
    T = 500
    obs, true_states, true_alphas, true_trans = generate_synthetic(T)
    print(f"\n• Generated T={T} observations on a 3-simplex")
    print(f"• Ground-truth states: {np.unique(true_states)}  (cycle 0→1→2→0…)")

    # ── 2. Train ─────────────────────────────────────────────────────────
    print(f"\n{'─' * 65}")
    print("  Training:  N_max=6,  K=2,  τ=0.5")
    print(f"{'─' * 65}\n")

    model = HigherOrderHMM(
        N_max=6, order_K=2, kl_threshold=0.5, n_dims=3, seed=0
    )
    model.fit(obs, max_iter=80, merge_interval=3, verbose=True)

    # ── 3. Results ───────────────────────────────────────────────────────
    print(f"\n{'─' * 65}")
    print("  Results")
    print(f"{'─' * 65}")
    print(f"  Final number of states : {model.N}  (target = 3)")

    decoded, log_p = model.predict(obs)
    print(f"  Viterbi log-prob       : {log_p:.2f}")

    # Label alignment via majority vote
    label_map = {}
    for pred_label in np.unique(decoded):
        votes = true_states[decoded == pred_label]
        label_map[pred_label] = Counter(votes.tolist()).most_common(1)[0][0]

    mapped = np.array([label_map[d] for d in decoded])
    acc = (mapped == true_states).mean()
    print(f"  Accuracy (mapped)      : {acc * 100:.1f} %")

    print("\n  Learned Dirichlet α per state:")
    for k in range(model.N):
        print(f"    state {k}: {model.emissions.alphas[k].round(2)}")

    print("\n  Ground-truth α:")
    for k in range(3):
        print(f"    state {k}: {true_alphas[k]}")

    print(f"\n  Decoded path (first 60): {decoded[:60].tolist()}")
    print(f"  True path   (first 60): {true_states[:60].tolist()}")
    print("=" * 65)


if __name__ == "__main__":
    main()

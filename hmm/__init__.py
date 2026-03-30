"""
hmm/__init__.py
---------------
Public API surface for the `hmm` package.

Typical usage
-------------
    from hmm import HigherOrderHMM
    from hmm.data import generate_synthetic

    obs, states, _, _ = generate_synthetic(T=500)
    model = HigherOrderHMM(N_max=6, order_K=2, kl_threshold=0.5, n_dims=3)
    model.fit(obs)
    decoded, log_p = model.predict(obs)
"""

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from .model import HigherOrderHMM
from .emissions import DirichletEmission
from .super_states import SuperStateSpace
from . import inference
from . import data

__all__ = [
    "HigherOrderHMM",
    "DirichletEmission",
    "SuperStateSpace",
    "inference",
    "data",
]

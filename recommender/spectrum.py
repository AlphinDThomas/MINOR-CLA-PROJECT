"""spectrum.py -- singular value spectrum of the mean-centred sparse ratings matrix (Phase 5).
Centring subtracts the global mean from OBSERVED entries only, so the matrix stays sparse.
(Unobserved entries then sit at 0 = 'average rating'; this is the usual convention for svds on ratings.)"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def centered(R, how="global"):
    """Return a sparse copy of R with observed ratings centred. how: 'global' | 'user'."""
    C = R.tocsr().astype(float).copy()
    if how == "global":
        C.data -= C.data.mean()
    elif how == "user":
        counts = np.diff(C.indptr)
        means = np.asarray(C.sum(axis=1)).ravel() / np.maximum(counts, 1)
        C.data -= np.repeat(means, counts)
    else:
        raise ValueError(how)
    return C


def top_singular_values(R, k=50, how="global", seed=42):
    C = centered(R, how)
    k = min(k, min(C.shape) - 1)
    v0 = np.random.default_rng(seed).normal(size=min(C.shape))          # fixed start -> reproducible
    s = svds(C, k=k, return_singular_vectors=False, v0=v0)
    return np.sort(s)[::-1]


def plot_spectrum(spectra, path, title="Top singular values of mean-centred ratings matrix"):
    """spectra: dict label -> array of singular values (e.g. {'clean': s1, 'attacked': s2})."""
    plt.figure(figsize=(6.5, 4.2))
    for label, s in spectra.items():
        plt.semilogy(np.arange(1, len(s) + 1), s, "o-", ms=3, label=label)
    plt.xlabel("index j"); plt.ylabel("singular value $\\sigma_j$ (log scale)")
    plt.title(title); plt.grid(alpha=.3, which="both")
    if len(spectra) > 1 or list(spectra)[0] != "":
        plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

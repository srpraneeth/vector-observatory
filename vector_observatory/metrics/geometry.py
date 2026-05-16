from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GeometryMetrics:
    anisotropy: float
    """Average cosine similarity between random embedding pairs.
    High value (→1.0) = embeddings occupy a narrow cone = collapse risk.
    Healthy range: < 0.3 for most embedding models."""

    isotropy_score: float
    """Partition function-based isotropy (Mu & Viswanath 2018).
    Higher = more uniform use of the embedding space."""

    intrinsic_dim: float
    """Two-NN estimate of the intrinsic dimensionality.
    How many dimensions does the data actually use?"""

    variance_per_dim: np.ndarray
    """Per-dimension variance. Near-zero dimensions are unused capacity."""

    @property
    def n_dead_dims(self, threshold: float = 1e-4) -> int:
        """Count dimensions with variance below threshold."""
        return int((self.variance_per_dim < threshold).sum())


def compute_geometry_metrics(embeddings: np.ndarray, n_pairs: int = 1000) -> GeometryMetrics:
    """Compute geometric health metrics for an embedding matrix.

    Args:
        embeddings: (N, D) float32 array.
        n_pairs: Number of random pairs to sample for anisotropy estimation.
    """
    rng = np.random.default_rng(42)

    anisotropy = _compute_anisotropy(embeddings, n_pairs, rng)
    isotropy = _compute_isotropy(embeddings, rng)
    intrinsic_dim = _compute_intrinsic_dim(embeddings)
    variance_per_dim = np.var(embeddings, axis=0)

    return GeometryMetrics(
        anisotropy=anisotropy,
        isotropy_score=isotropy,
        intrinsic_dim=intrinsic_dim,
        variance_per_dim=variance_per_dim,
    )


def _compute_anisotropy(embeddings: np.ndarray, n_pairs: int, rng: np.random.Generator) -> float:
    """Average cosine similarity between random pairs."""
    N = len(embeddings)
    n_pairs = min(n_pairs, N * (N - 1) // 2)
    idx_a = rng.integers(0, N, size=n_pairs)
    idx_b = rng.integers(0, N, size=n_pairs)
    same = idx_a == idx_b
    idx_b[same] = (idx_b[same] + 1) % N

    a = embeddings[idx_a]
    b = embeddings[idx_b]
    norm_a = np.linalg.norm(a, axis=1, keepdims=True) + 1e-10
    norm_b = np.linalg.norm(b, axis=1, keepdims=True) + 1e-10
    cosine_sims = (a / norm_a * b / norm_b).sum(axis=1)
    return float(cosine_sims.mean())


def _compute_isotropy(embeddings: np.ndarray, rng: np.random.Generator) -> float:
    """Partition function isotropy score (Mu & Viswanath 2018).

    I(W) = min_c Z(c) / max_c Z(c), where Z(c) = mean exp(w · c) over embeddings.
    Range: 0 (fully anisotropic) → 1 (perfectly uniform over the hypersphere).
    """
    N, D = embeddings.shape
    n_samples = min(N, 500)
    idx = rng.choice(N, size=n_samples, replace=False)
    E = embeddings[idx]

    norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-10
    E_normalized = E / norms

    random_dirs = rng.standard_normal((200, D))
    random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True) + 1e-10

    # Z(c) for each of the 200 directions
    Z = np.exp(E_normalized @ random_dirs.T).mean(axis=0)  # shape (200,)
    return float(Z.min() / Z.max()) if Z.max() > 0 else 0.0


def _compute_intrinsic_dim(embeddings: np.ndarray) -> float:
    """Two-NN estimator of intrinsic dimensionality (Facco et al. 2017)."""
    from sklearn.neighbors import NearestNeighbors

    N = len(embeddings)
    if N < 4:
        return float(embeddings.shape[1])

    nn = NearestNeighbors(n_neighbors=3).fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)

    r1 = distances[:, 1]  # distance to 1st neighbor
    r2 = distances[:, 2]  # distance to 2nd neighbor

    # Avoid log(0)
    valid = (r1 > 1e-10) & (r2 > r1)
    if valid.sum() < 10:
        return float(embeddings.shape[1])

    mu = r2[valid] / r1[valid]
    return float(1.0 / np.log(mu).mean())

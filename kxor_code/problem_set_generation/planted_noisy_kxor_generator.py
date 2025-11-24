

from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any
import numpy as np

from kxor_code.problem_set_generation.kxor_instance import KXORInstance


class PlantedNoisyKXORGenerator:
    """
    Generator for the distributions R_{n,k}(m) and P^z_{n,k}(m, ρ).
    """

    def __init__(self, n: int, k: int, rng: Optional[np.random.Generator] = None):
        assert k >= 2 and k <= n
        self.n = n
        self.k = k
        self.rng = rng if rng is not None else np.random.default_rng()

    # ---------- basic helpers ----------

    def _sample_scopes(self, m: int) -> np.ndarray:
        """
        Sample m scopes S_i ⊆ [n] of size k each.
        Each row is a sorted array of variable indices (0-based).
        """
        scopes = np.empty((m, self.k), dtype=int)
        for i in range(m):
            scopes[i] = np.sort(self.rng.choice(self.n, size=self.k, replace=False))
        return scopes

    def _sample_rademacher(self, size: int) -> np.ndarray:
        """
        Sample ±1 with equal probability (Rademacher).
        """
        # rng.integers(0, 2) gives 0 or 1; map to -1, +1:
        bits = self.rng.integers(0, 2, size=size, dtype=int)
        return 2 * bits - 1

    # ---------- distributions from the paper ----------

    def sample_random(self, m: int) -> KXORInstance:
        """
        Sample an instance from R_{n,k}(m):
          - scopes S_i uniform
          - b_i i.i.d. Rademacher
        """
        scopes = self._sample_scopes(m)
        b = self._sample_rademacher(m)

        return KXORInstance(
            n=self.n,
            k=self.k,
            m=m,
            scopes=scopes,
            b=b,
            is_planted=False,
            rho=None,
            z=None,
        )

    def sample_planted(
        self,
        m: int,
        rho: float,
        z: Optional[np.ndarray] = None,
    ) -> KXORInstance:
        """
        Sample an instance from P^z_{n,k}(m, ρ):

            S_i ~ uniform k-subsets of [n]
            η_i ∈ {±1} with E[η_i] = ρ
            b_i = η_i * ∏_{j∈S_i} z_j

        Parameters
        ----------
        m : int
            Number of constraints.
        rho : float
            Planted advantage ρ = 1 - 2η (in [0,1]).
        z : Optional[np.ndarray]
            Secret assignment z ∈ {±1}^n.
            If None, a random one is generated.
        """
        assert 0.0 <= rho <= 1.0
        scopes = self._sample_scopes(m)

        # Secret assignment z ∈ {±1}^n
        if z is None:
            z = self._sample_rademacher(self.n)
        else:
            assert z.shape == (self.n,)

        # noise η_i with E[η_i] = ρ
        # P(η = +1) = (1+ρ)/2,  P(η = -1) = (1-ρ)/2
        p_plus = (1.0 + rho) / 2.0
        u = self.rng.random(m)
        eta = np.where(u < p_plus, 1, -1).astype(int)

        # compute z_{S_i} = ∏_{j∈S_i} z_j  (over {±1})
        z_S = np.prod(z[scopes], axis=1)  # shape (m,)

        # b_i = η_i * z_{S_i}
        b = eta * z_S

        return KXORInstance(
            n=self.n,
            k=self.k,
            m=m,
            scopes=scopes,
            b=b,
            is_planted=True,
            rho=rho,
            z=z,
        )

    # ---------- optional: Poissonized version ˜R, ˜P ----------

    def sample_random_poisson(self, m_bar: float) -> KXORInstance:
        """
        Poissonized version: m ~ Poi(m_bar), then sample from R_{n,k}(m).
        """
        m = int(self.rng.poisson(m_bar))
        return self.sample_random(m)

    def sample_planted_poisson(
        self, m_bar: float, rho: float, z: Optional[np.ndarray] = None
    ) -> KXORInstance:
        """
        Poissonized version: m ~ Poi(m_bar), then sample from P^z_{n,k}(m, ρ).
        """
        m = int(self.rng.poisson(m_bar))
        return self.sample_planted(m, rho, z=z)
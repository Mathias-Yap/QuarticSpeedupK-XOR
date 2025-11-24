from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any
import numpy as np


@dataclass
class KXORInstance:
    """
    A k-XOR instance.

    Variables: x_1,...,x_n in {±1}
    Constraints: (S_i, b_i) with |S_i| = k, b_i in {±1}
    """
    n: int                 # number of variables
    k: int                 # clause size
    m: int                 # number of constraints
    scopes: np.ndarray     # shape (m, k), each row is a subset S_i ⊆ {0,...,n-1}
    b: np.ndarray          # shape (m,), entries in {+1, -1}
    is_planted: bool       # True if drawn from planted distribution
    rho: Optional[float]   # planted advantage ρ (None for random instances)
    z: Optional[np.ndarray]  # secret assignment z ∈ {±1}^n if planted, else None


    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            n=self.n,
            k=self.k,
            m=self.m,
            scopes=self.scopes,
            b=self.b,
            is_planted=self.is_planted,
            rho=-1.0 if self.rho is None else self.rho,
            has_rho=self.rho is not None,
            z=self.z if self.z is not None else np.array([], dtype=np.int8),
            has_z=self.z is not None,
        )
    @classmethod
    def create(
        cls,
        scopes: np.ndarray,
        b: np.ndarray,
        is_planted: bool = False,
        rho: Optional[float] = None,
        z: Optional[np.ndarray] = None,
    ) -> "KXORInstance":

        scopes = np.ascontiguousarray(scopes)
        b = np.ascontiguousarray(b, dtype=np.int8)

        m, k = scopes.shape
        n = int(scopes.max()) + 1 if scopes.size > 0 else 0

        return cls(
            n=n,
            k=k,
            m=m,
            scopes=scopes,
            b=b,
            is_planted=is_planted,
            rho=rho,
            z=z,
        )
        
    @classmethod
    def load(cls, path: str) -> "KXORInstance":
        data = np.load(path, allow_pickle=False)
        
        verify_keys = [
            "n", "k", "m", "scopes", "b",
            "is_planted", "has_rho", "rho",
            "has_z", "z"
        ]
        for key in verify_keys:
            if key not in data:
                raise ValueError(f"Missing key '{key}' in loaded KXORInstance data.")
        
        rho = None if not data["has_rho"] else float(data["rho"])
        z = data["z"]
        if not data["has_z"]:
            z = None

        return cls.create(
            scopes=data["scopes"],
            b=data["b"],
            is_planted=bool(data["is_planted"]),
            rho=rho,
            z=z,
        )
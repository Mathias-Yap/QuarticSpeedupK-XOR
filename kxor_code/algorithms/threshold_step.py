from typing import Any, Dict, Optional
import numpy as np
from kxor_code.algorithms.base_alg_step import BaseAlgorithmStep, ProblemRecord, StepContext, StepStats
from scipy.sparse import dok_matrix
from scipy.special import comb

class ThresholdStep(BaseAlgorithmStep):
    """This class computes the threshold of the eigenvalues for the problem instance as well as an upper bound on the number of
    eigenvalues above the threshold which helps to reduce computational load for computing overlap with cutoff eigenspace experiments."""
    requires_fields = ["kikuchi_matrix", "ell", "kappa"]
    produces_fields = ["num_eigenvalues", "threshold"]

    def _run(self, problem: ProblemRecord, context: StepContext, stats: StepStats) -> Optional[Dict[str, Any]]:
        k = problem.instance.k
        n = problem.instance.n
        m = problem.instance.m
        ell = problem.get_field("ell")
        kappa = problem.get_field("kappa")
        delta_lnk = comb(k,k/2) * ((comb(n - k, ell - k/2))/comb(n, ell))
        d = delta_lnk * m
        threshold = kappa * d
        kikuchi_matrix = problem.get_field("kikuchi_matrix")
        if d <= 0:
            raise ValueError("Threshold d must be greater than zero.")

        # Calculate the squared Frobenius norm: sum of squares of non-zero entries
        frobenius_norm_sq = sum(val**2 for val in kikuchi_matrix.values())
        
        # The bound derived from N * d^2 <= sum(lambda^2)
        upper_bound = int(np.floor(frobenius_norm_sq / (d**2)))
        
        # The number of eigenvalues cannot exceed the dimension of the matrix
        n = kikuchi_matrix.shape[0] #type: ignore

        return {
            "num_eigenvalues": min(upper_bound, n), #type: ignore
            "threshold": threshold
        }
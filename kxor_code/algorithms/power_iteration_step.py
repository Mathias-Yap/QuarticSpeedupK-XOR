from kxor_code.algorithms.base_alg_step import BaseAlgorithmStep, ProblemRecord, StepContext, StepStats
import numpy as np
from typing import Optional


class PowerIterationStep(BaseAlgorithmStep):
    requires_fields = ["kikuchi_matrix"]
    produces_fields = ["dominant_eigenvector"]
    num_iterations: int = 1000
    starting_vector: Optional[np.array] = None #type: ignore
    
    
    def _run(self, problem: ProblemRecord, context: StepContext, stats: StepStats) -> Optional[dict]:
        """Perform power iteration on the Kikuchi matrix to find the dominant eigenvector.

        Parameters:
            kikuchi_matrix (np.ndarray): The Kikuchi matrix.
            num_iterations (int): Number of power iterations to perform.
            starting_vector (Optional[np.ndarray]): Optional starting vector for the iteration.

        Returns:
            np.ndarray: The dominant eigenvector found by power iteration.
        """
        kikuchi_matrix = problem.get_field("kikuchi_matrix")
        n = kikuchi_matrix.shape[0]

        if self.starting_vector is not None:
            b_k = self.starting_vector
        else:
            b_k = np.random.rand(n)

        for _ in range(self.num_iterations):
            # Calculate the matrix-by-vector product
            b_k1 = kikuchi_matrix @ b_k

            # Normalize the vector
            b_k1_norm = np.linalg.norm(b_k1)
            b_k = b_k1 / b_k1_norm

        return {"dominant_eigenvector": b_k}
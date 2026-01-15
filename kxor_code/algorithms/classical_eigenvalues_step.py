from kxor_code.algorithms.base_alg_step import BaseAlgorithmStep, ProblemRecord, StepContext, StepStats
from typing import Any, Dict, Optional
from scipy.sparse.linalg import eigsh

class ClassicalEigenvaluesStep(BaseAlgorithmStep):
    requires_fields = ["kikuchi_matrix", "num_eigenvalues"]
    produces_fields = ["eigenvalues", "eigenvectors"]

    def _run(self, problem: ProblemRecord, context: StepContext, stats: StepStats) -> Optional[Dict[str, Any]]:
        """Compute the eigenvalues and eigenvectors of the Kikuchi matrix. Utilizes sparse matrix methods for efficiency. ARPACK wrapper: https://github.com/opencollab/arpack-ng

        Parameters:
            kikuchi_matrix (dok_matrix): The Kikuchi matrix.
            num_eigenvalues (int): The number of eigenvalues and eigenvectors to compute. If -1, compute all.
        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the eigenvalues and eigenvectors.
        """
        kikuchi_matrix = problem.get_field("kikuchi_matrix")
        num_eigenvalues = problem.get_field("num_eigenvalues")
        if not kikuchi_matrix.shape:
            raise ValueError("Kikuchi matrix is empty.")
        if num_eigenvalues == -1:
            eigvalues_count = kikuchi_matrix.shape[0]
        else:
            eigvalues_count = num_eigenvalues
        eigenvalues, eigenvectors = eigsh(kikuchi_matrix, k=eigvalues_count)
        return {"eigenvalues": eigenvalues, "eigenvectors": eigenvectors}
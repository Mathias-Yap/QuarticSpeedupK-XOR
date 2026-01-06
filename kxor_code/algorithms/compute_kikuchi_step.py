from itertools import combinations
from typing import Any, Dict, Optional
from kxor_code.algorithms.base_alg_step import BaseAlgorithmStep, ProblemRecord, StepContext, StepStats
from scipy.sparse import csr_matrix, dok_matrix

class ComputeKikuchiStep(BaseAlgorithmStep):
    requires_fields = ["ell"]
    produces_fields = ["kikuchi_matrix"]
    
    
    def _run(self, problem: ProblemRecord, context: StepContext, stats: StepStats) -> Optional[Dict[str, Any]]:
        problem_instance = problem.instance
        n = problem_instance.n
        scopes = problem_instance.scopes
        signs = problem_instance.b
        ell = problem.get_field("ell")

        clauses = [(frozenset(scope), sign) for scope, sign in zip(scopes, signs)]

        subsets = [frozenset(s) for s in combinations(range(n), ell)]
        subset_index = {s: i for i, s in enumerate(subsets)}
        m = len(subsets)

        incidence = dok_matrix((m, m), dtype=int)

        for i, S in enumerate(subsets):
            for C, sign in clauses:
                T = S.symmetric_difference(C)
                j = subset_index.get(T)
                if j is not None and j >= i:
                    incidence[i, j] = sign
                    incidence[j, i] = sign
        
        
        return {"kikuchi_matrix": incidence}

            

from itertools import combinations

from .k_xor_instance import KXORInstance, PlantedNoisyKXORGenerator
import numpy as np
from scipy.sparse import csr_matrix, dok_matrix
from scipy.sparse.linalg import eigsh

def compute_kikuchi_matrix(problem_instance: KXORInstance, ell: int): 
    pass 

def compute_matchings(problem_instance: KXORInstance, ell, int):
    combs = combinations(range(problem_instance.n), ell)
    # for clause, result in problem_instance.scopes, problem_instance.b:
    #     pass
    
def two_xor_matrix(problem_instance: KXORInstance):
    """
    Compute the adjacency matrix of a 2-XOR problem instance.
    Parameters: 
        problem_instance (KXORInstance): The 2-XOR problem instance.
    Returns:
        csr_matrix: The adjacency matrix in sparse (csr) format.
    """
    
    if problem_instance.k != 2:
        raise ValueError("The provided problem instance is not a 2-XOR instance.")
    scopes_array = problem_instance.scopes 
    edge_signs_array = problem_instance.b

    incidence = np.zeros((problem_instance.m, problem_instance.n), dtype=int)
    for i, (clause, sign) in enumerate(zip(scopes_array, edge_signs_array)):
        clause = np.asarray(clause, dtype=int)
        incidence[i, clause] = sign
    
    return csr_matrix(incidence)


def kikuchi_matrix_sets(problem_instance, ell: int):
    if problem_instance.k < 2:
        raise ValueError("Kikuchi matrix is only defined for k >= 2.")
    if ell < problem_instance.k / 2:
        raise ValueError("ell must be at least k/2.")

    n = problem_instance.n
    scopes = problem_instance.scopes
    signs = problem_instance.b

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

    return csr_matrix(incidence)

def kikuchi_matrix(problem_instance: KXORInstance, ell: int):
    """
    Compute the Kikuchi matrix for a given K-XOR problem instance and ell.
    Parameters: 
        problem_instance (KXORInstance): The K-XOR problem instance.
        ell (int): The size of the subsets to consider.
    Returns:
        csr_matrix: The Kikuchi matrix in sparse (csr) format.
    """
    if problem_instance.k < 2: 
        raise ValueError("Kikuchi matrix is only defined for k >= 2.")
    if ell < problem_instance.k/2:
        raise ValueError("ell must be at least k/2.")
    
    # Generate all subsets of size ell
    subsets = list(combinations(range(problem_instance.n), ell))
    subsets = [set(s) for s in subsets] 

    # Initialize Kikuchi Graph incidence matrix
    incidence = np.zeros((len(subsets), len(subsets)), dtype=int)
    for subset_1 in range(len(subsets)):
        for subset_2 in range(subset_1, len(subsets)):
            sym_diff = set(subsets[subset_1]).symmetric_difference(set(subsets[subset_2]))
            if len(sym_diff) == problem_instance.k:
                clause = tuple(sorted(sym_diff))
                for i, (clause_inst, sign) in enumerate(zip(problem_instance.scopes, problem_instance.b)):
                    if tuple(sorted(clause_inst)) == clause:
                        incidence[subset_1, subset_2] = sign
                        incidence[subset_2, subset_1] = sign
                        break
    return csr_matrix(incidence) 

def find_eigenvalues_and_vectors(kikuchi_matrix: csr_matrix, num_eigenvalues: int) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = eigsh(kikuchi_matrix, k=num_eigenvalues)
    return eigenvalues, eigenvectors
        

def find_matchings(clause: tuple, subsets: list, sign: int) -> list[tuple[int,int,int]]:
    """Given a clause and a list of subsets, find all pairs of subsets such that
    their symmetric difference equals the clause."""
    matchings = []
    clause_set = set(clause)
    for i in range(len(subsets)):
        for j in range(i+1, len(subsets)):
            subset_i = set(subsets[i])
            subset_j = set(subsets[j])
            sym_diff = subset_i.symmetric_difference(subset_j)
            if sym_diff == clause_set:
                matchings.append((i, j, sign))
    return matchings
    
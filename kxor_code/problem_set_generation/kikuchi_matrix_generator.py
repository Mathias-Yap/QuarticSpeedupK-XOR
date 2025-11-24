from itertools import combinations

from .kxor_instance import KXORInstance
from .planted_noisy_kxor_generator import PlantedNoisyKXORGenerator
import numpy as np
from scipy.sparse import csr_matrix, dok_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
import pandas as pd

class ClassicalKikuchiSolver:
    """
    Class for generating and analyzing the Kikuchi matrix of K-XOR problem instances.
    """
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


def compute_kikuchi_matrix(problem_instance: KXORInstance, ell: int) -> dok_matrix:
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

    return incidence


def find_eigenvalues_and_vectors(kikuchi_matrix: dok_matrix, num_eigenvalues: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """Compute the eigenvalues and eigenvectors of the Kikuchi matrix. Utilizes sparse matrix methods for efficiency.
    ARPACK wrapper: https://github.com/opencollab/arpack-ng

    Parameters:
        kikuchi_matrix (dok_matrix): The Kikuchi matrix.
        num_eigenvalues (int): The number of eigenvalues and eigenvectors to compute. If -1, compute all.
    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the eigenvalues and eigenvectors.
    """
    if not kikuchi_matrix.shape:
        raise ValueError("Kikuchi matrix is empty.")
    if num_eigenvalues == -1:
        num_eigenvalues = kikuchi_matrix.shape[0]
    eigenvalues, eigenvectors = eigsh(kikuchi_matrix, k=num_eigenvalues)
    return eigenvalues, eigenvectors


def find_matchings(clause: tuple, subsets: list, sign: int) -> list[tuple[int,int,int]]:
    """Given a clause and a list of subsets, find all pairs of subsets such that
    their symmetric difference equals the clause."""
    matchings = []
    clause_set = set(clause)
    for i in range(len(subsets)):
        subset_i = set(subsets[i])
        subset_j = subset_i.symmetric_difference(clause)
        if len(subset_j)== len(subsets[0]):
            matchings.append((subset_i, subset_j, sign))
    return matchings

def average_degree(kikuchi_matrix: dok_matrix) -> float:
    """Compute the average degree of the Kikuchi matrix.
    Parameters:
        kikuchi_matrix (dok_matrix): The Kikuchi matrix.
    Returns:
        float: The average degree.
    """
    degrees = kikuchi_matrix.getnnz(axis=1)
    return np.mean(degrees)

def power_iteration(kikuchi_matrix: dok_matrix, num_iterations: int = 1000):
    """Compute the largest eigenvalue and corresponding eigenvector of the Kikuchi matrix using the power iteration method.
    Parameters:
        kikuchi_matrix (dok_matrix): The Kikuchi matrix.
        num_iterations (int): The number of iterations to perform.
    Returns:
        tuple[float, np.ndarray]: A tuple containing the largest eigenvalue and corresponding eigenvector.
    """
    if not kikuchi_matrix.shape:
        raise ValueError("Kikuchi matrix is empty.")
    n = kikuchi_matrix.shape[0]
    b_k = np.random.rand(n)
    
    for _ in range(num_iterations):
        b_k1 = kikuchi_matrix.dot(b_k)
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm
    
    return b_k

def constraint_density(problem_instance: KXORInstance) -> float:
    """Compute the constraint density of a K-XOR problem instance.
    Parameters:
        problem_instance (KXORInstance): The K-XOR problem instance.
    Returns:
        float: The constraint density (m/n).
    """
    return problem_instance.m / problem_instance.n

        
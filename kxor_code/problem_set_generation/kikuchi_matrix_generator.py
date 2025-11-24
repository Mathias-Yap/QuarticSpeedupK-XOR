from itertools import combinations
from math import ceil, comb, log
import math
from pathlib import Path

from .kxor_instance import KXORInstance
from .planted_noisy_kxor_generator import PlantedNoisyKXORGenerator
import numpy as np
from scipy.sparse import csr_matrix, dok_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
import pandas as pd

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

def _find_kappa(error: float) -> float:
    """Helper function to compute the k constant used in Alice's theorem. It is a value satisfying the equation:

        k<=1 AND error<k/(2+k)

    Parameters:
        error (float): The error parameter."""
    if error <= 0:
        raise ValueError("Error must be positive non-zero.")
    k = (2 * error) / (1 - error)
    if k > 1.0:
        return 1.0
    return k

def get_kikuchi_parameters(problem_instance: KXORInstance, error: float):
    """Compute the parameters needed for the Kikuchi matrix based on Alice's theorem.
    Parameters:
        problem_instance (KXORInstance): The K-XOR problem instance.
        error (float): The error parameter.
    Returns:
        tuple[int, float]: A tuple containing ell and the eigenvalue threshold.
    """
    

def check_alice_theorem(ell: int, n: int, k: int, kappa: float, eps: float, m: int, delta_lnk=None, n_factor=None):
    """
    Check the conditions of Theorem 23 for given parameters.
    
    Parameters
    ----------
    l : int
        ell in the theorem (ell).
    n : int
        Number of variables.
    k : int
        Degree / arity, must be even.
    kappa : float
        κ in (0, 1]; the theorem assumes κ ≤ 1.
    eps : float
        ε with 0 < ε ≤ κ/(2+κ).
    m : int or float
        Number of constraints.
    delta_lnk : float, optional
        δ_{ell,n,k}. Only used to report d = δ_{ell,n,k} * m.
    n_factor : float, optional
        If set, we enforce the heuristic “n ≫ k*ell” as n ≥ n_factor * k*ell.
        If None, we just report the value of n / (k*ell).
        
    Returns
    -------
    result : dict
        {
            "basic_constraints_ok": bool,
            "n_asymptotic_ok": bool or None,
            "Delta": float,
            "C_kappa": float,
            "Delta_condition_ok": bool,
            "failure_probability_bound": float,
            "d": float or None,
            "all_ok": bool
        }
    """
    # --- basic structural constraints ---
    basic = True
    
    if ell <= 0 or n <= 0 or k <= 0:
        basic = False
    if k % 2 != 0:
        basic = False  # k must be even
    if ell < k / 2:
        basic = False  # ell ≥ k/2
    if not (0 < kappa <= 1):
        basic = False  # κ ≤ 1 and positive
    if not (0 < eps <= kappa / (2 + kappa)):
        basic = False  # 0 < ε ≤ κ/(2+κ)
    
    # --- n ≫ k*ell (heuristic check if requested) ---
    if n_factor is None:
        n_asymptotic_ok = None  # we just report the ratio
    else:
        n_asymptotic_ok = n >= n_factor * k * ell
    
    # --- main inequality for Δ ---
    Delta = m / n
    
    C_kappa = (
        2 * (1 + eps) * (1 + kappa) / (kappa ** 2)
        * (1 / comb(k, k // 2))
        * math.log(n)      # natural log
    )
    
    delta_threshold = C_kappa * (n / ell) ** ((k - 2) / 2)
    delta_condition_ok = Delta >= delta_threshold
    
    # --- probability bound in the theorem ---
    failure_prob_bound = 3 * (n ** (-eps * ell))
    
    # --- d = δ_{ell,n,k} * m (if given) ---
    d = delta_lnk * m if delta_lnk is not None else None
    
    all_ok = basic and (delta_condition_ok) and (n_asymptotic_ok in (True, None))
    
    return {
        "basic_constraints_ok": basic,
        "n_asymptotic_ok": n_asymptotic_ok,
        "n_over_k_l": n / (k * ell),
        "Delta": Delta,
        "Delta_threshold": delta_threshold,
        "C_kappa": C_kappa,
        "Delta_condition_ok": delta_condition_ok,
        "failure_probability_bound": failure_prob_bound,
        "d": d,
        "all_ok": all_ok,
    }

def search_kappa_ell(m: int, n: int, k: int, eps: float, n_factor=None):
    """
    Try to find κ and ell that satisfy Theorem 23 for given m, n, k, ε.

    Parameters
    ----------
    m : int or float
        Number of constraints.
    n : int
        Number of variables (appears in Δ and ln n).
    k : int
        Arity / degree. Must be even.
    eps : float
        ε. Theorem needs 0 < ε ≤ κ/(2+κ) with κ ∈ (0,1].
    n_factor : float or None, optional
        If given, we enforce the heuristic “n ≫ k*ell” as n ≥ n_factor * k * ell.
        If None, we don't enforce that asymptotic condition.
    ell_max : int, optional
        Maximum ell to consider. If -1, no upper limit (other than n).

    Returns
    -------
    result : dict
        If success:
            {
              "success": True,
              "kappa": κ,
              "ell": ell,
              "Delta": Δ,
              "Delta_threshold": RHS,
              "failure_probability_bound": 3 * n**(-eps * ell)
            }
        If impossible:
            { "success": False, "reason": "...text..." }
    """

    # Basic sanity checks
    if m <= 0 or n <= 0 or k <= 0:
        return {"success": False, "reason": "m, n, k must be positive."}
    if k % 2 != 0:
        return {"success": False, "reason": "k must be even."}

    # Check ε range for existence of any κ in (0,1]
    if not (0 < eps <= 1/3):
        return {
            "success": False,
            "reason": "No κ ∈ (0,1] can satisfy ε ≤ κ/(2+κ) when ε ∉ (0, 1/3]."
        }

    # Best choice: κ = 1 (gives smallest C_κ)
    kappa = 1.0

    # Check that ε actually satisfies the κ-condition (it should if ε ≤ 1/3)
    if eps > kappa / (2 + kappa):
        return {
            "success": False,
            "reason": "Given ε does not satisfy ε ≤ κ/(2+κ) even with κ = 1."
        }

    Delta = m / n
    exponent = (k - 2) / 2.0

    # Constant C_κ from the theorem:
    # C_κ = 2(1+ε)(1+κ)/κ² * (k choose k/2)^(-1) * ln n
    binom_mid = comb(k, k // 2)
    C_kappa = 2 * (1 + eps) * (1 + kappa) / (kappa ** 2) * (1 / binom_mid) * log(n)

    # ell must be ≥ k/2 and (optionally) small enough so that n ≥ n_factor * k*ell.
    ell_min = ceil(k / 2)
    if n_factor is None:
        ell_max = n  
    else:
        ell_max = int(n / (n_factor * k))
        if ell_max < ell_min:
            return {
                "success": False,
                "reason": "No ell satisfies ell ≥ k/2 and n ≥ n_factor * k*ell."
            }

    # If k = 2, exponent = 0, so the inequality doesn’t depend on ell.
    if exponent == 0:
        rhs = C_kappa  # (n/ell)^0 = 1
        if Delta >= rhs:
            ell = ell_min  # any ell in [ell_min, ell_max] works; pick the smallest
            failure_prob = 3 * (n ** (-eps * ell))
            return {
                "success": True,
                "kappa": kappa,
                "ell": ell,
                "Delta": Delta,
                "Delta_threshold": rhs,
                "failure_probability_bound": failure_prob,
            }
        else:
            return {
                "success": False,
                "reason": "Δ < C_κ, so the density condition fails for all ell when k=2."
            }

    # For k > 2, RHS decreases as ell increases, so we can simply search ell upward.
    best_ell = None
    best_rhs = None

    for ell in range(ell_min, ell_max + 1):
        rhs = C_kappa * (n / ell) ** exponent
        if Delta >= rhs:
            best_ell = ell
            best_rhs = rhs
            break

    if best_ell is None:
        return {
            "success": False,
            "reason": "No ell in [ell_min, ell_max] satisfies the density inequality."
        }

    failure_prob = 3 * (n ** (-eps * best_ell))
    return {
        "success": True,
        "kappa": kappa,
        "ell": best_ell,
        "Delta": Delta,
        "Delta_threshold": best_rhs,
        "failure_probability_bound": failure_prob,
    }



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

        
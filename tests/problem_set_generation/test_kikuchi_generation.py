import numpy as np
from kxor_code.problem_set_generation.kikuchi_matrix_generator import two_xor_matrix, compute_kikuchi_matrix
from kxor_code.problem_set_generation.kxor_instance import KXORInstance
from kxor_code.problem_set_generation.planted_noisy_kxor_generator import PlantedNoisyKXORGenerator
from math import comb
import pytest


def test_compute_kikuchi_matrix():
    rng = np.random.default_rng(42)
    test_instance_generator = PlantedNoisyKXORGenerator(n=10,k=3,rng = rng)
    test_instance = test_instance_generator.sample_random(m=15)
    kikuchimatrix = compute_kikuchi_matrix(test_instance, ell=2)

    # assert its shape is correct
    assert kikuchimatrix.shape is not None 
    assert kikuchimatrix.shape[0] == kikuchimatrix.shape[1]
    assert(kikuchimatrix.shape[0] == comb(test_instance.n, 2))

    # assert it's symmetric
    assert (kikuchimatrix != kikuchimatrix.T).nnz == 0  # Kikuchi matrix must be symmetric



def test_two_xor_matrix():
    rng = np.random.default_rng(42)
    test_instance_generator = PlantedNoisyKXORGenerator(n=10,k=3,rng = rng)
    test_instance = test_instance_generator.sample_random(m=15)
    adj_matrix = two_xor_matrix(test_instance)
    assert adj_matrix.shape == (test_instance.m, test_instance.n)
    
def test_adjacency_matrix_contents():
    rng = np.random.default_rng(42)
    test_instance_generator = PlantedNoisyKXORGenerator(n=10,k=3,rng = rng)
    test_instance = test_instance_generator.sample_random(m=15)
    adj_matrix = two_xor_matrix(test_instance)
    for i, scope in enumerate(test_instance.scopes):
        for j in range(test_instance.n):
            if j in scope:
                assert adj_matrix[i, j] == test_instance.b[i]
            else:
                assert adj_matrix[i, j] == 0
                
def test_adjacency_matrix_symmetry():
    rng = np.random.default_rng(42)
    test_instance_generator = PlantedNoisyKXORGenerator(n=10,k=3,rng = rng)
    test_instance = test_instance_generator.sample_random(m=15)
    adj_matrix = two_xor_matrix(test_instance)
    assert (adj_matrix != adj_matrix.T).nnz == 0  # Adjacency matrix must be symmetric
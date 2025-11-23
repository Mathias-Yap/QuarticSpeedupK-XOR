import numpy as np
import kxor_code
from kxor_code.problem_set_generation.kikuchi_matrix_generator import compute_kikuchi_matrix, two_xor_matrix
from kxor_code.problem_set_generation.k_xor_instance import KXORInstance, PlantedNoisyKXORGenerator
import pytest

test_instance_generator = PlantedNoisyKXORGenerator(n=3,k=2,seed = 42)
test_instance = test_instance_generator.sample_random(m=3)
test_instance.scopes = np.ndarray(shape=(3,), dtype=object)
test_instance.scopes[0] = [1, 0]
test_instance.scopes[1] = [1, 2]
test_instance.scopes[2] = [0, 2]
test_instance.b = np.array([1, 0, 1], dtype=int)
adj_matrix = two_xor_matrix(test_instance)
adj_matrix.todense()

def test_compute_kikuchi_matrix():
    pass

def test_two_xor_matrix():
    adj_matrix = two_xor_matrix(test_instance)
    assert adj_matrix.shape == (test_instance.m, test_instance.n)
    
def test_adjacency_matrix_contents():
    adj_matrix = two_xor_matrix(test_instance)
    for i, scope in enumerate(test_instance.scopes):
        for j in range(test_instance.n):
            if j in scope:
                assert adj_matrix[i, j] == test_instance.b[i]
            else:
                assert adj_matrix[i, j] == 0
                
def test_adjacency_matrix_symmetry():
    adj_matrix = two_xor_matrix(test_instance)
    assert (adj_matrix != adj_matrix.T).nnz == 0  # Adjacency matrix must be symmetric
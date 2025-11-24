
import pytest
import numpy as np
import tempfile
from kxor_code.problem_set_generation.kxor_instance import KXORInstance
from pathlib import Path


def test_create_kxor_instance():
    scopes = np.array([[0, 1], [1, 2], [0, 2]])
    b = np.array([1, -1, 1], dtype=np.int8)
    instance = KXORInstance.create(scopes=scopes, b=b, is_planted=True, rho=0.5, z=np.array([1, -1, 1], dtype=np.int8))

    assert instance.n == 3
    assert instance.k == 2
    assert instance.m == 3
    assert np.array_equal(instance.scopes, scopes)
    assert np.array_equal(instance.b, b)
    assert instance.is_planted is True
    assert instance.rho == 0.5
    assert instance.z is not None
    assert np.array_equal(instance.z, np.array([1, -1, 1], dtype=np.int8))

def test_save_and_load_kxor_instance():
    # Create a sample KXORInstance
    scopes = np.array([[0, 1], [1, 2], [0, 2]])
    b = np.array([1, 0, 1], dtype=np.int8)
    instance = KXORInstance.create(scopes=scopes, b=b, is_planted=True, rho=0.5, z=np.array([1, -1, 1], dtype=np.int8))

    # Save the instance to a temporary file
    instance.save("tests/problem_set_generation/temp_kxor_instance.npz")

    # Load the instance back
    loaded_instance = KXORInstance.load("tests/problem_set_generation/temp_kxor_instance.npz")

    Path("tests/problem_set_generation/temp_kxor_instance.npz").unlink()
    # Verify that the loaded instance matches the original
    assert loaded_instance.n == instance.n
    assert loaded_instance.k == instance.k
    assert loaded_instance.m == instance.m
    assert np.array_equal(loaded_instance.scopes, instance.scopes)
    assert np.array_equal(loaded_instance.b, instance.b)
    assert loaded_instance.is_planted == instance.is_planted
    assert loaded_instance.rho == instance.rho
    assert (loaded_instance.z is None) == (instance.z is None)
    if loaded_instance.z is not None and instance.z is not None:
        assert np.array_equal(loaded_instance.z, instance.z)
    
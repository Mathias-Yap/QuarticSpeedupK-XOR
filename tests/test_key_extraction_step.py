import numpy as np
import pytest

from kxor_code.algorithms.base_alg_step import ProblemRecord
from kxor_code.algorithms.compute_kikuchi_step import ComputeKikuchiStep
from kxor_code.algorithms.key_extraction_step import KeyExtractionStep
from kxor_code.problem_set_generation.kxor_instance import KXORInstance


def test_key_extraction_step_synthetic_state():
    # Choose instance params so KeyExtractionStep computes r=2.
    # r = ceil(log2(max(2, n*m))) with n=3, m=1 -> ceil(log2(3)) = 2
    instance = KXORInstance(
        n=3,
        k=2,
        m=1,
        scopes=np.array([[0, 1]]),
        b=np.array([1]),
        is_planted=False,
        rho=None,
        z=None,
    )

    problem = ProblemRecord(problem_id="p", instance=instance)
    problem.add_field("ell", 1)
    problem.add_field("threshold", 0.5)

    # Build a synthetic (phase, system) state with:
    # - r=2 phase qubits => 4 phases
    # - system_qubits=2 => dim 4
    # The step will compute thr_idx=floor(0.5/(2π)*4)=0, so good_phases=[1,2,3].
    r = 2
    system_qubits = 2
    tensor = np.zeros((1 << r, 1 << system_qubits), dtype=complex)
    good_phases = [1, 2, 3]

    # Make the extracted system vector proportional to [1, 2, 0, 0]
    # so the voting matrix is non-zero and the top index is deterministic.
    # Total norm^2 = 3 * (1^2 + 2^2) * scale^2 = 15 * scale^2 => scale = 1/sqrt(15).
    scale = 1 / np.sqrt(15)
    for p in good_phases:
        tensor[p, 0] = 1 * scale
        tensor[p, 1] = 2 * scale
    state = tensor.reshape(-1)

    def quartic_quantum_circuit():
        return state

    problem.add_field("quartic_quantum_circuit", quartic_quantum_circuit)

    step = KeyExtractionStep()
    step.execute(problem)

    assert problem.get_field("good_phases") == good_phases
    V = problem.get_field("voting_matrix")
    assert V.shape == (3, 3)
    assert np.allclose(V, V.conj().T)

    x_hat = int(problem.get_field("x_hat"))
    assert 0 <= x_hat < 3


def test_key_extraction_stage2_classical_eigsh_backend():
    pytest.importorskip("scipy")
    # Same setup as the synthetic-state test, but force stage-2 to use eigsh.
    instance = KXORInstance(
        n=3,
        k=2,
        m=1,
        scopes=np.array([[0, 1]]),
        b=np.array([1]),
        is_planted=False,
        rho=None,
        z=None,
    )

    problem = ProblemRecord(problem_id="p", instance=instance)
    problem.add_field("ell", 1)
    problem.add_field("threshold", 0.5)

    r = 2
    system_qubits = 2
    tensor = np.zeros((1 << r, 1 << system_qubits), dtype=complex)
    good_phases = [1, 2, 3]
    scale = 1 / np.sqrt(15)
    for p in good_phases:
        tensor[p, 0] = 1 * scale
        tensor[p, 1] = 2 * scale
    state = tensor.reshape(-1)

    def quartic_quantum_circuit():
        return state

    problem.add_field("quartic_quantum_circuit", quartic_quantum_circuit)

    step = KeyExtractionStep(stage2_backend="classical_eigsh", stage2_num_eigenvalues=2)
    step.raise_on_error = True
    stats = step.execute(problem)
    assert stats.failed is False

    V = problem.get_field("voting_matrix")
    assert V.shape == (3, 3)
    assert np.allclose(V, V.conj().T)
    x_hat = int(problem.get_field("x_hat"))
    assert 0 <= x_hat < 3


def test_key_extraction_stage2_schmidhuber_circuit_backend_smoke():
    pytest.importorskip("pennylane")

    instance = KXORInstance(
        n=3,
        k=2,
        m=1,
        scopes=np.array([[0, 1]]),
        b=np.array([1]),
        is_planted=False,
        rho=None,
        z=None,
    )

    problem = ProblemRecord(problem_id="p", instance=instance)
    problem.add_field("ell", 1)
    problem.add_field("threshold", 0.5)

    r = 2
    system_qubits = 2
    tensor = np.zeros((1 << r, 1 << system_qubits), dtype=complex)
    good_phases = [1, 2, 3]
    scale = 1 / np.sqrt(15)
    for p in good_phases:
        tensor[p, 0] = 1 * scale
        tensor[p, 1] = 2 * scale
    state = tensor.reshape(-1)

    def quartic_quantum_circuit():
        return state

    problem.add_field("quartic_quantum_circuit", quartic_quantum_circuit)

    step = KeyExtractionStep(
        stage2_backend="schmidhuber_stage2_circuit",
        stage2_circuit_phase_qubits=2,
        stage2_circuit_iters=1,
        stage2_circuit_neighborhood=1,
    )
    step.raise_on_error = True
    stats = step.execute(problem)
    assert stats.failed is False
    x_hat = int(problem.get_field("x_hat"))
    assert 0 <= x_hat < 3


def test_key_extraction_metrics_advantage_and_hamming():
    instance = KXORInstance(
        n=4,
        k=2,
        m=2,
        scopes=np.array([[0, 1], [2, 3]]),
        b=np.array([1, -1]),
        is_planted=True,
        rho=0.5,
        z=np.array([1, 1, 1, -1]),
    )

    # Perfect assignment for this instance satisfies both constraints => advantage 1.
    x = np.array([1, 1, 1, -1])
    adv = KeyExtractionStep._kxor_advantage(instance, x)
    assert np.isclose(adv, 1.0)

    # All-ones satisfies first but violates second => advantage 0.
    adv0 = KeyExtractionStep._kxor_advantage(instance, np.array([1, 1, 1, 1]))
    assert np.isclose(adv0, 0.0)

    # Hamming distance on {±1} vectors.
    ham = KeyExtractionStep._hamming_distance_pm1(np.array([1, 1, 1, -1]), np.array([1, -1, 1, -1]))
    assert ham == 1

    # Global sign resolution chooses the closer of z_hat and -z_hat.
    z_true = np.array([1, 1, 1, -1])
    z_hat = -z_true
    z_best, flipped = KeyExtractionStep._best_global_sign_match(z_hat, z_true)
    assert flipped is True
    assert np.array_equal(z_best, z_true)


def test_key_extraction_stage1_backend_quartic_step_adapter_smoke():
    pytest.importorskip("pennylane")
    pytest.importorskip("scipy")

    instance = KXORInstance(
        n=3,
        k=2,
        m=1,
        scopes=np.array([[0, 1]]),
        b=np.array([1]),
        is_planted=False,
        rho=None,
        z=None,
    )

    problem = ProblemRecord(problem_id="p", instance=instance)
    problem.add_field("ell", 1)
    problem.add_field("threshold", 0.0)

    # Provide kikuchi_matrix for the adapter backend.
    ComputeKikuchiStep().execute(problem)

    step = KeyExtractionStep(
        stage1_backend="quartic_step_adapter",
        stage2_backend="classical_eigsh",
        stage2_num_eigenvalues=2,
        random_seed=0,
    )
    step.raise_on_error = True
    stats = step.execute(problem)
    assert stats.failed is False

    x_hat = int(problem.get_field("x_hat"))
    assert 0 <= x_hat < 3

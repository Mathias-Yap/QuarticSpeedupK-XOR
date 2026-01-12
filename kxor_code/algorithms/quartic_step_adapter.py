import numpy as np
import pennylane as qml


def pad_and_normalize_state(state: np.ndarray, target_dim: int) -> np.ndarray:
    """Pad with zeros up to target_dim and normalize."""
    state = np.asarray(state, dtype=complex).reshape(-1)
    if state.size > target_dim:
        raise ValueError(f"State dimension {state.size} exceeds target_dim {target_dim}.")
    if state.size < target_dim:
        padded = np.zeros((target_dim,), dtype=complex)
        padded[: state.size] = state
        state = padded
    nrm = np.linalg.norm(state)
    if nrm == 0:
        raise ValueError("Guiding state has zero norm.")
    return state / nrm


def guiding_state_from_quartic_step(*, clauses, n: int, ell: int, k: int, target_dim: int) -> np.ndarray:
    """Get the guiding state from the pipeline step and embed it into target_dim.

    `clauses` is a list of (scope, rhs) pairs. The result is a normalized vector of
    length `target_dim`.
    """
    # Import here so the toy code still runs without dragging in the whole pipeline.
    from kxor_code.algorithms.quartic_quantum_algorithm_step import QuarticQuantumAlgorithm

    guiding_state = QuarticQuantumAlgorithm.prepare_guiding_state(clauses, n, ell, k)
    return pad_and_normalize_state(guiding_state, target_dim)


# Backwards-compat name: keep the old symbol without adding another call layer.
build_guiding_state_from_quartic_step = guiding_state_from_quartic_step


def extract_system_vector_from_state_no_ancillas(
    state: np.ndarray,
    *,
    phase_qubits: int,
    system_qubits: int,
    good_phases,
    normalize: bool = True,
) -> np.ndarray:
    """Extract a system-register vector by summing amplitudes over selected phase outcomes.

    This is for circuits laid out as (phase register, then system register) with no extra
    ancillas (i.e. the quartic step circuit).
    """
    state = np.asarray(state, dtype=complex).reshape(-1)
    r = int(phase_qubits)
    n = int(system_qubits)
    expected = 1 << (r + n)
    if state.size != expected:
        raise ValueError(f"State length {state.size} does not match 2**(r+n)={expected}.")

    tensor = state.reshape((1 << r, 1 << n))

    good = [int(p) for p in good_phases]
    if any((p < 0 or p >= (1 << r)) for p in good):
        raise ValueError("good_phases contains an out-of-range phase index.")

    sys_vec = np.sum(tensor[good, :], axis=0)
    if normalize:
        nrm = np.linalg.norm(sys_vec)
        if nrm > 0:
            sys_vec = sys_vec / nrm
    return sys_vec


def as_pauli_hamiltonian(H):
    """Convert a dense matrix H into a Pauli Hamiltonian for ApproxTimeEvolution."""
    if getattr(H, "pauli_rep", None) is not None:
        return H

    H = np.asarray(H, dtype=complex)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("H must be a square matrix or a PennyLane Hamiltonian/Pauli representation.")

    try:
        return qml.pauli_decompose(H)
    except Exception as e:
        raise ValueError(
            "Failed to convert dense matrix H into a Pauli-representation required by ApproxTimeEvolution. "
            "Provide H as `qml.Hamiltonian` / PauliSentence, or ensure `qml.pauli_decompose(H)` works. "
            f"Original error: {e!r}"
        )


def shift_hamiltonian_wires(H, *, offset: int):
    """Shift the wires inside a Hamiltonian by `offset`.

    The quartic step calls ApproxTimeEvolution(H, ...) without passing wires, so H needs to
    already be defined on the target wires (which start after the phase register).
    """
    if int(offset) == 0:
        return H

    try:
        wires = list(getattr(H, "wires", []))
        wire_map = {w: w + int(offset) for w in wires}
        return qml.map_wires(H, wire_map)
    except Exception as e:
        raise ValueError(
            "Failed to shift Hamiltonian wires for quartic-step adapter. "
            "Provide H with wires that already match the step's target_wires, or ensure qml.map_wires works. "
            f"Original error: {e!r}"
        )


def stage1_system_vector_from_quartic_step(
    *,
    clauses,
    n: int,
    ell: int,
    k: int,
    H,
    threshold: float,
    evolution_time: float = 1.0,
    trotter_steps: int = 1,
    phase_qubits: int | None = None,
    grover_iterations: int | None = None,
):
    """Run stage-1 via QuarticQuantumAlgorithm and return a usable system vector.

    Returns (v_top_sys, good_phases, qnode) where v_top_sys is the system-register vector
    you can feed into the voting matrix.
    """
    from kxor_code.algorithms.quartic_quantum_algorithm_step import QuarticQuantumAlgorithm

    guiding_raw = QuarticQuantumAlgorithm.prepare_guiding_state(clauses, n, ell, k)
    guiding_raw = np.asarray(guiding_raw, dtype=complex).reshape(-1)
    if guiding_raw.size == 0:
        raise ValueError("QuarticQuantumAlgorithm.prepare_guiding_state returned an empty vector.")

    target_dim = 1 << int(np.ceil(np.log2(guiding_raw.size)))
    system_qubits = int(np.log2(target_dim))
    guiding = pad_and_normalize_state(guiding_raw, target_dim)

    if phase_qubits is None:
        m = len(clauses)
        phase_qubits = int(np.ceil(np.log2(max(2, int(n) * max(1, int(m))))))

    if grover_iterations is None:
        grover_iterations = int(np.ceil(np.log2(max(2, int(n)))))

    H_pauli = as_pauli_hamiltonian(H)
    H_pauli = shift_hamiltonian_wires(H_pauli, offset=int(phase_qubits))

    qnode = QuarticQuantumAlgorithm.phase_estimation_with_amplitude_amplification(
        guiding,
        H_pauli,
        float(evolution_time),
        int(trotter_steps),
        int(phase_qubits),
        int(system_qubits),
        int(grover_iterations),
        float(threshold),
    )

    state = qnode()

    thr_idx = int(np.floor(float(threshold) * float(evolution_time) / (2 * np.pi) * (1 << int(phase_qubits))))
    good_phases = range(thr_idx + 1, 1 << int(phase_qubits))

    v_top_sys = extract_system_vector_from_state_no_ancillas(
        state,
        phase_qubits=int(phase_qubits),
        system_qubits=int(system_qubits),
        good_phases=good_phases,
        normalize=True,
    )

    return v_top_sys, list(good_phases), qnode


# Backwards-compat name: keep the old symbol without adding another call layer.
stage1_v_top_from_quartic_step = stage1_system_vector_from_quartic_step

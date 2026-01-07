import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

from kxor_code.algorithms.base_alg_step import BaseAlgorithmStep

class QuarticQuantumAlgorithm(BaseAlgorithmStep):
    requires_fields = ["ell", "kikuchi_matrix", "threshold"]
    produces_fields = ["quartic_quantum_circuit"]
    """
    Creates an equal superposition over the values that satisfies a clause. 
    """
    def prepare_xor_quantum_state(clause_qbits, value, n):
    """
    clause_qbits: list of qubit indices participating in XOR
    value:        0 or 1
    n:            total number of variables
    """
    pivot = clause_qbits[0]              # choose one as pivot
    others = clause_qbits[1:]            # other clause variables

    dev = qml.device("default.qubit", wires = n)

    @qml.qnode(dev)
    def circuit():
        # Excluding pivot, equal superposition of all states
        for q in range(n):
            if q != pivot:
                qml.Hadamard(wires = q)

        # Apply X on pivot if RHS = 1
        if value == 1:
            qml.PauliX(wires = pivot)

        # Compute pivot based on clause
        for q in others:
            qml.CNOT(wires=[q, pivot])

        return qml.state()

    return circuit

    """
    Constructs |phi>, as an equal superposition of the clauses
    """
    def prepare_phi(clauses, n, m, a, t):
    """
    clauses: list of tuples ([qubit_indices], target_value)
    n:       number of variables
    """
    dev = qml.device("default.qubit", wires=t)

    @qml.qnode(dev)
    def circuit():
        # Prepare Each Clause
        for i, (qubits, rhs) in enumerate(clauses):
            pivot = qubits[0]              # choose first as pivot
            other = qubits[1:]             # rest clause variables

            # Excluding pivot, equal superposition of all states in clause
            for q in range(n):
                wire = (i * n) + q
                if q != pivot:
                    qml.Hadamard(wires = wire)
                elif rhs == 1:
                    qml.PauliX(wires = wire)
            # Compute pivot based on clause
            for q in other:
                wire_q = (i * n) + q
                wire_p = (i * n) + pivot
                qml.CNOT(wires=[wire_q, wire_p])

        # Prepare ancilla in equal superposition
        a_start = (m + 1) * n # ancilla qubits start at this position
        for q in range(a):
            qml.Hadamard(wires=a_start + q)

        # Controlled swap each clause register into target register
        for i in range(m):
            # Binary string for ancilla control
            bin_str = format(i, f"0{a}b")
            ancilla_qubits = [a_start + j for j in range(a)]
            flip_qubits = [q for q, b in zip(ancilla_qubits, bin_str) if b == '0']
            # Flip zeros
            for q in flip_qubits:
                qml.PauliX(wires=q)

            for j in range(n):
                # Controlled swap on all ancilla
                qml.ctrl(lambda: qml.SWAP(wires=[(i * n) + j, (m * n) + j]), control=ancilla_qubits)()

            # Controlled-x on success ancilla
            qml.MultiControlledX(
                control_wires=list(range(a_start, a_start + a)),
                wires=a_start + a
            )

            # Undo flips
            for q in flip_qubits:
                qml.PauliX(wires=q)

        return qml.state(), qml.measure(a_start + a, postselect=1)


    return circuit

    def extract_and_repeat_qubits(quantum_state, total_wires, qubits_to_extract, copies):
    """
    Extract a subset of qubits from a full statevector and repeat them multiple times.

    Args:
        full_state (np.ndarray): 1D array of size 2^total_wires, the full quantum state
        total_wires (int): total number of qubits in full_state
        qubits_to_extract (list[int]): indices of qubits to extract
        copies (int): number of times to repeat the extracted qubits

    Returns:
        new_state (np.ndarray): 1D array of size 2^(len(qubits_to_extract)*copies)
    """

    # --- Step 1: Compute reduced density matrix for the qubits of interest ---
    # Compute full density matrix
    rho_full = np.outer(full_state, np.conj(full_state))

    # Qubits to trace out
    qubits_to_trace = [i for i in range(total_wires) if i not in qubits_to_extract]

    def partial_trace(rho, total_wires, qubits_to_trace):
        """Trace out unwanted qubits."""
        for q in sorted(qubits_to_trace, reverse=True):
            rho = np.trace(rho.reshape([2]*2*total_wires), axis1=q, axis2=q+total_wires)
            total_wires -= 1
        return rho

    rho_reduced = partial_trace(rho_full, total_wires, qubits_to_trace)

    # --- Step 2: Extract a representative pure state ---
    # Take the eigenvector with largest eigenvalue
    eigvals, eigvecs = np.linalg.eigh(rho_reduced)
    state_3 = eigvecs[:, np.argmax(eigvals)]

    # --- Step 3: Tensor product to create multiple copies ---
    new_state = state_3
    for _ in range(copies - 1):
        new_state = np.kron(new_state, state_3)

    return new_state

    """
    Constructs |phi> nl/k times, as an equal superposition of the clauses
    """
    def prepare_guiding_state(clauses, n, ell, k):
    """
    clauses: list of tuples ([qubit_indices], target_value)
    n:       number of variables
    """
    copies = n * ell / k

    m = len(clauses)                  # number of clauses
    a = int(np.ceil(np.log2(m)))      # ancilla qubits
    s = 1                             # ancilla qubit for success
    t = (m + 1) * n + a + s           # total qubits: m * n (n for each clause) + n (target) + a (ancilla) + s (success)

    phi = prepare_phi(clauses, n, m, a, t)
    guiding_state = extract_and_repeat_qubits(phi, t, list(range(m * n, (m + 1) * n)), ell / k)

    return guiding_state

    def hamiltonian_simulation(K, t=1.0, n_steps=1):
    """
    Hamiltonian simulation for a Hermitian matrix K using PennyLane.

    Args:
        K (np.ndarray): Hermitian matrix of size 2^n x 2^n
        t (float): evolution time
        n_steps (int): number of Trotter steps

    Returns:
        QNode: PennyLane QNode performing the simulation
    """
    # --- Validate Hermitian ---
    assert np.allclose(K, K.conj().T), "Matrix K must be Hermitian"

    # --- Determine number of qubits ---
    N = K.shape[0]
    n_qubits = int(np.log2(N))
    assert 2**n_qubits == N, "Matrix size must be a power of 2"

    wires = list(range(n_qubits))

    # --- Convert matrix K to PennyLane Hamiltonian ---
    pauli_terms, coeffs = qml.utils.decompose_hamiltonian(K)
    hamiltonian = qml.Hamiltonian(coeffs, pauli_terms)

    # --- Device ---
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit():
        # Start in |0...0>
        qml.BasisState(np.zeros(n_qubits, dtype=int), wires=wires)

        # Apply Hamiltonian evolution
        qml.templates.ApproxTimeEvolution(hamiltonian, time=t, n=n_steps)

        return qml.state()

    return circuit

def phase_estimation(unitary_circuit, n_phase_qubits, n_target_qubits, target_initial_state):
    """
    Perform QPE using a given unitary circuit and a guiding state.

    Args:
        unitary_circuit: QNode implementing U on target qubits
        n_phase_qubits: number of phase (ancilla) qubits
        n_target_qubits: number of target qubits
        target_initial_state: statevector of target qubits

    Returns:
        QNode measuring the phase qubits
    """
    total_wires = n_phase_qubits + n_target_qubits
    wires_phase = list(range(n_phase_qubits))
    wires_target = list(range(n_phase_qubits, total_wires))

    dev = qml.device("default.qubit", wires=total_wires)

    @qml.qnode(dev)
    def qpe_circuit():
        # Prepare phase register in |+> states
        for w in wires_phase:
            qml.Hadamard(wires=w)

        # Prepare target state
        qml.QubitStateVector(target_initial_state, wires=wires_target)

        # Apply controlled-U^{2^k} for each phase qubit
        for k, w in enumerate(wires_phase):
            power = 2 ** k
            # Controlled-U^{2^k} via repeated application
            for _ in range(power):
                qml.ctrl(unitary_circuit, control=w)(*wires_target)

        # Apply inverse QFT to phase qubits
        qml.templates.InverseQFT(wires=wires_phase)

        # Measure phase qubits
        return qml.probs(wires=wires_phase)

    return qpe_circuit

    def _run(self, problem: ProblemRecord, context: StepContext):
        n = problem.instance.n
        k = problem.instance.k
        scopes = problem.instance.scopes
        b = problem.instance.b
        clauses = [(list(scopes[scopes != -1]), int(val)) for scopes, val in zip(scopes, b)]
        ell = problem.fields["ell"]
        kikuchi_matrix = problem.fields["kikuchi_matrix"]
        threshold = problem.fields["threshold"]
        guiding_state = prepare_guiding_state(clauses, n, ell, k)
        hamiltonian = hamiltonian_simulation(kikuchi_matrix)
        n_phase_qubits = 20
        circuit = phase_estimation(hamiltonian, n_phase_qubits, n * ell / k, guiding_state)
        # TODO Amplitude Amplification
        return {
            "quartic_quantum_circuit": circuit
        }

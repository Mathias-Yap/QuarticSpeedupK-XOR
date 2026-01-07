from typing import Optional
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
import matplotlib.pyplot as plt

from kxor_code.algorithms.base_alg_step import BaseAlgorithmStep, ProblemRecord, StepContext, StepStats

class QuarticQuantumAlgorithm(BaseAlgorithmStep):
    requires_fields = ["ell", "kikuchi_matrix", "threshold"]
    produces_fields = ["quartic_quantum_circuit"]
    """
    Creates an equal superposition over the values that satisfies a clause. 
        clause_qbits: list of qubit indices participating in XOR
        value:        1 or 0/-1 (works both ways)
        n:            total number of variables
    """
    @staticmethod
    def prepare_xor_quantum_state(clause_qbits, value, n):
        pivot = clause_qbits[0]              # choose one as pivot
        others = clause_qbits[1:]            # other clause variables
        
        for q in range(n):
            if q != pivot:
                qml.Hadamard(wires = q)

        # Apply X on pivot if RHS = 1
        if value == 1:
            qml.PauliX(wires = pivot)

        # Compute pivot based on clause
        for q in others:
            qml.CNOT(wires=[q, pivot])

    """
    Constructs |phi>, as an equal superposition of the clauses
        wires:   the wires in the quantum circuit
        clauses: list of tuples ([qubit_indices], target_value)
        n:       number of variables
    """
    @staticmethod
    def prepare_phi(wires, clauses, n):
        m = len(clauses)                  # number of clauses
        a = int(np.ceil(np.log2(m)))      # ancilla qubits
        s = 1                             # ancilla qubit for success
        t = (m + 1) * n + a + s           # total qubits: m * n (n for each clause) + n (target) + a (ancilla) + s (success)
        assert len(wires) >= t, "Wire list too short"

        # Wires
        clause_wires = [wires[i*n:(i+1)*n] for i in range(m)]
        target_wires = wires[m*n:(m+1)*n]
        ancilla_wires = wires[(m+1)*n:(m+1)*n + a]
        success_wire = wires[-1]

        # Prepare Each Clause
        for i, (qubits, rhs) in enumerate(clauses):
            # Apply XOR clause using the fixed function
            # Map clause_qubits to local indices in clause_wires[i]
            local_mapping = {q: idx for idx, q in enumerate(qubits)}
            clause_indices = [local_mapping[q] for q in qubits]
            QuarticQuantumAlgorithm.prepare_xor_quantum_state(clause_wires[i], clause_indices, rhs)

        # Prepare ancilla in equal superposition
        for q in ancilla_wires:
            qml.Hadamard(wires=q)

        for i in range(m):
            # Binary string for ancilla control
            bin_str = format(i, f"0{a}b")
            flip_qubits = [q for q, b in zip(ancilla_wires, bin_str) if b == '0']
            # Flip zeros
            for q in flip_qubits:
                qml.PauliX(wires=q)
            # Controlled swap 
            for j in range(n):
                qml.ctrl(lambda: qml.SWAP(wires=[(i * n) + j, (m * n) + j]), control=ancilla_wires)()
            # Multi-controlled X on success qubit
            # qml.MultiControlledX(control_wires = ancilla_wires, wires=success_wire) #type: ignore
            qml.ctrl(lambda: qml.PauliX(wires=success_wire), control=ancilla_wires)()

            # Undo flips
            for q in flip_qubits:
                qml.PauliX(wires=q)

    """
    Extract a subset of qubits from a full statevector and repeat them multiple times.
        quantum_state:     1D array of size 2^total_wires
        total_wires:       total number of qubits in quantum_state
        qubits_to_extract: indices of qubits to extract
        copies:            number of times to repeat the extracted qubits
    """
    @staticmethod
    def extract_and_repeat_qubits(quantum_state, total_wires, qubits_to_extract, copies=1):
        # Reshape state and extract qubits ---
        state_reshaped = quantum_state.reshape([2]*total_wires)
        # Axes to keep
        keep_axes = qubits_to_extract
        # Axes to trace out
        trace_axes = [i for i in range(total_wires) if i not in qubits_to_extract]

        # Trace out unwanted axes
        rho = np.tensordot(state_reshaped, state_reshaped.conj(), axes=(trace_axes, trace_axes))
        # Take largest eigenvector
        eigvals, eigvecs = np.linalg.eigh(rho)
        extracted_state = eigvecs[:, np.argmax(eigvals)]

        # Repeat copies using tensor product ---
        final_state = extracted_state
        for _ in range(copies - 1):
            final_state = np.kron(final_state, extracted_state)

        # Normalize
        final_state = final_state / np.linalg.norm(final_state)

        return final_state

    """
    Constructs guiding state |phi> repeated nl/k times.
        clauses: list of tuples ([qubit_indices], target_value)
        n:       number of variables
        ell:     field "ell" from problem
        k:       problem parameter for repetition
    """
    @staticmethod
    def prepare_guiding_state(clauses, n, ell, k):
        copies = int(np.ceil(n * ell / k))  # number of times to repeat target register

        m = len(clauses)
        a = int(np.ceil(np.log2(m)))  # ancilla qubits
        s = 1                         # success qubit
        t = (m + 1) * n + a + s       # total qubits

        # Prepare wires
        wires = list(range(t))

        # Build the state using prepare_phi
        phi_qnode = qml.device("default.qubit", wires=t)

        @qml.qnode(phi_qnode)
        def phi_circuit():
            QuarticQuantumAlgorithm.prepare_phi(wires, clauses, n)
            return qml.state()

        phi_state = phi_circuit()

        # Extract target register and repeat
        target_wires = list(range(m * n, (m + 1) * n))
        guiding_state = QuarticQuantumAlgorithm.extract_and_repeat_qubits(phi_state, t, target_wires, copies)

        return guiding_state


    """
    Quartic Quantum Algorithm with Guiding State, Phase Estimation, and Amplitude Amplification 
        guiding_state:     array of guiding state
        H:                 Kikuchi Matrix (Hermitian)
        t:                 evolution time
        steps:             Trotter steps for Hamiltonian simulation
        r:                 number of phase qubits
        n:                 number of target qubits
        grover_iterations: number of amplitude amplification steps
    """
    @staticmethod
    def phase_estimation_with_amplitude_amplification(guiding_state, H, t, steps, r, n: int, grover_iterations, threshold):
        total_qubits = r + n
        phase_wires = list(range(r))
        target_wires = list(range(r, total_qubits))
        
        dev = qml.device("default.qubit", wires=total_qubits)
        
        @qml.qnode(dev)
        def circuit():
            # Prepare guiding state
            qml.templates.MottonenStatePreparation(guiding_state, wires=target_wires)
            # TODO: Cheat this, just copy the state over in qml.

            ### Phase Estimation
            # Apply Hadamard on phase register
            for q in phase_wires:
                qml.Hadamard(wires=q)
            # Controlled-Trotter evolutions for QPE
            for j, q in enumerate(phase_wires):
                qml.ctrl(
                    lambda: qml.templates.ApproxTimeEvolution(H, t * 2**j, steps),
                    control=q
                )()
            # Inverse QFT on phase register
            qml.adjoint(qml.templates.QFT)(wires=phase_wires)

            ### Amplitude Amplification
            # Grover iterations (includes QPE)
            for _ in range(grover_iterations):
                # Oracle marks "good" states (phase > threshold)
                # Note: threshold changes because we converted hermitian to unitary
                new_threshold = int(np.floor(threshold * t / (2 * np.pi) * 2**r))
                QuarticQuantumAlgorithm.phase_threshold_oracle(phase_wires, new_threshold)
                # Reflection about guiding state
                QuarticQuantumAlgorithm.reflection_around_guiding_state(guiding_state, target_wires)

            return qml.state()
        
        return circuit

    """
    Oracle that flips the sign of all phase register states >= threshold.
        phase_wires: phase register wires
        threshold:   threshold
    """
    @staticmethod
    def phase_threshold_oracle(phase_wires, threshold_index):
        r = len(phase_wires)
        # Loop over states > threshold
        for i in range(threshold_index + 1, 2**r):
            bin_str = format(i, f"0{r}b")
            for wire, bit in zip(phase_wires, bin_str):
                if bit == '0':
                    qml.PauliX(wires=wire)
            # Multi-controlled Z implemented via controlled PauliX on last qubit
            qml.ctrl(lambda: qml.PauliX(wires=phase_wires[:-1]), control=phase_wires[-1])()
            for wire, bit in zip(phase_wires, bin_str):
                if bit == '0':
                    qml.PauliX(wires=wire)

    """
    Implements the reflection about the guiding state:
        R = 2 |phi><phi| - I
    """
    @staticmethod
    def reflection_around_guiding_state(state, wires):
        dim = len(state)
        phi = state.reshape((dim,1))
        I = np.eye(dim, dtype=complex)
        # Reflection matrix
        R = 2 * (phi @ np.conj(phi.T)) - I
        qml.QubitUnitary(R, wires=wires)

    def _run(self, problem: ProblemRecord, context: StepContext, stats: StepStats) -> Optional[dict]:
        # n, k
        n = problem.instance.n
        k = problem.instance.k

        # clauses
        scopes = problem.instance.scopes
        b = problem.instance.b
        clauses = [(scope, int(val)) for scope, val in zip(scopes, b)]

        # ell, kikuchim, threshold
        ell = problem.fields["ell"]
        kikuchi_matrix = problem.fields["kikuchi_matrix"]
        threshold = problem.fields["threshold"]

        guiding_state = self.prepare_guiding_state(clauses, n, ell, k)

        # quartic circuit
        t = 1.0                             # evolution time
        steps = 1                           # Trotter steps
        r = int(np.ceil(np.log2(n * len(clauses))))        # number of phase qubits
        grover_iterations = int(np.log2(n)) # simple default
        circuit = QuarticQuantumAlgorithm.phase_estimation_with_amplitude_amplification(
            guiding_state,
            kikuchi_matrix,
            t,
            steps,
            r,
            int(n * ell / k),
            grover_iterations,
            threshold
        )
        return {"quartic_quantum_circuit": circuit}

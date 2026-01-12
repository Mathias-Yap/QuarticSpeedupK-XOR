import numpy as np
import pennylane as qml

from kxor_code.algorithms.Quartic_Schmidhuber_Quantum import Circuit
from kxor_code.algorithms.quartic_step_adapter import guiding_state_from_quartic_step


def main():
    """Small end-to-end demo for the toy circuit + stage-2.

    This is mainly here so I can sanity-check wiring, output shapes, etc., without
    cluttering the main module.
    """
    t, y = 2, 1
    H = np.array([[1, 0], [0, -1]], dtype=complex)
    tau = 0.3

    # Keep this tiny so the guiding-state prep circuit stays fast.
    # The step code uses a=ceil(log2(m)); when m=1 that becomes 0 and can be awkward,
    # so we just use m>=2 here.
    n = 1
    ell = 1
    k = 1
    clauses = [([0], 0), ([0], 1)]

    guiding = guiding_state_from_quartic_step(clauses=clauses, n=n, ell=ell, k=k, target_dim=2**y)

    c = Circuit(H=H, t=t, y=y, tau=tau, guiding_state=guiding, guiding_state_policy="strict")
    c.threshold_int = 2  # in [0, 2**t) = [0, 4)

    qpe_qnode = c.circuit()
    print(qml.draw(qpe_qnode, show_all_wires=True)())

    print("\nGuiding state used on system register:")
    print(guiding)

    aa_qnode = c.amplitude_amplification(n_iters=1)
    print(qml.draw(aa_qnode, show_all_wires=True)())

    print(qml.specs(aa_qnode)())
    print(f"\nPennylane version: {qml.__version__}")

    x_hat, V, v_top_sys, evals, good_phases = c.recover_index_small(
        n_iters=1,
        ell=1,
        stage1_quartic_kwargs={
            "clauses": clauses,
            "n": n,
            "ell": ell,
            "k": k,
            "threshold": 0.1,
            "evolution_time": 1.0,
            "trotter_steps": 1,
            "phase_qubits": t,
            "grover_iterations": 1,
        },
    )
    print("\nGood phases used:", good_phases)
    print("\nExtracted system vector (postselected):", v_top_sys)
    print("Voting matrix V:\n", V)
    print("Eigenvalues of V:", evals)
    print("Recovered index x_hat:", x_hat)

    x_hat2, v2, good_phases2, c2 = c.run_stage2_recover_index_small(V, n_iters=1, neighborhood=1)
    print("\n[Stage-2 quantum] Good phases used:", good_phases2)
    print("[Stage-2 quantum] Extracted system vector:", v2)
    print("[Stage-2 quantum] Recovered index x_hat2:", x_hat2)

    print("\nStage-2 circuit drawing:")
    stage2_qnode = c2.amplitude_amplification(n_iters=1, return_state=False, good_phases=good_phases2)
    print(qml.draw(stage2_qnode, show_all_wires=True)())


if __name__ == "__main__":
    main()

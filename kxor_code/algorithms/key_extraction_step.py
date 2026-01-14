from __future__ import annotations

import itertools
import math
from typing import Any, Dict, Optional

import numpy as np

from kxor_code.algorithms.base_alg_step import BaseAlgorithmStep, ProblemRecord, StepContext, StepStats
from kxor_code.algorithms.quartic_step_adapter import (
    extract_system_vector_from_state_no_ancillas,
    stage1_system_vector_from_quartic_step,
)
from kxor_code.algorithms.voting_matrix import form_voting_matrix_common_remainder


class KeyExtractionStep(BaseAlgorithmStep):
    """Pipeline step that turns the quartic circuit output into an extracted index.

    This step is intended to run *after* [kxor_code/algorithms/quartic_quantum_algorithm_step.py](kxor_code/algorithms/quartic_quantum_algorithm_step.py)
    and consumes its `quartic_quantum_circuit` output.

    Produced fields are deliberately simple (mostly numpy arrays / ints) so downstream
    steps can reuse them.
    """

    # Note: stage-1 can either consume a pre-built `quartic_quantum_circuit` (pipeline mode)
    # or build/run stage-1 itself via `stage1_system_vector_from_quartic_step` (adapter mode).
    requires_fields = ["ell", "threshold"]
    produces_fields = [
        "x_hat",
        "v_top_sys",
        "good_phases",
        "voting_matrix",
        "voting_eigenvalues",
    ]

    def __init__(
        self,
        *args,
        stage1_backend: str = "pipeline_qnode",
        evolution_time: float = 1.0,
        stage2_backend: str = "dense",
        stage2_num_eigenvalues: int = 5,
        stage2_circuit_iters: int = 1,
        stage2_circuit_neighborhood: int = 1,
        stage2_circuit_phase_qubits: int | None = None,
        evaluate: bool = False,
        random_baseline_trials: int = 0,
        random_seed: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        stage1_backend = str(stage1_backend)
        if stage1_backend in {"pipeline", "pipeline_circuit", "quartic_quantum_circuit"}:
            stage1_backend = "pipeline_qnode"
        if stage1_backend in {"adapter", "quartic_step", "quartic_step_adapter", "schmidhuber_stage1"}:
            stage1_backend = "quartic_step_adapter"
        self.stage1_backend = stage1_backend

        self.evolution_time = float(evolution_time)
        # Historical note: we used to expose a separate "dense" stage-2 backend.
        # The eigsh-style path is the general one; keep "dense" as an alias for compatibility.
        stage2_backend = str(stage2_backend)
        if stage2_backend == "dense":
            stage2_backend = "classical_eigsh"
        # Backwards-compatible alias for the stage-2 circuit backend.
        if stage2_backend == "schmidhuber_stage2_quantum":
            stage2_backend = "schmidhuber_stage2_circuit"
        self.stage2_backend = stage2_backend
        self.stage2_num_eigenvalues = int(stage2_num_eigenvalues)
        self.stage2_circuit_iters = int(stage2_circuit_iters)
        self.stage2_circuit_neighborhood = int(stage2_circuit_neighborhood)
        self.stage2_circuit_phase_qubits = (
            None if stage2_circuit_phase_qubits is None else int(stage2_circuit_phase_qubits)
        )
        self.evaluate = bool(evaluate)
        self.random_baseline_trials = int(random_baseline_trials)
        self.random_seed = int(random_seed)

        if self.stage1_backend not in {"pipeline_qnode", "quartic_step_adapter"}:
            raise ValueError(
                f"Unknown stage1_backend={self.stage1_backend!r}. Expected 'pipeline_qnode' or 'quartic_step_adapter'."
            )

        if self.stage2_backend not in {"classical_eigsh", "schmidhuber_stage2_circuit"}:
            raise ValueError(
                f"Unknown stage2_backend={self.stage2_backend!r}. Expected 'classical_eigsh' or 'schmidhuber_stage2_circuit'."
            )

    @staticmethod
    def _round_vector_to_pm1(v: np.ndarray) -> np.ndarray:
        """Round a real/complex vector to a {±1}^n assignment via sign(real(v))."""
        v = np.asarray(v)
        if v.ndim != 1:
            v = v.reshape(-1)
        signs = np.sign(np.real(v)).astype(int)
        signs[signs == 0] = 1
        return signs

    @staticmethod
    def _kxor_advantage(instance: Any, x_pm1: np.ndarray) -> float:
        """Paper definition: adv_I(x) = avg_{(S,b) in I} b * prod_{i in S} x_i."""
        x_pm1 = np.asarray(x_pm1, dtype=int).reshape(-1)
        scopes = np.asarray(instance.scopes, dtype=int)
        b = np.asarray(instance.b, dtype=int).reshape(-1)

        if x_pm1.size != int(instance.n):
            raise ValueError("x_pm1 must have length n")
        if scopes.ndim != 2 or scopes.shape[0] != b.size:
            raise ValueError("instance.scopes and instance.b shapes are inconsistent")
        if b.size == 0:
            return 0.0

        x_S = np.prod(x_pm1[scopes], axis=1)
        return float(np.mean(b * x_S))

    @staticmethod
    def _hamming_distance_pm1(a_pm1: np.ndarray, b_pm1: np.ndarray) -> int:
        a_pm1 = np.asarray(a_pm1, dtype=int).reshape(-1)
        b_pm1 = np.asarray(b_pm1, dtype=int).reshape(-1)
        if a_pm1.shape != b_pm1.shape:
            raise ValueError("Shapes do not match")
        return int(np.count_nonzero(a_pm1 != b_pm1))

    @staticmethod
    def _best_global_sign_match(z_hat_pm1: np.ndarray, z_pm1: np.ndarray) -> tuple[np.ndarray, bool]:
        """Resolve global sign ambiguity by choosing z_hat or -z_hat with smaller Hamming distance."""
        z_hat_pm1 = np.asarray(z_hat_pm1, dtype=int).reshape(-1)
        z_pm1 = np.asarray(z_pm1, dtype=int).reshape(-1)

        d0 = KeyExtractionStep._hamming_distance_pm1(z_hat_pm1, z_pm1)
        d1 = KeyExtractionStep._hamming_distance_pm1(-z_hat_pm1, z_pm1)
        if d1 < d0:
            return -z_hat_pm1, True
        return z_hat_pm1, False

    @staticmethod
    def _phase_qubits_for_problem(n: int, m: int) -> int:
        # Mirrors the default in QuarticQuantumAlgorithm.
        return int(np.ceil(np.log2(max(2, int(n) * max(1, int(m))))))

    @staticmethod
    def _threshold_index(threshold: float, *, evolution_time: float, phase_qubits: int) -> int:
        # Mirrors the conversion in quartic_quantum_algorithm_step.py.
        return int(
            np.floor(float(threshold) * float(evolution_time) / (2 * np.pi) * (1 << int(phase_qubits)))
        )

    @staticmethod
    def _form_voting_matrix_from_vector(v_top: np.ndarray, *, n: int, ell: int) -> np.ndarray:
        """Backward-compatible wrapper around the shared voting-matrix implementation."""
        return form_voting_matrix_common_remainder(v_top, n=int(n), ell=int(ell))

    def _run(self, problem: ProblemRecord, context: StepContext, stats: StepStats) -> Optional[Dict[str, Any]]:
        ell = int(problem.get_field("ell"))
        threshold = float(problem.get_field("threshold"))

        n = int(problem.instance.n)
        m_clauses = int(problem.instance.m)

        r = self._phase_qubits_for_problem(n=n, m=m_clauses)

        context.logger.info(
            "Running key extraction (n=%d, m=%d, ell=%d, phase_qubits=%d)",
            n,
            m_clauses,
            ell,
            r,
        )

        if self.stage1_backend == "pipeline_qnode":
            quartic_qnode = problem.get_field("quartic_quantum_circuit")
            if quartic_qnode is None or not callable(quartic_qnode):
                raise TypeError(
                    "stage1_backend='pipeline_qnode' requires problem.fields['quartic_quantum_circuit'] "
                    "to be a callable QNode."
                )

            state = np.asarray(quartic_qnode(), dtype=complex).reshape(-1)
            if state.size == 0:
                raise ValueError("quartic_quantum_circuit returned an empty statevector")

            total_qubits_f = np.log2(state.size)
            if abs(total_qubits_f - round(total_qubits_f)) > 1e-9:
                raise ValueError(f"Statevector length {state.size} is not a power of 2.")
            total_qubits = int(round(total_qubits_f))
            if total_qubits <= r:
                raise ValueError(
                    f"Statevector has {total_qubits} qubits but expected > phase_qubits={r}."
                )

            system_qubits = total_qubits - r
            thr_idx = self._threshold_index(threshold, evolution_time=self.evolution_time, phase_qubits=r)
            thr_idx = max(-1, min(thr_idx, (1 << r) - 1))
            good_phases = list(range(thr_idx + 1, 1 << r))

            v_top_sys = extract_system_vector_from_state_no_ancillas(
                state,
                phase_qubits=r,
                system_qubits=system_qubits,
                good_phases=good_phases,
                normalize=True,
            )
        else:
            kikuchi_matrix = problem.get_field("kikuchi_matrix")
            if kikuchi_matrix is None:
                raise TypeError(
                    "stage1_backend='quartic_step_adapter' requires problem.fields['kikuchi_matrix'] "
                    "(produced by ComputeKikuchiStep)."
                )

            scopes = np.asarray(problem.instance.scopes, dtype=int)
            b = np.asarray(problem.instance.b, dtype=int).reshape(-1)
            clauses = [(scope, int(val)) for scope, val in zip(scopes, b)]

            v_top_sys, good_phases, _qnode = stage1_system_vector_from_quartic_step(
                clauses=clauses,
                n=n,
                ell=ell,
                k=int(problem.instance.k),
                H=kikuchi_matrix,
                threshold=threshold,
                evolution_time=self.evolution_time,
                phase_qubits=r,
            )

            # stage1_system_vector_from_quartic_step returns a system vector of dimension 2**system_qubits.
            sys_dim = int(np.asarray(v_top_sys).size)
            sys_qubits_f = np.log2(sys_dim)
            if abs(sys_qubits_f - round(sys_qubits_f)) > 1e-9:
                raise ValueError(f"Stage-1 adapter returned system dimension {sys_dim} that is not a power of 2.")
            system_qubits = int(round(sys_qubits_f))
            total_qubits = int(system_qubits + r)
            thr_idx = self._threshold_index(threshold, evolution_time=self.evolution_time, phase_qubits=r)
            thr_idx = max(-1, min(thr_idx, (1 << r) - 1))

        m_subsets = math.comb(n, ell)
        v_top_subsets = v_top_sys[:m_subsets]

        # Use the shared implementation (also used by the Schmidhuber Circuit demo code).
        V = form_voting_matrix_common_remainder(v_top_subsets, n=n, ell=ell, logger=context.logger)

        n_v = int(V.shape[0])
        if n_v <= 2:
            raise ValueError(f"Stage-2 requires voting matrix dimension N>=3 (got N={n_v}).")

        if self.stage2_backend == "classical_eigsh":
            # Stage-2 via ARPACK (eigsh) in the same style as ClassicalEigenvaluesStep.
            # We reuse ClassicalEigenvaluesStep without modifying it by feeding a shifted matrix:
            #   V' = V + alpha*I with alpha > max_i sum_j |V_ij|
            # so all eigenvalues are positive and "largest magnitude" == "largest eigenvalue".
            # Eigenvectors are unchanged by shifting.

            # Lazy imports so environments without SciPy can still import this module.
            # For this backend, SciPy is required because ClassicalEigenvaluesStep uses eigsh.
            try:
                from scipy.sparse import csr_matrix, identity
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "stage2_backend='classical_eigsh' requires SciPy (scipy). Install it."
                ) from exc

            from kxor_code.algorithms.classical_eigenvalues_step import ClassicalEigenvaluesStep

            row_sum = float(np.max(np.sum(np.abs(V), axis=1)))
            alpha = row_sum + 1.0

            V_sparse = csr_matrix(V)
            V_shift = V_sparse + (alpha * identity(n_v, dtype=complex, format="csr"))

            # Build a minimal ProblemRecord for the classical step.
            tmp_problem = ProblemRecord(problem_id=problem.problem_id, instance=problem.instance)
            tmp_problem.add_field("kikuchi_matrix", V_shift)

            classical_step = ClassicalEigenvaluesStep()
            # ARPACK requires k < N-1. Also keep the request reasonable.
            classical_step.num_eigenvalues = int(min(max(1, self.stage2_num_eigenvalues), n_v - 2))
            out = classical_step._run(tmp_problem, context, stats) or {}

            evals_shift = np.asarray(out["eigenvalues"], dtype=float).reshape(-1)
            evecs = np.asarray(out["eigenvectors"], dtype=complex)
            evals = evals_shift - alpha

            top_idx = int(np.argmax(evals))
            top_vec = evecs[:, top_idx]
            x_hat = int(np.argmax(np.abs(top_vec)))

            # Note: in this backend we only compute `stage2_num_eigenvalues` eigenpairs.
            stats.add_data("stage2_alpha_shift", alpha)
            stats.add_data("stage2_eigs_computed", int(evals.size))
        else:
            # Stage-2 using the existing Schmidhuber "second circuit" implementation.
            # The circuit requires a power-of-two Hamiltonian dimension, so we embed the
            # shifted voting matrix (V + alpha I) into a larger zero-padded matrix.
            # The shift preserves eigenvectors; the zero padding preserves the top eigenspace.

            try:
                from kxor_code.algorithms.Quartic_Schmidhuber_Quantum_KeyExtraction import Circuit
            except Exception as exc:
                raise ModuleNotFoundError(
                    "stage2_backend='schmidhuber_stage2_circuit' requires the Schmidhuber Circuit module "
                    "and PennyLane (pennylane)."
                ) from exc

            row_sum = float(np.max(np.sum(np.abs(V), axis=1)))
            alpha = row_sum + 1.0
            V_shift_dense = V + (alpha * np.eye(n_v, dtype=complex))

            y2 = int(np.ceil(np.log2(max(1, n_v))))
            dim = 1 << y2
            V_pad = np.zeros((dim, dim), dtype=complex)
            V_pad[:n_v, :n_v] = V_shift_dense

            t2 = int(self.stage2_circuit_phase_qubits) if self.stage2_circuit_phase_qubits is not None else int(r)
            t2 = max(1, t2)

            c2 = Circuit(H=V_pad, t=t2, y=y2, tau=self.evolution_time)
            good_phases2 = c2.choose_good_phases_top_eigen(neighborhood=int(max(0, self.stage2_circuit_neighborhood)))

            _c2, stage2_qnode = c2.stage2_circuit_from_voting_matrix(
                V_pad,
                tau2=self.evolution_time,
                phase_qubits=t2,
                good_phases=good_phases2,
                n_iters=int(max(0, self.stage2_circuit_iters)),
                return_state=True,
            )

            state2 = stage2_qnode()
            v2_full = _c2.extract_system_vector_from_state(state2, good_phases=good_phases2)
            v2_full = Circuit._normalize_global_phase(v2_full)

            top_vec = np.asarray(v2_full, dtype=complex).reshape(-1)[:n_v]
            x_hat = int(np.argmax(np.abs(top_vec)))

            # No eigenvalues are computed in this backend.
            evals = np.array([], dtype=float)
            stats.add_data("stage2_alpha_shift", alpha)
            stats.add_data("stage2_circuit_dim", int(dim))
            stats.add_data("stage2_circuit_phase_qubits", int(t2))
            stats.add_data("stage2_circuit_good_phases_count", int(len(good_phases2)))

        if self.evaluate:
            z_hat = self._round_vector_to_pm1(top_vec)
            adv_hat = self._kxor_advantage(problem.instance, z_hat)
            adv_hat_flipped = self._kxor_advantage(problem.instance, -z_hat)
            if adv_hat_flipped > adv_hat:
                z_hat = -z_hat
                adv_hat = adv_hat_flipped
                flipped_for_advantage = True
            else:
                flipped_for_advantage = False

            stats.add_data("eval_advantage", adv_hat)
            stats.add_data("eval_flipped_for_advantage", flipped_for_advantage)

            z_true = getattr(problem.instance, "z", None)
            if z_true is not None and np.asarray(z_true).size == n:
                z_true = np.asarray(z_true, dtype=int).reshape(-1)
                z_hat_for_z, flipped_for_z = self._best_global_sign_match(z_hat, z_true)
                ham = self._hamming_distance_pm1(z_hat_for_z, z_true)
                ham_frac = float(ham) / float(n) if n else 0.0
                corr = float(np.mean(z_hat_for_z * z_true)) if n else 0.0
                stats.add_data("eval_hamming", ham)
                stats.add_data("eval_hamming_frac", ham_frac)
                stats.add_data("eval_correlation", corr)
                stats.add_data("eval_flipped_for_z", flipped_for_z)

                context.logger.info(
                    "Eval: adv=%.4f, hamming=%d/%d (%.3f), corr=%.3f",
                    adv_hat,
                    ham,
                    n,
                    ham_frac,
                    corr,
                )
            else:
                context.logger.info("Eval: adv=%.4f (no ground-truth z available)", adv_hat)

            if self.random_baseline_trials > 0:
                rng = np.random.default_rng(self.random_seed)
                advs = []
                for _ in range(self.random_baseline_trials):
                    x_rand = rng.choice([-1, 1], size=n, replace=True)
                    advs.append(self._kxor_advantage(problem.instance, x_rand))
                stats.add_data("eval_random_adv_mean", float(np.mean(advs)))
                stats.add_data("eval_random_adv_std", float(np.std(advs)))

        stats.add_data("stage2_backend", self.stage2_backend)
        stats.add_data("stage1_backend", self.stage1_backend)
        stats.add_data("phase_qubits", r)
        stats.add_data("system_qubits", system_qubits)
        stats.add_data("total_qubits", total_qubits)
        stats.add_data("threshold_index", thr_idx)
        stats.add_data("good_phases_count", len(good_phases))
        stats.add_data("subset_count", m_subsets)
        stats.add_data("x_hat", x_hat)

        context.logger.info("Key extraction complete: x_hat=%d", x_hat)

        return {
            "x_hat": x_hat,
            "v_top_sys": v_top_sys,
            "good_phases": good_phases,
            "voting_matrix": V,
            "voting_eigenvalues": evals,
        }

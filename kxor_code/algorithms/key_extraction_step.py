from __future__ import annotations

import itertools
import math
from typing import Any, Dict, Optional

import numpy as np

from kxor_code.algorithms.base_alg_step import BaseAlgorithmStep, ProblemRecord, StepContext, StepStats
from kxor_code.algorithms.quartic_step_adapter import extract_system_vector_from_state_no_ancillas


class KeyExtractionStep(BaseAlgorithmStep):
    """Pipeline step that turns the quartic circuit output into an extracted index.

    This step is intended to run *after* [kxor_code/algorithms/quartic_quantum_algorithm_step.py](kxor_code/algorithms/quartic_quantum_algorithm_step.py)
    and consumes its `quartic_quantum_circuit` output.

    Produced fields are deliberately simple (mostly numpy arrays / ints) so downstream
    steps can reuse them.
    """

    requires_fields = ["ell", "threshold", "quartic_quantum_circuit"]
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
        evolution_time: float = 1.0,
        evaluate: bool = False,
        random_baseline_trials: int = 0,
        random_seed: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.evolution_time = float(evolution_time)
        self.evaluate = bool(evaluate)
        self.random_baseline_trials = int(random_baseline_trials)
        self.random_seed = int(random_seed)

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
        """Build the (n x n) voting matrix using the common-remainder formula."""
        n = int(n)
        ell = int(ell)
        if n <= 0:
            raise ValueError("n must be positive")
        if ell <= 0 or ell > n:
            raise ValueError("ell must satisfy 1 <= ell <= n")

        v_top = np.asarray(v_top, dtype=complex).reshape(-1)

        subset_index = {tuple(s): i for i, s in enumerate(itertools.combinations(range(n), ell))}
        expected = math.comb(n, ell)
        if v_top.size < expected:
            raise ValueError(f"v_top too small: need at least C(n,ell)={expected}, got {v_top.size}.")

        V = np.zeros((n, n), dtype=complex)
        vertices = set(range(n))

        for i in range(n):
            for j in range(i + 1, n):
                total = 0.0 + 0.0j
                for R in itertools.combinations(vertices - {i, j}, ell - 1):
                    S_i = tuple(sorted(R + (i,)))
                    S_j = tuple(sorted(R + (j,)))
                    total += v_top[subset_index[S_i]].conjugate() * v_top[subset_index[S_j]]
                V[i, j] = 0.5 * total
                V[j, i] = np.conj(V[i, j])

        return V

    def _run(self, problem: ProblemRecord, context: StepContext, stats: StepStats) -> Optional[Dict[str, Any]]:
        ell = int(problem.get_field("ell"))
        threshold = float(problem.get_field("threshold"))

        quartic_qnode = problem.get_field("quartic_quantum_circuit")
        if quartic_qnode is None or not callable(quartic_qnode):
            raise TypeError("problem.fields['quartic_quantum_circuit'] must be a callable QNode.")

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

        m_subsets = math.comb(n, ell)
        v_top_subsets = v_top_sys[:m_subsets]

        V = self._form_voting_matrix_from_vector(v_top_subsets, n=n, ell=ell)
        evals, evecs = np.linalg.eigh(V)
        top_vec = evecs[:, int(np.argmax(evals))]
        x_hat = int(np.argmax(np.abs(top_vec)))

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

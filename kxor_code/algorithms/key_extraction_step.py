from __future__ import annotations

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
    """
        ---USAGE INSTRUCTIONS---
    
        Key extraction / key recovery pipeline step.

        This step is the project’s "distinguish → recover" extension: it produces a *key estimate*
        $z\_\text{hat} \in \{\pm 1\}^n$ from the stage-2 output vector.

        It is designed to be used inside the pipeline framework (via `execute(...)`), but the
        core work happens in `_run(problem, context, stats)`.

        ------------------------------------------------------------------------------
        Manual (How to run / configure)
        ------------------------------------------------------------------------------

        **Inputs expected in `problem`**
        - Always required fields:
            - `ell` (int): Kikuchi/voting subset size.
            - `threshold` (float): phase-estimation threshold used to define "good phases".
        - Required pipeline instance data:
            - `problem.instance.n`, `problem.instance.m`, `problem.instance.k`,
                `problem.instance.scopes`, `problem.instance.b`.

        **Stage-1 backends (how `v_top_sys` is produced)**
        - `stage1_backend='pipeline_qnode'` (default)
            - Requires `problem.fields['quartic_quantum_circuit']` (callable returning a full
                statevector as produced by `QuarticQuantumAlgorithmStep`).
            - This is the standard pipeline mode when you already ran the quartic step.
            - No PennyLane import is required here (the QNode can still be PennyLane under the hood).

        - `stage1_backend='quartic_step_adapter'`
            - Runs stage-1 via `stage1_system_vector_from_quartic_step(...)` from
                `kxor_code.algorithms.quartic_step_adapter`.
            - Requires `problem.fields['kikuchi_matrix']` (typically produced by `ComputeKikuchiStep`).
            - Dependency notes:
                - Requires PennyLane at runtime.
                - If `kikuchi_matrix` is SciPy sparse, the adapter densifies *small* instances.

        - `stage1_backend='precomputed_eigenvector'`
            - Uses a precomputed eigenvector stored on the record (typically from a prior
              `ClassicalEigenvaluesStep` run):
                - `problem.fields['eigenvectors']` and optionally `problem.fields['eigenvalues']`.
            - Recommended for analyzing saved compact records that already contain eigenpairs.

        Aliases accepted via class variables (see `STAGE1_BACKEND_ALIASES`).

        **Stage-2 backends (how we get `top_vec` from the voting matrix)**
        - `stage2_backend='schmidhuber_stage2_circuit'` (default)
            - Uses the "second circuit" implementation in
                `kxor_code.algorithms.Quartic_Schmidhuber_Quantum_KeyExtraction.Circuit`.
            - Requires PennyLane.
            - This backend does not compute eigenvalues explicitly; `voting_eigenvalues` is empty.
            - Parameters:
                - `stage2_circuit_phase_qubits` (int|None): defaults to stage-1 phase qubits.
                - `stage2_circuit_iters` (int): amplitude amplification iterations in stage-2.
                - `stage2_circuit_neighborhood` (int): how wide a neighborhood of good phases to mark.

        - `stage2_backend='classical_eigsh'`
            - Computes a few top eigenpairs using ARPACK via `ClassicalEigenvaluesStep`.
            - Requires SciPy.
            - Parameter: `stage2_num_eigenvalues` (int): number of eigenpairs requested
                (internally capped to satisfy ARPACK constraint `k < N-1`).
            - Note: internally we shift the matrix by `alpha I` so "largest magnitude" matches
                "largest eigenvalue"; eigenvectors are unchanged by shifting.

        Aliases accepted via class variables (see `STAGE2_BACKEND_ALIASES`).

        **Key recovery output (what we recover)**
        - Stage-2 produces a vector `top_vec`.
        - We map it to a key estimate via
            `recover_key_from_top_vector(instance, top_vec)` in
            `kxor_code.algorithms.Quartic_Schmidhuber_Quantum_KeyExtraction`:
            - Round to ±1 via `sign(real(top_vec))`.
            - Resolve the global sign ambiguity by picking the sign that maximizes *advantage*.
        - `x_hat = argmax(|top_vec|)` is still produced for compatibility/diagnostics.

        **Evaluation metrics (optional)**
        - Enable with `evaluate=True`.
        - Always computed when enabled:
            - `eval_advantage`: advantage of the recovered key `z_hat`.
        - Only computed if `problem.instance.z` exists and has length n:
            - `eval_hamming`, `eval_hamming_frac`, `eval_correlation`.
        - Optional random baseline: set `random_baseline_trials > 0`.

        **Produced fields**
        - `z_hat` (np.ndarray shape (n,), entries in {±1})  <-- main recovery artifact
        - `x_hat` (int)                                    <-- diagnostic/legacy
        - `v_top_sys` (np.ndarray)
        - `good_phases` (list[int])
        - `voting_matrix` (np.ndarray)
        - `voting_eigenvalues` (np.ndarray; empty for circuit backend)

        **Example configurations**
        - Default (stage-2 circuit):
            `KeyExtractionStep()`
        - Force classical stage-2 eigensolver:
            `KeyExtractionStep(stage2_backend='classical_eigsh', stage2_num_eigenvalues=5)`
        - Run stage-1 via adapter + stage-2 via circuit:
            `KeyExtractionStep(stage1_backend='quartic_step_adapter')`
    """

    # --- class-level configuration knobs (so settings are discoverable in one place) ---
    DEFAULT_STAGE1_BACKEND = "pipeline_qnode"
    DEFAULT_STAGE2_BACKEND = "schmidhuber_stage2_circuit"

    STAGE1_BACKEND_ALIASES = {
        "pipeline": "pipeline_qnode",
        "pipeline_circuit": "pipeline_qnode",
        "quartic_quantum_circuit": "pipeline_qnode",
        "adapter": "quartic_step_adapter",
        "quartic_step": "quartic_step_adapter",
        "quartic_step_adapter": "quartic_step_adapter",
        "schmidhuber_stage1": "quartic_step_adapter",
        "precomputed": "precomputed_eigenvector",
        "precomputed_eigenvector": "precomputed_eigenvector",
        "eigenvector": "precomputed_eigenvector",
        "eigenvectors": "precomputed_eigenvector",
    }
    VALID_STAGE1_BACKENDS = {"pipeline_qnode", "quartic_step_adapter", "precomputed_eigenvector"}

    STAGE2_BACKEND_ALIASES = {
        # Historical aliases
        "dense": "classical_eigsh",
        "schmidhuber_stage2_quantum": "schmidhuber_stage2_circuit",
    }
    VALID_STAGE2_BACKENDS = {"classical_eigsh", "schmidhuber_stage2_circuit"}

    DEFAULT_STAGE2_NUM_EIGENVALUES = 5
    DEFAULT_STAGE2_CIRCUIT_ITERS = 1
    DEFAULT_STAGE2_CIRCUIT_NEIGHBORHOOD = 1

    # Note: stage-1 can either consume a pre-built `quartic_quantum_circuit` (pipeline mode)
    # or build/run stage-1 itself via `stage1_system_vector_from_quartic_step` (adapter mode).
    requires_fields = ["ell", "threshold"]
    produces_fields = [
        "x_hat",
        "z_hat",
        "v_top_sys",
        "good_phases",
        "voting_matrix",
        "voting_eigenvalues",
    ]

    def __init__(
        self,
        *args,
        stage1_backend: str | None = None,
        evolution_time: float = 1.0,
        stage2_backend: str | None = None,
        stage2_num_eigenvalues: int | None = None,
        stage2_circuit_iters: int | None = None,
        stage2_circuit_neighborhood: int | None = None,
        stage2_circuit_phase_qubits: int | None = None,
        evaluate: bool = False,
        random_baseline_trials: int = 0,
        random_seed: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        stage1_backend = self.DEFAULT_STAGE1_BACKEND if stage1_backend is None else str(stage1_backend)
        stage1_backend = self.STAGE1_BACKEND_ALIASES.get(stage1_backend, stage1_backend)
        self.stage1_backend = stage1_backend

        self.evolution_time = float(evolution_time)
        # Historical note: we used to expose a separate "dense" stage-2 backend.
        # The eigsh-style path is the general one; keep "dense" as an alias for compatibility.

        stage2_backend = self.DEFAULT_STAGE2_BACKEND if stage2_backend is None else str(stage2_backend)
        stage2_backend = self.STAGE2_BACKEND_ALIASES.get(stage2_backend, stage2_backend)
        self.stage2_backend = stage2_backend
        self.stage2_num_eigenvalues = int(
            self.DEFAULT_STAGE2_NUM_EIGENVALUES if stage2_num_eigenvalues is None else stage2_num_eigenvalues
        )
        self.stage2_circuit_iters = int(
            self.DEFAULT_STAGE2_CIRCUIT_ITERS if stage2_circuit_iters is None else stage2_circuit_iters
        )
        self.stage2_circuit_neighborhood = int(
            self.DEFAULT_STAGE2_CIRCUIT_NEIGHBORHOOD
            if stage2_circuit_neighborhood is None
            else stage2_circuit_neighborhood
        )
        self.stage2_circuit_phase_qubits = (
            None if stage2_circuit_phase_qubits is None else int(stage2_circuit_phase_qubits)
        )
        self.evaluate = bool(evaluate)
        self.random_baseline_trials = int(random_baseline_trials)
        self.random_seed = int(random_seed)

        if self.stage1_backend not in self.VALID_STAGE1_BACKENDS:
            raise ValueError(
                f"Unknown stage1_backend={self.stage1_backend!r}. Expected 'pipeline_qnode' or 'quartic_step_adapter'."
            )

        if self.stage2_backend not in self.VALID_STAGE2_BACKENDS:
            raise ValueError(
                f"Unknown stage2_backend={self.stage2_backend!r}. Expected 'classical_eigsh' or 'schmidhuber_stage2_circuit'."
            )

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

        # Stage-1: obtain the (post-selected) system vector v_top_sys.
        #
        # This vector is what the later voting-matrix construction consumes.
        # Depending on backend, it is either extracted from a statevector that the
        # pipeline already computed (pipeline_qnode), or computed here by calling
        # the quartic adapter (quartic_step_adapter).
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

            # Convert the user-facing threshold (continuous) into a discrete phase-index cutoff.
            # We then mark all indices above that cutoff as "good".
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
        elif self.stage1_backend == "precomputed_eigenvector":
            # For saved records that already contain eigenpairs, we can reuse the top eigenvector
            # directly instead of re-running the quartic stage-1 circuit.
            evecs = problem.get_field("eigenvectors")
            if evecs is None:
                raise TypeError(
                    "stage1_backend='precomputed_eigenvector' requires problem.fields['eigenvectors']"
                )
            evecs = np.asarray(evecs)
            if evecs.ndim != 2 or evecs.shape[0] == 0 or evecs.shape[1] == 0:
                raise ValueError(
                    f"problem.fields['eigenvectors'] must be a non-empty 2D array (got shape {evecs.shape})"
                )

            evals = problem.get_field("eigenvalues")
            if evals is not None:
                evals = np.asarray(evals, dtype=float).reshape(-1)
                if evals.size != evecs.shape[1]:
                    raise ValueError(
                        "problem.fields['eigenvalues'] length must match eigenvectors column count "
                        f"(got {evals.size} vs {evecs.shape[1]})"
                    )
                top_idx = int(np.argmax(evals))
            else:
                top_idx = int(evecs.shape[1] - 1)

            v_top_sys = np.asarray(evecs[:, top_idx], dtype=complex).reshape(-1)

            # For consistency in stats, still compute the threshold index and phase qubit count,
            # even though we didn't actually run QPE here.
            thr_idx = self._threshold_index(threshold, evolution_time=self.evolution_time, phase_qubits=r)
            thr_idx = max(-1, min(thr_idx, (1 << r) - 1))
            good_phases = []

            sys_dim = int(v_top_sys.size)
            system_qubits = int(np.ceil(np.log2(max(1, sys_dim))))
            total_qubits = int(system_qubits + r)
        else:
            # Adapter backend: build the clauses list, run stage-1 (quartic step), and return the
            # extracted system vector directly.
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

            # Keep the same threshold bookkeeping as the pipeline backend so that stats are
            # comparable across backends.
            thr_idx = self._threshold_index(threshold, evolution_time=self.evolution_time, phase_qubits=r)
            thr_idx = max(-1, min(thr_idx, (1 << r) - 1))

        m_subsets = math.comb(n, ell)

        # Voting matrix construction only uses the first C(n, ell) amplitudes, corresponding
        # to the ell-subsets in the common-remainder encoding.
        v_top_subsets = v_top_sys[:m_subsets]

        # Use the shared implementation.
        V = form_voting_matrix_common_remainder(v_top_subsets, n=n, ell=ell, logger=context.logger)

        n_v = int(V.shape[0])
        if n_v <= 2:
            raise ValueError(f"Stage-2 requires voting matrix dimension N>=3 (got N={n_v}).")

        # Stage-2: extract a top-eigenvector direction (or a proxy for it) from the voting matrix.
        #
        # We keep x_hat = argmax(|top_vec|) for diagnostics and backwards compatibility.
        # The actual key recovery is done below from top_vec.
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
                    "stage2_backend='classical_eigsh' requires package: SciPy (conda/pip install scipy)."
                ) from exc

            from kxor_code.algorithms.classical_eigenvalues_step import ClassicalEigenvaluesStep

            row_sum = float(np.max(np.sum(np.abs(V), axis=1)))
            alpha = row_sum + 1.0

            # Shift by +alpha*I so all eigenvalues are positive and ARPACK's "largest magnitude"
            # selection matches the top-eigenvalue direction we care about. The shift does not
            # change eigenvectors.

            V_sparse = csr_matrix(V)
            V_shift = V_sparse + (alpha * identity(n_v, dtype=complex, format="csr"))

            # Build a minimal ProblemRecord for the classical step.
            tmp_problem = ProblemRecord(problem_id=problem.problem_id, instance=problem.instance)
            tmp_problem.add_field("kikuchi_matrix", V_shift)

            # ClassicalEigenvaluesStep reads `num_eigenvalues` from the ProblemRecord.
            eigs_requested = int(min(max(1, self.stage2_num_eigenvalues), n_v - 2))
            tmp_problem.add_field("num_eigenvalues", eigs_requested)

            classical_step = ClassicalEigenvaluesStep()
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
            # Stage-2 using the existing Schmidhuber-extended "second circuit" implementation.
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

            # Circuit code requires a power-of-two dimension. Embed V into the top-left block of a
            # 2**y2 matrix by zero-padding.
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

            # Drop the padded part; only the original n_v entries correspond to the original
            # voting matrix.
            top_vec = np.asarray(v2_full, dtype=complex).reshape(-1)[:n_v]
            x_hat = int(np.argmax(np.abs(top_vec)))

            # No eigenvalues are computed in this backend.
            evals = np.array([], dtype=float)
            stats.add_data("stage2_alpha_shift", alpha)
            stats.add_data("stage2_circuit_dim", int(dim))
            stats.add_data("stage2_circuit_phase_qubits", int(t2))
            stats.add_data("stage2_circuit_good_phases_count", int(len(good_phases2)))

        # Key recovery: keep the key-recovery logic in the Schmidhuber module.
        from kxor_code.algorithms.Quartic_Schmidhuber_Quantum_KeyExtraction import recover_key_from_top_vector

        z_hat, adv_hat, flipped_for_advantage = recover_key_from_top_vector(problem.instance, top_vec)

        if self.evaluate:
            # Evaluation focuses on the recovered key z_hat.
            #
            # - Always compute advantage of z_hat (this does not require ground truth).
            # - If ground-truth z exists, compute sign-corrected Hamming/correlation.
            #   (There is a global sign ambiguity: z and -z are equivalent for the planted structure.
            #    For reporting we pick the sign that matches z best.)
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
                # Optional sanity check: compare recovered advantage against random ±1 keys.
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
            "z_hat": z_hat,
            "v_top_sys": v_top_sys,
            "good_phases": good_phases,
            "voting_matrix": V,
            "voting_eigenvalues": evals,
        }

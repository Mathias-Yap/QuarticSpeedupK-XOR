from __future__ import annotations

import functools
import itertools
import logging
import time
from math import comb
from typing import Callable, Iterable, Sequence

import numpy as np

try:
    import pennylane as qml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    qml = None
 

from kxor_code.algorithms.quartic_step_adapter import (
    pad_and_normalize_state,
    build_guiding_state_from_quartic_step,
    guiding_state_from_quartic_step,
    extract_system_vector_from_state_no_ancillas,
    stage1_system_vector_from_quartic_step,
    stage1_v_top_from_quartic_step,
)


__all__ = [
    "Circuit",
    "recover_key_from_top_vector",
    # Re-export adapter helpers so older imports keep working.
    "pad_and_normalize_state",
    "build_guiding_state_from_quartic_step",
    "guiding_state_from_quartic_step",
    "extract_system_vector_from_state_no_ancillas",
    "stage1_system_vector_from_quartic_step",
    "stage1_v_top_from_quartic_step",
]


_LOGGER = logging.getLogger(__name__)


def _round_vector_to_pm1(v: np.ndarray) -> np.ndarray:
    """Round a real/complex vector to a {±1}^n assignment via sign(real(v))."""
    v = np.asarray(v)
    if v.ndim != 1:
        v = v.reshape(-1)
    signs = np.sign(np.real(v)).astype(int)
    signs[signs == 0] = 1
    return signs


def _kxor_advantage(instance, x_pm1: np.ndarray) -> float:
    """Compute advantage Adv(x) = avg_i b_i * prod_{j in S_i} x_j for x in {±1}^n."""
    x_pm1 = np.asarray(x_pm1, dtype=int).reshape(-1)
    n = int(instance.n)
    if x_pm1.size != n:
        raise ValueError(f"x_pm1 must have length n={n} (got {x_pm1.size})")

    scopes = np.asarray(instance.scopes, dtype=int)
    b = np.asarray(instance.b, dtype=int).reshape(-1)
    if scopes.ndim != 2 or scopes.shape[0] != b.size:
        raise ValueError("instance.scopes and instance.b shapes are inconsistent")
    if b.size == 0:
        return 0.0

    x_S = np.prod(x_pm1[scopes], axis=1)
    return float(np.mean(b * x_S))


def recover_key_from_top_vector(instance, top_vec: np.ndarray) -> tuple[np.ndarray, float, bool]:
    """Recover a planted-key estimate z_hat ∈ {±1}^n from a stage-2 vector.

    The vector `top_vec` is interpreted as a proxy for the planted direction (top eigenspace).
    We round to ±1 by sign(real(.)) and resolve global sign ambiguity by choosing the sign
    that maximizes advantage on the instance clauses.

    Returns
    -------
    z_hat : np.ndarray
        Length-n vector with entries in {±1}.
    adv : float
        Advantage achieved by z_hat.
    flipped : bool
        Whether we flipped the global sign (-z_hat_raw) to maximize advantage.
    """
    z_hat_raw = _round_vector_to_pm1(top_vec)
    n = int(instance.n)
    if z_hat_raw.size != n:
        raise ValueError(
            f"Recovered vector has length {z_hat_raw.size}, but instance has n={n}. Cannot form z_hat."
        )

    adv = _kxor_advantage(instance, z_hat_raw)
    adv_flipped = _kxor_advantage(instance, -z_hat_raw)
    if adv_flipped > adv:
        return -z_hat_raw, adv_flipped, True
    return z_hat_raw, adv, False


def _unitary_from_hermitian(H: np.ndarray, tau: float) -> np.ndarray:
    """Compute U = exp(-i H tau) for Hermitian H via eigendecomposition.

    This avoids a hard SciPy dependency while remaining numerically stable for the
    Hermitian matrices used throughout this module.
    """
    H = np.asarray(H, dtype=complex)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("H must be a square matrix")
    if not np.allclose(H, H.conj().T, atol=1e-8):
        raise ValueError("H must be Hermitian (within numerical tolerance)")

    evals, evecs = np.linalg.eigh(H)
    phases = np.exp(-1j * evals * float(tau))
    return (evecs * phases) @ evecs.conj().T


def _qpe_aa_template(
    *,
    qpe: Callable[[], None],
    use_aa: bool,
    n_iters: int,
    oracle: Callable[[], None],
    reflect_zero: Callable[[], None],
    return_state: bool,
    phase_wires: Sequence[int],
):
    """Shared QPE (+ optional amplitude amplification) template.

    Small module-level helper so we can build QNodes without nested defs.
    The actual quantum ops are passed in as callables.
    """
    qpe()

    if use_aa:
        for _ in range(int(n_iters)):
            oracle()
            qml.adjoint(qpe)()
            reflect_zero()
            qpe()

    if return_state:
        return qml.state()
    return qml.probs(wires=phase_wires)


# quartic-step adapter helpers live in `kxor_code.algorithms.quartic_step_adapter`.
# Imported above and re-exported here so older code keeps working.


class Circuit:
    # Default stage-1 backend used by example/demo helpers when the caller doesn't specify.
    # - 'quartic_step': use quartic_quantum_algorithm_step for stage-1, then extract a system vector (i.e. import circuit).
    # - 'toy': use this module's dense-matrix QPE+AA for stage-1 (i.e. import only guiding state).
    DEFAULT_STAGE1_BACKEND = "quartic_step"
    # How to handle guiding states whose dimension doesn't match 2**y.
    # - "pad": pad with zeros up to 2**y and normalize (current behavior)
    # - "strict": require guiding_state.size == 2**y (raise otherwise)
    GUIDING_STATE_POLICY = "pad"

    def __init__(
        self,
        H: np.ndarray,
        t: int,
        y: int,
        tau: float,
        stage1_backend: str | None = None,
        guiding_state: np.ndarray | None = None,
        guiding_state_policy: str | None = None,
        *,
        logger: logging.Logger | None = None,
    ):
        r"""Create a toy QPE(+AA) circuit instance.

        Parameters
        ----------
        H : np.ndarray
            Hermitian $(2^y \times 2^y)$ matrix.
        t : int
            Number of phase-register qubits.
        y : int
            Number of system-register qubits.
        tau : float
            Evolution time for $U = \exp(-i H \tau)$.
        """
        if qml is None:  # pragma: no cover
            raise ModuleNotFoundError(
                "PennyLane (pennylane) is required to construct Circuit objects. Install it to use circuit backends."
            )

        self.H = H
        self.t = t  # number of phase qubits used in first register
        self.y = y  # number of system qubits in second register
        self.tau = tau  # evolution time parameter for unitary U = exp(-i H tau)

        self.U = _unitary_from_hermitian(H, tau)  # exp(-i H tau)

        # Logging is optional; we don't set up handlers here.
        # If you want output, configure logging in your entrypoint / notebook.
        self.logger = logger or _LOGGER

        # Stage-1 backend selection; used when recover_index_small(...) doesn't get an override.
        self.stage1_backend = str(stage1_backend) if stage1_backend is not None else str(Circuit.DEFAULT_STAGE1_BACKEND)

        # wire layout: phase qubits from 0 to t-1, system qubits from t to t+y-1
        self.phase_wires = list(range(t))  # QPE phase estimation register wires
        self.sys_wires = list(range(t, t + y))  # system register wires

        # --- ancillas for a coherent (no-measurement) threshold oracle on the phase register ---
        # lt_wire will hold the predicate "phase < threshold" (1 means less-than).
        # eq_wires implement a prefix-equality chain needed for a reversible comparator.
        base = t + y
        self.lt_wire = base
        self.eq_wires = list(range(base + 1, base + 1 + (t + 1)))  # t+1 ancillas, eq_wires[0] used as eq=1 seed

        # device initialization with total wires = t + y + ancillas for threshold oracle
        self.dev = qml.device("default.qubit", wires=t + y + 1 + (t + 1))

        # Pre-bind the QPE block as a plain callable so we don't need a wrapper method.
        # This also plays nicely with qml.adjoint(self._qpe)() later.
        self._qpe = functools.partial(
            Circuit.qpe_block_ops,
            U=self.U,
            phase_wires=self.phase_wires,
            sys_wires=self.sys_wires,
            prepare_system_fn=self.prepare_system,
        )

        self.guiding_state_policy = (
            str(guiding_state_policy) if guiding_state_policy is not None else str(Circuit.GUIDING_STATE_POLICY)
        )

        self.guiding_state = None
        if guiding_state is not None:
            self.set_guiding_state(guiding_state, policy=self.guiding_state_policy)

        self.logger.debug(
            "Initialized Circuit(t=%d, y=%d, wires_total=%d, stage1_backend=%s)",
            self.t,
            self.y,
            len(self.dev.wires),
            self.stage1_backend,
        )

    def set_guiding_state(self, guiding_state: np.ndarray | None, *, policy: str | None = None):
        """Set (or clear) the system-register guiding state.

        If set, `prepare_system()` uses amplitude encoding (Möttönen state prep).
        """
        if guiding_state is None:
            self.guiding_state = None
            self.logger.debug("Cleared guiding state")
            return

        policy = str(policy) if policy is not None else str(self.guiding_state_policy)
        target_dim = 2 ** int(self.y)
        guiding_state = np.asarray(guiding_state, dtype=complex).reshape(-1)

        if policy == "strict":
            if guiding_state.size != target_dim:
                raise ValueError(
                    f"Guiding state dimension must equal 2**y={target_dim} (got {guiding_state.size}). "
                    "Either set y to match, or use guiding_state_policy='pad'."
                )
            nrm = np.linalg.norm(guiding_state)
            if nrm == 0:
                raise ValueError("Guiding state has zero norm.")
            self.guiding_state = guiding_state / nrm
            self.logger.debug(
                "Set guiding state (policy=strict, dim=%d, norm=%.6g)",
                target_dim,
                float(nrm),
            )
            return

        if policy == "pad":
            self.guiding_state = pad_and_normalize_state(guiding_state, target_dim)
            self.logger.debug(
                "Set guiding state (policy=pad, input_dim=%d -> target_dim=%d)",
                int(guiding_state.size),
                target_dim,
            )
            return

        raise ValueError(f"Unknown guiding_state_policy={policy!r}. Expected 'pad' or 'strict'.")

    def prepare_system(self) -> None:
        """Prepare the system register before QPE.

        Default is a uniform superposition. If `self.guiding_state` is set, we
        instead amplitude-encode that vector on the system wires.
        """
        if self.guiding_state is not None:
            qml.templates.MottonenStatePreparation(self.guiding_state, wires=self.sys_wires)
            return

        for w in self.sys_wires:
            qml.Hadamard(w)

    @staticmethod
    def qpe_block_ops(
        *,
        U: np.ndarray,
        phase_wires: Sequence[int],
        sys_wires: Sequence[int],
        prepare_system_fn: Callable[[], None] | None = None,
    ) -> None:
        """Apply QPE with everything passed in explicitly.

        Parameters
        ----------
        U : np.ndarray
            Unitary to be phase-estimated (dimension 2**len(sys_wires)).
        phase_wires : Sequence[int]
            Wires for the phase/QPE register.
        sys_wires : Sequence[int]
            Wires for the system register on which U acts.
        prepare_system_fn : callable or None
            Optional state preparation function. If provided, it is called before QPE.
            Must be a PennyLane quantum function (applies ops, returns None).
        """
        if prepare_system_fn is not None:
            prepare_system_fn()

        # Hadamards on the phase register
        for w in phase_wires:
            qml.Hadamard(w)

        # Controlled-U^(2^j) operations
        for j, ctrl in enumerate(phase_wires):
            U_pow = np.linalg.matrix_power(U, 2**j)
            qml.ctrl(qml.QubitUnitary, control=ctrl)(U_pow, wires=sys_wires)

        # Inverse QFT on the phase register
        qml.adjoint(qml.QFT)(wires=phase_wires)

    def circuit(self):
        """QPE circuit that returns phase-register probabilities (eigenvalue info only)."""

        self.logger.debug("Building QPE probs QNode")

        @qml.qnode(self.dev)
        def _circuit():
            """Run the QPE block and return phase probabilities."""
            self._qpe()
            return qml.probs(wires=self.phase_wires)

        return _circuit

    def circuit_state(self):
        """QPE circuit that returns the *full* statevector (toy sizes only)."""

        self.logger.debug("Building QPE state QNode")

        @qml.qnode(self.dev)
        def _circuit_state():
            """Run the QPE block and return the full statevector."""
            self._qpe()
            return qml.state()

        return _circuit_state

    def oracle(self):
        """
        Coherent (no-measurement) phase-flip oracle for amplitude amplification.

        Marks "good" states where the *phase register interpreted as an integer* satisfies:
            phase >= threshold_int

        Implementation strategy:
          1) Reversibly compute lt = [phase < threshold_int] into ancilla self.lt_wire
             using a prefix-equality chain eq_wires.
          2) Apply a phase flip to the NOT-lt subspace (i.e., lt=0) via X-Z-X on lt.
          3) Uncompute lt and eq_wires back to |0...0>.

        Notes:
          - This oracle acts only on the phase register + ancillas; the system register is untouched.
          - You must set `self.threshold_int` before calling amplitude amplification, e.g.
                c.threshold_int = 12
        """
        if not hasattr(self, "threshold_int") or self.threshold_int is None:
            raise ValueError("Set `self.threshold_int` (an int in [0, 2**t)) before calling oracle().")

        t = self.t
        threshold_int = int(self.threshold_int)
        if threshold_int < 0 or threshold_int >= 2**t:
            raise ValueError(f"threshold_int must be in [0, 2**t). Got {threshold_int} for t={t}.")

        # Interpret phase_wires as MSB -> LSB for the comparator.
        phase = self.phase_wires
        lt = self.lt_wire
        eq = self.eq_wires  # length t+1

        # Threshold bits MSB -> LSB
        thr_bits = [(threshold_int >> (t - 1 - i)) & 1 for i in range(t)]

        # --- initialize eq[0]=1 (others are 0 by default), lt=0 by default ---
        qml.PauliX(eq[0])

        # --- forward pass: compute lt = [phase < threshold] and the eq prefix chain ---
        # eq[i]=1 means: the first i most significant bits of `phase` equal those of `threshold`.
        for i in range(t):
            # If threshold bit is 1, then (eq[i]=1 AND phase[i]=0) implies phase < threshold at this first differing bit.
            if thr_bits[i] == 1:
                # negative control on phase[i]=0 implemented by X on phase[i]
                qml.PauliX(phase[i])
                qml.Toffoli(wires=[eq[i], phase[i], lt])  # lt ^= eq[i] & (~phase[i]_original)
                qml.PauliX(phase[i])

            # Compute eq[i+1] = eq[i] & (phase[i] == thr_bits[i]) into eq[i+1] (XOR-style compute, but eq[i+1] starts at 0)
            if thr_bits[i] == 1:
                # need phase[i]=1 for equality with threshold bit 1
                qml.Toffoli(wires=[eq[i], phase[i], eq[i + 1]])
            else:
                # need phase[i]=0 for equality with threshold bit 0 (negative control)
                qml.PauliX(phase[i])
                qml.Toffoli(wires=[eq[i], phase[i], eq[i + 1]])
                qml.PauliX(phase[i])

        # --- phase flip on "good" subspace: good iff NOT lt (i.e., phase >= threshold) ---
        # X-Z-X on lt applies a -1 phase exactly when lt == 0.
        qml.PauliX(lt)
        qml.PauliZ(lt)
        qml.PauliX(lt)

        # --- uncompute: reverse pass to clean ancillas back to |0...0> ---
        for i in reversed(range(t)):
            # Uncompute eq[i+1]
            if thr_bits[i] == 1:
                qml.Toffoli(wires=[eq[i], phase[i], eq[i + 1]])
            else:
                qml.PauliX(phase[i])
                qml.Toffoli(wires=[eq[i], phase[i], eq[i + 1]])
                qml.PauliX(phase[i])

            # Uncompute lt toggle (only existed when thr_bits[i] == 1)
            if thr_bits[i] == 1:
                qml.PauliX(phase[i])
                qml.Toffoli(wires=[eq[i], phase[i], lt])
                qml.PauliX(phase[i])

        # reset eq[0] back to 0
        qml.PauliX(eq[0])

    def oracle_mark_phases(self, good_phases: Iterable[int]) -> None:
        """Phase-flip oracle that marks an explicit set of phase-register basis states.

        For small instances we can avoid comparator subtleties by directly marking the desired
        phase outcomes (integers in [0, 2**t)).

        Implementation: for each phase value p, temporarily map |p> -> |11..1> with X on 0-bits,
        toggle an ancilla via MCX, apply Z on that ancilla to kick back a phase, then uncompute.

        Notes:
          - Uses `self.lt_wire` as the work/target ancilla (it is returned to |0>).
          - Requires `good_phases` to be iterable of ints.
        """
        t = self.t
        phase = self.phase_wires
        anc = self.lt_wire

        for p in good_phases:
            p = int(p)
            if p < 0 or p >= 2 ** t:
                raise ValueError(f"good phase {p} out of range for t={t}")

            # Bits MSB->LSB (same convention as in the comparator oracle)
            bits = [(p >> (t - 1 - i)) & 1 for i in range(t)]

            # Map |p> -> |11..1>
            for i, b in enumerate(bits):
                if b == 0:
                    qml.PauliX(phase[i])

            # Flag match -> phase kickback -> uncompute
            qml.MultiControlledX(wires=phase + [anc], work_wires=None)
            qml.PauliZ(anc)
            qml.MultiControlledX(wires=phase + [anc], work_wires=None)

            # Undo mapping
            for i, b in enumerate(bits):
                if b == 0:
                    qml.PauliX(phase[i])

    @staticmethod
    def _bit_reverse(k: int, t: int) -> int:
        """Reverse the lowest-t bits of integer k."""
        out = 0
        for _ in range(t):
            out = (out << 1) | (k & 1)
            k >>= 1
        return out

    def choose_good_phases_top_eigen(self, neighborhood: int = 1) -> list[int]:
        """Pick likely QPE phase outcomes for the top eigenvalue of H (small instances).

        For U = exp(-i H tau), an eigenvalue λ maps to eigenphase φ = (-λ * tau) mod 2π.
        QPE ideally returns an integer close to frac * 2^t where frac = φ/(2π).

        Because endianness / QFT conventions can vary, include both the direct index and its
        bit-reversal, plus a small neighborhood.
        """
        evals, _ = np.linalg.eigh(self.H)
        lam = float(np.max(evals))

        phi = (-lam * float(self.tau)) % (2 * np.pi)
        frac = phi / (2 * np.pi)

        k = int(np.round(frac * (2 ** self.t))) % (2 ** self.t)
        k_br = self._bit_reverse(k, self.t)

        cand = set()
        for base in (k, k_br):
            for d in range(-int(neighborhood), int(neighborhood) + 1):
                cand.add((base + d) % (2 ** self.t))

        out = sorted(cand)
        self.logger.debug(
            "Picked good phases around top eigenvalue (n=%d, neighborhood=%d)",
            len(out),
            int(neighborhood),
        )
        return out

    def reflect_zero(self) -> None:
        r"""
        Reflection about the |0...0> state on phase+system qubits.

        Implements: R0 = 2|0...0><0...0| - I

        Standard construction:
          X^{\otimes n} · (2|1...1><1...1| - I) · X^{\otimes n}
        where the middle reflection is implemented via a multi-controlled-Z on |1...1>.

        Note: this is the standard reflection used for amplitude amplification.
        """
        wires = self.phase_wires + self.sys_wires

        # Map |0...0> -> |1...1>
        for w in wires:
            qml.PauliX(w)

        # Apply a phase flip (-1) to |1...1> using an MCZ implemented via H-MCX-H on the target.
        target = wires[-1]
        qml.Hadamard(target)
        qml.MultiControlledX(wires=wires, work_wires=None)  # controls=wires[:-1], target=wires[-1]
        qml.Hadamard(target)

        # Unmap back
        for w in wires:
            qml.PauliX(w)

    def _reshape_state(self, state: np.ndarray) -> np.ndarray:
        """Reshape a flat statevector into a tensor with one axis per wire (wire order = 0..N-1)."""
        n_wires = len(self.dev.wires)
        return np.reshape(state, (2,) * n_wires)

    def extract_system_vector_from_state(
        self,
        state: np.ndarray,
        good_phases: Iterable[int] | None = None,
        require_ancillas_zero: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Toy-size helper: extract a *classical* vector for the system register by postselecting
        on a set of 'good' phase outcomes.

        Parameters
        ----------
        state : np.ndarray
            Full statevector returned by qml.state().
        good_phases : iterable[int] or None
            Phase outcomes (0..2**t-1) to keep. If None, uses `range(self.threshold_int, 2**self.t)`.
            (This matches the current oracle definition: good iff phase >= threshold_int.)
        require_ancillas_zero : bool
            If True, we additionally postselect ancillas to be all-zeros. This should be valid if
            the oracle uncomputation was correct.
        normalize : bool
            If True, normalize the extracted system vector.

        Returns
        -------
        sys_vec : np.ndarray of shape (2**y,)
            The extracted (postselected and phase-summed) system-register amplitudes.
        """
        if good_phases is None:
            if not hasattr(self, "threshold_int") or self.threshold_int is None:
                raise ValueError("Set self.threshold_int or pass good_phases explicitly.")
            good_phases = range(int(self.threshold_int), 2 ** self.t)

        # Layout is: [phase wires][system wires][ancillas]
        t = self.t
        y = self.y
        anc = 1 + (t + 1)  # lt + eq_wires

        tensor = self._reshape_state(state)

        # Group axes into (phase, system, ancillas)
        tensor = np.reshape(tensor, (2 ** t, 2 ** y, 2 ** anc))

        # Optionally postselect ancillas = |0...0>
        if require_ancillas_zero:
            tensor = tensor[:, :, 0]

        # Sum amplitudes over the selected phase outcomes to get a system vector.
        sys_vec = np.zeros((2 ** y,), dtype=complex)
        for p in good_phases:
            sys_vec += tensor[p, :]

        if normalize:
            nrm = np.linalg.norm(sys_vec)
            if nrm > 0:
                sys_vec = sys_vec / nrm

        self.logger.debug(
            "Extracted system vector (dim=%d, normalized=%s, ancillas_zero=%s)",
            int(sys_vec.size),
            bool(normalize),
            bool(require_ancillas_zero),
        )

        return sys_vec

    def amplitude_amplification(self, n_iters: int, return_state: bool = False, good_phases=None):
        """Build the AA QNode around the QPE block.

        If return_state is True we return the full statevector (toy sizes only).
        """
        try:
            n_good = None if good_phases is None else len(good_phases)
        except TypeError:
            n_good = "?"

        self.logger.debug(
            "Building AA QNode (n_iters=%d, return_state=%s, good_phases=%s)",
            int(n_iters),
            bool(return_state),
            n_good,
        )

        @qml.qnode(self.dev)
        def amp_amp_circuit():
            """Run QPE, then n_iters Grover iterations, then measure."""
            # Apply initial QPE block (operator A)
            self._qpe()

            for _ in range(n_iters):
                # Grover iterate: apply oracle (phase flip of good states)
                if good_phases is None:
                    self.oracle()
                else:
                    self.oracle_mark_phases(good_phases)

                # Apply A† (inverse QPE block)
                qml.adjoint(self._qpe)()

                # Reflection about |0...0> state
                self.reflect_zero()

                # Apply A (QPE block) again
                self._qpe()

            # Return either the full statevector (toy sizes) or the phase-register probs
            if return_state:
                return qml.state()
            return qml.probs(wires=self.phase_wires)

        return amp_amp_circuit

    # --- stage-2 utilities (second circuit)/KEY EXTRACTION ---

    @staticmethod
    def _normalize_global_phase(vec: np.ndarray) -> np.ndarray:
        """Fix a consistent global phase for a statevector (so comparisons are stable)."""
        vec = np.asarray(vec, dtype=complex)
        k = np.argmax(np.abs(vec))
        if np.abs(vec[k]) == 0:
            return vec
        return vec * np.exp(-1j * np.angle(vec[k]))

    def stage2_circuit_from_voting_matrix(
        self,
        V: np.ndarray,
        tau2: float | None = None,
        phase_qubits: int | None = None,
        good_phases=None,
        n_iters: int = 1,
        return_state: bool = False,
    ):
        """
        Build and return a *second* Circuit instance that runs QPE (+ optional amplitude amplification)
        for H := V (the voting matrix), on a fresh device with its own wires/ancillas.

        Parameters
        ----------
        V : np.ndarray
            Voting matrix; must be square Hermitian with dimension 2**y2 for some integer y2.
        tau2 : float or None
            Evolution time for stage-2 Hamiltonian simulation. Defaults to self.tau.
        phase_qubits : int or None
            Number of QPE phase qubits for stage-2. Defaults to self.t.
        good_phases : iterable[int] or None
            If provided, amplitude amplification will mark exactly these phases.
            If None, the comparator-threshold oracle is used (requires setting c2.threshold_int).
        n_iters : int
            Number of amplitude amplification iterations in stage-2.
        return_state : bool
            Whether the returned qnode returns full statevector (toy) or phase probs.

        Returns
        -------
        c2 : Circuit
            The stage-2 circuit object.
        qnode : callable
            The stage-2 qnode (either plain QPE if n_iters==0, else amplitude amplification qnode).
        """
        V = np.asarray(V, dtype=complex)
        if V.ndim != 2 or V.shape[0] != V.shape[1]:
            raise ValueError("V must be a square matrix.")
        if not np.allclose(V, V.conj().T, atol=1e-8):
            raise ValueError("V must be Hermitian (within numerical tolerance).")

        dim = V.shape[0]
        y2 = int(np.log2(dim))
        if 2**y2 != dim:
            raise ValueError(f"V dimension must be a power of 2. Got {dim}.")

        t2 = int(self.t if phase_qubits is None else phase_qubits)
        tau2 = float(self.tau if tau2 is None else tau2)

        self.logger.debug(
            "Building stage-2 circuit (dim=%d, y2=%d, t2=%d, use_aa=%s)",
            int(dim),
            int(y2),
            int(t2),
            bool(n_iters and n_iters > 0),
        )

        c2 = Circuit(H=V, t=t2, y=y2, tau=tau2)

        # c2 already exposes the correctly bound QPE block as a plain callable.
        qpe2 = c2._qpe

        use_aa = bool(n_iters and n_iters > 0)

        if good_phases is None:
            oracle2 = c2.oracle
        else:
            oracle2 = functools.partial(c2.oracle_mark_phases, good_phases)

        qnode = qml.QNode(
            functools.partial(
                _qpe_aa_template,
                qpe=qpe2,
                use_aa=use_aa,
                n_iters=int(n_iters),
                oracle=oracle2,
                reflect_zero=c2.reflect_zero,
                return_state=return_state,
                phase_wires=c2.phase_wires,
            ),
            c2.dev,
        )

        return c2, qnode

if __name__ == "__main__":
    raise SystemExit(
        "This module is intended to be imported. "
        "Run the demo script instead: `python tests/manual/quartic_schmidhuber_demo.py`"
    )
"""Shared tests for the state backends (NumPy statevector, QSim, STIM).

Each backend gets a small adapter that builds states and reps with its own
constructors, and reads a state back through ``probe``, a Z-basis measurement
of every qubit on a copy. Backend-specific behaviour lives in each backend's
own test file.
"""

import copy

import numpy as np
import pytest

from loqs.backends.reps import (
    KrausGateRep,
    OutcomeOperationDictInstrumentRep,
    ProbabilisticStimGateRep,
    QSimSuperopGateRep,
    StimCircuitGateRep,
    StimCircuitInstrumentRep,
    UnitaryGateRep,
    ZBasisPrePostInstrumentRep,
    ZBasisProjectionInstrumentRep,
)

LABELS = ["Q0", "Q1"]
SEED = 20241016


class _Adapter:
    """Builds states and reps for one backend."""

    cls = None
    input_reps: set = set()

    def make(self, n_or_bits, labels, seed=None):
        raise NotImplementedError

    def gate(self, name, qubits):
        raise NotImplementedError

    def labels(self, state):
        return list(state.qubit_labels)

    def flush(self, state):
        """Force any operations a backend defers to be applied."""

    def probe(self, state):
        """Z-basis outcome of every qubit, measured on a copy."""
        labels = self.labels(state)
        rep = ZBasisProjectionInstrumentRep(None, True, labels)
        outs = state.copy().apply_reps_inplace([rep])
        return [outs[label][0] for label in labels]


class _NpsvAdapter(_Adapter):
    from loqs.backends import NumpyStatevectorQuantumState as cls

    input_reps = {
        UnitaryGateRep,
        KrausGateRep,
        ZBasisProjectionInstrumentRep,
        ZBasisPrePostInstrumentRep,
        OutcomeOperationDictInstrumentRep,
    }
    _H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    _MATS = {
        "X": np.array([[0, 1], [1, 0]]),
        "H": _H,
        "I": np.eye(2),
        "CZ": np.diag([1, 1, 1, -1]),
        "P0": np.diag([1.0, 0.0]),
        "P1": np.diag([0.0, 1.0]),
        "R10": np.array([[0.0, 1.0], [0.0, 0.0]]),
        "HP0": _H @ np.diag([1.0, 0.0]),
        "HR10": _H @ np.array([[0.0, 1.0], [0.0, 0.0]]),
    }

    def make(self, n_or_bits, labels, seed=None):
        return self.cls(n_or_bits, labels, seed=seed)

    def gate(self, name, qubits):
        return UnitaryGateRep(self._MATS[name], qubits)

    def populations(self, state):
        return np.abs(state.state.flatten()) ** 2

    foreign_gate = StimCircuitGateRep("I 0", ["Q0"])
    foreign_instrument = StimCircuitInstrumentRep("M 0", ["Q0"])


class _QSimAdapter(_Adapter):
    def __init__(self):
        from quantumsim import ptm

        from loqs.backends import QSimQuantumState

        self.cls = QSimQuantumState
        self.input_reps = {
            QSimSuperopGateRep,
            ZBasisProjectionInstrumentRep,
            ZBasisPrePostInstrumentRep,
            OutcomeOperationDictInstrumentRep,
        }
        effect0 = np.array([[1, 0, 0, 0]])
        effect1 = np.array([[0, 0, 0, 1]])
        r10 = effect0.T @ effect1
        self._mats = {
            "X": ptm.rotate_x_ptm(np.pi),
            "H": ptm.hadamard_ptm(),
            "I": ptm.rotate_x_ptm(0),
            "CZ": QSimQuantumState(2).state._cphase_ptm,
            "P0": effect0.T @ effect0,
            "P1": effect1.T @ effect1,
            "R10": r10,
            "HP0": ptm.hadamard_ptm() @ (effect0.T @ effect0),
            "HR10": ptm.hadamard_ptm() @ r10,
        }
        self.foreign_gate = UnitaryGateRep(np.eye(2), ["Q0"])
        self.foreign_instrument = StimCircuitInstrumentRep("M 0", ["Q0"])

    def make(self, n_or_bits, labels, seed=None):
        if isinstance(n_or_bits, int):
            return self.cls(n_or_bits, labels, seed=seed)
        state = self.cls(len(n_or_bits), labels, seed=seed)
        for label, bit in zip(labels, n_or_bits):
            if bit:
                state.state.set_bit(label, 1)
        return state

    def gate(self, name, qubits):
        return QSimSuperopGateRep(self._mats[name], qubits)

    def labels(self, state):
        return list(state.state.names)

    def flush(self, state):
        for label in self.labels(state):
            state.state.combine_and_apply_single_ptm(label)

    def populations(self, state):
        test = state.copy()
        self.flush(test)
        for label in self.labels(test):
            test.state.ensure_dense(label)
        return test.state.full_dm.get_diag()


class _StimAdapter(_Adapter):
    def __init__(self):
        from loqs.backends import STIMQuantumState

        self.cls = STIMQuantumState
        self.input_reps = {
            StimCircuitGateRep,
            ProbabilisticStimGateRep,
            ZBasisProjectionInstrumentRep,
            ZBasisPrePostInstrumentRep,
            StimCircuitInstrumentRep,
        }
        self.foreign_gate = UnitaryGateRep(np.eye(2), ["Q0"])
        self.foreign_instrument = OutcomeOperationDictInstrumentRep(
            {}, True, ["Q0"]
        )

    def make(self, n_or_bits, labels, seed=None):
        return self.cls(n_or_bits, labels, seed=seed)

    def gate(self, name, qubits):
        text = {"X": "X 0", "H": "H 0", "I": "", "CZ": "CZ 0 1"}[name]
        return StimCircuitGateRep(text, qubits)


def _build(name):
    if name == "npsv":
        return _NpsvAdapter()
    if name == "qsim":
        pytest.importorskip("quantumsim")
        return _QSimAdapter()
    pytest.importorskip("stim")
    return _StimAdapter()


@pytest.fixture(params=["npsv", "qsim", "stim"])
def adapter(request):
    return _build(request.param)


@pytest.fixture(params=["npsv", "qsim"])
def odict_adapter(request):
    return _build(request.param)


def _batch_outcomes(adapter, reps, seed=SEED):
    """Outcomes of `reps` applied to a fresh |0> state with a given seed."""
    state = adapter.make(1, ["Q0"], seed=seed)
    return state.apply_reps_inplace(reps)["Q0"]


def test_init_and_copy(adapter):
    assert adapter.probe(adapter.make(2, LABELS)) == [0, 0]

    source = adapter.make([1, 0], LABELS)
    assert adapter.probe(adapter.cls(source, LABELS)) == [1, 0]

    for dup in (source.copy(), copy.deepcopy(source)):
        assert dup is not source
        assert adapter.probe(dup) == [1, 0]
        # Editing the copy must leave the source alone
        dup.apply_reps_inplace([adapter.gate("X", ["Q1"])])
        assert adapter.probe(dup) == [1, 1]
        assert adapter.probe(source) == [1, 0]


def test_apply_gates(adapter):
    x = adapter.gate("X", ["Q0"])
    test = adapter.make(1, ["Q0"])
    test.apply_reps_inplace([x])
    assert adapter.probe(test) == [1]

    test2, outcomes = adapter.make(1, ["Q0"]).apply_reps([x])
    assert adapter.probe(test2) == [1]
    assert len(outcomes) == 0

    # CNOT as H CZ H, control Q0 and target Q1
    cnot = [
        adapter.gate("H", ["Q1"]),
        adapter.gate("CZ", LABELS),
        adapter.gate("H", ["Q1"]),
    ]
    test3, _ = adapter.make([1, 0], LABELS).apply_reps(cnot)
    assert adapter.probe(test3) == [1, 1]


def test_unsupported_rep_raises(adapter):
    state = adapter.make(1, ["Q0"])
    with pytest.raises(NotImplementedError):
        state.apply_reps([adapter.foreign_gate])
    with pytest.raises(NotImplementedError):
        state.apply_reps_inplace([adapter.foreign_instrument])


def test_input_reps(adapter):
    assert set(adapter.make(1, ["Q0"]).input_reps) == adapter.input_reps


def test_apply_instruments(adapter):
    q = ["Q0"]
    h = adapter.gate("H", q)
    idle = adapter.gate("I", q)
    proj = ZBasisProjectionInstrumentRep(None, True, q)
    reset = ZBasisProjectionInstrumentRep(0, True, q)

    # A projection without reset leaves the state in the measured outcome
    state = adapter.make(1, q, seed=SEED)
    for _ in range(10):
        outs = state.apply_reps_inplace([h, proj])
        assert adapter.probe(state) == outs["Q0"]

    # The same seed gives the same outcomes in a loop and in one batch
    looped = adapter.make(1, q, seed=SEED)
    outcomes = []
    for _ in range(10):
        outcomes += looped.apply_reps_inplace([h, reset])["Q0"]
    assert adapter.probe(looped) == [0]
    assert _batch_outcomes(adapter, [h, reset] * 10) == outcomes

    # Pre-op and post-op variants
    pre = ZBasisPrePostInstrumentRep(0, True, h, idle, q)
    assert _batch_outcomes(adapter, [pre] * 10) == outcomes
    post = ZBasisPrePostInstrumentRep(0, True, idle, h, q)
    assert _batch_outcomes(adapter, [h] + [post] * 10) == outcomes

    # Without outcomes, nothing is recorded
    proj_no = ZBasisProjectionInstrumentRep(None, False, q)
    pre_no = ZBasisPrePostInstrumentRep(0, False, h, idle, q)
    for reps in ([h, proj_no] * 10, [pre_no] * 10):
        state = adapter.make(1, q, seed=SEED)
        assert len(state.apply_reps_inplace(reps)) == 0


@pytest.mark.parametrize("reset_to", [0, 1])
def test_projection_reset(adapter, reset_to):
    hs = [adapter.gate("H", ["Q0"]), adapter.gate("H", ["Q1"])]
    rep = ZBasisProjectionInstrumentRep(reset_to, True, LABELS)

    state = adapter.make(2, LABELS, seed=SEED)
    for _ in range(5):
        outs = state.apply_reps_inplace(hs + [rep])
        assert set(outs.keys()) == set(LABELS)
        assert len(outs["Q0"]) == 1 and len(outs["Q1"]) == 1
        assert adapter.probe(state) == [reset_to, reset_to]


def test_odict_instrument(odict_adapter):
    a = odict_adapter
    q = ["Q0"]
    h = a.gate("H", q)
    reset = ZBasisProjectionInstrumentRep(0, True, q)
    expected = _batch_outcomes(a, [h, reset] * 10)

    # An ideal projector map records the same outcomes as a projection
    ideal = OutcomeOperationDictInstrumentRep(
        {0: a.gate("P0", q), 1: a.gate("P1", q)}, True, q
    )
    assert _batch_outcomes(a, [h, ideal] * 10) == expected

    # A map that resets to 0 and applies a gate after each outcome
    noisy_reset = OutcomeOperationDictInstrumentRep(
        {0: a.gate("HP0", q), 1: a.gate("HR10", q)}, True, q
    )
    assert _batch_outcomes(a, [h] + [noisy_reset] * 10) == expected

    # Without outcomes nothing is recorded, and the state still collapses
    ideal_no = OutcomeOperationDictInstrumentRep(
        {0: a.gate("P0", q), 1: a.gate("P1", q)}, False, q
    )
    state = a.make(1, q, seed=SEED)
    assert len(state.apply_reps_inplace([h, ideal_no])) == 0
    assert np.isclose(max(a.populations(state)), 1)


def test_serialization_round_trip(adapter, make_temp_path):
    # CNOT as H CZ H, split by a write and read after the CZ
    state = adapter.make([1, 0], LABELS)
    state.apply_reps_inplace([adapter.gate("H", ["Q1"])])
    adapter.flush(state)
    state.apply_reps_inplace([adapter.gate("CZ", LABELS)])
    # QSim defers single-qubit gates, so this X is still pending at write
    state.apply_reps_inplace([adapter.gate("X", ["Q0"])])

    with make_temp_path(suffix=".json") as tmp_path:
        state.write(tmp_path)
        loaded = adapter.cls.read(tmp_path)

    assert isinstance(loaded, adapter.cls)
    loaded.apply_reps_inplace([adapter.gate("H", ["Q1"])])
    assert adapter.probe(loaded) == [0, 1]


def test_serialization_after_instrument(adapter, make_temp_path):
    # The measurement outcome itself is a random draw, so only the reset
    # value is checked
    state = adapter.make(1, ["Q0"], seed=SEED)
    state.apply_reps_inplace(
        [
            adapter.gate("H", ["Q0"]),
            ZBasisProjectionInstrumentRep(0, True, ["Q0"]),
        ]
    )
    adapter.flush(state)

    with make_temp_path(suffix=".json") as tmp_path:
        state.write(tmp_path)
        loaded = adapter.cls.read(tmp_path)

    assert adapter.probe(loaded) == [0]

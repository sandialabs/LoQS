"""Tester for loqs.backends.model.pygstimodel"""

import numpy as np
import pytest

pygsti = pytest.importorskip("pygsti")

from pygsti.baseobjs import Label
from pygsti.baseobjs.statespace import QubitSpace
from pygsti.modelmembers.instruments import Instrument
from pygsti.modelmembers.operations import FullArbitraryOp
from pygsti.models import ExplicitOpModel

from loqs.backends.model.dictmodel import DictNoiseModel
from loqs.backends.model.pygstimodel import PyGSTiNoiseModel
from loqs.backends.reps import GateRep, InstrumentRep

# TP-preserving (sum_i K_i K_i^dagger = I) amplitude-damping Kraus operators
# -- a genuinely non-unitary channel, used to exercise the GateRep fallback
# loop (GateRep.UNITARY fails for it; GateRep.PTM/QSIM_SUPEROPERATOR/
# KRAUS_OPERATORS succeed).
_GAMMA = 0.1
_K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1 - _GAMMA)]], dtype=complex)
_K1 = np.array([[0.0, np.sqrt(_GAMMA)], [0.0, 0.0]], dtype=complex)
_AMP_DAMP_SUPEROP = FullArbitraryOp.from_kraus_operators([_K0, _K1], "pp").to_dense()

_ZBASIS_P0 = np.diag([1, 0, 0, 1]).astype(complex)
_ZBASIS_P1 = np.diag([0, 1, 1, 0]).astype(complex)


def _build_explicit_model():
    model = ExplicitOpModel(state_space=QubitSpace(["Q0"]), basis="pp")
    model.operations[Label("Gxpi", "Q0")] = FullArbitraryOp(np.eye(4), basis="pp")
    model.operations[Label("Gad", "Q0")] = FullArbitraryOp(_AMP_DAMP_SUPEROP, basis="pp")
    model.instruments[Label("Iz", "Q0")] = Instrument(
        {"0": _ZBASIS_P0, "1": _ZBASIS_P1}
    )
    return model


def _build_implicit_model():
    pspec = pygsti.processors.QubitProcessorSpec(
        1,
        gate_names=["Gxpi"],
        qubit_labels=["Q0"],
        availability={"Gxpi": "all-permutations"},
    )
    model = pygsti.models.create_crosstalk_free_model(pspec)
    model.operation_blks["layers"][Label("Gad", "Q0")] = FullArbitraryOp(
        _AMP_DAMP_SUPEROP, basis="pp"
    )
    model.instrument_blks["layers"][Label("Iz", "Q0")] = Instrument(
        {"0": _ZBASIS_P0, "1": _ZBASIS_P1}
    )
    return model


class TestConstruction:
    def test_from_explicit_op_model(self):
        pgm = PyGSTiNoiseModel(_build_explicit_model())
        assert ("Gxpi", ["Q0"]) in pgm.gate_keys
        assert ("Iz", ["Q0"]) in pgm.instrument_keys
        assert pgm.use_embedded_op is False

    def test_from_implicit_op_model(self):
        pgm = PyGSTiNoiseModel(_build_implicit_model())
        assert ("Gxpi", ["Q0"]) in pgm.gate_keys
        assert ("Iz", ["Q0"]) in pgm.instrument_keys
        assert pgm.use_embedded_op is True

    def test_default_qubit_aliases_are_identity(self):
        pgm = PyGSTiNoiseModel(_build_explicit_model())
        assert pgm.qubit_aliases == {"Q0": "Q0"}

    def test_qubit_aliases_mapping_renames_qubits(self):
        pgm = PyGSTiNoiseModel(_build_explicit_model(), qubit_aliases={"Q0": "MyQubit"})
        assert pgm.gate_keys == [("Gxpi", ["MyQubit"]), ("Gad", ["MyQubit"])]

    def test_qubit_aliases_sequence_renames_qubits(self):
        pgm = PyGSTiNoiseModel(_build_explicit_model(), qubit_aliases=["MyQubit"])
        assert pgm.gate_keys == [("Gxpi", ["MyQubit"]), ("Gad", ["MyQubit"])]

    def test_invalid_qubit_aliases_type_raises(self):
        with pytest.raises(TypeError, match="Invalid type for qubit aliases"):
            PyGSTiNoiseModel(_build_explicit_model(), qubit_aliases=42)

    def test_non_conforming_qubit_label_without_alias_raises(self):
        model = ExplicitOpModel(state_space=QubitSpace(["A0"]), basis="pp")
        model.operations[Label("Gxpi", "A0")] = FullArbitraryOp(np.eye(4), basis="pp")
        with pytest.raises(AssertionError, match="Model must use int or str"):
            PyGSTiNoiseModel(model)

    def test_qubit_aliases_do_not_bypass_the_label_format_check(self):
        """Documents a discovered discrepancy: the code comment above this
        assertion (`pygstimodel.py` ~line 320) says the label-format check
        exists "but allow[s] qubit label aliasing" for non-conforming
        labels -- but the assertion actually checks the model's *raw*
        `state_space.qubit_labels` unconditionally, before `qubit_aliases`
        is even considered. So providing `qubit_aliases` does NOT let you
        use a model with non-conforming raw qubit labels (e.g. `"A0"`); it
        only lets you rename already-conforming labels (see
        `test_qubit_aliases_mapping_renames_qubits` above). This test pins
        down that current (arguably surprising, contrary to the docstring)
        behavior; not fixed here since it wasn't part of this pass's scope.
        """
        model = ExplicitOpModel(state_space=QubitSpace(["A0"]), basis="pp")
        model.operations[Label("Gxpi", "A0")] = FullArbitraryOp(np.eye(4), basis="pp")
        with pytest.raises(AssertionError, match="Model must use int or str"):
            PyGSTiNoiseModel(model, qubit_aliases={"A0": "Q0"})

    # The next three tests document a single, shared discovered bug: the
    # qubit-label-format assertion (`pygstimodel.py` ~line 324,
    # `model.state_space.qubit_labels`) runs *unconditionally*, before the
    # later `if isinstance(model, ExplicitOpModel): ... elif ...: ...`
    # dispatch chain (~line 354 onwards) that's clearly intended to handle
    # `PyGSTiNoiseModel` (copy), `DictNoiseModel` (NotImplementedError), and
    # any other type (TypeError with a specific message) as special cases.
    # Because none of those three input types have a `.state_space`
    # attribute of their own, every one of them instead raises an opaque,
    # unrelated `AttributeError` from line 324 -- the entire dispatch chain
    # below `ExplicitOpModel`/`ImplicitOpModel` is effectively dead code.
    # Not fixed here since it wasn't part of this pass's scope; pinned down
    # so a future fix has regression tests to flip from
    # `pytest.raises(AttributeError)` to the actually-intended behavior.

    def test_copy_constructor_is_currently_broken(self):
        pgm = PyGSTiNoiseModel(_build_explicit_model())
        with pytest.raises(AttributeError, match="state_space"):
            PyGSTiNoiseModel(pgm)

    def test_invalid_model_type_error_message_is_currently_unreachable(self):
        with pytest.raises(AttributeError, match="state_space"):
            PyGSTiNoiseModel(42)

    def test_dictnoisemodel_not_implemented_error_is_currently_unreachable(self):
        with pytest.raises(AttributeError, match="state_space"):
            PyGSTiNoiseModel(DictNoiseModel(({}, {})))


class TestGetGateRep:
    @pytest.fixture
    def pgm(self):
        return PyGSTiNoiseModel(_build_explicit_model())

    def test_unitary(self, pgm):
        rep, reptype = pgm._get_gate_rep("Gxpi", ["Q0"], [GateRep.UNITARY])
        assert reptype == GateRep.UNITARY
        assert np.shape(rep) == (2, 2)

    def test_kraus_operators(self, pgm):
        rep, reptype = pgm._get_gate_rep("Gad", ["Q0"], [GateRep.KRAUS_OPERATORS])
        assert reptype == GateRep.KRAUS_OPERATORS
        assert len(rep) >= 1

    def test_ptm(self, pgm):
        rep, reptype = pgm._get_gate_rep("Gad", ["Q0"], [GateRep.PTM])
        assert reptype == GateRep.PTM
        assert np.shape(rep) == (4, 4)

    def test_qsim_superoperator(self, pgm):
        rep, reptype = pgm._get_gate_rep("Gad", ["Q0"], [GateRep.QSIM_SUPEROPERATOR])
        assert reptype == GateRep.QSIM_SUPEROPERATOR
        assert np.shape(rep) == (4, 4)

    def test_fallback_skips_failing_candidate(self, pgm):
        """The priority-ordered fallback loop this refactor (issue 72)
        specifically wants to replace with a dedicated exception: request
        GateRep.UNITARY first (fails for the non-unitary amplitude-damping
        channel) and confirm it falls through to GateRep.QSIM_SUPEROPERATOR
        rather than raising."""
        rep, reptype = pgm._get_gate_rep(
            "Gad", ["Q0"], [GateRep.UNITARY, GateRep.QSIM_SUPEROPERATOR]
        )
        assert reptype == GateRep.QSIM_SUPEROPERATOR
        assert np.shape(rep) == (4, 4)

    def test_no_valid_candidate_raises(self, pgm):
        with pytest.raises(ValueError, match="Failed to create gate rep for any of"):
            pgm._get_gate_rep("Gad", ["Q0"], [GateRep.UNITARY])


class TestGetInstrumentRep:
    @pytest.fixture
    def pgm(self):
        return PyGSTiNoiseModel(_build_explicit_model())

    def test_zbasis_projection(self, pgm):
        rep, reptype = pgm._get_instrument_rep(
            "Iz", ["Q0"], [InstrumentRep.ZBASIS_PROJECTION]
        )
        assert reptype == InstrumentRep.ZBASIS_PROJECTION
        reset, include_outcomes = rep
        assert reset == 0  # zbasis_proj_resets=True by default
        assert include_outcomes is True

    def test_zbasis_outcome_operation_dict(self, pgm):
        rep, reptype = pgm._get_instrument_rep(
            "Iz", ["Q0"], [InstrumentRep.ZBASIS_OUTCOME_OPERATION_DICT]
        )
        assert reptype == InstrumentRep.ZBASIS_OUTCOME_OPERATION_DICT
        outcome_dict, include_outcomes = rep
        assert include_outcomes is True
        assert set(outcome_dict.keys()) == {(0,), (1,)}

    def test_no_valid_candidate_raises(self, pgm):
        # STIM_CIRCUIT_STR is a real InstrumentRep member, but _get_rep's
        # if/elif only ever handles ZBASIS_PROJECTION/
        # ZBASIS_OUTCOME_OPERATION_DICT, so this exercises the same
        # "no candidate worked" final-failure path as the gate-rep case.
        with pytest.raises(
            ValueError, match="Failed to create instrument rep for any of"
        ):
            pgm._get_instrument_rep("Iz", ["Q0"], [InstrumentRep.STIM_CIRCUIT_STR])

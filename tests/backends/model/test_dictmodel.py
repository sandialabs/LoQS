"""Tester for loqs.backends.model.dictmodel"""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from loqs.backends.circuit.listcircuit import ListPhysicalCircuit
from loqs.backends.model.dictmodel import DictNoiseModel
from loqs.backends.reps import GateRep, InstrumentRep, RepTuple
from loqs.internal.serializable import Serializable

FIXTURES_DIR = Path(__file__).parent / "fixtures"

_UNITARY_1Q = np.eye(2)
_GAMMA = 0.1
_TP_K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1 - _GAMMA)]])
_TP_K1 = np.array([[0.0, 0.0], [np.sqrt(_GAMMA), 0.0]])
_KRAUS_SEQ = ((_TP_K0, None), (_TP_K1, None))
_PROB_STIM_SEQ = (("X 0", 0.5), ("Y 0", 0.5))


class TestConstruction:
    def test_from_tuple_of_dicts(self):
        model = DictNoiseModel(({}, {}))
        assert model.gate_dict == {}
        assert model.inst_dict == {}

    def test_copy_constructor(self):
        gate_dict = {("X", ("Q0",)): np.eye(4)}
        original = DictNoiseModel((gate_dict, {}))
        copy = DictNoiseModel(original)
        assert copy.gate_dict.keys() == original.gate_dict.keys()
        assert copy is not original
        assert copy.gate_dict is not original.gate_dict

    def test_invalid_type_raises_type_error(self):
        with pytest.raises(TypeError, match="Can only other NoiseModels"):
            DictNoiseModel(42)

    def test_pygsti_duck_typed_conversion(self):
        """`DictNoiseModel.__init__` dispatches on `type(...).__name__ ==
        "PyGSTiNoiseModel"` (not `isinstance`), specifically so it can
        convert a real `PyGSTiNoiseModel` without hard-requiring `pygsti` to
        be importable. A minimal duck-typed stand-in exercises the same
        code path without needing pygsti installed."""

        class FakeGateKey:
            def __init__(self, name, qubits):
                self.name = name
                self.qubits = qubits

        class PyGSTiNoiseModel:  # name matters -- see docstring above
            gate_keys = [FakeGateKey("X", ("Q0",))]
            instrument_keys = [FakeGateKey("M", ("Q0",))]

            def get_reps(self, circ, gatereps, instreps):
                label = circ.circuit[0][0]
                if label[0] == "X":
                    return [[RepTuple(np.eye(4), label[1], GateRep.QSIM_SUPEROPERATOR)]]
                return [[RepTuple((None, True), label[1], InstrumentRep.ZBASIS_PROJECTION)]]

        model = DictNoiseModel(PyGSTiNoiseModel())
        assert ("X", ("Q0",)) in model.gate_dict
        assert model.gate_dict[("X", ("Q0",))].reptype == GateRep.QSIM_SUPEROPERATOR
        assert ("M", ("Q0",)) in model.inst_dict
        assert model.inst_dict[("M", ("Q0",))].reptype == InstrumentRep.ZBASIS_PROJECTION


class TestGateDispatch:
    def test_ndarray_uses_gaterep_array_cast_rep(self):
        model = DictNoiseModel(
            ({("X", ("Q0",)): _UNITARY_1Q}, {}), gaterep_array_cast_rep=GateRep.UNITARY
        )
        assert model.gate_dict[("X", ("Q0",))].reptype == GateRep.UNITARY

    def test_ndarray_default_cast_rep_is_qsim_superoperator(self):
        model = DictNoiseModel(({("X", ("Q0",)): np.eye(4)}, {}))
        assert model.gate_dict[("X", ("Q0",))].reptype == GateRep.QSIM_SUPEROPERATOR

    def test_str_becomes_stim_circuit_str(self):
        model = DictNoiseModel(
            ({("X", ("Q0",)): "X 0"}, {}), gatereps=[GateRep.STIM_CIRCUIT_STR]
        )
        assert model.gate_dict[("X", ("Q0",))].reptype == GateRep.STIM_CIRCUIT_STR
        assert model.gate_dict[("X", ("Q0",))].rep == "X 0"

    def test_kraus_sequence_becomes_kraus_operators(self):
        model = DictNoiseModel(
            ({("X", ("Q0",)): _KRAUS_SEQ}, {}), gatereps=[GateRep.KRAUS_OPERATORS]
        )
        rt = model.gate_dict[("X", ("Q0",))]
        assert rt.reptype == GateRep.KRAUS_OPERATORS
        assert len(rt.rep) == 2

    def test_probabilistic_stim_sequence_becomes_probabilistic_stim_operations(self):
        model = DictNoiseModel(
            ({("X", ("Q0",)): _PROB_STIM_SEQ}, {}),
            gatereps=[GateRep.PROBABILISTIC_STIM_OPERATIONS],
        )
        rt = model.gate_dict[("X", ("Q0",))]
        assert rt.reptype == GateRep.PROBABILISTIC_STIM_OPERATIONS
        assert rt.rep == (("X 0", 0.5), ("Y 0", 0.5))

    def test_reptuple_passthrough(self):
        rt = RepTuple(np.eye(4), ("Q0",), GateRep.QSIM_SUPEROPERATOR)
        model = DictNoiseModel(({("X", ("Q0",)): rt}, {}))
        assert model.gate_dict[("X", ("Q0",))] is rt

    def test_reptuple_with_disallowed_reptype_raises(self):
        rt = RepTuple("X 0", ("Q0",), GateRep.STIM_CIRCUIT_STR)
        with pytest.raises(AssertionError, match="not provided gatereps"):
            DictNoiseModel(
                ({("X", ("Q0",)): rt}, {}), gatereps=[GateRep.QSIM_SUPEROPERATOR]
            )

    def test_sequence_matching_neither_validator_raises(self):
        with pytest.raises(AssertionError, match="failed to upgrade to a RepTuple"):
            DictNoiseModel(({("X", ("Q0",)): ("not", "a", "valid", "shape")}, {}))


class TestInstrumentDispatch:
    def test_bare_string_becomes_stim_circuit_str(self):
        model = DictNoiseModel(({}, {("M", ("Q0",)): "M 0"}))
        rt = model.inst_dict[("M", ("Q0",))]
        assert rt.reptype == InstrumentRep.STIM_CIRCUIT_STR
        assert rt.rep == "M 0"

    def test_zbasis_projection_tuple(self):
        model = DictNoiseModel(({}, {("M", ("Q0",)): (0, True)}))
        rt = model.inst_dict[("M", ("Q0",))]
        assert rt.reptype == InstrumentRep.ZBASIS_PROJECTION
        assert rt.rep == (0, True)

    def test_two_element_non_projection_becomes_pre_post_operations(self):
        model = DictNoiseModel(
            ({}, {("M", ("Q0",)): (_UNITARY_1Q, _UNITARY_1Q)}),
            gatereps=[GateRep.QSIM_SUPEROPERATOR],
            instreps=[InstrumentRep.ZBASIS_PRE_POST_OPERATIONS],
            instrep_cast_reset=0,
            instrep_cast_include_outcomes=False,
        )
        rt = model.inst_dict[("M", ("Q0",))]
        assert rt.reptype == InstrumentRep.ZBASIS_PRE_POST_OPERATIONS
        reset, include_outcomes, preop, postop = rt.rep
        assert reset == 0
        assert include_outcomes is False
        assert isinstance(preop, RepTuple) and isinstance(postop, RepTuple)

    def test_pre_post_operations_without_instrep_declared_raises(self):
        with pytest.raises(AssertionError, match="ZBASIS_PRE_POST_OPERATIONS not passed"):
            DictNoiseModel(
                ({}, {("M", ("Q0",)): (_UNITARY_1Q, _UNITARY_1Q)}),
                instreps=[InstrumentRep.ZBASIS_PROJECTION],
            )

    def test_mapping_becomes_outcome_operation_dict(self):
        model = DictNoiseModel(
            ({}, {("M", ("Q0",)): {0: _UNITARY_1Q, 1: _UNITARY_1Q}}),
            gatereps=[GateRep.QSIM_SUPEROPERATOR],
            instreps=[InstrumentRep.ZBASIS_OUTCOME_OPERATION_DICT],
        )
        rt = model.inst_dict[("M", ("Q0",))]
        assert rt.reptype == InstrumentRep.ZBASIS_OUTCOME_OPERATION_DICT
        outcome_dict, include_outcomes = rt.rep
        assert include_outcomes is True
        assert set(outcome_dict.keys()) == {0, 1}
        assert all(isinstance(v, RepTuple) for v in outcome_dict.values())

    def test_outcome_operation_dict_without_instrep_declared_raises(self):
        with pytest.raises(AssertionError, match="ZBASIS_OUTCOME_OPERATION_DICT not passed"):
            DictNoiseModel(
                ({}, {("M", ("Q0",)): {0: _UNITARY_1Q, 1: _UNITARY_1Q}}),
                instreps=[InstrumentRep.ZBASIS_PROJECTION],
            )

    def test_reptuple_passthrough(self):
        rt = RepTuple((None, True), ("Q0",), InstrumentRep.ZBASIS_PROJECTION)
        model = DictNoiseModel(({}, {("M", ("Q0",)): rt}))
        assert model.inst_dict[("M", ("Q0",))] is rt

    def test_reptuple_with_disallowed_reptype_raises(self):
        rt = RepTuple("M 0", ("Q0",), InstrumentRep.STIM_CIRCUIT_STR)
        with pytest.raises(AssertionError, match="reptype not in instreps"):
            DictNoiseModel(
                ({}, {("M", ("Q0",)): rt}), instreps=[InstrumentRep.ZBASIS_PROJECTION]
            )


class TestGetReps:
    def test_exact_label_match(self):
        model = DictNoiseModel(({("X", ("Q0",)): np.eye(4)}, {}))
        circuit = ListPhysicalCircuit([[("X", ("Q0",))]])
        reps = model.get_reps(circuit, [GateRep.QSIM_SUPEROPERATOR], [])
        assert len(reps) == 1
        assert reps[0].qubits == ("Q0",)

    def test_generic_name_only_fallback_for_gates(self):
        model = DictNoiseModel(({"X": np.eye(4)}, {}))
        circuit = ListPhysicalCircuit([[("X", ("Q1",))]])
        reps = model.get_reps(circuit, [GateRep.QSIM_SUPEROPERATOR], [])
        assert len(reps) == 1
        assert reps[0].qubits == ("Q1",)
        assert reps[0].reptype == GateRep.QSIM_SUPEROPERATOR

    def test_generic_name_only_fallback_for_instruments(self):
        model = DictNoiseModel(({}, {"M": (None, True)}))
        circuit = ListPhysicalCircuit([[("M", ("Q1",))]])
        reps = model.get_reps(circuit, [], [InstrumentRep.ZBASIS_PROJECTION])
        assert len(reps) == 1
        assert reps[0].qubits == ("Q1",)
        assert reps[0].reptype == InstrumentRep.ZBASIS_PROJECTION

    def test_lookup_failure_raises(self):
        model = DictNoiseModel(({}, {}))
        circuit = ListPhysicalCircuit([[("X", ("Q0",))]])
        with pytest.raises(AssertionError, match="Failed to look up"):
            model.get_reps(circuit, [], [])


class TestDictModelFixtureRoundTrip:
    """Round-trip `tests/backends/model/fixtures/dictmodel_v1.{json,h5}`
    (generated by `generate_model_fixtures.py` from the current code)."""

    @pytest.fixture(params=["json", "hdf5"])
    def decoded(self, request):
        if request.param == "json":
            with open(FIXTURES_DIR / "dictmodel_v1.json") as f:
                return Serializable.decode(json.load(f), format="json")
        else:
            with h5py.File(FIXTURES_DIR / "dictmodel_v1.h5", "r") as f:
                return Serializable.decode(f["root"], format="hdf5")

    def test_decodes_to_dictnoisemodel_with_correct_content(self, decoded):
        assert isinstance(decoded, DictNoiseModel)
        assert set(decoded.gate_dict.keys()) == {("X", ("Q0",)), ("KRAUS", ("Q0",))}
        assert set(decoded.inst_dict.keys()) == {("M", ("Q0",))}
        assert decoded.gate_dict[("X", ("Q0",))].reptype == GateRep.QSIM_SUPEROPERATOR
        assert decoded.gate_dict[("KRAUS", ("Q0",))].reptype == GateRep.KRAUS_OPERATORS
        assert decoded.inst_dict[("M", ("Q0",))].reptype == InstrumentRep.ZBASIS_PROJECTION
        # _from_decoded_attrs reconstructs _gatereps/_instreps via
        # GateRep(v)/InstrumentRep(v) -- confirm these are real enum
        # members, not raw ints, post-decode.
        assert decoded.output_gate_reps == [
            GateRep.QSIM_SUPEROPERATOR,
            GateRep.KRAUS_OPERATORS,
        ]
        assert all(isinstance(g, GateRep) for g in decoded.output_gate_reps)
        assert decoded.output_instrument_reps == [InstrumentRep.ZBASIS_PROJECTION]
        assert all(isinstance(i, InstrumentRep) for i in decoded.output_instrument_reps)

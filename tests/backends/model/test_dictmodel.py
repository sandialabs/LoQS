"""Tester for loqs.backends.model.dictmodel"""

import json
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

from loqs.backends.circuit.listcircuit import ListPhysicalCircuit
from loqs.backends.model.dictmodel import DictNoiseModel, add_command_aliases
from loqs.backends.model.stimdictmodel import STIMDictNoiseModel
from loqs.backends.reps import (
    GateRep,
    InstrumentRep,
    KrausGateRep,
    ProbabilisticStimGateRep,
    QSimSuperoperatorGateRep,
    StimCircuitGateRep,
    StimCircuitInstrumentRep,
    UnitaryGateRep,
    ZBasisOutcomeOperationDictInstrumentRep,
    ZBasisPrePostInstrumentRep,
    ZBasisProjectionInstrumentRep,
)
from loqs.internal.serializable import Serializable

FIXTURES_DIR = Path(__file__).parent / "fixtures"

_UNITARY_1Q = np.eye(2)
_GAMMA = 0.1
_TP_K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1 - _GAMMA)]])
_TP_K1 = np.array([[0.0, 0.0], [np.sqrt(_GAMMA), 0.0]])
_KRAUS_SEQ = ((_TP_K0, None), (_TP_K1, None))
_PROB_STIM_SEQ = (("X 0", 0.5), ("Y 0", 0.5))

try:
    import stim
    from loqs.backends.circuit.stimcircuit import STIMPhysicalCircuit

    NO_STIM = False
except ImportError:
    NO_STIM = True


class TestConstruction:
    def test_from_tuple_of_dicts(self):
        model = DictNoiseModel(({}, {}))
        assert model.gate_dict == {}
        assert model.inst_dict == {}

    def test_str(self):
        model = DictNoiseModel(({}, {}))
        assert str(model) == f"Physical {model.name} noise model\n"

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
                    return [[QSimSuperoperatorGateRep(np.eye(4), label[1])]]
                return [[ZBasisProjectionInstrumentRep(None, True, label[1])]]

        model = DictNoiseModel(PyGSTiNoiseModel())
        assert ("X", ("Q0",)) in model.gate_dict
        assert isinstance(
            model.gate_dict[("X", ("Q0",))], QSimSuperoperatorGateRep
        )
        assert ("M", ("Q0",)) in model.inst_dict
        assert isinstance(
            model.inst_dict[("M", ("Q0",))], ZBasisProjectionInstrumentRep
        )


class TestGateDispatch:
    def test_ndarray_uses_gaterep_array_cast_rep(self):
        model = DictNoiseModel(
            ({("X", ("Q0",)): _UNITARY_1Q}, {}),
            gaterep_array_cast_rep=UnitaryGateRep,
        )
        assert isinstance(model.gate_dict[("X", ("Q0",))], UnitaryGateRep)

    def test_ndarray_default_cast_rep_is_qsim_superoperator(self):
        model = DictNoiseModel(({("X", ("Q0",)): np.eye(4)}, {}))
        assert isinstance(
            model.gate_dict[("X", ("Q0",))], QSimSuperoperatorGateRep
        )

    def test_str_becomes_stim_circuit_str(self):
        model = DictNoiseModel(
            ({("X", ("Q0",)): "X 0"}, {}), gatereps=[StimCircuitGateRep]
        )
        rep = model.gate_dict[("X", ("Q0",))]
        assert isinstance(rep, StimCircuitGateRep)
        assert rep.circuit_str == "X 0"

    def test_kraus_sequence_becomes_kraus_operators(self):
        model = DictNoiseModel(
            ({("X", ("Q0",)): _KRAUS_SEQ}, {}), gatereps=[KrausGateRep]
        )
        rep = model.gate_dict[("X", ("Q0",))]
        assert isinstance(rep, KrausGateRep)
        assert len(rep.kraus_operators) == 2

    def test_probabilistic_stim_sequence_becomes_probabilistic_stim_operations(self):
        model = DictNoiseModel(
            ({("X", ("Q0",)): _PROB_STIM_SEQ}, {}),
            gatereps=[ProbabilisticStimGateRep],
        )
        rep = model.gate_dict[("X", ("Q0",))]
        assert isinstance(rep, ProbabilisticStimGateRep)
        assert rep.operations == (("X 0", 0.5), ("Y 0", 0.5))

    def test_rep_passthrough(self):
        rep = QSimSuperoperatorGateRep(np.eye(4), ("Q0",))
        model = DictNoiseModel(({("X", ("Q0",)): rep}, {}))
        assert model.gate_dict[("X", ("Q0",))] is rep

    def test_rep_with_disallowed_type_raises(self):
        rep = StimCircuitGateRep("X 0", ("Q0",))
        with pytest.raises(AssertionError, match="not provided gatereps"):
            DictNoiseModel(
                ({("X", ("Q0",)): rep}, {}),
                gatereps=[QSimSuperoperatorGateRep],
            )

    def test_sequence_matching_no_gaterep_raises(self):
        with pytest.raises(Exception, match="does not match any known rep class"):
            DictNoiseModel(({("X", ("Q0",)): ("not", "a", "valid", "shape")}, {}))


class TestInstrumentDispatch:
    def test_bare_string_becomes_stim_circuit_str(self):
        model = DictNoiseModel(({}, {("M", ("Q0",)): "M 0"}))
        rep = model.inst_dict[("M", ("Q0",))]
        assert isinstance(rep, StimCircuitInstrumentRep)
        assert rep.circuit_str == "M 0"

    def test_zbasis_projection_tuple(self):
        model = DictNoiseModel(({}, {("M", ("Q0",)): (0, True)}))
        rep = model.inst_dict[("M", ("Q0",))]
        assert isinstance(rep, ZBasisProjectionInstrumentRep)
        assert rep.reset == 0
        assert rep.include_outcome is True

    def test_two_element_non_projection_becomes_pre_post_operations(self):
        model = DictNoiseModel(
            ({}, {("M", ("Q0",)): (_UNITARY_1Q, _UNITARY_1Q)}),
            gatereps=[QSimSuperoperatorGateRep],
            instreps=[ZBasisPrePostInstrumentRep],
            instrep_cast_reset=0,
            instrep_cast_include_outcomes=False,
        )
        rep = model.inst_dict[("M", ("Q0",))]
        assert isinstance(rep, ZBasisPrePostInstrumentRep)
        assert rep.reset == 0
        assert rep.include_outcome is False
        assert isinstance(rep.pre_op, QSimSuperoperatorGateRep)
        assert isinstance(rep.post_op, QSimSuperoperatorGateRep)

    def test_pre_post_operations_without_instrep_declared_raises(self):
        with pytest.raises(
            AssertionError, match="ZBasisPrePostInstrumentRep not passed"
        ):
            DictNoiseModel(
                ({}, {("M", ("Q0",)): (_UNITARY_1Q, _UNITARY_1Q)}),
                instreps=[ZBasisProjectionInstrumentRep],
            )

    def test_mapping_becomes_outcome_operation_dict(self):
        model = DictNoiseModel(
            ({}, {("M", ("Q0",)): {0: _UNITARY_1Q, 1: _UNITARY_1Q}}),
            gatereps=[QSimSuperoperatorGateRep],
            instreps=[ZBasisOutcomeOperationDictInstrumentRep],
        )
        rep = model.inst_dict[("M", ("Q0",))]
        assert isinstance(rep, ZBasisOutcomeOperationDictInstrumentRep)
        assert rep.include_outcome is True
        assert set(rep.outcome_ops.keys()) == {0, 1}
        assert all(
            isinstance(v, QSimSuperoperatorGateRep)
            for v in rep.outcome_ops.values()
        )

    def test_outcome_operation_dict_without_instrep_declared_raises(self):
        with pytest.raises(
            AssertionError, match="ZBasisOutcomeOperationDictInstrumentRep not passed"
        ):
            DictNoiseModel(
                ({}, {("M", ("Q0",)): {0: _UNITARY_1Q, 1: _UNITARY_1Q}}),
                instreps=[ZBasisProjectionInstrumentRep],
            )

    def test_rep_passthrough(self):
        rep = ZBasisProjectionInstrumentRep(None, True, ("Q0",))
        model = DictNoiseModel(({}, {("M", ("Q0",)): rep}))
        assert model.inst_dict[("M", ("Q0",))] is rep

    def test_rep_with_disallowed_type_raises(self):
        rep = StimCircuitInstrumentRep("M 0", ("Q0",))
        with pytest.raises(AssertionError, match="reptype not in instreps"):
            DictNoiseModel(
                ({}, {("M", ("Q0",)): rep}),
                instreps=[ZBasisProjectionInstrumentRep],
            )


class TestGetReps:
    def test_exact_label_match(self):
        model = DictNoiseModel(({("X", ("Q0",)): np.eye(4)}, {}))
        circuit = ListPhysicalCircuit([[("X", ("Q0",))]])
        reps = model.get_reps(circuit, [QSimSuperoperatorGateRep], [])
        assert len(reps) == 1
        assert reps[0].qubits == ("Q0",)

    def test_generic_name_only_fallback_for_gates(self):
        model = DictNoiseModel(({"X": np.eye(4)}, {}))
        circuit = ListPhysicalCircuit([[("X", ("Q1",))]])
        reps = model.get_reps(circuit, [QSimSuperoperatorGateRep], [])
        assert len(reps) == 1
        assert reps[0].qubits == ("Q1",)
        assert isinstance(reps[0], QSimSuperoperatorGateRep)

    def test_generic_name_only_fallback_for_instruments(self):
        model = DictNoiseModel(({}, {"M": (None, True)}))
        circuit = ListPhysicalCircuit([[("M", ("Q1",))]])
        reps = model.get_reps(circuit, [], [ZBasisProjectionInstrumentRep])
        assert len(reps) == 1
        assert reps[0].qubits == ("Q1",)
        assert isinstance(reps[0], ZBasisProjectionInstrumentRep)

    def test_lookup_failure_raises(self):
        model = DictNoiseModel(({}, {}))
        circuit = ListPhysicalCircuit([[("X", ("Q0",))]])
        with pytest.raises(AssertionError, match="Failed to look up"):
            model.get_reps(circuit, [], [])


class TestDictModelFixtureRoundTrip:
    """Round-trip `tests/backends/model/fixtures/dictmodel_v1.{json,h5}`
    (frozen, pre-refactor bytes generated by `generate_model_fixtures.py`)
    to confirm old `DictNoiseModel` files -- including their bare
    `GateRep`/`InstrumentRep` enum-member `_gatereps`/`_instreps` tags,
    never wrapped in a `RepTuple` -- still decode correctly."""

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
        assert isinstance(
            decoded.gate_dict[("X", ("Q0",))], QSimSuperoperatorGateRep
        )
        assert isinstance(decoded.gate_dict[("KRAUS", ("Q0",))], KrausGateRep)
        assert isinstance(
            decoded.inst_dict[("M", ("Q0",))], ZBasisProjectionInstrumentRep
        )
        # _from_decoded_attrs reconstructs _gatereps/_instreps via
        # upgrade_legacy_gaterep_tag -- confirm these are real classes,
        # not raw ints or legacy enum tags, post-decode.
        assert decoded.output_gate_reps == [
            QSimSuperoperatorGateRep,
            KrausGateRep,
        ]
        assert decoded.output_instrument_reps == [ZBasisProjectionInstrumentRep]


@pytest.mark.skipif(NO_STIM, reason="Skipping STIM backend tests due to failed import")
class TestSTIMGetReps:
    """`DictNoiseModel.get_reps`'s `STIMPhysicalCircuit`-registered
    implementation (formerly `STIMDictNoiseModel.get_reps`)."""

    def test_init_basic(self):
        model = DictNoiseModel(({}, {}))
        assert isinstance(model.gate_dict, dict)
        assert isinstance(model.inst_dict, dict)

    def test_case_normalization_at_lookup_time(self):
        """STIM command names are looked up case-insensitively -- unlike
        the pre-refactor `STIMDictNoiseModel`, which normalized case at
        *construction* time, `DictNoiseModel` doesn't need a STIM-specific
        `__init__` override at all: normalization happens purely in the
        registered `get_reps` implementation, at lookup time."""
        model = DictNoiseModel(
            ({"x": "x 0", "h": "h 0"}, {}), gatereps=[StimCircuitGateRep]
        )
        # Keys are stored exactly as given -- no construction-time
        # normalization.
        assert "x" in model.gate_dict
        assert "X" not in model.gate_dict

        circuit = STIMPhysicalCircuit("X 0", ["Q0"])
        reps = model.get_reps(
            circuit, [StimCircuitGateRep], [ZBasisProjectionInstrumentRep]
        )
        assert len(reps) == 1
        assert reps[0].circuit_str == "x 0"

    def test_add_command_aliases_resolution_through_get_reps(self):
        """A gate registered only under its alias-source name (`CNOT`,
        aliased by `add_command_aliases` to `CX` at lookup time) must
        resolve when the input circuit uses the name STIM itself
        normalizes to (`CX`)."""
        gate_dict = {
            ("CNOT", ("Q0", "Q1")): StimCircuitGateRep(
                "CX 0 1", ("Q0", "Q1")
            ),
        }
        model = DictNoiseModel((gate_dict, {}), gatereps=[StimCircuitGateRep])
        circuit = STIMPhysicalCircuit("CNOT 0 1", ["Q0", "Q1"])
        reps = model.get_reps(
            circuit, [StimCircuitGateRep], [ZBasisProjectionInstrumentRep]
        )
        assert len(reps) == 1
        assert reps[0].circuit_str == "CX 0 1"

    def test_get_reps_basic_circuit(self):
        gate_dict = {"X": StimCircuitGateRep("X 0", ("Q0",))}
        model = DictNoiseModel((gate_dict, {}), gatereps=[StimCircuitGateRep])
        circuit = STIMPhysicalCircuit("X 0", ["Q0"])
        reps = model.get_reps(
            circuit, [StimCircuitGateRep], [ZBasisProjectionInstrumentRep]
        )
        assert len(reps) == 1
        assert isinstance(reps[0], StimCircuitGateRep)

    def test_get_reps_complex_circuit(self):
        gate_dict = {
            "X": StimCircuitGateRep("X 0", ("Q0",)),
            "H": StimCircuitGateRep("H 0", ("Q0",)),
            "CNOT": StimCircuitGateRep("CNOT 0 1", ("Q0", "Q1")),
        }
        model = DictNoiseModel((gate_dict, {}), gatereps=[StimCircuitGateRep])
        circuit = STIMPhysicalCircuit("H 0\nCNOT 0 1\nX 0", ["Q0", "Q1"])
        reps = model.get_reps(
            circuit, [StimCircuitGateRep], [ZBasisProjectionInstrumentRep]
        )
        assert len(reps) > 1
        for rep in reps:
            assert isinstance(rep, StimCircuitGateRep)

    def test_get_reps_with_instruments(self):
        gate_dict = {"X": StimCircuitGateRep("X 0", ("Q0",))}
        inst_dict = {
            "M": ZBasisProjectionInstrumentRep(None, True, ("Q0",)),
        }
        model = DictNoiseModel(
            (gate_dict, inst_dict),
            gatereps=[StimCircuitGateRep],
            instreps=[StimCircuitInstrumentRep, ZBasisProjectionInstrumentRep],
        )
        circuit = STIMPhysicalCircuit("X 0\nM 0", ["Q0"])
        reps = model.get_reps(
            circuit, [StimCircuitGateRep], [ZBasisProjectionInstrumentRep]
        )
        assert len(reps) == 2

    def test_warnings_for_noise_channels(self):
        model = DictNoiseModel(({}, {}))
        circuit = STIMPhysicalCircuit("X_ERROR(0.1) 0", ["Q0"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            reps = model.get_reps(
                circuit, [StimCircuitGateRep], [ZBasisProjectionInstrumentRep]
            )
            assert len(w) > 0
            assert "Noise channel" in str(w[0].message)
            assert len(reps) >= 1

    def test_warnings_for_measure_noise(self):
        inst_dict = {"M": ZBasisProjectionInstrumentRep(None, True, ("Q0",))}
        model = DictNoiseModel(
            ({}, inst_dict),
            instreps=[StimCircuitInstrumentRep, ZBasisProjectionInstrumentRep],
        )
        circuit = STIMPhysicalCircuit("M(0.1) 0", ["Q0"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.get_reps(
                circuit, [StimCircuitGateRep], [ZBasisProjectionInstrumentRep]
            )
            assert len(w) > 0
            assert "Measure noise" in str(w[0].message)

    def test_qubit_label_mapping(self):
        gate_dict = {"X": StimCircuitGateRep("X 0", ("Q0",))}
        model = DictNoiseModel((gate_dict, {}), gatereps=[StimCircuitGateRep])
        circuit = STIMPhysicalCircuit("X 0", ["A0"])
        reps = model.get_reps(
            circuit, [StimCircuitGateRep], [ZBasisProjectionInstrumentRep]
        )
        assert len(reps) == 1
        assert isinstance(reps[0], StimCircuitGateRep)

    def test_empty_circuit(self):
        model = DictNoiseModel(({}, {}))
        circuit = STIMPhysicalCircuit("", [])
        reps = model.get_reps(
            circuit, [StimCircuitGateRep], [ZBasisProjectionInstrumentRep]
        )
        assert isinstance(reps, list)

    def test_circuit_with_comments(self):
        """`stim` itself strips `#`-comment lines when parsing a circuit
        string, so `get_reps` never actually sees them (verified below via
        `circuit._unroll_repeats()`)."""
        gate_dict = {"X": StimCircuitGateRep("X 0", ("Q0",))}
        model = DictNoiseModel((gate_dict, {}), gatereps=[StimCircuitGateRep])
        circuit = STIMPhysicalCircuit("# This is a comment\nX 0", ["Q0"])
        assert circuit._unroll_repeats() == "X 0"
        reps = model.get_reps(
            circuit, [StimCircuitGateRep], [ZBasisProjectionInstrumentRep]
        )
        assert len(reps) == 1
        assert reps[0].circuit_str == "X 0"
        assert reps[0].qubits == ("Q0",)

    def test_tuple_key_aliasing(self):
        assert STIMPhysicalCircuit.stim_command_aliases.get("CNOT") == "CX"
        gate_dict = {
            ("CNOT", ("Q0", "Q1")): StimCircuitGateRep(
                "CNOT 0 1", ("Q0", "Q1")
            ),
        }
        model = DictNoiseModel((gate_dict, {}), gatereps=[StimCircuitGateRep])
        circuit = STIMPhysicalCircuit("CX 0 1", ["Q0", "Q1"])
        reps = model.get_reps(
            circuit, [StimCircuitGateRep], [ZBasisProjectionInstrumentRep]
        )
        assert len(reps) == 1
        assert reps[0].circuit_str == "CNOT 0 1"

    def test_negated_qubit_mapping(self):
        inst_dict = {"M": ZBasisProjectionInstrumentRep(None, True, ("Q0",))}
        model = DictNoiseModel(
            ({}, inst_dict), instreps=[ZBasisProjectionInstrumentRep]
        )
        circuit = STIMPhysicalCircuit("M !0", ["Q0"])
        reps = model.get_reps(
            circuit, [StimCircuitGateRep], [ZBasisProjectionInstrumentRep]
        )
        assert len(reps) == 1
        assert reps[0].qubits == ("!Q0",)

    def test_generic_instrument_merging_multiqubit(self):
        inst_dict = {"M": ZBasisProjectionInstrumentRep(None, True, ("Q0",))}
        model = DictNoiseModel(
            ({}, inst_dict), instreps=[ZBasisProjectionInstrumentRep]
        )
        circuit = STIMPhysicalCircuit("M 0 1 2", ["Q0", "Q1", "Q2"])
        reps = model.get_reps(
            circuit, [StimCircuitGateRep], [ZBasisProjectionInstrumentRep]
        )
        assert len(reps) == 1
        assert reps[0].qubits == ("Q0", "Q1", "Q2")
        assert reps[0].reset is None
        assert reps[0].include_outcome is True

    def test_repeat_block_unrolling_through_get_reps(self):
        gate_dict = {"X": StimCircuitGateRep("X 0", ("Q0",))}
        model = DictNoiseModel((gate_dict, {}), gatereps=[StimCircuitGateRep])
        circuit = STIMPhysicalCircuit("REPEAT 3 {\nX 0\n}", ["Q0"])
        reps = model.get_reps(
            circuit, [StimCircuitGateRep], [ZBasisProjectionInstrumentRep]
        )
        assert len(reps) == 3
        assert all(r.circuit_str == "X 0" for r in reps)

    def test_lookup_failure_raises_clear_error(self):
        model = DictNoiseModel(({}, {}))
        circuit = STIMPhysicalCircuit("X 0", ["Q0"])
        with pytest.raises(AssertionError, match="Failed to look up"):
            model.get_reps(
                circuit, [StimCircuitGateRep], [ZBasisProjectionInstrumentRep]
            )

    def test_multiple_same_command_combining(self):
        gate_dict = {"X": StimCircuitGateRep("X 0", ("Q0",))}
        model = DictNoiseModel((gate_dict, {}), gatereps=[StimCircuitGateRep])
        circuit = STIMPhysicalCircuit("X 0\nX 1\nX 2", ["Q0", "Q1", "Q2"])
        reps = model.get_reps(
            circuit, [StimCircuitGateRep], [ZBasisProjectionInstrumentRep]
        )
        combined_reps = [r for r in reps if len(r.qubits) > 1]
        assert len(combined_reps) == 1
        merged = combined_reps[0]
        assert merged.circuit_str == "X 0 1 2"
        assert merged.qubits == ("Q0", "Q1", "Q2")

    def test_common_command_combining_requires_self_indexed_template(self):
        gate_dict = {"X": StimCircuitGateRep("X", ("Q0",))}
        model = DictNoiseModel((gate_dict, {}), gatereps=[StimCircuitGateRep])
        circuit = STIMPhysicalCircuit("X 0\nX 1\nX 2", ["Q0", "Q1", "Q2"])
        with pytest.raises(ValueError, match="must already reference its own qubit"):
            model.get_reps(
                circuit, [StimCircuitGateRep], [ZBasisProjectionInstrumentRep]
            )

    def test_individual_command_no_combining(self):
        gate_dict = {
            ("X", ("Q0",)): StimCircuitGateRep("X 0", ("Q0",)),
            ("X", ("Q1",)): StimCircuitGateRep("X 1", ("Q1",)),
            ("X", ("Q2",)): StimCircuitGateRep("X 2", ("Q2",)),
        }
        model = DictNoiseModel((gate_dict, {}), gatereps=[StimCircuitGateRep])
        circuit = STIMPhysicalCircuit("X 0\nX 1\nX 2", ["Q0", "Q1", "Q2"])
        reps = model.get_reps(
            circuit, [StimCircuitGateRep], [ZBasisProjectionInstrumentRep]
        )
        assert len(reps) >= 3
        individual_reps = [r for r in reps if len(r.qubits) == 1]
        assert len(individual_reps) >= 3

    def test_exact_entry_takes_priority_over_generic(self):
        gate_dict = {
            "X": StimCircuitGateRep("X 0", ("Q0",)),
            ("X", ("Q1",)): StimCircuitGateRep("Y 0", ("Q1",)),
        }
        model = DictNoiseModel((gate_dict, {}), gatereps=[StimCircuitGateRep])
        circuit = STIMPhysicalCircuit("X 1", ["Q0", "Q1"])
        reps = model.get_reps(
            circuit, [StimCircuitGateRep], [ZBasisProjectionInstrumentRep]
        )
        assert len(reps) == 1
        assert reps[0] is model.gate_dict[("X", ("Q1",))]
        assert reps[0].circuit_str == "Y 0"

    def test_twoq_gate_multiple_pairs_one_line(self):
        gate_dict = {"CX": StimCircuitGateRep("CX 0 1", ("Q0", "Q1"))}
        model = DictNoiseModel((gate_dict, {}), gatereps=[StimCircuitGateRep])
        circuit = STIMPhysicalCircuit("CX 0 1 2 3", ["Q0", "Q1", "Q2", "Q3"])
        reps = model.get_reps(
            circuit, [StimCircuitGateRep], [ZBasisProjectionInstrumentRep]
        )
        assert len(reps) == 1
        assert reps[0].circuit_str == "CX 0 1 2 3"
        assert reps[0].qubits == ("Q0", "Q1", "Q2", "Q3")

    def test_multiple_generic_commands_merge_independently(self):
        """`get_reps`'s generic-command combining (the `common` dict) is
        scoped to a single circuit line, not the whole circuit."""
        gate_dict = {
            "X": StimCircuitGateRep("X 0", ("Q0",)),
            "Y": StimCircuitGateRep("Y 0", ("Q0",)),
        }
        model = DictNoiseModel((gate_dict, {}), gatereps=[StimCircuitGateRep])
        circuit = STIMPhysicalCircuit(
            "X 0\nY 1\nX 2\nY 3", ["Q0", "Q1", "Q2", "Q3"]
        )
        assert circuit._unroll_repeats() == "X 0\nY 1\nX 2\nY 3"
        reps = model.get_reps(
            circuit, [StimCircuitGateRep], [ZBasisProjectionInstrumentRep]
        )
        assert len(reps) == 4
        assert all(len(r.qubits) == 1 for r in reps)
        x_reps = [r for r in reps if r.circuit_str == "X 0"]
        y_reps = [r for r in reps if r.circuit_str == "Y 0"]
        assert {r.qubits for r in x_reps} == {("Q0",), ("Q2",)}
        assert {r.qubits for r in y_reps} == {("Q1",), ("Q3",)}


@pytest.mark.skipif(NO_STIM, reason="Skipping STIM backend tests due to failed import")
class TestMergeCommonRep:
    """Directly test `_merge_common_rep`, the module-level helper that
    merges generic (name-only) dict entries across multiple qubits."""

    def test_multiline_template(self):
        from loqs.backends.model.dictmodel import _merge_common_rep

        generic = StimCircuitGateRep("H 0\nS 0", ("Q0",))
        common: dict = {}
        _merge_common_rep("HS", ("Q0",), generic, common)
        _merge_common_rep("HS", ("Q1",), generic, common)
        _merge_common_rep("HS", ("Q2",), generic, common)
        merged = common["HS"]
        assert merged.circuit_str == "H 0 1 2\nS 0 1 2"
        assert merged.qubits == ("Q0", "Q1", "Q2")

    def test_instrument_reptype_merges_by_concatenating_qubits(self):
        from loqs.backends.model.dictmodel import _merge_common_rep

        generic = ZBasisProjectionInstrumentRep(None, True, ("Q0",))
        common: dict = {}
        _merge_common_rep("M", ("Q0",), generic, common)
        _merge_common_rep("M", ("Q1",), generic, common)
        merged = common["M"]
        assert merged.reset is None
        assert merged.include_outcome is True
        assert merged.qubits == ("Q0", "Q1")

    def test_stim_circuit_instrumentrep_also_merges_by_appending_indices(self):
        """`StimCircuitInstrumentRep` shares `StimCircuitPayloadMixin` with
        `StimCircuitGateRep`, so it takes the same trailing-index-appending
        merge path as the gate case above, not the plain-concatenation path
        used by other instrument reps like `ZBasisProjectionInstrumentRep`.
        """
        from loqs.backends.model.dictmodel import _merge_common_rep

        generic = StimCircuitInstrumentRep("M 0", ("Q0",))
        common: dict = {}
        _merge_common_rep("M", ("Q0",), generic, common)
        _merge_common_rep("M", ("Q1",), generic, common)
        merged = common["M"]
        assert merged.circuit_str == "M 0 1"
        assert merged.qubits == ("Q0", "Q1")


@pytest.mark.skipif(NO_STIM, reason="Skipping STIM backend tests due to failed import")
class TestAddCommandAliases:
    def test_adds_aliased_key_alongside_original(self):
        d = {"CNOT": "value"}
        add_command_aliases(d)
        assert d["CNOT"] == "value"
        assert d["CX"] == "value"

    def test_tuple_keys(self):
        d = {("CNOT", ("Q0", "Q1")): "value"}
        add_command_aliases(d)
        assert ("CNOT", ("Q0", "Q1")) in d
        assert ("CX", ("Q0", "Q1")) in d


@pytest.mark.skipif(NO_STIM, reason="Skipping STIM backend tests due to failed import")
class TestSTIMDictNoiseModelDecodeOnlyShim:
    """`STIMDictNoiseModel` is eliminated as a usable class -- see
    `loqs.backends.model.stimdictmodel`'s module docstring. This tests its
    decode-only compatibility shim directly."""

    def test_construction_raises_type_error(self):
        with pytest.raises(TypeError, match="STIMDictNoiseModel is deprecated"):
            STIMDictNoiseModel(({}, {}))

    @pytest.fixture(params=["json", "hdf5"])
    def decoded(self, request):
        if request.param == "json":
            with open(FIXTURES_DIR / "stimdictmodel_v1.json") as f:
                return Serializable.decode(json.load(f), format="json")
        else:
            with h5py.File(FIXTURES_DIR / "stimdictmodel_v1.h5", "r") as f:
                return Serializable.decode(f["root"], format="hdf5")

    def test_decodes_to_plain_dictnoisemodel(self, decoded):
        """Old `class: "STIMDictNoiseModel"`-tagged files decode to a
        plain `DictNoiseModel`, not an instance of the (now-nonexistent
        as a usable class) `STIMDictNoiseModel`."""
        assert isinstance(decoded, DictNoiseModel)
        assert not isinstance(decoded, STIMDictNoiseModel)
        # "CX" is present alongside "CNOT" because add_command_aliases ran
        # at construction time in the pre-refactor code that produced this
        # fixture -- confirms aliasing is captured in the frozen bytes.
        assert set(decoded.gate_dict.keys()) == {"X", "H", "CNOT", "CX"}
        assert decoded.gate_dict["CNOT"].circuit_str == "CNOT 0 1"
        assert decoded.gate_dict["CX"].circuit_str == "CNOT 0 1"
        assert set(decoded.inst_dict.keys()) == {"M"}
        assert isinstance(
            decoded.inst_dict["M"], ZBasisProjectionInstrumentRep
        )

    def test_get_reps_works_on_decoded_model(self, decoded):
        """Confirm a decoded model isn't just structurally correct but is
        actually still usable end-to-end -- including dispatching through
        to the STIM-specific `get_reps` implementation."""
        circuit = STIMPhysicalCircuit("X 0\nM 0", ["Q0"])
        reps = decoded.get_reps(
            circuit, [StimCircuitGateRep], [ZBasisProjectionInstrumentRep]
        )
        assert len(reps) == 2
        assert reps[0].circuit_str == "X 0"
        assert isinstance(reps[1], ZBasisProjectionInstrumentRep)

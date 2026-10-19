"""Tester for loqs.backends.model.dictmodel"""

import json
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import pytest

from loqs.backends.circuit.listcircuit import ListPhysicalCircuit
from loqs.backends.model.dictmodel import DictNoiseModel, add_command_aliases
from loqs.backends.reps import (
    KrausGateRep,
    ProbabilisticStimGateRep,
    QSimSuperopGateRep,
    StimCircuitGateRep,
    StimCircuitInstrumentRep,
    UnitaryGateRep,
    OutcomeOperationDictInstrumentRep,
    ZBasisPrePostInstrumentRep,
    ZBasisProjectionInstrumentRep,
)
from loqs.internal.serializable import Serializable

FIXTURES_DIR = Path(__file__).parent / "fixtures"

_UNITARY_1Q = np.eye(2)
_SUPEROP_1Q = np.eye(4)  # shape-correct QSimSuperopGateRep payload for 1 qubit
_SUPEROP_1Q_ALT = np.diag([1.0, -1.0, -1.0, 1.0])  # distinguishable from above
_GAMMA = 0.1
_TP_K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1 - _GAMMA)]])
_TP_K1 = np.array([[0.0, np.sqrt(_GAMMA)], [0.0, 0.0]])
_KRAUS_SEQ = ((_TP_K0, None), (_TP_K1, None))
_PROB_STIM_SEQ = (("X 0", 0.5), ("Y 0", 0.5))

try:
    import stim  # noqa: F401 -- import itself is the skip-guard for missing stim
    from loqs.backends.circuit.stimcircuit import STIMPhysicalCircuit

    NO_STIM = False
except ImportError:
    NO_STIM = True

requires_stim = pytest.mark.skipif(
    NO_STIM, reason="Skipping STIM backend tests due to failed import"
)


def test_copy_constructor():
    gate_dict = {("X", ("Q0",)): np.eye(4)}
    original = DictNoiseModel(gate_dict, {})
    copy = DictNoiseModel.from_model(original)
    assert copy.gate_dict.keys() == original.gate_dict.keys()
    assert copy is not original
    assert copy.gate_dict is not original.gate_dict


class TestGateDispatch:
    @pytest.mark.parametrize(
        "payload, kwargs, rep_class, check",
        [
            (
                _UNITARY_1Q,
                {"gaterep_array_cast_rep": UnitaryGateRep},
                UnitaryGateRep,
                lambda rep: True,
            ),
            (
                "X 0",
                {"gatereps": [StimCircuitGateRep]},
                StimCircuitGateRep,
                lambda rep: rep.circuit_str == "X 0",
            ),
            (
                _KRAUS_SEQ,
                {"gatereps": [KrausGateRep]},
                KrausGateRep,
                lambda rep: len(rep.kraus_operators) == 2,
            ),
            (
                _PROB_STIM_SEQ,
                {"gatereps": [ProbabilisticStimGateRep]},
                ProbabilisticStimGateRep,
                lambda rep: rep.operations == (("X 0", 0.5), ("Y 0", 0.5)),
            ),
        ],
        ids=["ndarray", "str", "kraus", "probabilistic-stim"],
    )
    def test_raw_gate_payload_cast(self, payload, kwargs, rep_class, check):
        model = DictNoiseModel({("X", ("Q0",)): payload}, {}, **kwargs)
        rep = model.gate_dict[("X", ("Q0",))]
        assert isinstance(rep, rep_class)
        assert check(rep)

    def test_rep_with_disallowed_type_raises(self):
        rep = StimCircuitGateRep("X 0", ("Q0",))
        with pytest.raises(AssertionError):
            DictNoiseModel(
                {("X", ("Q0",)): rep},
                {},
                gatereps=[QSimSuperopGateRep],
            )


class TestInstrumentDispatch:
    @pytest.mark.parametrize(
        "payload, kwargs, rep_class, check",
        [
            (
                "M 0",
                {},
                StimCircuitInstrumentRep,
                lambda rep: rep.circuit_str == "M 0",
            ),
            (
                (0, True),
                {},
                ZBasisProjectionInstrumentRep,
                lambda rep: rep.reset == 0 and rep.include_outcome is True,
            ),
            (
                {0: _SUPEROP_1Q, 1: _SUPEROP_1Q},
                {
                    "gatereps": [QSimSuperopGateRep],
                    "instreps": [OutcomeOperationDictInstrumentRep],
                },
                OutcomeOperationDictInstrumentRep,
                lambda rep: rep.include_outcome is True
                and set(rep.outcome_ops) == {0, 1}
                and all(
                    isinstance(v, QSimSuperopGateRep)
                    for v in rep.outcome_ops.values()
                ),
            ),
        ],
        ids=["str", "projection-tuple", "mapping"],
    )
    def test_raw_instrument_payload_cast(
        self, payload, kwargs, rep_class, check
    ):
        model = DictNoiseModel({}, {("M", ("Q0",)): payload}, **kwargs)
        rep = model.inst_dict[("M", ("Q0",))]
        assert isinstance(rep, rep_class)
        assert check(rep)

    def test_two_element_non_projection_becomes_pre_post_operations(self):
        # pre_op/post_op payloads are deliberately distinct so a test
        # failure would show if they got swapped.
        model = DictNoiseModel(
            {},
            {("M", ("Q0",)): (_SUPEROP_1Q, _SUPEROP_1Q_ALT)},
            gatereps=[QSimSuperopGateRep],
            instreps=[ZBasisPrePostInstrumentRep],
            instrep_cast_reset=0,
            instrep_cast_include_outcomes=False,
        )
        rep = model.inst_dict[("M", ("Q0",))]
        assert isinstance(rep, ZBasisPrePostInstrumentRep)
        assert rep.reset == 0
        assert rep.include_outcome is False
        assert isinstance(rep.pre_op, QSimSuperopGateRep)
        assert isinstance(rep.post_op, QSimSuperopGateRep)
        assert np.array_equal(rep.pre_op.superop, _SUPEROP_1Q)
        assert np.array_equal(rep.post_op.superop, _SUPEROP_1Q_ALT)


@pytest.mark.parametrize("kind", ["gate", "instrument"])
def test_rep_passthrough(kind):
    if kind == "gate":
        rep = QSimSuperopGateRep(np.eye(4), ("Q0",))
        model = DictNoiseModel({("X", ("Q0",)): rep}, {})
        assert model.gate_dict[("X", ("Q0",))] is rep
    else:
        rep = ZBasisProjectionInstrumentRep(None, True, ("Q0",))
        model = DictNoiseModel({}, {("M", ("Q0",)): rep})
        assert model.inst_dict[("M", ("Q0",))] is rep


class TestGetReps:
    def test_exact_label_match(self):
        model = DictNoiseModel({("X", ("Q0",)): np.eye(4)}, {})
        circuit = ListPhysicalCircuit([[("X", ("Q0",))]])
        reps = model.get_reps(circuit, [QSimSuperopGateRep], [])
        assert len(reps) == 1
        assert reps[0].qubit_labels == ("Q0",)

    @pytest.mark.parametrize("kind", ["gate", "instrument"])
    def test_generic_name_only_fallback(self, kind):
        if kind == "gate":
            model = DictNoiseModel({"X": np.eye(4)}, {})
            circuit = ListPhysicalCircuit([[("X", ("Q1",))]])
            reps = model.get_reps(circuit, [QSimSuperopGateRep], [])
            rep_class = QSimSuperopGateRep
        else:
            model = DictNoiseModel({}, {"M": (None, True)})
            circuit = ListPhysicalCircuit([[("M", ("Q1",))]])
            reps = model.get_reps(circuit, [], [ZBasisProjectionInstrumentRep])
            rep_class = ZBasisProjectionInstrumentRep
        assert len(reps) == 1
        assert reps[0].qubit_labels == ("Q1",)
        assert isinstance(reps[0], rep_class)

    @pytest.mark.parametrize(
        "kind",
        ["list", pytest.param("stim", marks=requires_stim)],
    )
    def test_lookup_failure_raises(self, kind):
        model = DictNoiseModel({}, {})
        if kind == "list":
            circuit = ListPhysicalCircuit([[("X", ("Q0",))]])
            gatereps, instreps = [], []
        else:
            circuit = STIMPhysicalCircuit("X 0", ["Q0"])
            gatereps = [StimCircuitGateRep]
            instreps = [ZBasisProjectionInstrumentRep]
        with pytest.raises(AssertionError):
            model.get_reps(circuit, gatereps, instreps)

    def test_get_reps_does_not_reconstruct_already_correct_circuit_type(self):
        """A `circuit` that's already a `ListPhysicalCircuit` is read
        directly, without constructing or copying a second one."""
        model = DictNoiseModel({("X", ("Q0",)): np.eye(4)}, {})
        circuit = ListPhysicalCircuit([[("X", ("Q0",))]])
        original_init = ListPhysicalCircuit.__init__
        call_count = [0]

        def counting_init(self, *args, **kwargs):
            call_count[0] += 1
            return original_init(self, *args, **kwargs)

        with mock.patch.object(ListPhysicalCircuit, "__init__", counting_init):
            model.get_reps(circuit, [QSimSuperopGateRep], [])
        assert call_count[0] == 0


def _decode_fixture(stem, fmt):
    if fmt == "json":
        with open(FIXTURES_DIR / f"{stem}.json") as f:
            return Serializable.decode(json.load(f), format="json")
    with h5py.File(FIXTURES_DIR / f"{stem}.h5", "r") as f:
        return Serializable.decode(f["root"], format="hdf5")


class TestDictModelFixtureRoundTrip:
    """Round-trip the frozen `dictmodel_v{1,2}.{json,h5}` fixtures to
    confirm saved `DictNoiseModel` files still decode, including the v1
    files' bare `GateRep`/`InstrumentRep` enum-member tags."""

    @pytest.mark.parametrize("fmt", ["json", "hdf5"])
    @pytest.mark.parametrize("version", ["v1", "v2"])
    def test_decodes_to_dictnoisemodel_with_correct_content(
        self, version, fmt
    ):
        decoded = _decode_fixture(f"dictmodel_{version}", fmt)
        assert isinstance(decoded, DictNoiseModel)
        assert set(decoded.gate_dict.keys()) == {
            ("X", ("Q0",)),
            ("KRAUS", ("Q0",)),
        }
        assert set(decoded.inst_dict.keys()) == {("M", ("Q0",))}
        assert isinstance(
            decoded.gate_dict[("X", ("Q0",))], QSimSuperopGateRep
        )
        assert isinstance(decoded.gate_dict[("KRAUS", ("Q0",))], KrausGateRep)
        assert isinstance(
            decoded.inst_dict[("M", ("Q0",))], ZBasisProjectionInstrumentRep
        )
        # Legacy tags are decoded to real classes, not raw ints or enums.
        assert decoded.output_gate_reps == [
            QSimSuperopGateRep,
            KrausGateRep,
        ]
        assert decoded.output_instrument_reps == [
            ZBasisProjectionInstrumentRep
        ]


@requires_stim
class TestStimFixtureRoundTrip:
    """Saved STIM-flavoured dict models (the v1 `STIMDictNoiseModel` files
    redirect to `DictNoiseModel` on decode) still load and run."""

    @pytest.mark.parametrize("fmt", ["json", "hdf5"])
    @pytest.mark.parametrize(
        "stem, gate_keys",
        [
            ("stimdictmodel_v1", {"X", "H", "CNOT", "CX"}),
            ("dictmodel_stim_v2", {"X", "H", "CNOT"}),
        ],
        ids=["v1", "v2"],
    )
    def test_decodes_and_runs(self, stem, gate_keys, fmt):
        decoded = _decode_fixture(stem, fmt)
        assert type(decoded) is DictNoiseModel
        assert set(decoded.gate_dict.keys()) == gate_keys
        for key in gate_keys & {"CNOT", "CX"}:
            assert decoded.gate_dict[key].circuit_str == "CNOT 0 1"
        assert set(decoded.inst_dict.keys()) == {"M"}
        assert isinstance(
            decoded.inst_dict["M"], ZBasisProjectionInstrumentRep
        )

        circuit = STIMPhysicalCircuit("X 0\nM 0", ["Q0"])
        reps = decoded.get_reps(
            circuit, [StimCircuitGateRep], [ZBasisProjectionInstrumentRep]
        )
        assert len(reps) == 2
        assert reps[0].circuit_str == "X 0"
        assert isinstance(reps[1], ZBasisProjectionInstrumentRep)


@requires_stim
class TestSTIMGetReps:
    """`DictNoiseModel.get_reps`'s `STIMPhysicalCircuit`-registered
    implementation."""

    def _get_reps(self, model, circuit):
        return model.get_reps(
            circuit, [StimCircuitGateRep], [ZBasisProjectionInstrumentRep]
        )

    def test_add_command_aliases_resolution_through_get_reps(self):
        """A gate registered only under its alias-source name (`CNOT`,
        aliased by `add_command_aliases` to `CX` at lookup time) must
        resolve when the input circuit uses the name STIM itself
        normalizes to (`CX`)."""
        gate_dict = {
            ("CNOT", ("Q0", "Q1")): StimCircuitGateRep("CX 0 1", ("Q0", "Q1")),
        }
        model = DictNoiseModel(gate_dict, {}, gatereps=[StimCircuitGateRep])
        circuit = STIMPhysicalCircuit("CNOT 0 1", ["Q0", "Q1"])
        reps = self._get_reps(model, circuit)
        assert len(reps) == 1
        assert reps[0].circuit_str == "CX 0 1"

    def test_get_reps_with_instruments(self):
        gate_dict = {"X": StimCircuitGateRep("X 0", ("Q0",))}
        inst_dict = {
            "M": ZBasisProjectionInstrumentRep(None, True, ("Q0",)),
        }
        model = DictNoiseModel(
            gate_dict,
            inst_dict,
            gatereps=[StimCircuitGateRep],
            instreps=[StimCircuitInstrumentRep, ZBasisProjectionInstrumentRep],
        )
        circuit = STIMPhysicalCircuit("X 0\nM 0", ["Q0"])
        assert len(self._get_reps(model, circuit)) == 2

    @pytest.mark.parametrize("kind", ["channel", "measure"])
    def test_double_noise_warns(self, kind):
        if kind == "channel":
            model = DictNoiseModel({}, {})
            circuit = STIMPhysicalCircuit("X_ERROR(0.1) 0", ["Q0"])
        else:
            inst_dict = {
                "M": ZBasisProjectionInstrumentRep(None, True, ("Q0",))
            }
            model = DictNoiseModel(
                {},
                inst_dict,
                instreps=[
                    StimCircuitInstrumentRep,
                    ZBasisProjectionInstrumentRep,
                ],
            )
            circuit = STIMPhysicalCircuit("M(0.1) 0", ["Q0"])
        with pytest.warns(UserWarning):
            self._get_reps(model, circuit)

    def test_qubit_label_mapping(self):
        gate_dict = {"X": StimCircuitGateRep("X 0", ("Q0",))}
        model = DictNoiseModel(gate_dict, {}, gatereps=[StimCircuitGateRep])
        circuit = STIMPhysicalCircuit("X 0", ["A0"])
        reps = self._get_reps(model, circuit)
        assert len(reps) == 1
        assert isinstance(reps[0], StimCircuitGateRep)

    def test_negated_qubit_mapping(self):
        inst_dict = {"M": ZBasisProjectionInstrumentRep(None, True, ("Q0",))}
        model = DictNoiseModel(
            {}, inst_dict, instreps=[ZBasisProjectionInstrumentRep]
        )
        circuit = STIMPhysicalCircuit("M !0", ["Q0"])
        reps = self._get_reps(model, circuit)
        assert len(reps) == 1
        assert reps[0].qubit_labels == ("!Q0",)

    def test_generic_instrument_merging_multiqubit(self):
        inst_dict = {"M": ZBasisProjectionInstrumentRep(None, True, ("Q0",))}
        model = DictNoiseModel(
            {}, inst_dict, instreps=[ZBasisProjectionInstrumentRep]
        )
        circuit = STIMPhysicalCircuit("M 0 1 2", ["Q0", "Q1", "Q2"])
        reps = self._get_reps(model, circuit)
        assert len(reps) == 1
        assert reps[0].qubit_labels == ("Q0", "Q1", "Q2")
        assert reps[0].reset is None
        assert reps[0].include_outcome is True

    def test_repeat_block_unrolling_through_get_reps(self):
        gate_dict = {"X": StimCircuitGateRep("X 0", ("Q0",))}
        model = DictNoiseModel(gate_dict, {}, gatereps=[StimCircuitGateRep])
        circuit = STIMPhysicalCircuit("REPEAT 3 {\nX 0\n}", ["Q0"])
        reps = self._get_reps(model, circuit)
        assert len(reps) == 3
        assert all(r.circuit_str == "X 0" for r in reps)

    def test_multiple_same_command_combining(self):
        gate_dict = {"X": StimCircuitGateRep("X 0", ("Q0",))}
        model = DictNoiseModel(gate_dict, {}, gatereps=[StimCircuitGateRep])
        circuit = STIMPhysicalCircuit("X 0\nX 1\nX 2", ["Q0", "Q1", "Q2"])
        reps = self._get_reps(model, circuit)
        combined_reps = [r for r in reps if len(r.qubit_labels) > 1]
        assert len(combined_reps) == 1
        merged = combined_reps[0]
        assert merged.circuit_str == "X 0 1 2"
        assert merged.qubit_labels == ("Q0", "Q1", "Q2")

    def test_common_command_combining_requires_self_indexed_template(self):
        gate_dict = {"X": StimCircuitGateRep("X", ("Q0",))}
        model = DictNoiseModel(gate_dict, {}, gatereps=[StimCircuitGateRep])
        circuit = STIMPhysicalCircuit("X 0\nX 1\nX 2", ["Q0", "Q1", "Q2"])
        with pytest.raises(ValueError):
            self._get_reps(model, circuit)

    def test_individual_command_no_combining(self):
        gate_dict = {
            ("X", ("Q0",)): StimCircuitGateRep("X 0", ("Q0",)),
            ("X", ("Q1",)): StimCircuitGateRep("X 1", ("Q1",)),
            ("X", ("Q2",)): StimCircuitGateRep("X 2", ("Q2",)),
        }
        model = DictNoiseModel(gate_dict, {}, gatereps=[StimCircuitGateRep])
        circuit = STIMPhysicalCircuit("X 0\nX 1\nX 2", ["Q0", "Q1", "Q2"])
        reps = self._get_reps(model, circuit)
        assert len(reps) >= 3
        individual_reps = [r for r in reps if len(r.qubit_labels) == 1]
        assert len(individual_reps) >= 3

    def test_exact_entry_takes_priority_over_generic(self):
        gate_dict = {
            "X": StimCircuitGateRep("X 0", ("Q0",)),
            ("X", ("Q1",)): StimCircuitGateRep("Y 0", ("Q1",)),
        }
        model = DictNoiseModel(gate_dict, {}, gatereps=[StimCircuitGateRep])
        circuit = STIMPhysicalCircuit("X 1", ["Q0", "Q1"])
        reps = self._get_reps(model, circuit)
        assert len(reps) == 1
        assert reps[0] is model.gate_dict[("X", ("Q1",))]
        assert reps[0].circuit_str == "Y 0"

    def test_twoq_gate_multiple_pairs_one_line(self):
        gate_dict = {"CX": StimCircuitGateRep("CX 0 1", ("Q0", "Q1"))}
        model = DictNoiseModel(gate_dict, {}, gatereps=[StimCircuitGateRep])
        circuit = STIMPhysicalCircuit("CX 0 1 2 3", ["Q0", "Q1", "Q2", "Q3"])
        reps = self._get_reps(model, circuit)
        assert len(reps) == 1
        assert reps[0].circuit_str == "CX 0 1 2 3"
        assert reps[0].qubit_labels == ("Q0", "Q1", "Q2", "Q3")


@requires_stim
class TestMergeCommonRep:
    """`_merge_common_rep` merges generic (name-only) dict entries across
    qubits."""

    def test_multiline_template(self):
        from loqs.backends.model.dictmodel import _merge_common_rep

        generic = StimCircuitGateRep("H 0\nS 0", ("Q0",))
        common: dict = {}
        _merge_common_rep("HS", ("Q0",), generic, common)
        _merge_common_rep("HS", ("Q1",), generic, common)
        _merge_common_rep("HS", ("Q2",), generic, common)
        merged = common["HS"]
        assert merged.circuit_str == "H 0 1 2\nS 0 1 2"
        assert merged.qubit_labels == ("Q0", "Q1", "Q2")


@requires_stim
class TestAddCommandAliases:
    def test_tuple_keys(self):
        d = {("CNOT", ("Q0", "Q1")): "value"}
        add_command_aliases(d)
        assert ("CNOT", ("Q0", "Q1")) in d
        assert ("CX", ("Q0", "Q1")) in d


@requires_stim
def test_legacy_stim_dict_noise_model_construction_warns():
    """Old `STIMDictNoiseModel(...)` user code still runs, returning a plain
    `DictNoiseModel` with STIM-appropriate rep defaults."""
    from loqs.backends.model.stimdictmodel import STIMDictNoiseModel

    with pytest.warns(DeprecationWarning):
        model = STIMDictNoiseModel(({"H": "H", "CX": "CX"}, {"M": "M"}))

    assert type(model) is DictNoiseModel
    assert isinstance(model.gate_dict["H"], StimCircuitGateRep)
    assert isinstance(model.gate_dict["CX"], StimCircuitGateRep)
    assert isinstance(model.inst_dict["M"], StimCircuitInstrumentRep)
    assert model.output_gate_reps == [
        StimCircuitGateRep,
        ProbabilisticStimGateRep,
    ]
    assert model.output_instrument_reps == [StimCircuitInstrumentRep]

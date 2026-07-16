"""Tester for loqs.backends.reps"""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from loqs.backends.reps import (
    ConcreteGateReps,
    ConcreteInstrumentReps,
    GateRep,
    InstrumentRep,
    RepTuple,
)
from loqs.internal.serializable import Serializable

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# TP-preserving (sum_i K_i K_i^dagger = I) single-qubit Kraus operators.
_GAMMA = 0.1
_TP_K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1 - _GAMMA)]])
_TP_K1 = np.array([[0.0, 0.0], [np.sqrt(_GAMMA), 0.0]])

# Deliberately non-TP: a single, non-unitary, non-normalized operator.
_NON_TP_K = np.eye(2) * 0.5


class TestSequenceIsKrausopRep:
    def test_valid_tp_kraus_no_warning(self, recwarn):
        gr = ((_TP_K0, None), (_TP_K1, None))
        assert ConcreteGateReps.sequence_is_krausop_rep(gr) is True
        assert len(recwarn) == 0

    def test_valid_with_precomputed_probabilities(self):
        gr = ((_TP_K0, 0.95), (_TP_K1, 0.05))
        assert ConcreteGateReps.sequence_is_krausop_rep(gr, tp_check_abstol=float("inf")) is True

    def test_non_tp_kraus_warns(self):
        with pytest.warns(UserWarning, match="do not constitute a TP channel"):
            result = ConcreteGateReps.sequence_is_krausop_rep(((_NON_TP_K, None),))
        assert result is True  # still a structurally valid Kraus rep, just non-TP

    def test_tp_check_disabled_by_infinite_tolerance(self, recwarn):
        result = ConcreteGateReps.sequence_is_krausop_rep(
            ((_NON_TP_K, None),), tp_check_abstol=float("inf")
        )
        assert result is True
        assert len(recwarn) == 0

    def test_empty_sequence_is_false(self):
        assert ConcreteGateReps.sequence_is_krausop_rep(()) is False

    @pytest.mark.parametrize(
        "malformed",
        [
            (_TP_K0,),  # element is not a (op, prob) tuple/list at all
            ((_TP_K0, 0.5, "extra"),),  # wrong length
            (("not an array", 0.5),),  # first entry not an ndarray
            ((_TP_K0, "not a float"),),  # second entry not float/None
        ],
    )
    def test_malformed_sequences_are_false(self, malformed):
        assert ConcreteGateReps.sequence_is_krausop_rep(malformed) is False


class TestSequenceIsProbabilisticStimRep:
    def test_valid_sequence(self):
        gr = (("X 0", 0.5), ("Y 0", 0.5))
        assert ConcreteGateReps.sequence_is_probabilisticstim_rep(gr) is True

    def test_accepts_int_probability(self):
        # int is explicitly allowed alongside float/np.floating
        assert ConcreteGateReps.sequence_is_probabilisticstim_rep((("X 0", 1),)) is True

    def test_empty_sequence_is_false(self):
        assert ConcreteGateReps.sequence_is_probabilisticstim_rep(()) is False

    @pytest.mark.parametrize(
        "malformed",
        [
            ("X 0",),  # element is not a (str, prob) tuple/list
            (("X 0", 0.5, "extra"),),  # wrong length
            ((0, 0.5),),  # first entry not a str
            (("X 0", "not a number"),),  # second entry not float/int
        ],
    )
    def test_malformed_sequences_are_false(self, malformed):
        assert ConcreteGateReps.sequence_is_probabilisticstim_rep(malformed) is False


class TestIsZBasisProjectionRep:
    @pytest.mark.parametrize("ir", [(None, True), (0, False), (1, True)])
    def test_valid_reps(self, ir):
        assert ConcreteInstrumentReps.is_zbasis_projection_rep(ir) is True

    def test_not_a_tuple_or_list(self):
        assert ConcreteInstrumentReps.is_zbasis_projection_rep("not a tuple") is False

    @pytest.mark.parametrize(
        "malformed",
        [
            (None,),  # wrong length
            (None, True, "extra"),  # wrong length
            ("not an int", True),  # first entry not int/None
            (0, "not a bool"),  # second entry not bool
        ],
    )
    def test_malformed_reps_are_false(self, malformed):
        assert ConcreteInstrumentReps.is_zbasis_projection_rep(malformed) is False


class TestRepTupleBasics:
    def test_single_qubit_str_is_wrapped_in_tuple(self):
        rt = RepTuple("X 0", "Q0", GateRep.STIM_CIRCUIT_STR)
        assert rt.qubits == ("Q0",)

    def test_tuple_like_access(self):
        rt = RepTuple("X 0", ("Q0",), GateRep.STIM_CIRCUIT_STR)
        assert len(rt) == 3
        assert rt[0] == "X 0"
        assert rt[1] == ("Q0",)
        assert rt[2] == GateRep.STIM_CIRCUIT_STR

    def test_cast_from_list(self):
        rt = RepTuple.cast(["X 0", ("Q0",), GateRep.STIM_CIRCUIT_STR])
        assert isinstance(rt, RepTuple)
        assert rt.rep == "X 0"
        assert rt.qubits == ("Q0",)
        assert rt.reptype == GateRep.STIM_CIRCUIT_STR

    def test_cast_from_dict(self):
        rt = RepTuple.cast(
            {"rep": "X 0", "qubits": ("Q0",), "reptype": GateRep.STIM_CIRCUIT_STR}
        )
        assert isinstance(rt, RepTuple)
        assert rt.reptype == GateRep.STIM_CIRCUIT_STR

    def test_cast_is_identity_for_existing_reptuple(self):
        rt = RepTuple("X 0", ("Q0",), GateRep.STIM_CIRCUIT_STR)
        assert RepTuple.cast(rt) is rt

    def test_reptype_must_be_a_rep_enum(self):
        with pytest.raises(AssertionError):
            RepTuple("X 0", ("Q0",), "not a RepEnum")


class TestRepsFixtureRoundTrip:
    """Round-trip the `tests/backends/fixtures/reps_v1.{json,h5}` fixtures
    (generated by `generate_reps_fixtures.py` from the current code) to
    guard against silent regressions in serialization of any GateRep/
    InstrumentRep member, including the two nested-RepTuple cases."""

    EXPECTED_REPTYPES = {
        "GATEREP_UNITARY": GateRep.UNITARY,
        "GATEREP_PTM": GateRep.PTM,
        "GATEREP_QSIM_SUPEROPERATOR": GateRep.QSIM_SUPEROPERATOR,
        "GATEREP_STIM_CIRCUIT_STR": GateRep.STIM_CIRCUIT_STR,
        "GATEREP_PROBABILISTIC_STIM_OPERATIONS": GateRep.PROBABILISTIC_STIM_OPERATIONS,
        "GATEREP_KRAUS_OPERATORS": GateRep.KRAUS_OPERATORS,
        "INSTRUMENTREP_ZBASIS_PROJECTION": InstrumentRep.ZBASIS_PROJECTION,
        "INSTRUMENTREP_ZBASIS_PRE_POST_OPERATIONS": InstrumentRep.ZBASIS_PRE_POST_OPERATIONS,
        "INSTRUMENTREP_ZBASIS_OUTCOME_OPERATION_DICT": InstrumentRep.ZBASIS_OUTCOME_OPERATION_DICT,
        "INSTRUMENTREP_STIM_CIRCUIT_STR": InstrumentRep.STIM_CIRCUIT_STR,
    }

    @pytest.fixture(params=["json", "hdf5"])
    def decoded(self, request):
        if request.param == "json":
            with open(FIXTURES_DIR / "reps_v1.json") as f:
                return Serializable.decode(json.load(f), format="json")
        else:
            with h5py.File(FIXTURES_DIR / "reps_v1.h5", "r") as f:
                return Serializable.decode(f["root"], format="hdf5")

    def test_all_members_present_with_correct_reptype_and_class(self, decoded):
        assert set(decoded.keys()) == set(self.EXPECTED_REPTYPES.keys())
        for name, expected_reptype in self.EXPECTED_REPTYPES.items():
            rt = decoded[name]
            assert isinstance(rt, RepTuple), f"{name} did not decode to a RepTuple"
            assert rt.reptype == expected_reptype
            assert rt.qubits == ("Q0",)

    def test_flat_gate_rep_payloads_round_trip(self, decoded):
        assert decoded["GATEREP_STIM_CIRCUIT_STR"].rep == "X 0"
        probs = decoded["GATEREP_PROBABILISTIC_STIM_OPERATIONS"].rep
        assert tuple(probs) == (("X 0", 0.5), ("Y 0", 0.5))
        kraus = decoded["GATEREP_KRAUS_OPERATORS"].rep
        assert len(kraus) == 2
        for op, prob in kraus:
            assert isinstance(op, np.ndarray)
            assert op.shape == (2, 2)
            assert prob is None
        assert isinstance(decoded["GATEREP_UNITARY"].rep, np.ndarray)
        assert decoded["GATEREP_UNITARY"].rep.shape == (2, 2)

    def test_nested_pre_post_operations_round_trip(self, decoded):
        rt = decoded["INSTRUMENTREP_ZBASIS_PRE_POST_OPERATIONS"]
        reset, include_outcomes, preop, postop = rt.rep
        assert reset == 0
        # NOTE: `include_outcomes` decodes as plain `int` (1), not `bool`
        # (True) -- see test_bool_payloads_are_coerced_to_int_on_round_trip
        # below for why -- so compare by value, not identity/type.
        assert include_outcomes == True  # noqa: E712
        assert isinstance(preop, RepTuple)
        assert isinstance(postop, RepTuple)
        assert preop.reptype == GateRep.UNITARY
        assert postop.reptype == GateRep.UNITARY
        assert preop.qubits == ("Q0",)
        assert postop.qubits == ("Q0",)

    def test_nested_outcome_operation_dict_round_trip(self, decoded, request):
        """NOTE: JSON round-trips the `{0: ..., 1: ...}` outcome dict's keys
        as *strings* (`"0"`, `"1"`), not ints -- an inherent JSON limitation
        (object keys are always strings), not a LoQS-specific bug. HDF5
        preserves the original int keys. Any future code reconstructing
        this dict from a JSON-decoded legacy file needs to `int()`-cast the
        keys explicitly; this is worth keeping in mind for the eventual
        `ZBASIS_OUTCOME_OPERATION_DICT` legacy-decode path.
        """
        rt = decoded["INSTRUMENTREP_ZBASIS_OUTCOME_OPERATION_DICT"]
        outcome_dict, include_outcomes = rt.rep
        assert include_outcomes == True  # noqa: E712
        if "json" in request.node.callspec.id:
            assert set(outcome_dict.keys()) == {"0", "1"}
            outcome_0, outcome_1 = outcome_dict["0"], outcome_dict["1"]
        else:
            assert set(outcome_dict.keys()) == {0, 1}
            outcome_0, outcome_1 = outcome_dict[0], outcome_dict[1]
        assert isinstance(outcome_0, RepTuple)
        assert isinstance(outcome_1, RepTuple)
        assert outcome_0.reptype == GateRep.PTM
        assert outcome_1.reptype == GateRep.QSIM_SUPEROPERATOR

    def test_bool_payloads_are_coerced_to_int_on_round_trip(self):
        """Documents a real, pre-existing (not #72-related) quirk of the
        serialization framework found while writing this test: `bool` values
        inside a `RepTuple` payload decode back as plain `int` (e.g. `True`
        -> `1`), not `bool`, because `JSONEncoder.encode_primitive`/its HDF5
        counterpart check `isinstance(x, Int)` and `bool` is a subclass of
        `int` in Python, with no separate bool branch. This affects every
        `include_outcomes: bool` field across `InstrumentRep` payloads. Not
        fixed here (it's a pre-existing, independent limitation of
        `loqs.internal.serializable`/`encoder`, out of scope for this
        change) -- pinned down explicitly so it isn't mistaken for a new
        regression later, and so any future serialization work is aware of
        it.
        """
        encoded = Serializable.encode((0, True), format="json", reset_encode_id=True)
        decoded = Serializable.decode(json.loads(json.dumps(encoded)), format="json")
        assert decoded == (0, 1)
        assert type(decoded[1]) is int

    def test_module_and_class_metadata(self):
        """The (module, class) metadata this fixture records is exactly the
        mechanism a future legacy-decode compatibility shim would depend on
        -- pin it down explicitly rather than only testing round-trip
        behavior end-to-end."""
        with open(FIXTURES_DIR / "reps_v1.json") as f:
            raw = json.load(f)
        entry = raw["items"]["GATEREP_UNITARY"]
        assert entry["module"] == "loqs.backends.reps"
        assert entry["class"] == "RepTuple"
        reptype_entry = entry["reptype"]
        assert reptype_entry["module"] == "loqs.backends.reps"
        assert reptype_entry["class"] == "GateRep"

"""Tester for loqs.backends.reps.instrumentreps"""

import numpy as np
import pytest

from loqs.backends.reps import (
    InstrumentRep,
    RepConstructionError,
    StimCircuitInstrumentRep,
    UnitaryGateRep,
    ZBasisOutcomeOperationDictInstrumentRep,
    ZBasisPrePostInstrumentRep,
    ZBasisProjectionInstrumentRep,
)


class TestZBasisProjectionInstrumentRep:
    @pytest.mark.parametrize("ir", [(None, True), (0, False), (1, True)])
    def test_matches_valid_reps(self, ir):
        assert ZBasisProjectionInstrumentRep.matches(ir) is True

    def test_does_not_match_non_tuple(self):
        assert ZBasisProjectionInstrumentRep.matches("not a tuple") is False

    @pytest.mark.parametrize(
        "malformed",
        [
            (None,),  # wrong length
            (None, True, "extra"),  # wrong length
            ("not an int", True),  # first entry not int/None
            (0, "not a bool"),  # second entry not bool
        ],
    )
    def test_malformed_reps_do_not_match(self, malformed):
        assert ZBasisProjectionInstrumentRep.matches(malformed) is False

    def test_from_raw_constructs_instance(self):
        rep = ZBasisProjectionInstrumentRep.from_raw((0, True), ("Q0",))
        assert isinstance(rep, ZBasisProjectionInstrumentRep)
        assert isinstance(rep, InstrumentRep)
        assert rep.reset == 0
        assert rep.include_outcome is True
        assert rep.qubits == ("Q0",)

    def test_from_raw_rejects_non_matching_payload(self):
        with pytest.raises(RepConstructionError):
            ZBasisProjectionInstrumentRep.from_raw("bad", ("Q0",))


class TestZBasisPrePostInstrumentRep:
    def test_matches_two_tuple(self):
        assert ZBasisPrePostInstrumentRep.matches((np.eye(2), np.eye(2))) is True

    def test_does_not_match_wrong_length(self):
        assert ZBasisPrePostInstrumentRep.matches((np.eye(2),)) is False

    def test_from_raw_requires_gate_upgrader(self):
        with pytest.raises(RepConstructionError, match="gate_upgrader"):
            ZBasisPrePostInstrumentRep.from_raw(
                (np.eye(2), np.eye(2)), ("Q0",)
            )

    def test_from_raw_rejects_non_matching_payload(self):
        def gate_upgrader(raw, qubits):
            return UnitaryGateRep(raw, qubits)

        with pytest.raises(RepConstructionError):
            ZBasisPrePostInstrumentRep.from_raw(
                "not a pair", ("Q0",), gate_upgrader=gate_upgrader
            )

    def test_from_raw_constructs_instance_with_recursively_upgraded_ops(self):
        def gate_upgrader(raw, qubits):
            return UnitaryGateRep(raw, qubits)

        pre_raw = np.eye(2)
        post_raw = np.eye(2) * 2
        rep = ZBasisPrePostInstrumentRep.from_raw(
            (pre_raw, post_raw),
            ("Q0",),
            reset=1,
            include_outcome=False,
            gate_upgrader=gate_upgrader,
        )
        assert isinstance(rep, ZBasisPrePostInstrumentRep)
        assert rep.reset == 1
        assert rep.include_outcome is False
        assert isinstance(rep.pre_op, UnitaryGateRep)
        assert isinstance(rep.post_op, UnitaryGateRep)
        assert np.array_equal(rep.pre_op.unitary, pre_raw)
        assert np.array_equal(rep.post_op.unitary, post_raw)
        assert rep.pre_op.qubits == ("Q0",)
        assert rep.post_op.qubits == ("Q0",)

    def test_from_raw_defaults_reset_none_include_outcome_true(self):
        def gate_upgrader(raw, qubits):
            return UnitaryGateRep(raw, qubits)

        rep = ZBasisPrePostInstrumentRep.from_raw(
            (np.eye(2), np.eye(2)), ("Q0",), gate_upgrader=gate_upgrader
        )
        assert rep.reset is None
        assert rep.include_outcome is True


class TestZBasisOutcomeOperationDictInstrumentRep:
    def test_matches_mapping(self):
        assert ZBasisOutcomeOperationDictInstrumentRep.matches({0: np.eye(2)}) is True

    def test_does_not_match_non_mapping(self):
        assert ZBasisOutcomeOperationDictInstrumentRep.matches([0, 1]) is False

    def test_from_raw_requires_gate_upgrader(self):
        with pytest.raises(RepConstructionError, match="gate_upgrader"):
            ZBasisOutcomeOperationDictInstrumentRep.from_raw(
                {0: np.eye(2)}, ("Q0",)
            )

    def test_from_raw_constructs_instance_with_recursively_upgraded_ops(self):
        def gate_upgrader(raw, qubits):
            return UnitaryGateRep(raw, qubits)

        raw_dict = {0: np.eye(2), 1: np.eye(2) * 2}
        rep = ZBasisOutcomeOperationDictInstrumentRep.from_raw(
            raw_dict,
            ("Q0",),
            include_outcome=False,
            gate_upgrader=gate_upgrader,
        )
        assert isinstance(rep, ZBasisOutcomeOperationDictInstrumentRep)
        assert rep.include_outcome is False
        assert set(rep.outcome_ops.keys()) == {0, 1}
        assert all(
            isinstance(v, UnitaryGateRep) for v in rep.outcome_ops.values()
        )
        assert np.array_equal(rep.outcome_ops[1].unitary, raw_dict[1])

    def test_from_raw_defaults_include_outcome_true(self):
        def gate_upgrader(raw, qubits):
            return UnitaryGateRep(raw, qubits)

        rep = ZBasisOutcomeOperationDictInstrumentRep.from_raw(
            {0: np.eye(2)}, ("Q0",), gate_upgrader=gate_upgrader
        )
        assert rep.include_outcome is True


class TestStimCircuitInstrumentRep:
    def test_matches_str(self):
        assert StimCircuitInstrumentRep.matches("M 0") is True

    def test_does_not_match_non_str(self):
        assert StimCircuitInstrumentRep.matches(42) is False

    def test_from_raw_constructs_instance(self):
        rep = StimCircuitInstrumentRep.from_raw("M 0", ("Q0",))
        assert isinstance(rep, StimCircuitInstrumentRep)
        assert rep.circuit_str == "M 0"

    def test_from_raw_rejects_non_matching_payload(self):
        with pytest.raises(RepConstructionError):
            StimCircuitInstrumentRep.from_raw(42, ("Q0",))

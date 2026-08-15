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
    @pytest.mark.parametrize("reset,include_outcome", [(None, True), (0, False), (1, True)])
    def test_constructs_instance(self, reset, include_outcome):
        rep = ZBasisProjectionInstrumentRep(reset, include_outcome, ("Q0",))
        assert isinstance(rep, ZBasisProjectionInstrumentRep)
        assert isinstance(rep, InstrumentRep)
        assert rep.reset == reset
        assert rep.include_outcome == include_outcome
        assert rep.qubit_labels == ("Q0",)

    @pytest.mark.parametrize(
        "reset,include_outcome",
        [
            ("not an int", True),  # reset not int/None
            (0, "not a bool"),  # include_outcome not bool
            (2, True),  # reset out of range
        ],
    )
    def test_rejects_invalid_values(self, reset, include_outcome):
        with pytest.raises(RepConstructionError):
            ZBasisProjectionInstrumentRep(reset, include_outcome, ("Q0",))

    def test_accepts_legacy_int_include_outcome(self):
        """Files serialized before `include_outcome` was decoded as a
        `bool` store it as a plain `1`/`0`; construction from decoded
        legacy data must still succeed."""
        rep = ZBasisProjectionInstrumentRep(0, 1, ("Q0",))
        assert rep.include_outcome == 1


class TestZBasisPrePostInstrumentRep:
    def test_constructs_instance(self):
        pre_op = UnitaryGateRep(np.eye(2), ("Q0",))
        post_op = UnitaryGateRep(np.eye(2) * 2, ("Q0",))
        rep = ZBasisPrePostInstrumentRep(1, False, pre_op, post_op, ("Q0",))
        assert isinstance(rep, ZBasisPrePostInstrumentRep)
        assert rep.reset == 1
        assert rep.include_outcome is False
        assert rep.pre_op is pre_op
        assert rep.post_op is post_op

    def test_rejects_non_gaterep_pre_op(self):
        post_op = UnitaryGateRep(np.eye(2), ("Q0",))
        with pytest.raises(RepConstructionError):
            ZBasisPrePostInstrumentRep(None, True, np.eye(2), post_op, ("Q0",))

    def test_rejects_non_gaterep_post_op(self):
        pre_op = UnitaryGateRep(np.eye(2), ("Q0",))
        with pytest.raises(RepConstructionError):
            ZBasisPrePostInstrumentRep(None, True, pre_op, np.eye(2), ("Q0",))

    def test_rejects_invalid_reset(self):
        pre_op = UnitaryGateRep(np.eye(2), ("Q0",))
        post_op = UnitaryGateRep(np.eye(2), ("Q0",))
        with pytest.raises(RepConstructionError):
            ZBasisPrePostInstrumentRep(2, True, pre_op, post_op, ("Q0",))

    def test_rejects_pre_op_qubits_mismatch(self):
        pre_op = UnitaryGateRep(np.eye(2), ("Q1",))
        post_op = UnitaryGateRep(np.eye(2), ("Q0",))
        with pytest.raises(RepConstructionError):
            ZBasisPrePostInstrumentRep(None, True, pre_op, post_op, ("Q0",))

    def test_rejects_post_op_qubits_mismatch(self):
        pre_op = UnitaryGateRep(np.eye(2), ("Q0",))
        post_op = UnitaryGateRep(np.eye(2), ("Q1",))
        with pytest.raises(RepConstructionError):
            ZBasisPrePostInstrumentRep(None, True, pre_op, post_op, ("Q0",))


class TestZBasisOutcomeOperationDictInstrumentRep:
    def test_constructs_instance(self):
        outcome_ops = {
            0: UnitaryGateRep(np.eye(2), ("Q0",)),
            1: UnitaryGateRep(np.eye(2) * 2, ("Q0",)),
        }
        rep = ZBasisOutcomeOperationDictInstrumentRep(outcome_ops, False, ("Q0",))
        assert isinstance(rep, ZBasisOutcomeOperationDictInstrumentRep)
        assert rep.include_outcome is False
        assert rep.outcome_ops == outcome_ops

    def test_rejects_non_mapping(self):
        with pytest.raises(RepConstructionError):
            ZBasisOutcomeOperationDictInstrumentRep([0, 1], True, ("Q0",))

    def test_rejects_non_gaterep_values(self):
        with pytest.raises(RepConstructionError):
            ZBasisOutcomeOperationDictInstrumentRep({0: np.eye(2)}, True, ("Q0",))


class TestStimCircuitInstrumentRep:
    def test_constructs_instance(self):
        rep = StimCircuitInstrumentRep("M 0", ("Q0",))
        assert isinstance(rep, StimCircuitInstrumentRep)
        assert rep.circuit_str == "M 0"

    def test_rejects_non_str(self):
        with pytest.raises(RepConstructionError):
            StimCircuitInstrumentRep(42, ("Q0",))

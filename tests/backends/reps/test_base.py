"""Tester for loqs.backends.reps.base"""

import numpy as np
import pytest

from loqs.backends.reps import (
    OperationRep,
    RepConstructionError,
    StimCircuitGateRep,
    StimCircuitInstrumentRep,
    StimCircuitPayloadMixin,
    UnitaryGateRep,
    OutcomeOperationDictInstrumentRep,
    ZBasisPrePostInstrumentRep,
    GateRep,
    is_rep_compatible,
)


def test_cannot_instantiate_operation_rep_directly():
    with pytest.raises(TypeError):
        OperationRep()  # type: ignore[abstract]


@pytest.mark.parametrize(
    "make,expected",
    [
        (lambda: UnitaryGateRep(np.eye(2), "Q0"), ("Q0",)),
        (lambda: UnitaryGateRep(np.eye(2), 0), (0,)),
        (lambda: UnitaryGateRep(np.eye(4), ["Q0", "Q1"]), ("Q0", "Q1")),
        (lambda: StimCircuitGateRep("X 0"), ()),
    ],
    ids=["str", "int", "sequence", "default"],
)
def test_qubits_normalization(make, expected):
    assert make().qubit_labels == expected


class TestWithQubits:
    def test_returns_new_instance_with_new_qubits(self):
        rep = UnitaryGateRep(np.eye(2), ("Q0",))
        retargeted = rep.with_qubit_labels(("Q1",))
        assert retargeted is not rep
        assert retargeted.qubit_labels == ("Q1",)
        assert rep.qubit_labels == ("Q0",)  # original untouched
        assert np.array_equal(
            retargeted.unitary, np.eye(2)
        )  # payload preserved

    def test_revalidates_against_new_qubit_count(self):
        """Retargeting reconstructs via `__init__`, so a payload that's
        no longer shape-consistent with the new qubit count is rejected
        immediately, rather than silently producing an inconsistent
        instance."""
        rep = UnitaryGateRep(np.eye(2), ("Q0",))
        with pytest.raises(RepConstructionError):
            rep.with_qubit_labels(("Q0", "Q1"))

    def test_cascades_to_nested_operation_rep_fields(self):
        """A composite rep's nested `OperationRep` fields (e.g. `pre_op`/
        `post_op`) are retargeted along with the outer rep, since
        reconstruction requires them to remain consistent with the new
        `qubit_labels`."""
        pre_op = UnitaryGateRep(np.eye(2))
        post_op = UnitaryGateRep(np.eye(2))
        rep = ZBasisPrePostInstrumentRep(None, True, pre_op, post_op)

        retargeted = rep.with_qubit_labels(("Q0",))

        assert retargeted.qubit_labels == ("Q0",)
        assert retargeted.pre_op.qubit_labels == ("Q0",)
        assert retargeted.post_op.qubit_labels == ("Q0",)
        # Originals untouched
        assert pre_op.qubit_labels == ()
        assert post_op.qubit_labels == ()

    def test_cascades_to_mapping_values(self):
        """Nested `OperationRep` values inside a `Mapping` field (e.g.
        `outcome_ops`) are retargeted too."""
        rep = OutcomeOperationDictInstrumentRep(
            {0: UnitaryGateRep(np.eye(2)), 1: UnitaryGateRep(np.eye(2))}, True
        )

        retargeted = rep.with_qubit_labels(("Q0",))

        assert retargeted.qubit_labels == ("Q0",)
        assert retargeted.outcome_ops[0].qubit_labels == ("Q0",)
        assert retargeted.outcome_ops[1].qubit_labels == ("Q0",)


@pytest.mark.parametrize(
    "rep_cls,accepted,expected",
    [
        (UnitaryGateRep, [UnitaryGateRep], True),
        (UnitaryGateRep, [StimCircuitGateRep], False),
        (StimCircuitGateRep, [GateRep], True),
    ],
    ids=["exact", "no-match", "coarse-grained"],
)
def test_is_rep_compatible(rep_cls, accepted, expected):
    assert is_rep_compatible(rep_cls, accepted) is expected


@pytest.mark.parametrize(
    "rep",
    [StimCircuitGateRep("X 0"), StimCircuitInstrumentRep("M 0")],
    ids=["gate", "instrument"],
)
def test_stim_circuit_reps_share_payload_mixin(rep):
    assert isinstance(rep, StimCircuitPayloadMixin)

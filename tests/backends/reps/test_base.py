"""Tester for loqs.backends.reps.base"""

import numpy as np
import pytest

from loqs.backends.reps import (
    GateRep,
    InstrumentRep,
    OperationRep,
    RepConstructionError,
    StimCircuitGateRep,
    StimCircuitInstrumentRep,
    StimCircuitPayloadMixin,
    UnitaryGateRep,
    ZBasisOutcomeOperationDictInstrumentRep,
    ZBasisPrePostInstrumentRep,
    is_rep_compatible,
)


class TestOperationRepAbstract:
    def test_cannot_instantiate_operation_rep_directly(self):
        with pytest.raises(TypeError):
            OperationRep()  # type: ignore[abstract]

    def test_cannot_instantiate_gaterep_directly(self):
        with pytest.raises(TypeError):
            GateRep()  # type: ignore[abstract]

    def test_cannot_instantiate_instrumentrep_directly(self):
        with pytest.raises(TypeError):
            InstrumentRep()  # type: ignore[abstract]


class TestQubitsNormalization:
    def test_single_str_qubit_is_wrapped_in_tuple(self):
        rep = UnitaryGateRep(np.eye(2), "Q0")
        assert rep.qubit_labels == ("Q0",)

    def test_single_int_qubit_is_wrapped_in_tuple(self):
        rep = UnitaryGateRep(np.eye(2), 0)
        assert rep.qubit_labels == (0,)

    def test_sequence_qubits_becomes_tuple(self):
        rep = UnitaryGateRep(np.eye(4), ["Q0", "Q1"])
        assert rep.qubit_labels == ("Q0", "Q1")

    def test_default_qubits_is_empty_tuple(self):
        rep = StimCircuitGateRep("X 0")
        assert rep.qubit_labels == ()


class TestStr:
    def test_str_includes_class_name_and_serialize_attrs(self):
        rep = UnitaryGateRep(np.eye(2), ("Q0",))
        s = str(rep)
        assert s.startswith("UnitaryGateRep(")
        assert "unitary=" in s
        assert "qubit_labels=('Q0',)" in s


class TestWithQubits:
    def test_returns_new_instance_with_new_qubits(self):
        rep = UnitaryGateRep(np.eye(2), ("Q0",))
        retargeted = rep.with_qubit_labels(("Q1",))
        assert retargeted is not rep
        assert retargeted.qubit_labels == ("Q1",)
        assert rep.qubit_labels == ("Q0",)  # original untouched
        assert np.array_equal(retargeted.unitary, np.eye(2))  # payload preserved

    def test_single_qubit_str_is_wrapped(self):
        rep = StimCircuitGateRep("X 0", ())
        retargeted = rep.with_qubit_labels("Q0")
        assert retargeted.qubit_labels == ("Q0",)

    def test_works_for_instrument_reps_too(self):
        rep = StimCircuitInstrumentRep("M 0", ())
        retargeted = rep.with_qubit_labels(("Q0",))
        assert retargeted.qubit_labels == ("Q0",)
        assert retargeted.circuit_str == "M 0"

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
        rep = ZBasisOutcomeOperationDictInstrumentRep(
            {0: UnitaryGateRep(np.eye(2)), 1: UnitaryGateRep(np.eye(2))}, True
        )

        retargeted = rep.with_qubit_labels(("Q0",))

        assert retargeted.qubit_labels == ("Q0",)
        assert retargeted.outcome_ops[0].qubit_labels == ("Q0",)
        assert retargeted.outcome_ops[1].qubit_labels == ("Q0",)


class TestIsRepCompatible:
    def test_exact_class_match(self):
        assert is_rep_compatible(UnitaryGateRep, [UnitaryGateRep]) is True

    def test_no_match(self):
        assert is_rep_compatible(UnitaryGateRep, [StimCircuitGateRep]) is False

    def test_coarse_grained_base_class_accepted(self):
        """A caller may declare a coarse-grained capability like `GateRep`
        to mean "accepts any gate representation" -- structurally
        impossible with the old exhaustive enum-value list."""
        assert is_rep_compatible(UnitaryGateRep, [GateRep]) is True
        assert is_rep_compatible(StimCircuitGateRep, [GateRep]) is True

    def test_empty_accepted_list_is_never_compatible(self):
        assert is_rep_compatible(UnitaryGateRep, []) is False


class TestStimCircuitPayloadMixin:
    """`StimCircuitGateRep`/`StimCircuitInstrumentRep` share their payload
    storage and construction logic via `StimCircuitPayloadMixin` (rather
    than duplicating it), since callers like `DictNoiseModel`'s STIM-text
    merging logic need to treat both uniformly regardless of whether the
    underlying rep is a gate or an instrument.
    """

    def test_stim_circuit_gaterep_is_a_mixin_instance(self):
        assert isinstance(StimCircuitGateRep("X 0"), StimCircuitPayloadMixin)

    def test_stim_circuit_instrumentrep_is_a_mixin_instance(self):
        assert isinstance(StimCircuitInstrumentRep("M 0"), StimCircuitPayloadMixin)

    def test_non_stim_circuit_reps_are_not_mixin_instances(self):
        assert not isinstance(UnitaryGateRep(np.eye(2)), StimCircuitPayloadMixin)

    def test_mixin_does_not_unify_gaterep_and_instrumentrep_dispatch(self):
        """The mixin is purely a shared-mechanics helper; `StimCircuitGateRep`
        and `StimCircuitInstrumentRep` remain unrelated to each other from a
        `GateRep`/`InstrumentRep` dispatch perspective."""
        assert not issubclass(StimCircuitGateRep, InstrumentRep)
        assert not issubclass(StimCircuitInstrumentRep, GateRep)
        assert issubclass(StimCircuitGateRep, GateRep)
        assert issubclass(StimCircuitInstrumentRep, InstrumentRep)

    def test_mixin_cannot_be_instantiated_as_an_operationrep(self):
        """`StimCircuitPayloadMixin` is not itself an `OperationRep` subclass."""
        assert not issubclass(StimCircuitPayloadMixin, OperationRep)


class TestRepConstructionError:
    def test_is_an_exception(self):
        assert issubclass(RepConstructionError, Exception)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(RepConstructionError, match="custom message"):
            raise RepConstructionError("custom message")

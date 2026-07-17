"""Tester for loqs.backends.reps.base"""

import pytest

from loqs.backends.reps import (
    GateRep,
    InstrumentRep,
    OperationRep,
    RepConstructionError,
    StimCircuitGateRep,
    StimCircuitInstrumentRep,
    UnitaryGateRep,
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
        rep = UnitaryGateRep(None, "Q0")
        assert rep.qubits == ("Q0",)

    def test_single_int_qubit_is_wrapped_in_tuple(self):
        rep = UnitaryGateRep(None, 0)
        assert rep.qubits == (0,)

    def test_sequence_qubits_becomes_tuple(self):
        rep = UnitaryGateRep(None, ["Q0", "Q1"])
        assert rep.qubits == ("Q0", "Q1")

    def test_default_qubits_is_empty_tuple(self):
        rep = StimCircuitGateRep("X 0")
        assert rep.qubits == ()


class TestStr:
    def test_str_includes_class_name_and_serialize_attrs(self):
        rep = UnitaryGateRep(None, ("Q0",))
        s = str(rep)
        assert s.startswith("UnitaryGateRep(")
        assert "unitary=None" in s
        assert "qubits=('Q0',)" in s


class TestWithQubits:
    def test_returns_shallow_copy_with_new_qubits(self):
        rep = UnitaryGateRep("original_payload", ("Q0",))
        retargeted = rep.with_qubits(("Q1", "Q2"))
        assert retargeted is not rep
        assert retargeted.qubits == ("Q1", "Q2")
        assert rep.qubits == ("Q0",)  # original untouched
        assert retargeted.unitary == "original_payload"  # payload preserved

    def test_single_qubit_str_is_wrapped(self):
        rep = StimCircuitGateRep("X 0", ())
        retargeted = rep.with_qubits("Q0")
        assert retargeted.qubits == ("Q0",)

    def test_works_for_instrument_reps_too(self):
        rep = StimCircuitInstrumentRep("M 0", ())
        retargeted = rep.with_qubits(("Q0",))
        assert retargeted.qubits == ("Q0",)
        assert retargeted.circuit_str == "M 0"


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


class TestRepConstructionError:
    def test_is_an_exception(self):
        assert issubclass(RepConstructionError, Exception)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(RepConstructionError, match="custom message"):
            raise RepConstructionError("custom message")

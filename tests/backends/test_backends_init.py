"""Tester for loqs.backends (package-level dispatch/availability helpers)."""

import numpy as np
import pytest

from loqs.backends import (
    DictNoiseModel,
    ListPhysicalCircuit,
    NumpyStatevectorQuantumState,
    propagate_state,
)
import loqs.backends as backends_module
from loqs.backends import (
    _check_backend_availability,
    get_available_backends,
    get_backend_error,
    is_backend_available,
)
from loqs.backends.reps import (
    UnitaryGateRep,
    QSimSuperopGateRep,
    ZBasisProjectionInstrumentRep,
)


class TestBackendAvailability:
    @pytest.fixture(autouse=True)
    def isolated_registry(self, monkeypatch):
        """Registrations made by these tests don't leak into the global
        registry."""
        monkeypatch.setattr(backends_module, "_backend_availability", {})

    def test_check_backend_availability_success(self):
        assert _check_backend_availability("_test_numpy", "numpy") is True
        assert is_backend_available("_test_numpy") is True
        assert get_backend_error("_test_numpy") is None
        assert "_test_numpy" in get_available_backends()

    def test_check_backend_availability_failure(self):
        assert (
            _check_backend_availability(
                "_test_missing", "loqs._no_such_module"
            )
            is False
        )
        assert is_backend_available("_test_missing") is False
        assert get_backend_error("_test_missing") is not None
        assert "_test_missing" not in get_available_backends()


def test_stim_dispatch_registered_when_stim_available():
    """`DictNoiseModel.get_reps` silently loses its STIM dispatch if
    `loqs.backends` imports its submodules before defining the availability
    helpers, because `dictmodel` swallows the resulting `ImportError`."""
    pytest.importorskip("stim")
    from loqs.backends.circuit.stimcircuit import STIMPhysicalCircuit

    registry = DictNoiseModel.__dict__["get_reps"].dispatcher.registry
    assert STIMPhysicalCircuit in registry


class TestLazyBackendImportErrors:
    """`loqs.backends.__getattr__` raises `ImportError` for an optional
    backend class whose third-party package isn't available."""

    def test_raises_import_error_when_unavailable(self, monkeypatch):
        monkeypatch.setitem(
            backends_module._backend_availability,
            "stim_state",
            backends_module.BackendAvailability(
                "stim_state", False, "forced off"
            ),
        )
        with pytest.raises(ImportError):
            getattr(backends_module, "STIMQuantumState")

    def test_lazy_import_succeeds_when_available(self):
        pytest.importorskip("pygsti")
        assert getattr(backends_module, "safe_time_dependent_evotype")

    def test_unknown_attribute_raises_attribute_error(self):
        with pytest.raises(AttributeError):
            getattr(backends_module, "NotARealBackendClass")


_X = np.array([[0, 1], [1, 0]], dtype=complex)


class TestPropagateState:
    def _build_model_and_state(self, instreps):
        model = DictNoiseModel(
            {"X": UnitaryGateRep(_X, ())},
            {"M": ZBasisProjectionInstrumentRep(None, True, ())},
            gatereps=[UnitaryGateRep],
            instreps=instreps,
        )
        state = NumpyStatevectorQuantumState(1, ["Q0"], seed=20260716)
        circuit = ListPhysicalCircuit([[("X", ("Q0",)), ("M", ("Q0",))]])
        return model, state, circuit

    def test_inplace_default(self):
        model, state, circuit = self._build_model_and_state(
            [ZBasisProjectionInstrumentRep]
        )
        result_state, outcomes = propagate_state(circuit, model, state)
        assert result_state is state
        assert outcomes["Q0"] == [1]

    def test_not_inplace_returns_new_state(self):
        model, state, circuit = self._build_model_and_state(
            [ZBasisProjectionInstrumentRep]
        )
        result_state, outcomes = propagate_state(
            circuit, model, state, inplace=False
        )
        assert result_state is not state
        assert outcomes["Q0"] == [1]
        # Original state must be untouched (X never applied to it)
        assert np.isclose(state.state[0], 1)

    def test_no_matching_gate_rep_raises(self):
        model = DictNoiseModel(
            {"X": QSimSuperopGateRep(np.eye(4), ())},
            {},
            gatereps=[QSimSuperopGateRep],
            instreps=[ZBasisProjectionInstrumentRep],
        )
        state = NumpyStatevectorQuantumState(1, ["Q0"])
        circuit = ListPhysicalCircuit([[("X", ("Q0",))]])
        with pytest.raises(AssertionError):
            propagate_state(circuit, model, state)

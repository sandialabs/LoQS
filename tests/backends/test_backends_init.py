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
    StimCircuitInstrumentRep,
    ZBasisPrePostInstrumentRep,
    ZBasisProjectionInstrumentRep,
)


class TestBackendAvailability:
    def test_check_backend_availability_success(self):
        assert _check_backend_availability("_test_numpy", "numpy") is True
        assert is_backend_available("_test_numpy") is True
        assert get_backend_error("_test_numpy") is None
        assert "_test_numpy" in get_available_backends()

    def test_check_backend_availability_failure(self):
        assert (
            _check_backend_availability("_test_missing", "loqs._no_such_module")
            is False
        )
        assert is_backend_available("_test_missing") is False
        assert get_backend_error("_test_missing") is not None
        assert "_test_missing" not in get_available_backends()

    def test_is_backend_available_unknown_name_is_false(self):
        assert is_backend_available("_never_registered") is False

    def test_get_backend_error_unknown_name_is_none(self):
        assert get_backend_error("_never_registered") is None

    def test_fresh_import_registers_stim_dispatch_when_stim_available(self):
        """`loqs/backends/__init__.py` must define `BackendAvailability`/
        `_check_backend_availability`/`is_backend_available`/
        `get_backend_error` *before* the `.reps`/`.circuit`/`.model`/
        `.state` imports (LoQS#72's model-backend consolidation, see
        `loqs.backends.model.dictmodel`, requires `dictmodel.py` to
        import `loqs.backends.circuit.stimcircuit` eagerly, and
        `stimcircuit.py` reaches back into `loqs.backends` for
        `is_backend_available`).

        If that ordering ever regresses, this does *not* crash the
        import -- `dictmodel.py`'s own `try: from
        loqs.backends.circuit.stimcircuit import STIMPhysicalCircuit
        except ImportError: STIMPhysicalCircuit = None` silently catches
        the resulting `ImportError` (it looks identical to `stim`
        genuinely not being installed), so `DictNoiseModel.get_reps`
        simply never gets its `STIMPhysicalCircuit` dispatch registered
        -- a silent, no-crash false negative, confirmed by directly
        reverting the import order and observing exactly this symptom
        (`STIMPhysicalCircuit not in
        DictNoiseModel.get_reps.dispatcher.registry`) rather than any
        exception. This test therefore checks that the registration
        actually happened (in a fresh subprocess, since Python caches
        modules and this refactor's own test suite has already imported
        `loqs.backends` once by the time any test runs), not merely that
        import doesn't raise."""
        pytest.importorskip("stim")
        import subprocess
        import sys

        script = (
            "import stim\n"
            "from loqs.backends.model.dictmodel import DictNoiseModel\n"
            "from loqs.backends.circuit.stimcircuit import STIMPhysicalCircuit\n"
            "registry = DictNoiseModel.__dict__['get_reps'].dispatcher.registry\n"
            "assert STIMPhysicalCircuit in registry, "
            "f'STIM dispatch not registered: {registry.keys()}'\n"
            "print('OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout


class TestLazyBackendImportErrors:
    """`loqs.backends.__getattr__` raises a clear ImportError (naming the
    specific missing dependency) for each optional backend class when its
    underlying third-party package isn't available."""

    @pytest.mark.parametrize(
        "attr_name,backend_key",
        [
            ("PyGSTiPhysicalCircuit", "pygsti_circuit"),
            ("STIMPhysicalCircuit", "stim_circuit"),
            ("PyGSTiNoiseModel", "pygsti_model"),
            ("safe_time_dependent_evotype", "pygsti_model"),
            ("PyGSTiEmbeddedOpMemoryWarning", "pygsti_model"),
            ("STIMQuantumState", "stim_state"),
            ("QSimQuantumState", "qsim_state"),
        ],
    )
    def test_raises_import_error_when_unavailable(
        self, attr_name, backend_key, monkeypatch
    ):
        original = backends_module._backend_availability[backend_key]
        monkeypatch.setitem(
            backends_module._backend_availability,
            backend_key,
            backends_module.BackendAvailability(backend_key, False, "forced off"),
        )
        try:
            with pytest.raises(ImportError, match="forced off"):
                getattr(backends_module, attr_name)
        finally:
            backends_module._backend_availability[backend_key] = original

    def test_unknown_attribute_raises_attribute_error(self):
        with pytest.raises(AttributeError, match="has no attribute"):
            getattr(backends_module, "NotARealBackendClass")

    @pytest.mark.parametrize(
        "attr_name",
        ["safe_time_dependent_evotype", "PyGSTiEmbeddedOpMemoryWarning"],
    )
    def test_lazy_import_succeeds_when_available(self, attr_name):
        """These two pyGSTi-model-backend attributes are less commonly
        imported directly than the main backend classes, so their
        successful-import branch is worth pinning explicitly."""
        pytest.importorskip("pygsti")
        assert is_backend_available("pygsti_model")
        obj = getattr(backends_module, attr_name)
        assert obj is not None


_X = np.array([[0, 1], [1, 0]], dtype=complex)


@pytest.mark.xfail(
    strict=True,
    reason="DictNoiseModel.__init__ does not yet accept gate_dict/inst_dict "
    "as separate parameters",
)
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

    def test_multiple_compatible_instreps_all_considered(self):
        """`model.output_instrument_reps` may list more than one compatible
        InstrumentRep; `propagate_state` must consider all of them (not
        just the first) when intersecting with `state.input_reps`."""
        model, state, circuit = self._build_model_and_state(
            [
                ZBasisProjectionInstrumentRep,
                ZBasisPrePostInstrumentRep,
            ]
        )
        result_state, outcomes = propagate_state(circuit, model, state)
        assert result_state is state
        assert outcomes["Q0"] == [1]

    def test_no_matching_gate_rep_raises(self):
        model = DictNoiseModel(
            {"X": QSimSuperopGateRep(np.eye(4), ())}, {},
            gatereps=[QSimSuperopGateRep],
            instreps=[ZBasisProjectionInstrumentRep],
        )
        state = NumpyStatevectorQuantumState(1, ["Q0"])
        circuit = ListPhysicalCircuit([[("X", ("Q0",))]])
        with pytest.raises(AssertionError, match="Could not find matching gate rep"):
            propagate_state(circuit, model, state)

    def test_no_matching_instrument_rep_raises(self):
        model = DictNoiseModel(
            {"X": UnitaryGateRep(np.eye(2), ())}, {"M": StimCircuitInstrumentRep("M 0", ())},
            gatereps=[UnitaryGateRep],
            instreps=[StimCircuitInstrumentRep],
        )
        state = NumpyStatevectorQuantumState(1, ["Q0"])
        circuit = ListPhysicalCircuit([[("X", ("Q0",))]])
        with pytest.raises(
            AssertionError, match="Could not find matching instrument rep"
        ):
            propagate_state(circuit, model, state)

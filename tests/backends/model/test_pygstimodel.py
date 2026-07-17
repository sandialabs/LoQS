"""Tester for loqs.backends.model.pygstimodel"""

import numpy as np
import pytest

pygsti = pytest.importorskip("pygsti")

from pygsti.baseobjs import Label
from pygsti.baseobjs.label import LabelStr
from pygsti.baseobjs.statespace import QubitSpace
from pygsti.modelmembers.instruments import Instrument
from pygsti.modelmembers.operations import FullArbitraryOp
from pygsti.models import ExplicitOpModel

import loqs.backends as backends_module
from loqs.backends.model.dictmodel import DictNoiseModel
from loqs.backends.model.pygstimodel import PyGSTiNoiseModel
from loqs.backends.reps import (
    GateRep,
    InstrumentRep,
    KrausGateRep,
    PTMGateRep,
    QSimSuperoperatorGateRep,
    RepConstructionError,
    StimCircuitGateRep,
    StimCircuitInstrumentRep,
    UnitaryGateRep,
    ZBasisOutcomeOperationDictInstrumentRep,
    ZBasisProjectionInstrumentRep,
)

# TP-preserving (sum_i K_i K_i^dagger = I) amplitude-damping Kraus operators
# -- a genuinely non-unitary channel, used to exercise the GateRep fallback
# loop (UnitaryGateRep fails for it; PTMGateRep/QSimSuperoperatorGateRep/
# KrausGateRep succeed).
_GAMMA = 0.1
_K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1 - _GAMMA)]], dtype=complex)
_K1 = np.array([[0.0, np.sqrt(_GAMMA)], [0.0, 0.0]], dtype=complex)
_AMP_DAMP_SUPEROP = FullArbitraryOp.from_kraus_operators([_K0, _K1], "pp").to_dense()

_ZBASIS_P0 = np.diag([1, 0, 0, 1]).astype(complex)
_ZBASIS_P1 = np.diag([0, 1, 1, 0]).astype(complex)


def _build_explicit_model():
    model = ExplicitOpModel(state_space=QubitSpace(["Q0"]), basis="pp")
    model.operations[Label("Gxpi", "Q0")] = FullArbitraryOp(np.eye(4), basis="pp")
    model.operations[Label("Gad", "Q0")] = FullArbitraryOp(_AMP_DAMP_SUPEROP, basis="pp")
    model.instruments[Label("Iz", "Q0")] = Instrument(
        {"0": _ZBASIS_P0, "1": _ZBASIS_P1}
    )
    return model


def _build_implicit_model():
    pspec = pygsti.processors.QubitProcessorSpec(
        1,
        gate_names=["Gxpi"],
        qubit_labels=["Q0"],
        availability={"Gxpi": "all-permutations"},
    )
    model = pygsti.models.create_crosstalk_free_model(pspec)
    model.operation_blks["layers"][Label("Gad", "Q0")] = FullArbitraryOp(
        _AMP_DAMP_SUPEROP, basis="pp"
    )
    model.instrument_blks["layers"][Label("Iz", "Q0")] = Instrument(
        {"0": _ZBASIS_P0, "1": _ZBASIS_P1}
    )
    return model


class TestConstruction:
    def test_raises_import_error_when_unavailable(self):
        original = backends_module._backend_availability["pygsti_model"]
        backends_module._backend_availability["pygsti_model"] = (
            backends_module.BackendAvailability("pygsti_model", False)
        )
        try:
            with pytest.raises(
                ImportError, match="PyGSTi model backend is not available"
            ):
                PyGSTiNoiseModel(_build_explicit_model())
        finally:
            backends_module._backend_availability["pygsti_model"] = original

    def test_gate_keys_labelstr_has_no_qubits(self):
        """A `LabelStr`-keyed gate (no qubit arguments at all, e.g. a
        global idle) must appear in `gate_keys` as a bare 1-tuple,
        skipping the qubit-aliasing step entirely."""
        model = ExplicitOpModel(state_space=QubitSpace(["Q0"]), basis="pp")
        model.operations[LabelStr("Gidle")] = FullArbitraryOp(
            np.eye(4), basis="pp"
        )
        pgm = PyGSTiNoiseModel(model)
        assert ("Gidle",) in pgm.gate_keys

    def test_from_explicit_op_model(self):
        pgm = PyGSTiNoiseModel(_build_explicit_model())
        assert ("Gxpi", ["Q0"]) in pgm.gate_keys
        assert ("Iz", ["Q0"]) in pgm.instrument_keys
        assert pgm.use_embedded_op is False

    def test_from_implicit_op_model(self):
        pgm = PyGSTiNoiseModel(_build_implicit_model())
        assert ("Gxpi", ["Q0"]) in pgm.gate_keys
        assert ("Iz", ["Q0"]) in pgm.instrument_keys
        assert pgm.use_embedded_op is True

    def test_default_qubit_aliases_are_identity(self):
        pgm = PyGSTiNoiseModel(_build_explicit_model())
        assert pgm.qubit_aliases == {"Q0": "Q0"}

    def test_qubit_aliases_mapping_renames_qubits(self):
        pgm = PyGSTiNoiseModel(_build_explicit_model(), qubit_aliases={"Q0": "MyQubit"})
        assert pgm.gate_keys == [("Gxpi", ["MyQubit"]), ("Gad", ["MyQubit"])]

    def test_qubit_aliases_sequence_renames_qubits(self):
        pgm = PyGSTiNoiseModel(_build_explicit_model(), qubit_aliases=["MyQubit"])
        assert pgm.gate_keys == [("Gxpi", ["MyQubit"]), ("Gad", ["MyQubit"])]

    def test_invalid_qubit_aliases_type_raises(self):
        with pytest.raises(TypeError, match="Invalid type for qubit aliases"):
            PyGSTiNoiseModel(_build_explicit_model(), qubit_aliases=42)

    def test_non_conforming_qubit_label_without_alias_raises(self):
        model = ExplicitOpModel(state_space=QubitSpace(["A0"]), basis="pp")
        model.operations[Label("Gxpi", "A0")] = FullArbitraryOp(np.eye(4), basis="pp")
        with pytest.raises(AssertionError, match="Model must use int or str"):
            PyGSTiNoiseModel(model)

    def test_qubit_aliases_do_not_bypass_the_underlying_model_label_check(self):
        """`qubit_aliases` lets you present different qubit labels to LoQS
        on top of a conforming model; it does not let you use a model with
        non-conforming raw labels (e.g. `"A0"`), since the label-format
        restriction applies to the underlying pyGSTi model itself."""
        model = ExplicitOpModel(state_space=QubitSpace(["A0"]), basis="pp")
        model.operations[Label("Gxpi", "A0")] = FullArbitraryOp(np.eye(4), basis="pp")
        with pytest.raises(AssertionError, match="Model must use int or str"):
            PyGSTiNoiseModel(model, qubit_aliases={"A0": "Q0"})

    def test_copy_constructor_copies_model_and_aliases(self):
        pgm = PyGSTiNoiseModel(_build_explicit_model(), qubit_aliases={"Q0": "MyQubit"})
        pgm_copy = PyGSTiNoiseModel(pgm)
        assert pgm_copy.model is pgm.model
        assert pgm_copy.gate_dict is pgm.gate_dict
        assert pgm_copy.inst_dict is pgm.inst_dict
        assert pgm_copy.use_embedded_op == pgm.use_embedded_op
        assert pgm_copy.qubit_aliases == pgm.qubit_aliases
        assert pgm_copy.gate_keys == pgm.gate_keys

    def test_copy_constructor_allows_overriding_qubit_aliases(self):
        pgm = PyGSTiNoiseModel(_build_explicit_model())
        pgm_copy = PyGSTiNoiseModel(pgm, qubit_aliases={"Q0": "OtherQubit"})
        assert pgm_copy.qubit_aliases == {"Q0": "OtherQubit"}

    def test_invalid_model_type_raises_type_error(self):
        with pytest.raises(TypeError, match="Cannot cast .* to PyGSTiNoiseModel"):
            PyGSTiNoiseModel(42)

    def test_dictnoisemodel_raises_not_implemented_error(self):
        with pytest.raises(NotImplementedError, match="Build explicit op model"):
            PyGSTiNoiseModel(DictNoiseModel(({}, {})))


class TestGetGateRep:
    @pytest.fixture
    def pgm(self):
        return PyGSTiNoiseModel(_build_explicit_model())

    def test_unitary(self, pgm):
        rep, reptype = pgm._get_gate_rep("Gxpi", ["Q0"], [UnitaryGateRep])
        assert reptype is UnitaryGateRep
        assert np.shape(rep) == (2, 2)

    def test_kraus_operators(self, pgm):
        rep, reptype = pgm._get_gate_rep("Gad", ["Q0"], [KrausGateRep])
        assert reptype is KrausGateRep
        assert len(rep) >= 1

    def test_ptm(self, pgm):
        rep, reptype = pgm._get_gate_rep("Gad", ["Q0"], [PTMGateRep])
        assert reptype is PTMGateRep
        assert np.shape(rep) == (4, 4)

    def test_qsim_superoperator(self, pgm):
        rep, reptype = pgm._get_gate_rep("Gad", ["Q0"], [QSimSuperoperatorGateRep])
        assert reptype is QSimSuperoperatorGateRep
        assert np.shape(rep) == (4, 4)

    def test_fallback_skips_failing_candidate(self, pgm):
        """Requesting UnitaryGateRep first (fails for the non-unitary
        amplitude-damping channel) falls through to
        QSimSuperoperatorGateRep rather than raising."""
        rep, reptype = pgm._get_gate_rep(
            "Gad", ["Q0"], [UnitaryGateRep, QSimSuperoperatorGateRep]
        )
        assert reptype is QSimSuperoperatorGateRep
        assert np.shape(rep) == (4, 4)

    def test_no_valid_candidate_raises(self, pgm):
        with pytest.raises(RepConstructionError, match="Failed to create gate rep for any of"):
            pgm._get_gate_rep("Gad", ["Q0"], [UnitaryGateRep])


class TestGetInstrumentRep:
    @pytest.fixture
    def pgm(self):
        return PyGSTiNoiseModel(_build_explicit_model())

    def test_zbasis_projection(self, pgm):
        rep, reptype = pgm._get_instrument_rep(
            "Iz", ["Q0"], [ZBasisProjectionInstrumentRep]
        )
        assert reptype is ZBasisProjectionInstrumentRep
        reset, include_outcomes = rep
        assert reset == 0  # zbasis_proj_resets=True by default
        assert include_outcomes is True

    def test_zbasis_outcome_operation_dict(self, pgm):
        rep, reptype = pgm._get_instrument_rep(
            "Iz", ["Q0"], [ZBasisOutcomeOperationDictInstrumentRep]
        )
        assert reptype is ZBasisOutcomeOperationDictInstrumentRep
        outcome_dict, include_outcomes = rep
        assert include_outcomes is True
        assert set(outcome_dict.keys()) == {(0,), (1,)}

    def test_no_valid_candidate_raises(self, pgm):
        # STIM_CIRCUIT_STR is a real InstrumentRep member, but _get_rep's
        # if/elif only ever handles ZBASIS_PROJECTION/
        # ZBASIS_OUTCOME_OPERATION_DICT, so this exercises the same
        # "no candidate worked" final-failure path as the gate-rep case.
        with pytest.raises(
            RepConstructionError, match="Failed to create instrument rep for any of"
        ):
            pgm._get_instrument_rep("Iz", ["Q0"], [StimCircuitInstrumentRep])


class TestTimeDependence:
    """`get_gate_duration`/`get_instrument_duration` (and the
    `add_gate_duration_to_layer`/`add_layer_duration_to_current_time`
    helpers they feed into via `get_reps`) are only active when
    `use_time_dependence=True`; otherwise they always return 0."""

    @pytest.fixture
    def pgm(self):
        return PyGSTiNoiseModel(_build_explicit_model())

    @pytest.fixture
    def pgm_time_dependent(self):
        return PyGSTiNoiseModel(
            _build_explicit_model(),
            use_time_dependence=True,
            default_gate_durations={Label("Gxpi", "Q0"): 5, "Gad": 7},
            default_instrument_durations={Label("Iz", "Q0"): 3},
        )

    @pytest.mark.parametrize(
        "method,label",
        [
            ("get_gate_duration", Label("Gxpi", "Q0")),
            ("get_instrument_duration", Label("Iz", "Q0")),
        ],
    )
    def test_disabled_always_returns_zero(self, pgm, method, label):
        assert getattr(pgm, method)(label) == 0

    def test_gate_duration_from_label_with_time(self, pgm_time_dependent):
        label = Label("Gxpi", "Q0", time=1.5)
        assert pgm_time_dependent.get_gate_duration(label) == 1.5

    def test_instrument_duration_from_label_with_time(self, pgm_time_dependent):
        label = Label("Iz", "Q0", time=2.5)
        assert pgm_time_dependent.get_instrument_duration(label) == 2.5

    @pytest.mark.parametrize(
        "method",
        ["get_gate_duration", "get_instrument_duration"],
    )
    def test_layer_label_raises_value_error(self, pgm_time_dependent, method):
        layer_label = Label(
            [Label("Gxpi", "Q0"), Label("Gypi", "Q1")], time=1.0
        )
        with pytest.raises(ValueError, match="LayerTupTupWithTime"):
            getattr(pgm_time_dependent, method)(layer_label)

    def test_gate_duration_no_defaults_raises_value_error(self):
        pgm = PyGSTiNoiseModel(
            _build_explicit_model(), use_time_dependence=True
        )
        with pytest.raises(ValueError, match="no default gate durations"):
            pgm.get_gate_duration(Label("Gxpi", "Q0"))

    def test_instrument_duration_no_defaults_raises_value_error(self):
        pgm = PyGSTiNoiseModel(
            _build_explicit_model(), use_time_dependence=True
        )
        with pytest.raises(ValueError, match="no default instrument durations"):
            pgm.get_instrument_duration(Label("Iz", "Q0"))

    def test_gate_duration_exact_label_match(self, pgm_time_dependent):
        assert pgm_time_dependent.get_gate_duration(Label("Gxpi", "Q0")) == 5

    def test_gate_duration_name_only_fallback(self, pgm_time_dependent):
        # "Gad" is registered by name only (no qubit), so an exact-label
        # lookup for Label("Gad", "Q0") must fall back to a name-only
        # lookup.
        assert pgm_time_dependent.get_gate_duration(Label("Gad", "Q0")) == 7

    def test_instrument_duration_exact_label_match(self, pgm_time_dependent):
        assert pgm_time_dependent.get_instrument_duration(Label("Iz", "Q0")) == 3

    def test_gate_duration_not_found_raises_key_error(self, pgm_time_dependent):
        with pytest.raises(KeyError, match="not available by label or name"):
            pgm_time_dependent.get_gate_duration(Label("Gunknown", "Q0"))

    def test_instrument_duration_not_found_raises_key_error(
        self, pgm_time_dependent
    ):
        with pytest.raises(KeyError, match="not available by label or name"):
            pgm_time_dependent.get_instrument_duration(Label("Iunknown", "Q0"))

    def test_get_reps_advances_current_time(self):
        """End-to-end: `get_reps` on a real (multi-layer) pyGSTi circuit
        with `use_time_dependence=True` must advance `current_time` by
        each layer's max gate/instrument duration."""
        from loqs.backends import PyGSTiPhysicalCircuit

        pgm = PyGSTiNoiseModel(
            _build_explicit_model(),
            use_time_dependence=True,
            default_gate_durations={"Gxpi": 5, "Gad": 2},
            default_instrument_durations={"Iz": 3},
        )
        assert pgm.current_time == 0.0

        circuit = PyGSTiPhysicalCircuit(
            [("Gxpi", "Q0"), ("Gad", "Q0"), ("Iz", "Q0")], ["Q0"]
        )
        pgm.get_reps(
            circuit,
            [UnitaryGateRep, QSimSuperoperatorGateRep],
            [ZBasisProjectionInstrumentRep],
        )

        # Gxpi (5) and Gad (2) can't share a layer (both act on Q0), and
        # Iz (3) is its own layer too -- so total elapsed time is 5+2+3=10.
        assert pgm.current_time == 10.0


class TestDenseEmbeddingWarningHelpers:
    def test_safe_time_dependent_evotype(self):
        from pygsti.evotypes import Evotype
        from loqs.backends.model.pygstimodel import safe_time_dependent_evotype

        evotype = safe_time_dependent_evotype("densitymx")
        assert isinstance(evotype, Evotype)

    def test_check_for_dense_embedding_issues_no_warning(self, recwarn):
        """A model with no `EmbeddedOp`s at all must not warn."""
        pgm = PyGSTiNoiseModel(_build_explicit_model())
        pgm.check_for_dense_embedding_issues()
        from loqs.backends.model.pygstimodel import (
            PyGSTiEmbeddedOpMemoryWarning,
        )

        assert not any(
            issubclass(w.category, PyGSTiEmbeddedOpMemoryWarning)
            for w in recwarn.list
        )


class TestGetRepsErrorPaths:
    def test_unhandled_component_prefix_raises(self):
        """`get_reps` only knows how to dispatch component names starting
        with `G` (gates) or `I` (instruments); anything else must raise a
        clear `NotImplementedError` rather than silently mis-dispatching."""
        from loqs.backends import PyGSTiPhysicalCircuit

        pgm = PyGSTiNoiseModel(_build_explicit_model())
        circuit = PyGSTiPhysicalCircuit([("Xpi", "Q0")], ["Q0"])
        with pytest.raises(NotImplementedError, match="G/I prefixes"):
            pgm.get_reps(
                circuit, [UnitaryGateRep], [ZBasisProjectionInstrumentRep]
            )

    def test_gate_rep_unsupported_reptype_raises(self):
        pgm = PyGSTiNoiseModel(_build_explicit_model())
        with pytest.raises(
            RepConstructionError, match="Failed to create gate rep for any of"
        ):
            pgm._get_gate_rep("Gxpi", ["Q0"], [StimCircuitGateRep])

    def test_gate_rep_qsim_superoperator_more_than_2_qubits_raises(self):
        model = ExplicitOpModel(state_space=QubitSpace(["Q0", "Q1", "Q2"]), basis="pp")
        model.operations[Label("Gccx", ("Q0", "Q1", "Q2"))] = FullArbitraryOp(
            np.eye(64), basis="pp"
        )
        pgm = PyGSTiNoiseModel(model)
        with pytest.raises(
            RepConstructionError, match="Failed to create gate rep for any of"
        ):
            pgm._get_gate_rep(
                "Gccx", ["Q0", "Q1", "Q2"], [QSimSuperoperatorGateRep]
            )

    def test_kraus_operators_identity_branch(self):
        """`_get_gate_rep`'s KRAUS_OPERATORS branch pre-computes a fixed
        probability whenever a Kraus operator is proportional to the
        identity once its own scale is divided out (always true for any
        *unitary* Kraus component, e.g. a probabilistic-bit-flip channel);
        the amplitude-damping channel used elsewhere in this file instead
        exercises the opposite (non-unitary, `prob=None`) branch."""
        U_X = np.array([[0, 1], [1, 0]], dtype=complex)
        p = 0.2
        K0 = np.sqrt(1 - p) * np.eye(2)
        K1 = np.sqrt(p) * U_X
        superop = FullArbitraryOp.from_kraus_operators([K0, K1], "pp").to_dense()

        model = ExplicitOpModel(state_space=QubitSpace(["Q0"]), basis="pp")
        model.operations[Label("Gbf", "Q0")] = FullArbitraryOp(superop, basis="pp")
        pgm = PyGSTiNoiseModel(model)

        rep, reptype = pgm._get_gate_rep("Gbf", ["Q0"], [KrausGateRep])
        assert reptype is KrausGateRep
        probs = sorted(prob for _, prob in rep)
        assert all(prob is not None for prob in probs)
        assert np.allclose(probs, [p, 1 - p])

    def test_instrument_rep_multiqubit_outcome_keys(self):
        """A multi-qubit instrument's outcome-dict keys are multi-character
        strings (e.g. `"01"`), exercising the multi-char branch of the
        outcome-key-to-tuple conversion (as opposed to the existing
        single-qubit fixture's single-character keys)."""
        model = ExplicitOpModel(state_space=QubitSpace(["Q0", "Q1"]), basis="pp")
        effects = {k: np.eye(16, dtype=complex) for k in ["00", "01", "10", "11"]}
        model.instruments[Label("Izz", ("Q0", "Q1"))] = Instrument(effects)
        pgm = PyGSTiNoiseModel(model)

        rep, reptype = pgm._get_instrument_rep(
            "Izz", ["Q0", "Q1"], [ZBasisOutcomeOperationDictInstrumentRep]
        )
        outcome_dict, include_outcomes = rep
        assert include_outcomes is True
        assert set(outcome_dict.keys()) == {(0, 0), (0, 1), (1, 0), (1, 1)}

    def test_instrument_rep_non_string_outcome_keys(self):
        """An instrument whose outcome-dict keys are already tuples (not
        strings) must pass them through unchanged, skipping the
        string-to-tuple parsing entirely."""
        model = ExplicitOpModel(state_space=QubitSpace(["Q0"]), basis="pp")
        model.instruments[Label("Iz", "Q0")] = Instrument(
            {(0,): np.eye(4, dtype=complex), (1,): np.eye(4, dtype=complex)}
        )
        pgm = PyGSTiNoiseModel(model)

        rep, reptype = pgm._get_instrument_rep(
            "Iz", ["Q0"], [ZBasisOutcomeOperationDictInstrumentRep]
        )
        outcome_dict, include_outcomes = rep
        assert set(outcome_dict.keys()) == {(0,), (1,)}

    def test_instrument_rep_bad_outcome_key_raises(self):
        model = ExplicitOpModel(state_space=QubitSpace(["Q0"]), basis="pp")
        model.instruments[Label("Iz", "Q0")] = Instrument(
            {"a": np.eye(4, dtype=complex), "b": np.eye(4, dtype=complex)}
        )
        pgm = PyGSTiNoiseModel(model)
        with pytest.raises(
            RepConstructionError, match="Failed to create instrument rep for any of"
        ):
            pgm._get_instrument_rep(
                "Iz", ["Q0"], [ZBasisOutcomeOperationDictInstrumentRep]
            )

    def test_get_reps_time_dependence_with_outcome_operation_dict(self):
        """Exercises the ZBASIS_OUTCOME_OPERATION_DICT-specific
        `op.set_time(...)` call, only reachable with `use_time_dependence`
        and that specific InstrumentRep requested (unlike
        `test_get_reps_advances_current_time`, which requests
        ZBASIS_PROJECTION)."""
        from loqs.backends import PyGSTiPhysicalCircuit

        pgm = PyGSTiNoiseModel(
            _build_explicit_model(),
            use_time_dependence=True,
            default_instrument_durations={"Iz": 1},
        )
        circuit = PyGSTiPhysicalCircuit([("Iz", "Q0")], ["Q0"])
        reps = pgm.get_reps(
            circuit, [UnitaryGateRep], [ZBasisOutcomeOperationDictInstrumentRep]
        )
        assert len(reps) == 1
        assert isinstance(reps[0], ZBasisOutcomeOperationDictInstrumentRep)


class TestSerialization:
    def test_round_trip(self, make_temp_path):
        pgm = PyGSTiNoiseModel(
            _build_explicit_model(), qubit_aliases={"Q0": "MyQubit"}
        )

        with make_temp_path(suffix=".json") as tmp_path:
            pgm.write(tmp_path)
            pgm2 = PyGSTiNoiseModel.read(tmp_path)

        assert isinstance(pgm2, PyGSTiNoiseModel)
        assert pgm2.qubit_aliases == {"Q0": "MyQubit"}
        assert pgm2.gate_keys == pgm.gate_keys
        assert pgm2.instrument_keys == pgm.instrument_keys

        rep, reptype = pgm2._get_gate_rep("Gxpi", ["Q0"], [UnitaryGateRep])
        assert reptype is UnitaryGateRep
        assert np.shape(rep) == (2, 2)

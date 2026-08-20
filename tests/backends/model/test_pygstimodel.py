"""Tester for loqs.backends.model.pygstimodel"""

import warnings
from unittest import mock

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
    QSimSuperopGateRep,
    RepConstructionError,
    StimCircuitGateRep,
    StimCircuitInstrumentRep,
    UnitaryGateRep,
    OutcomeOperationDictInstrumentRep,
    ZBasisProjectionInstrumentRep,
)

try:
    import stim  # noqa: F401

    NO_STIM = False
except ImportError:
    NO_STIM = True

# TP-preserving (sum_i K_i K_i^dagger = I) amplitude-damping Kraus operators
# -- a genuinely non-unitary channel, used to exercise the GateRep fallback
# loop (UnitaryGateRep fails for it; PTMGateRep/QSimSuperopGateRep/
# KrausGateRep succeed).
_GAMMA = 0.1
_K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1 - _GAMMA)]], dtype=complex)
_K1 = np.array([[0.0, np.sqrt(_GAMMA)], [0.0, 0.0]], dtype=complex)
_AMP_DAMP_SUPEROP = FullArbitraryOp.from_kraus_operators([_K0, _K1], "pp").to_dense()

# Individual Z-basis measurement branches |0><0|(.)|0><0| and |1><1|(.)|1><1|
# (each a single, non-trace-preserving-on-its-own Kraus operator; their sum
# is the fully-dephasing channel, i.e. the identity restricted to the
# computational basis).
_ZBASIS_P0 = FullArbitraryOp.from_kraus_operators(
    [np.diag([1.0, 0.0]).astype(complex)], "pp"
).to_dense()
_ZBASIS_P1 = FullArbitraryOp.from_kraus_operators(
    [np.diag([0.0, 1.0]).astype(complex)], "pp"
).to_dense()


def _build_explicit_model():
    model = ExplicitOpModel(state_space=QubitSpace(["Q0"]), basis="pp")
    model.operations[Label("Gxpi", "Q0")] = FullArbitraryOp(np.eye(4), basis="pp")
    model.operations[Label("Gad", "Q0")] = FullArbitraryOp(_AMP_DAMP_SUPEROP, basis="pp")
    model.instruments[Label("Imrz", "Q0")] = Instrument(
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
    model.instrument_blks["layers"][Label("Imrz", "Q0")] = Instrument(
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
        assert ("Imrz", ["Q0"]) in pgm.instrument_keys
        assert pgm.use_embedded_op is False

    def test_from_implicit_op_model(self):
        pgm = PyGSTiNoiseModel(_build_implicit_model())
        assert ("Gxpi", ["Q0"]) in pgm.gate_keys
        assert ("Imrz", ["Q0"]) in pgm.instrument_keys
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

    def test_copy_constructor_preserves_instrument_outcome_qubits(self):
        pgm = PyGSTiNoiseModel(
            _build_explicit_model(),
            instrument_outcome_qubits={"Ipc": "synd_Q0Q1"},
        )
        pgm_copy = PyGSTiNoiseModel(pgm)
        assert pgm_copy.instrument_outcome_qubits == {"Ipc": "synd_Q0Q1"}

    def test_copy_constructor_allows_overriding_instrument_outcome_qubits(self):
        pgm = PyGSTiNoiseModel(
            _build_explicit_model(),
            instrument_outcome_qubits={"Ipc": "synd_Q0Q1"},
        )
        pgm_copy = PyGSTiNoiseModel(
            pgm, instrument_outcome_qubits={"Ipc": "other_label"}
        )
        assert pgm_copy.instrument_outcome_qubits == {"Ipc": "other_label"}

    def test_invalid_model_type_raises_type_error(self):
        with pytest.raises(TypeError, match="Cannot cast .* to PyGSTiNoiseModel"):
            PyGSTiNoiseModel(42)

    def test_dictnoisemodel_raises_not_implemented_error(self):
        with pytest.raises(NotImplementedError, match="Build explicit op model"):
            PyGSTiNoiseModel(DictNoiseModel({}, {}))

    def test_output_gate_reps_property(self):
        pgm = PyGSTiNoiseModel(_build_explicit_model())
        assert pgm.output_gate_reps == [
            UnitaryGateRep,
            KrausGateRep,
            PTMGateRep,
            QSimSuperopGateRep,
        ]

    def test_output_instrument_reps_property(self):
        pgm = PyGSTiNoiseModel(_build_explicit_model())
        assert pgm.output_instrument_reps == [
            ZBasisProjectionInstrumentRep,
            OutcomeOperationDictInstrumentRep,
        ]


class TestGetGateRep:
    @pytest.fixture
    def pgm(self):
        return PyGSTiNoiseModel(_build_explicit_model())

    def test_unitary(self, pgm):
        rep = pgm._get_gate_rep("Gxpi", ["Q0"], [UnitaryGateRep])
        assert isinstance(rep, UnitaryGateRep)
        assert rep.unitary.shape == (2, 2)

    def test_kraus_operators(self, pgm):
        rep = pgm._get_gate_rep("Gad", ["Q0"], [KrausGateRep])
        assert isinstance(rep, KrausGateRep)
        assert len(rep.kraus_operators) >= 1

    def test_ptm(self, pgm):
        rep = pgm._get_gate_rep("Gad", ["Q0"], [PTMGateRep])
        assert isinstance(rep, PTMGateRep)
        assert rep.ptm.shape == (4, 4)

    def test_qsim_superoperator(self, pgm):
        rep = pgm._get_gate_rep("Gad", ["Q0"], [QSimSuperopGateRep])
        assert isinstance(rep, QSimSuperopGateRep)
        assert rep.superop.shape == (4, 4)

    def test_fallback_skips_failing_candidate(self, pgm):
        """Requesting UnitaryGateRep first (fails for the non-unitary
        amplitude-damping channel) falls through to
        QSimSuperopGateRep rather than raising."""
        rep = pgm._get_gate_rep(
            "Gad", ["Q0"], [UnitaryGateRep, QSimSuperopGateRep]
        )
        assert isinstance(rep, QSimSuperopGateRep)
        assert rep.superop.shape == (4, 4)

    def test_no_valid_candidate_raises(self, pgm):
        with pytest.raises(RepConstructionError, match="Failed to create gate rep for any of"):
            pgm._get_gate_rep("Gad", ["Q0"], [UnitaryGateRep])

    def test_embedded_op_gate_with_naturally_compact_rep_no_warning(self, recwarn):
        """On a large-enough crosstalk-free model, pyGSTi's own `Evotype`
        selection already prefers the compact 'embedded' representation
        over 'dense', so the blowup check must recognize this and not warn
        -- distinct from the smaller model below, where pyGSTi picks
        'dense' but the parent space is small enough to be cheap."""
        from loqs.backends.model.pygstimodel import PyGSTiEmbeddedOpMemoryWarning

        pspec = pygsti.processors.QubitProcessorSpec(
            4,
            gate_names=["Gxpi"],
            qubit_labels=[f"Q{i}" for i in range(4)],
            availability={"Gxpi": "all-permutations"},
        )
        model = pygsti.models.create_crosstalk_free_model(pspec)
        pgm = PyGSTiNoiseModel(model)

        rep = pgm._get_gate_rep("Gxpi", ["Q0"], [UnitaryGateRep])
        assert isinstance(rep, UnitaryGateRep)
        assert not any(
            issubclass(w.category, PyGSTiEmbeddedOpMemoryWarning)
            for w in recwarn.list
        )

    def test_embedded_op_gate_on_multiqubit_crosstalk_free_model(self):
        """On a >1-qubit crosstalk-free (implicit) model, pyGSTi wraps each
        per-qubit gate in an `EmbeddedOp`; `use_embedded_op=True` must
        unwrap it to the local operator before converting, rather than
        converting the full multi-qubit-sized `EmbeddedOp` PTM."""
        pspec = pygsti.processors.QubitProcessorSpec(
            2,
            gate_names=["Gxpi"],
            qubit_labels=["Q0", "Q1"],
            availability={"Gxpi": "all-permutations"},
        )
        model = pygsti.models.create_crosstalk_free_model(pspec)
        pgm = PyGSTiNoiseModel(model)
        assert pgm.use_embedded_op is True

        rep = pgm._get_gate_rep("Gxpi", ["Q0"], [UnitaryGateRep])
        assert isinstance(rep, UnitaryGateRep)
        # Unwrapped to the local 1-qubit operator, not the 2-qubit embedding.
        assert rep.unitary.shape == (2, 2)

    def test_caches_result_when_not_time_dependent(self, pgm):
        assert pgm._gate_rep_cache == {}
        rep = pgm._get_gate_rep("Gxpi", ["Q0"], [UnitaryGateRep])
        assert pgm._gate_rep_cache == {(("Gxpi", "Q0"), UnitaryGateRep): rep}
        # Second call reuses the cached instance rather than recomputing.
        assert pgm._get_gate_rep("Gxpi", ["Q0"], [UnitaryGateRep]) is rep

    def test_does_not_cache_result_when_time_dependent(self):
        pgm = PyGSTiNoiseModel(
            _build_explicit_model(), use_time_dependence=True
        )
        pgm._get_gate_rep("Gxpi", ["Q0"], [UnitaryGateRep])
        assert pgm._gate_rep_cache == {}

    def test_repeated_gate_lookup_skips_rechecking_dense_embedding(self):
        """With time dependence (so results aren't cached), a repeated
        lookup of the same gate must skip the dense-embedding blowup
        recheck the second time."""
        pgm = PyGSTiNoiseModel(
            _build_explicit_model(), use_time_dependence=True
        )
        pgm._get_gate_rep("Gxpi", ["Q0"], [UnitaryGateRep])
        checked = set(pgm._dense_embedding_checked_gate_keys)
        assert checked

        rep = pgm._get_gate_rep("Gxpi", ["Q0"], [UnitaryGateRep])
        assert isinstance(rep, UnitaryGateRep)
        assert pgm._dense_embedding_checked_gate_keys == checked


class TestGetInstrumentRep:
    @pytest.fixture
    def pgm(self):
        return PyGSTiNoiseModel(_build_explicit_model())

    def test_zbasis_projection(self, pgm):
        rep = pgm._get_instrument_rep(
            "Imrz", ["Q0"], [ZBasisProjectionInstrumentRep]
        )
        assert isinstance(rep, ZBasisProjectionInstrumentRep)
        assert rep.reset == 0  # zbasis_proj_resets=True by default
        assert rep.include_outcome is True

    def test_zbasis_outcome_operation_dict(self, pgm):
        rep = pgm._get_instrument_rep(
            "Imrz", ["Q0"], [OutcomeOperationDictInstrumentRep]
        )
        assert isinstance(rep, OutcomeOperationDictInstrumentRep)
        assert rep.include_outcome is True
        # One classical channel matching the single physical qubit, so
        # keys collapse to bare ints rather than length-1 tuples.
        assert set(rep.outcome_ops.keys()) == {0, 1}
        assert rep.outcome_qubits == ("Q0",)

    def test_no_valid_candidate_raises(self, pgm):
        # STIM_CIRCUIT_STR is a real InstrumentRep member, but _make_rep's
        # if/elif only ever handles ZBASIS_PROJECTION/
        # ZBASIS_OUTCOME_OPERATION_DICT, so this exercises the same
        # "no candidate worked" final-failure path as the gate-rep case.
        with pytest.raises(
            RepConstructionError, match="Failed to create instrument rep for any of"
        ):
            pgm._get_instrument_rep("Imrz", ["Q0"], [StimCircuitInstrumentRep])

    def test_caches_result_when_not_time_dependent(self, pgm):
        assert pgm._inst_rep_cache == {}
        rep = pgm._get_instrument_rep(
            "Imrz", ["Q0"], [ZBasisProjectionInstrumentRep]
        )
        assert pgm._inst_rep_cache == {
            (("Imrz", "Q0"), ZBasisProjectionInstrumentRep): rep
        }
        assert (
            pgm._get_instrument_rep("Imrz", ["Q0"], [ZBasisProjectionInstrumentRep])
            is rep
        )

    def test_does_not_cache_result_when_time_dependent(self):
        pgm = PyGSTiNoiseModel(
            _build_explicit_model(), use_time_dependence=True
        )
        pgm._get_instrument_rep("Imrz", ["Q0"], [ZBasisProjectionInstrumentRep])
        assert pgm._inst_rep_cache == {}

    def test_repeated_outcome_operation_dict_lookup_skips_rechecking(self):
        """With time dependence (so results aren't cached), repeated
        lookups of the same ZBASIS_OUTCOME_OPERATION_DICT instrument must
        skip the dense-embedding blowup recheck the second time."""
        pgm = PyGSTiNoiseModel(
            _build_explicit_model(), use_time_dependence=True
        )
        pgm._get_instrument_rep(
            "Imrz", ["Q0"], [OutcomeOperationDictInstrumentRep]
        )
        checked = set(pgm._dense_embedding_checked_inst_keys)
        assert checked

        rep = pgm._get_instrument_rep(
            "Imrz", ["Q0"], [OutcomeOperationDictInstrumentRep]
        )
        assert isinstance(rep, OutcomeOperationDictInstrumentRep)
        assert pgm._dense_embedding_checked_inst_keys == checked

    def test_get_reps_joint_instrument_end_to_end_on_npsvstate(self):
        """A declared joint (2Q parity-check) instrument survives
        `get_reps`, and the resulting rep can be simulated directly by
        `SVState`: the outcome is recorded under the declared classical
        register, and superposition within the measured parity sector is
        preserved (unlike two independent single-qubit Z measurements,
        which would collapse it)."""
        from loqs.backends import NumpyStatevectorQuantumState as SVState
        from loqs.backends import PyGSTiPhysicalCircuit

        even_ptm = FullArbitraryOp.from_kraus_operators(
            [np.diag([1.0, 0, 0, 1.0]).astype(complex)], "pp"
        ).to_dense()
        odd_ptm = FullArbitraryOp.from_kraus_operators(
            [np.diag([0, 1.0, 1.0, 0]).astype(complex)], "pp"
        ).to_dense()
        model = ExplicitOpModel(state_space=QubitSpace(["Q0", "Q1"]), basis="pp")
        model.instruments[Label("Ipc", ("Q0", "Q1"))] = Instrument(
            {"even": even_ptm, "odd": odd_ptm}
        )
        pgm = PyGSTiNoiseModel(
            model, instrument_outcome_qubits={"Ipc": "synd_Q0Q1"}
        )

        circuit = PyGSTiPhysicalCircuit([("Ipc", "Q0", "Q1")], ["Q0", "Q1"])
        reps = pgm.get_reps(
            circuit, [UnitaryGateRep], [OutcomeOperationDictInstrumentRep]
        )
        assert len(reps) == 1
        rep = reps[0]
        assert rep.qubit_labels == ("Q0", "Q1")
        assert rep.outcome_qubits == ("synd_Q0Q1",)

        bell_state = np.array([1, 0, 0, 1]) / np.sqrt(2)
        for trial in range(5):
            state = SVState(bell_state.copy(), ["Q0", "Q1"], seed=20260815 + trial)
            outs = state.apply_reps_inplace([rep])
            assert outs["synd_Q0Q1"] == [0]
            # Up to global phase: converting through pyGSTi's PTM
            # representation and back doesn't preserve an overall sign.
            assert np.allclose(np.abs(state.state.flatten()), bell_state)


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
            default_instrument_durations={Label("Imrz", "Q0"): 3},
        )

    @pytest.mark.parametrize(
        "method,label",
        [
            ("get_gate_duration", Label("Gxpi", "Q0")),
            ("get_instrument_duration", Label("Imrz", "Q0")),
        ],
    )
    def test_disabled_always_returns_zero(self, pgm, method, label):
        assert getattr(pgm, method)(label) == 0

    def test_gate_duration_from_label_with_time(self, pgm_time_dependent):
        label = Label("Gxpi", "Q0", time=1.5)
        assert pgm_time_dependent.get_gate_duration(label) == 1.5

    def test_instrument_duration_from_label_with_time(self, pgm_time_dependent):
        label = Label("Imrz", "Q0", time=2.5)
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
            pgm.get_instrument_duration(Label("Imrz", "Q0"))

    def test_gate_duration_exact_label_match(self, pgm_time_dependent):
        assert pgm_time_dependent.get_gate_duration(Label("Gxpi", "Q0")) == 5

    def test_gate_duration_name_only_fallback(self, pgm_time_dependent):
        # "Gad" is registered by name only (no qubit), so an exact-label
        # lookup for Label("Gad", "Q0") must fall back to a name-only
        # lookup.
        assert pgm_time_dependent.get_gate_duration(Label("Gad", "Q0")) == 7

    def test_instrument_duration_exact_label_match(self, pgm_time_dependent):
        assert pgm_time_dependent.get_instrument_duration(Label("Imrz", "Q0")) == 3

    def test_gate_duration_not_found_raises_key_error(self, pgm_time_dependent):
        with pytest.raises(KeyError, match="not available by label or name"):
            pgm_time_dependent.get_gate_duration(Label("Gunknown", "Q0"))

    def test_instrument_duration_not_found_raises_key_error(
        self, pgm_time_dependent
    ):
        with pytest.raises(KeyError, match="not available by label or name"):
            pgm_time_dependent.get_instrument_duration(Label("Iunknown", "Q0"))

    def test_gate_duration_not_found_with_renamed_iz_name_hints_loqs_migrate(
        self, pgm_time_dependent
    ):
        with pytest.raises(KeyError, match="Imrz.*v1\\.2"):
            pgm_time_dependent.get_gate_duration(Label("Iz", "Q0"))

    def test_instrument_duration_not_found_with_renamed_iz_name_hints_loqs_migrate(
        self, pgm_time_dependent
    ):
        with pytest.raises(KeyError, match="Imrz.*v1\\.2"):
            pgm_time_dependent.get_instrument_duration(Label("Iz", "Q0"))

    def test_get_reps_advances_current_time(self):
        """End-to-end: `get_reps` on a real (multi-layer) pyGSTi circuit
        with `use_time_dependence=True` must advance `current_time` by
        each layer's max gate/instrument duration."""
        from loqs.backends import PyGSTiPhysicalCircuit

        pgm = PyGSTiNoiseModel(
            _build_explicit_model(),
            use_time_dependence=True,
            default_gate_durations={"Gxpi": 5, "Gad": 2},
            default_instrument_durations={"Imrz": 3},
        )
        assert pgm.current_time == 0.0

        circuit = PyGSTiPhysicalCircuit(
            [("Gxpi", "Q0"), ("Gad", "Q0"), ("Imrz", "Q0")], ["Q0"]
        )
        pgm.get_reps(
            circuit,
            [UnitaryGateRep, QSimSuperopGateRep],
            [ZBasisProjectionInstrumentRep],
        )

        # Gxpi (5) and Gad (2) can't share a layer (both act on Q0), and
        # Imrz (3) is its own layer too -- so total elapsed time is 5+2+3=10.
        assert pgm.current_time == 10.0

    def test_get_reps_does_not_reconstruct_already_correct_circuit_type(self):
        """A `circuit` that's already a `PyGSTiPhysicalCircuit` is read
        directly, without constructing or copying a second one."""
        from loqs.backends import PyGSTiPhysicalCircuit

        pgm = PyGSTiNoiseModel(_build_explicit_model())
        circuit = PyGSTiPhysicalCircuit([("Gxpi", "Q0")], ["Q0"])
        original_init = PyGSTiPhysicalCircuit.__init__
        call_count = [0]

        def counting_init(self, *args, **kwargs):
            call_count[0] += 1
            return original_init(self, *args, **kwargs)

        with mock.patch.object(PyGSTiPhysicalCircuit, "__init__", counting_init):
            pgm.get_reps(circuit, [UnitaryGateRep, QSimSuperopGateRep], [])
        assert call_count[0] == 0

    def test_get_reps_casts_non_pygsti_circuit(self):
        """A `circuit` that isn't already a `PyGSTiPhysicalCircuit` (e.g. a
        `ListPhysicalCircuit`) must be cast to one before use."""
        from loqs.backends import ListPhysicalCircuit

        pgm = PyGSTiNoiseModel(_build_explicit_model())
        circuit = ListPhysicalCircuit([("Gxpi", "Q0")], ["Q0"])
        reps = pgm.get_reps(circuit, [UnitaryGateRep, QSimSuperopGateRep], [])
        assert len(reps) == 1
        assert isinstance(reps[0], UnitaryGateRep)


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

    def test_iter_embedded_ops_handles_object_without_submembers(self):
        """`_iter_embedded_ops` duck-types on `submembers`; an object with
        neither that attribute nor `EmbeddedOp`-ness (not a real pyGSTi
        `ModelMember`, which always exposes `submembers`) must yield
        nothing rather than raising."""
        from loqs.backends.model.pygstimodel import _iter_embedded_ops

        assert list(_iter_embedded_ops(object())) == []

    def test_check_op_for_dense_embedding_blowup_warns(self):
        """An `EmbeddedOp` whose parent state space exceeds the dimension
        threshold and whose `_rep_type` is `'dense'` must trigger
        `PyGSTiEmbeddedOpMemoryWarning`. `_rep_type` is force-set here since
        the installed pyGSTi's own `Evotype.cast` already hardcodes this
        same dimension cutoff to prefer the compact 'embedded' rep instead,
        so this exercises the defensive check in isolation."""
        from pygsti.modelmembers.operations import FullArbitraryOp, EmbeddedOp
        from pygsti.baseobjs.statespace import QubitSpace
        from loqs.backends.model.pygstimodel import (
            PyGSTiEmbeddedOpMemoryWarning,
            _check_op_for_dense_embedding_blowup,
        )

        child_space = QubitSpace(["Q0"])
        child_op = FullArbitraryOp(np.eye(4), basis="pp", state_space=child_space)
        parent_space = QubitSpace([f"Q{i}" for i in range(5)])  # dim = 4**5 = 1024
        embedded_op = EmbeddedOp(parent_space, ["Q0"], child_op)
        embedded_op._rep_type = "dense"

        with pytest.warns(PyGSTiEmbeddedOpMemoryWarning, match="TestLabel"):
            _check_op_for_dense_embedding_blowup(embedded_op, "TestLabel")

    def test_check_op_for_dense_embedding_blowup_skips_small_or_non_embedding(self):
        """Neither a small parent state space nor a child that's (almost)
        as large as its parent counts as a memory-blowup symptom, even
        with `_rep_type == 'dense'`."""
        from pygsti.modelmembers.operations import FullArbitraryOp, EmbeddedOp
        from pygsti.baseobjs.statespace import QubitSpace
        from loqs.backends.model.pygstimodel import (
            PyGSTiEmbeddedOpMemoryWarning,
            _check_op_for_dense_embedding_blowup,
        )

        # Small parent (dim 16, below the 64 threshold).
        small_parent = QubitSpace(["Q0", "Q1"])
        small_child_op = FullArbitraryOp(
            np.eye(4), basis="pp", state_space=QubitSpace(["Q0"])
        )
        small_embedded = EmbeddedOp(small_parent, ["Q0"], small_child_op)
        small_embedded._rep_type = "dense"

        # Large parent, but the "embedded" op spans (almost) the whole
        # space -- not actually a wasteful embedding.
        large_parent = QubitSpace([f"Q{i}" for i in range(5)])
        large_child_op = FullArbitraryOp(
            np.eye(4**5), basis="pp", state_space=large_parent
        )
        non_embedding = EmbeddedOp(
            large_parent, [f"Q{i}" for i in range(5)], large_child_op
        )
        non_embedding._rep_type = "dense"

        with warnings.catch_warnings():
            warnings.simplefilter("error", PyGSTiEmbeddedOpMemoryWarning)
            _check_op_for_dense_embedding_blowup(small_embedded, "Small")
            _check_op_for_dense_embedding_blowup(non_embedding, "NonEmbedding")

    def test_check_for_dense_embedding_issues_second_call_skips_rechecking(self):
        """Each gate/instrument key is only ever checked once; a second
        call must skip straight past the already-checked keys rather than
        re-running the (potentially expensive) blowup check."""
        pgm = PyGSTiNoiseModel(_build_explicit_model())
        pgm.check_for_dense_embedding_issues()
        checked_gate_keys = set(pgm._dense_embedding_checked_raw_gate_keys)
        checked_inst_keys = set(pgm._dense_embedding_checked_raw_inst_keys)
        assert checked_gate_keys and checked_inst_keys

        # No error, and the bookkeeping sets are unchanged by the repeat.
        pgm.check_for_dense_embedding_issues()
        assert pgm._dense_embedding_checked_raw_gate_keys == checked_gate_keys
        assert pgm._dense_embedding_checked_raw_inst_keys == checked_inst_keys


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
        """`Gad` (amplitude damping) is not unitary, so it can never reach
        `StimCircuitGateRep` regardless of whether `stim` is installed --
        unlike `Gxpi` (see `test_clifford_gate_reaches_stim_circuit_gaterep`
        below), which is reachable since it's exactly Clifford (trivially
        so, in this fixture, since it's actually the identity -- see
        `_build_explicit_model`)."""
        pgm = PyGSTiNoiseModel(_build_explicit_model())
        with pytest.raises(
            RepConstructionError, match="Failed to create gate rep for any of"
        ):
            pgm._get_gate_rep("Gad", ["Q0"], [StimCircuitGateRep])

    @pytest.mark.skipif(NO_STIM, reason="stim is not installed")
    def test_clifford_gate_reaches_stim_circuit_gaterep(self):
        """Any exactly-Clifford gate -- including `Gxpi`, which in this
        fixture's `_build_explicit_model` is actually the identity
        superoperator -- is reachable via `PTMGateRep -> UnitaryGateRep ->
        StimCircuitGateRep`, since a `UnitaryGateRep <-> StimCircuitGateRep`
        conversion edge exists whenever `stim` is installed."""
        pgm = PyGSTiNoiseModel(_build_explicit_model())
        rep = pgm._get_gate_rep("Gxpi", ["Q0"], [StimCircuitGateRep])
        assert isinstance(rep, StimCircuitGateRep)
        assert isinstance(rep.circuit_str, str)

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
                "Gccx", ["Q0", "Q1", "Q2"], [QSimSuperopGateRep]
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

        rep = pgm._get_gate_rep("Gbf", ["Q0"], [KrausGateRep])
        assert isinstance(rep, KrausGateRep)
        probs = sorted(prob for _, prob in rep.kraus_operators)
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

        rep = pgm._get_instrument_rep(
            "Izz", ["Q0", "Q1"], [OutcomeOperationDictInstrumentRep]
        )
        assert rep.include_outcome is True
        assert set(rep.outcome_ops.keys()) == {(0, 0), (0, 1), (1, 0), (1, 1)}

    def test_instrument_rep_non_string_outcome_keys(self):
        """An instrument whose outcome-dict keys are already tuples (not
        strings) must pass them through unchanged, skipping the
        string-to-tuple parsing entirely. The single classical channel
        this resolves to (matching the single physical qubit) then
        collapses the length-1 tuple keys to bare ints, same as the
        string-keyed fixture."""
        model = ExplicitOpModel(state_space=QubitSpace(["Q0"]), basis="pp")
        model.instruments[Label("Imrz", "Q0")] = Instrument(
            {(0,): np.eye(4, dtype=complex), (1,): np.eye(4, dtype=complex)}
        )
        pgm = PyGSTiNoiseModel(model)

        rep = pgm._get_instrument_rep(
            "Imrz", ["Q0"], [OutcomeOperationDictInstrumentRep]
        )
        assert set(rep.outcome_ops.keys()) == {0, 1}

    def test_instrument_rep_arbitrary_outcome_keys_on_matching_qubit_count(self):
        """A single-qubit instrument with non-numeric outcome labels (e.g.
        'a'/'b') is still just one classical channel matching its one
        physical qubit, so it's accepted with those labels used as-is --
        arbitrary hashable keys are only restricted once there's more than
        one physical qubit and no declared `instrument_outcome_qubits`."""
        model = ExplicitOpModel(state_space=QubitSpace(["Q0"]), basis="pp")
        model.instruments[Label("Imrz", "Q0")] = Instrument(
            {"a": np.eye(4, dtype=complex), "b": np.eye(4, dtype=complex)}
        )
        pgm = PyGSTiNoiseModel(model)

        rep = pgm._get_instrument_rep(
            "Imrz", ["Q0"], [OutcomeOperationDictInstrumentRep]
        )
        assert set(rep.outcome_ops.keys()) == {"a", "b"}
        assert rep.outcome_qubits == ("Q0",)

    def test_instrument_rep_joint_outcome_without_declaration_raises(self):
        """A 2Q parity-check instrument (one joint classical channel on
        two physical qubits) with no matching `instrument_outcome_qubits`
        entry must raise -- there's no physical qubit to default the
        classical label to."""
        model = ExplicitOpModel(state_space=QubitSpace(["Q0", "Q1"]), basis="pp")
        model.instruments[Label("Ipc", ("Q0", "Q1"))] = Instrument(
            {"even": np.eye(16, dtype=complex), "odd": np.eye(16, dtype=complex)}
        )
        pgm = PyGSTiNoiseModel(model)
        with pytest.raises(
            RepConstructionError, match="Failed to create instrument rep for any of"
        ):
            pgm._get_instrument_rep(
                "Ipc", ["Q0", "Q1"], [OutcomeOperationDictInstrumentRep]
            )

    def test_instrument_rep_joint_outcome_bare_name_lookup(self):
        """`instrument_outcome_qubits` keyed by a bare instrument name
        applies wherever that name is used."""
        model = ExplicitOpModel(state_space=QubitSpace(["Q0", "Q1"]), basis="pp")
        model.instruments[Label("Ipc", ("Q0", "Q1"))] = Instrument(
            {"even": np.eye(16, dtype=complex), "odd": np.eye(16, dtype=complex)}
        )
        pgm = PyGSTiNoiseModel(
            model, instrument_outcome_qubits={"Ipc": "synd_Q0Q1"}
        )
        rep = pgm._get_instrument_rep(
            "Ipc", ["Q0", "Q1"], [OutcomeOperationDictInstrumentRep]
        )
        assert set(rep.outcome_ops.keys()) == {"even", "odd"}
        assert rep.outcome_qubits == ("synd_Q0Q1",)

    def test_instrument_rep_joint_outcome_name_and_qubits_lookup(self):
        """A more specific `(name, aliased_qubits)` entry in
        `instrument_outcome_qubits` takes priority over a bare-name entry
        for the same instrument name."""
        model = ExplicitOpModel(state_space=QubitSpace(["Q0", "Q1"]), basis="pp")
        model.instruments[Label("Ipc", ("Q0", "Q1"))] = Instrument(
            {"even": np.eye(16, dtype=complex), "odd": np.eye(16, dtype=complex)}
        )
        pgm = PyGSTiNoiseModel(
            model,
            instrument_outcome_qubits={
                "Ipc": "fallback_label",
                ("Ipc", ("Q0", "Q1")): "synd_Q0Q1",
            },
        )
        rep = pgm._get_instrument_rep(
            "Ipc", ["Q0", "Q1"], [OutcomeOperationDictInstrumentRep]
        )
        assert rep.outcome_qubits == ("synd_Q0Q1",)

    def test_instrument_rep_joint_outcome_uses_aliased_qubits_for_lookup(self):
        """The `(name, aliased_qubits)` lookup key uses the user-facing
        aliased qubit labels, not the underlying pyGSTi model's own."""
        model = ExplicitOpModel(state_space=QubitSpace(["Q0", "Q1"]), basis="pp")
        model.instruments[Label("Ipc", ("Q0", "Q1"))] = Instrument(
            {"even": np.eye(16, dtype=complex), "odd": np.eye(16, dtype=complex)}
        )
        pgm = PyGSTiNoiseModel(
            model,
            qubit_aliases={"Q0": "A", "Q1": "B"},
            instrument_outcome_qubits={("Ipc", ("A", "B")): "synd_AB"},
        )
        rep = pgm._get_instrument_rep(
            "Ipc", ["Q0", "Q1"], [OutcomeOperationDictInstrumentRep]
        )
        assert rep.outcome_qubits == ("synd_AB",)

    def test_instrument_rep_channel_count_matching_neither_raises(self):
        """Outcome labels that consistently resolve to some channel count
        that's neither 1 nor `len(qubits)` (e.g. 3-bit labels on a 2-qubit
        instrument) are rejected -- there's no sensible physical or joint
        interpretation for them."""
        model = ExplicitOpModel(state_space=QubitSpace(["Q0", "Q1"]), basis="pp")
        model.instruments[Label("Ibad", ("Q0", "Q1"))] = Instrument(
            {"000": np.eye(16, dtype=complex), "111": np.eye(16, dtype=complex)}
        )
        pgm = PyGSTiNoiseModel(model)
        with pytest.raises(
            RepConstructionError, match="Failed to create instrument rep for any of"
        ):
            pgm._get_instrument_rep(
                "Ibad", ["Q0", "Q1"], [OutcomeOperationDictInstrumentRep]
            )

    def test_instrument_rep_inconsistent_channel_counts_raises(self):
        """Outcome labels that don't all share one channel count (e.g. a
        mix of single-bit and two-bit labels) are rejected outright."""
        model = ExplicitOpModel(state_space=QubitSpace(["Q0", "Q1"]), basis="pp")
        model.instruments[Label("Ibad", ("Q0", "Q1"))] = Instrument(
            {"0": np.eye(16, dtype=complex), "01": np.eye(16, dtype=complex)}
        )
        pgm = PyGSTiNoiseModel(model)
        with pytest.raises(
            RepConstructionError, match="Failed to create instrument rep for any of"
        ):
            pgm._get_instrument_rep(
                "Ibad", ["Q0", "Q1"], [OutcomeOperationDictInstrumentRep]
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
            default_instrument_durations={"Imrz": 1},
        )
        circuit = PyGSTiPhysicalCircuit([("Imrz", "Q0")], ["Q0"])
        reps = pgm.get_reps(
            circuit, [UnitaryGateRep], [OutcomeOperationDictInstrumentRep]
        )
        assert len(reps) == 1
        assert isinstance(reps[0], OutcomeOperationDictInstrumentRep)


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

        rep = pgm2._get_gate_rep("Gxpi", ["Q0"], [UnitaryGateRep])
        assert isinstance(rep, UnitaryGateRep)
        assert rep.unitary.shape == (2, 2)

"""Tester for loqs.backends.model.pygstimodel"""

from unittest import mock

import numpy as np
import pytest

pygsti = pytest.importorskip("pygsti")

from pygsti.baseobjs import Label
from pygsti.baseobjs.statespace import QubitSpace
from pygsti.modelmembers.instruments import Instrument
from pygsti.modelmembers.operations import FullArbitraryOp
from pygsti.models import ExplicitOpModel

from loqs.backends.circuit.listcircuit import ListPhysicalCircuit
from loqs.backends.model.dictmodel import DictNoiseModel
from loqs.backends.model.pygstimodel import PyGSTiNoiseModel
from loqs.backends.reps import (
    KrausGateRep,
    QSimSuperopGateRep,
    RepConstructionError,
    StimCircuitGateRep,
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
_AMP_DAMP_SUPEROP = FullArbitraryOp.from_kraus_operators(
    [_K0, _K1], "pp"
).to_dense()

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
    model.operations[Label("Gxpi", "Q0")] = FullArbitraryOp(
        np.eye(4), basis="pp"
    )
    model.operations[Label("Gad", "Q0")] = FullArbitraryOp(
        _AMP_DAMP_SUPEROP, basis="pp"
    )
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


def _build_implicit_model_with_idle_names(idle_names):
    # Only an implicit (crosstalk-free/local-noise) model can hold gate
    # names like "GiMCM" at all -- unlike ExplicitOpModel, its member
    # dicts don't validate that a key round-trips through pyGSTi's own
    # string-form label parser at assignment time (see
    # `Model-member key ... does not round-trip through parse_label`).
    all_names = ["Gxpi", *idle_names]
    pspec = pygsti.processors.QubitProcessorSpec(
        1,
        gate_names=all_names,
        nonstd_gate_unitaries={name: np.eye(2) for name in idle_names},
        qubit_labels=["Q0"],
        availability={name: "all-permutations" for name in all_names},
    )
    return pygsti.models.create_crosstalk_free_model(pspec)


class TestConstruction:
    @pytest.mark.parametrize(
        "builder, use_embedded_op",
        [(_build_explicit_model, False), (_build_implicit_model, True)],
        ids=["explicit", "implicit"],
    )
    def test_from_op_model(self, builder, use_embedded_op):
        pgm = PyGSTiNoiseModel(builder())
        assert ("Gxpi", ["Q0"]) in pgm.gate_keys
        assert ("Imrz", ["Q0"]) in pgm.instrument_keys
        assert pgm.use_embedded_op is use_embedded_op

    @pytest.mark.parametrize(
        "qubit_aliases", [{"Q0": "MyQubit"}, ["MyQubit"]], ids=["map", "seq"]
    )
    def test_qubit_aliases_rename_qubits(self, qubit_aliases):
        pgm = PyGSTiNoiseModel(
            _build_explicit_model(), qubit_aliases=qubit_aliases
        )
        assert pgm.gate_keys == [("Gxpi", ["MyQubit"]), ("Gad", ["MyQubit"])]

    def test_non_conforming_qubit_label_without_alias_raises(self):
        model = ExplicitOpModel(state_space=QubitSpace(["A0"]), basis="pp")
        model.operations[Label("Gxpi", "A0")] = FullArbitraryOp(
            np.eye(4), basis="pp"
        )
        with pytest.raises(AssertionError):
            PyGSTiNoiseModel(model)

    def test_copy_constructor_copies_model_aliases_and_outcome_qubits(self):
        pgm = PyGSTiNoiseModel(
            _build_explicit_model(),
            qubit_aliases={"Q0": "MyQubit"},
            instrument_outcome_qubits={"Ipc": "synd_Q0Q1"},
        )
        pgm_copy = PyGSTiNoiseModel(pgm)
        assert pgm_copy.model is pgm.model
        assert pgm_copy.use_embedded_op == pgm.use_embedded_op
        assert pgm_copy.qubit_aliases == pgm.qubit_aliases
        assert pgm_copy.gate_keys == pgm.gate_keys
        assert pgm_copy.instrument_outcome_qubits == {"Ipc": "synd_Q0Q1"}


class TestGetGateRep:
    @pytest.fixture
    def pgm(self):
        return PyGSTiNoiseModel(_build_explicit_model())

    @pytest.mark.parametrize(
        "gate, rep_class",
        [
            ("Gxpi", UnitaryGateRep),
            ("Gad", KrausGateRep),
            ("Gad", QSimSuperopGateRep),
            pytest.param(
                "Gxpi",
                StimCircuitGateRep,
                marks=pytest.mark.skipif(
                    NO_STIM, reason="stim is not installed"
                ),
            ),
        ],
        ids=["unitary", "kraus", "qsim-superop", "stim"],
    )
    def test_gate_reptype(self, pgm, gate, rep_class):
        rep = pgm._get_gate_rep(gate, ["Q0"], [rep_class])
        assert isinstance(rep, rep_class)
        if rep_class is UnitaryGateRep:
            assert rep.unitary.shape == (2, 2)
        elif rep_class is KrausGateRep:
            assert len(rep.kraus_operators) >= 1
        elif rep_class is QSimSuperopGateRep:
            assert rep.superop.shape == (4, 4)
        else:
            assert isinstance(rep.circuit_str, str)

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
        with pytest.raises(RepConstructionError):
            pgm._get_gate_rep("Gad", ["Q0"], [UnitaryGateRep])


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
        model = ExplicitOpModel(
            state_space=QubitSpace(["Q0", "Q1"]), basis="pp"
        )
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
            state = SVState(
                bell_state.copy(), ["Q0", "Q1"], seed=20260815 + trial
            )
            outs = state.apply_reps_inplace([rep])
            assert outs["synd_Q0Q1"] == [0]
            # Up to global phase: converting through pyGSTi's PTM
            # representation and back doesn't preserve an overall sign.
            assert np.allclose(np.abs(state.state.flatten()), bell_state)


class TestRepCache:
    """Reps are cached per (label, rep type) unless time dependence is on,
    where a cached rep would be stale."""

    @staticmethod
    def _lookup(pgm, kind):
        if kind == "gate":
            return pgm._get_gate_rep("Gxpi", ["Q0"], [UnitaryGateRep])
        return pgm._get_instrument_rep(
            "Imrz", ["Q0"], [ZBasisProjectionInstrumentRep]
        )

    @pytest.mark.parametrize("kind", ["gate", "instrument"])
    def test_caches_result_when_not_time_dependent(self, kind):
        pgm = PyGSTiNoiseModel(_build_explicit_model())
        assert self._lookup(pgm, kind) is self._lookup(pgm, kind)

    @pytest.mark.parametrize("kind", ["gate", "instrument"])
    def test_does_not_cache_result_when_time_dependent(self, kind):
        pgm = PyGSTiNoiseModel(
            _build_explicit_model(), use_time_dependence=True
        )
        assert self._lookup(pgm, kind) is not self._lookup(pgm, kind)


class TestTimeDependence:
    """`get_gate_duration`/`get_instrument_duration` (and the
    `add_gate_duration_to_layer`/`add_layer_duration_to_current_time`
    helpers they feed into via `get_reps`) are only active when
    `use_time_dependence=True`; otherwise they always return 0."""

    @pytest.fixture
    def pgm_time_dependent(self):
        return PyGSTiNoiseModel(
            _build_explicit_model(),
            use_time_dependence=True,
            default_gate_durations={Label("Gxpi", "Q0"): 5, "Gad": 7},
            default_instrument_durations={Label("Imrz", "Q0"): 3},
        )

    @pytest.mark.parametrize(
        "method, name, duration",
        [
            ("get_gate_duration", "Gxpi", 1.5),
            ("get_instrument_duration", "Imrz", 2.5),
        ],
        ids=["gate", "instrument"],
    )
    def test_duration_from_label_with_time(
        self, pgm_time_dependent, method, name, duration
    ):
        label = Label(name, "Q0", time=duration)
        assert getattr(pgm_time_dependent, method)(label) == duration

    def test_dict_model_from_time_dependent_model_raises(
        self, pgm_time_dependent
    ):
        with pytest.raises(NotImplementedError):
            DictNoiseModel.from_model(pgm_time_dependent)

    def test_gate_duration_exact_label_match(self, pgm_time_dependent):
        assert pgm_time_dependent.get_gate_duration(Label("Gxpi", "Q0")) == 5

    def test_gate_duration_not_found_raises_key_error(
        self, pgm_time_dependent
    ):
        with pytest.raises(KeyError):
            pgm_time_dependent.get_gate_duration(Label("Gunknown", "Q0"))

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

        with mock.patch.object(
            PyGSTiPhysicalCircuit, "__init__", counting_init
        ):
            pgm.get_reps(circuit, [UnitaryGateRep, QSimSuperopGateRep], [])
        assert call_count[0] == 0


class TestDenseEmbeddingWarningHelpers:
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
        child_op = FullArbitraryOp(
            np.eye(4), basis="pp", state_space=child_space
        )
        parent_space = QubitSpace(
            [f"Q{i}" for i in range(5)]
        )  # dim = 4**5 = 1024
        embedded_op = EmbeddedOp(parent_space, ["Q0"], child_op)
        embedded_op._rep_type = "dense"

        with pytest.warns(PyGSTiEmbeddedOpMemoryWarning):
            _check_op_for_dense_embedding_blowup(embedded_op, "TestLabel")


class TestGetRepsErrorPaths:
    def test_unhandled_component_prefix_raises(self):
        """`get_reps` only knows how to dispatch component names starting
        with `G` (gates) or `I` (instruments); anything else must raise a
        clear `NotImplementedError` rather than silently mis-dispatching."""
        from loqs.backends import PyGSTiPhysicalCircuit

        pgm = PyGSTiNoiseModel(_build_explicit_model())
        circuit = PyGSTiPhysicalCircuit([("Xpi", "Q0")], ["Q0"])
        with pytest.raises(NotImplementedError):
            pgm.get_reps(
                circuit, [UnitaryGateRep], [ZBasisProjectionInstrumentRep]
            )

    def test_instrument_rep_multiqubit_outcome_keys(self):
        """A multi-qubit instrument's outcome-dict keys are multi-character
        strings (e.g. `"01"`), exercising the multi-char branch of the
        outcome-key-to-tuple conversion (as opposed to the existing
        single-qubit fixture's single-character keys)."""
        model = ExplicitOpModel(
            state_space=QubitSpace(["Q0", "Q1"]), basis="pp"
        )
        effects = {
            k: np.eye(16, dtype=complex) for k in ["00", "01", "10", "11"]
        }
        model.instruments[Label("Izz", ("Q0", "Q1"))] = Instrument(effects)
        pgm = PyGSTiNoiseModel(model)

        rep = pgm._get_instrument_rep(
            "Izz", ["Q0", "Q1"], [OutcomeOperationDictInstrumentRep]
        )
        assert rep.include_outcome is True
        assert set(rep.outcome_ops.keys()) == {(0, 0), (0, 1), (1, 0), (1, 1)}

    @staticmethod
    def _joint_parity_model():
        model = ExplicitOpModel(
            state_space=QubitSpace(["Q0", "Q1"]), basis="pp"
        )
        model.instruments[Label("Ipc", ("Q0", "Q1"))] = Instrument(
            {
                "even": np.eye(16, dtype=complex),
                "odd": np.eye(16, dtype=complex),
            }
        )
        return model

    def test_instrument_rep_joint_outcome_without_declaration_raises(self):
        """A 2Q parity-check instrument (one joint classical channel on
        two physical qubits) with no matching `instrument_outcome_qubits`
        entry must raise -- there's no physical qubit to default the
        classical label to."""
        pgm = PyGSTiNoiseModel(self._joint_parity_model())
        with pytest.raises(RepConstructionError):
            pgm._get_instrument_rep(
                "Ipc", ["Q0", "Q1"], [OutcomeOperationDictInstrumentRep]
            )

    def test_instrument_rep_joint_outcome_lookup(self):
        """A `(name, qubits)` entry in `instrument_outcome_qubits` takes
        priority over a bare-name entry, and its qubits are the user-facing
        aliased labels, not the underlying pyGSTi model's own."""
        pgm = PyGSTiNoiseModel(
            self._joint_parity_model(),
            instrument_outcome_qubits={
                "Ipc": "fallback_label",
                ("Ipc", ("Q0", "Q1")): "synd_Q0Q1",
            },
        )
        rep = pgm._get_instrument_rep(
            "Ipc", ["Q0", "Q1"], [OutcomeOperationDictInstrumentRep]
        )
        assert rep.outcome_qubits == ("synd_Q0Q1",)

        pgm = PyGSTiNoiseModel(
            self._joint_parity_model(),
            qubit_aliases={"Q0": "A", "Q1": "B"},
            instrument_outcome_qubits={("Ipc", ("A", "B")): "synd_AB"},
        )
        rep = pgm._get_instrument_rep(
            "Ipc", ["Q0", "Q1"], [OutcomeOperationDictInstrumentRep]
        )
        assert rep.outcome_qubits == ("synd_AB",)


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

    def test_round_trip_gatename_needing_pygsti_safe_alias(
        self, make_temp_path
    ):
        # "GiMCM" doesn't survive pyGSTi's own string-form label parser
        # unaliased -- it truncates a name at its first uppercase letter
        # after the initial character, so "GiMCM" would otherwise
        # round-trip as "Gi" and collide with the real "Gi" gate below,
        # which pyGSTi's own model deserialization raises on.
        model = _build_implicit_model_with_idle_names(["Gi", "GiMCM"])
        pgm = PyGSTiNoiseModel(model)

        encoded = pgm._get_encoding_attr("model")
        assert isinstance(encoded, dict)
        assert encoded["gatename_renames"] == {"GiMCM": "Gimcm"}

        with make_temp_path(suffix=".json") as tmp_path:
            pgm.write(tmp_path)
            pgm2 = PyGSTiNoiseModel.read(tmp_path)

        assert pgm2.model.is_similar(model)
        assert set(pgm2.model.operation_blks["gates"].keys()) == set(
            model.operation_blks["gates"].keys()
        )
        assert set(pgm2.model.operation_blks["layers"].keys()) == set(
            model.operation_blks["layers"].keys()
        )

    def test_gatename_alias_collision_raises(self):
        # "GiMCM" and "Gimcm" both alias to the same pyGSTi-safe name --
        # serializing both in one model can't be reversed unambiguously.
        model = _build_implicit_model_with_idle_names(["GiMCM", "Gimcm"])
        pgm = PyGSTiNoiseModel(model)
        with pytest.raises(ValueError):
            pgm._get_encoding_attr("model")


class TestGateContentWithAliases:
    """A PyGSTi model's `Gcnot` reps have the right control and target,
    through `qubit_aliases`, and match a `DictNoiseModel` of the same
    unitaries."""

    QUBITS = ["A0", "D3", "D4"]
    CNOT = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=float
    )
    CNOT_PAIRS = [("A0", "D3"), ("D3", "A0"), ("D3", "D4")]

    @pytest.fixture
    def pgm(self):
        pspec = pygsti.processors.QubitProcessorSpec(
            3,
            gate_names=["Gcnot", "Gh"],
            qubit_labels=["Q0", "Q1", "Q2"],
            availability={
                "Gcnot": [("Q0", "Q1"), ("Q1", "Q0"), ("Q1", "Q2")],
                "Gh": "all-permutations",
            },
        )
        model = pygsti.models.create_crosstalk_free_model(pspec)
        return PyGSTiNoiseModel(
            model, qubit_aliases={"Q0": "A0", "Q1": "D3", "Q2": "D4"}
        )

    @staticmethod
    def _gate_reps(model, label):
        from loqs.backends import ListPhysicalCircuit

        circuit = ListPhysicalCircuit([[label]])
        return model.get_reps(circuit, [UnitaryGateRep], [])

    def test_cnot_rep_control_is_first_aliased_qubit(self, pgm):
        for pair in self.CNOT_PAIRS:
            (rep,) = self._gate_reps(pgm, ("Gcnot", pair))
            assert tuple(rep.qubit_labels) == pair
            # Unitary is in the rep's own qubit order: first label controls
            assert np.allclose(rep.unitary, self.CNOT)

    def test_reps_act_like_dict_model_on_basis_states(self, pgm):
        from loqs.backends import NumpyStatevectorQuantumState as SVState

        dict_model = DictNoiseModel(
            {("Gcnot", pair): self.CNOT for pair in self.CNOT_PAIRS},
            gatereps=[UnitaryGateRep],
            gaterep_array_cast_rep=UnitaryGateRep,
        )

        for pair in self.CNOT_PAIRS:
            (pgsti_rep,) = self._gate_reps(pgm, ("Gcnot", pair))
            (dict_rep,) = self._gate_reps(dict_model, ("Gcnot", pair))
            for index in range(8):
                bits = {
                    q: (index >> (2 - i)) & 1
                    for i, q in enumerate(self.QUBITS)
                }
                outs = []
                for rep in (pgsti_rep, dict_rep):
                    state = SVState(
                        np.eye(8)[index].astype(complex), self.QUBITS
                    )
                    state.apply_reps_inplace([rep])
                    outs.append(state.state.flatten())
                assert np.allclose(outs[0], outs[1])

                # The control is the pair's first qubit: the target flips
                # exactly when the control bit is set.
                control, target = pair
                expected = dict(bits)
                expected[target] ^= bits[control]
                expected_index = sum(
                    expected[q] << (2 - i) for i, q in enumerate(self.QUBITS)
                )
                assert np.allclose(outs[0], np.eye(8)[expected_index])


def test_dict_model_from_pygsti_model():
    pgm = PyGSTiNoiseModel(_build_explicit_model(), qubit_aliases={"Q0": "A0"})
    dict_model = DictNoiseModel.from_model(
        pgm, [QSimSuperopGateRep], [OutcomeOperationDictInstrumentRep]
    )
    circuit = ListPhysicalCircuit([[("Gad", ("A0",))], [("Imrz", ("A0",))]])
    args = ([QSimSuperopGateRep], [OutcomeOperationDictInstrumentRep])
    gate_a, inst_a = pgm.get_reps(circuit, *args)
    gate_b, inst_b = dict_model.get_reps(circuit, *args)
    assert np.allclose(gate_a.superop, gate_b.superop)
    assert inst_a.outcome_ops.keys() == inst_b.outcome_ops.keys()
    for key, op in inst_a.outcome_ops.items():
        assert np.allclose(op.ptm, inst_b.outcome_ops[key].ptm)

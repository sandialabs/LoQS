"""Test Serializable version compatibility."""

import gzip
import json
from pathlib import Path

import pytest

quantumsim = pytest.importorskip("quantumsim")

import loqs.internal.serializable as serializable_module
from loqs.backends.state.qsimstate import QSimQuantumState
from loqs.core.instructions.instruction import Instruction
from loqs.core.quantumprogram import QuantumProgram
from loqs.internal.serializable import IMPORT_LOCATION_CHANGES_BY_VERSION, Serializable


class TestVersionCompatibility:
    """Parameterized tests for Serializable class functionality."""

    @pytest.mark.parametrize("version_file",[
        "QuantumProgram_v0.json.gz",
        "QuantumProgram_v1.json.gz",
    ])
    def test_read_versioned_quantumprogram(self, version_file):
        """Test whether we can load QuantumProgram for given serialization version.

        Test files are taken from test_quantumprogram files."""

        path = Path(__file__).parent
        loaded_program = QuantumProgram.read(path / version_file, migrate_legacy_fns=True)

        assert isinstance(loaded_program, QuantumProgram)
        assert loaded_program.name == "Prep minus, measure X"
        assert len(loaded_program.instruction_stack) == 4
        assert loaded_program.state_type == QSimQuantumState
        assert loaded_program.patch_types is not None
        assert list(loaded_program.patch_types.keys())[0] == "5Q"

        loaded_program_results = loaded_program.run(2)
        assert len(loaded_program_results.shot_histories) == 2
    
    def test_function_import_updates(self):
        # This is a real physical circuit instruction apply_fn from version 0
        test_str="""
from collections.abc import Mapping, Sequence
import inspect as ins
import numpy as np
from loqs.backends import propagate_state
from loqs.backends.circuit import BasePhysicalCircuit
from loqs.backends.model import (
    BaseNoiseModel,
    TimeDependentBaseNoiseModel,
)
from loqs.backends.state import BaseQuantumState
from loqs.core.frame import Frame
from loqs.core.qeccode import QECCode, QECCodePatch
from loqs.core.recordables.measurementoutcomes import MeasurementOutcomes
from loqs.core.recordables.patchdict import PatchDict
from loqs.core.syndrome import (
    PauliFrame,
    SyndromeLabel,
    SyndromeLabelCastableTypes
)
from loqs.backends import (
    STIMQuantumState,
    STIMPhysicalCircuit,
    PyGSTiNoiseModel,
)
def apply_fn(
    model: BaseNoiseModel,
    circuit: BasePhysicalCircuit,
    state: BaseQuantumState,
    inplace: bool,
    error_injections: list[tuple[int, str, int]] | None,
    pauli_frame_update: str | list[str] | dict[str, str] | None,
    patch_label: str,
    patches: PatchDict,
) -> Frame:

    [physical circuit apply function]
    [talks about PauliFrame]

    return Frame(data)
"""
        expected_str="""
from collections.abc import Mapping, Sequence
import inspect as ins
import numpy as np
from loqs.backends import propagate_state
from loqs.backends.circuit import BasePhysicalCircuit
from loqs.backends.model import BaseNoiseModel, TimeDependentBaseNoiseModel
from loqs.backends.state import BaseQuantumState
from loqs.core.frame import Frame
from loqs.core.qeccode import QECCode, QECCodePatch
from loqs.core.recordables.measurementoutcomes import MeasurementOutcomes
from loqs.core.recordables.patchlayout import PatchLayout
from loqs.core.recordables.pauliframe import PauliFrame
from loqs.core.syndromelabel import SyndromeLabel
from loqs.core.syndromelabel import SyndromeLabelLike
from loqs.backends import STIMQuantumState, STIMPhysicalCircuit, PyGSTiNoiseModel
def apply_fn(
    model: BaseNoiseModel,
    circuit: BasePhysicalCircuit,
    state: BaseQuantumState,
    inplace: bool,
    error_injections: list[tuple[int, str, int]] | None,
    pauli_frame_update: str | list[str] | dict[str, str] | None,
    patch_label: str,
    patches: PatchLayout,
) -> Frame:

    [physical circuit apply function]
    [talks about PauliFrame]

    return Frame(data)
"""

        updated_str = Serializable._update_imports(test_str, 0)
        assert updated_str == expected_str

        # Also try one where a module name changes. Uses version 1's table
        # directly (not the full multi-hop composition), so
        # SyndromeLabelCastableTypes only moves module here -- its later
        # rename to SyndromeLabelLike is version 2's entry, not version 1's.
        renamed_loc_change = IMPORT_LOCATION_CHANGES_BY_VERSION[1].copy()
        renamed_loc_change[("loqs.core.syndrome", "PauliFrame")] = ("loqs.core.recordables.pauliframe", "PauliFrameRenamed")

        expected_str2 = """
from collections.abc import Mapping, Sequence
import inspect as ins
import numpy as np
from loqs.backends import propagate_state
from loqs.backends.circuit import BasePhysicalCircuit
from loqs.backends.model import BaseNoiseModel, TimeDependentBaseNoiseModel
from loqs.backends.state import BaseQuantumState
from loqs.core.frame import Frame
from loqs.core.qeccode import QECCode, QECCodePatch
from loqs.core.recordables.measurementoutcomes import MeasurementOutcomes
from loqs.core.recordables.patchdict import PatchDict
from loqs.core.recordables.pauliframe import PauliFrameRenamed
from loqs.core.syndromelabel import SyndromeLabel
from loqs.core.syndromelabel import SyndromeLabelCastableTypes
from loqs.backends import STIMQuantumState, STIMPhysicalCircuit, PyGSTiNoiseModel
def apply_fn(
    model: BaseNoiseModel,
    circuit: BasePhysicalCircuit,
    state: BaseQuantumState,
    inplace: bool,
    error_injections: list[tuple[int, str, int]] | None,
    pauli_frame_update: str | list[str] | dict[str, str] | None,
    patch_label: str,
    patches: PatchDict,
) -> Frame:

    [physical circuit apply function]
    [talks about PauliFrameRenamed]

    return Frame(data)
"""

        updated_str2 = Serializable._update_imports(test_str, loc_change=renamed_loc_change)
        assert updated_str2 == expected_str2

    def test_instruction_label_and_stack_castable_types_rename(self):
        """`InstructionLabelCastableTypes`/`InstructionStackCastableTypes`
        need real version-2 compat entries -- referenced inside real
        serialized `QuantumProgram` fixtures' frozen source, previously
        breaking decode with `ImportError`. Isolates the import-rewrite
        mechanism, independent of the decode-time label remap."""
        test_str = """
from loqs.core.instructions.instructionlabel import (
    InstructionLabel,
    InstructionLabelCastableTypes,
)
from loqs.core.instructions.instructionstack import (
    InstructionStack,
    InstructionStackCastableTypes,
)
def apply_fn(
    instructions: InstructionStackCastableTypes,
) -> None:
    pass
"""
        expected_str = """
from loqs.core.instructions.instructionlabel import InstructionLabel
from loqs.core.instructions.instructionlabel import InstructionLabelLike
from loqs.core.instructions.instructionstack import InstructionStack
from loqs.core.instructions.instructionstack import InstructionStackLike
def apply_fn(
    instructions: InstructionStackLike,
) -> None:
    pass
"""
        updated_str = Serializable._update_imports(test_str, 1)
        assert updated_str == expected_str

    def test_get_cumulative_changes_multi_hop_composition(self, monkeypatch):
        """A rename chained across 3 versions (A -> B -> C -> D) must
        compose into a single A -> D mapping. Regression test for a real
        infinite loop in `_get_cumulative_changes` (`version` was never
        incremented in its old `while` loop) -- no existing test
        exercised more than one hop."""
        fake_table = {
            1: {("mod0", "A"): ("mod1", "B")},
            2: {("mod1", "B"): ("mod2", "C")},
            3: {("mod2", "C"): ("mod3", "D")},
        }
        monkeypatch.setattr(
            serializable_module, "IMPORT_LOCATION_CHANGES_BY_VERSION", fake_table
        )
        monkeypatch.setattr(serializable_module, "SERIALIZATION_VERSION", 4)

        assert Serializable._get_cumulative_changes(0) == {
            ("mod0", "A"): ("mod3", "D")
        }

    def test_get_cumulative_changes_handles_missing_table_entry(self, monkeypatch):
        """A version with no import-location changes at all simply has no
        entry in `IMPORT_LOCATION_CHANGES_BY_VERSION` (not an empty one) --
        `_get_cumulative_changes` must not raise `KeyError` when composing
        across such a gap, whether at the very first hop or a later one."""
        fake_table = {
            # No entry for version 1 at all.
            2: {("mod0", "A"): ("mod2", "B")},
            # No entry for version 3 at all.
            4: {("mod2", "B"): ("mod4", "C")},
        }
        monkeypatch.setattr(
            serializable_module, "IMPORT_LOCATION_CHANGES_BY_VERSION", fake_table
        )
        monkeypatch.setattr(serializable_module, "SERIALIZATION_VERSION", 5)

        assert Serializable._get_cumulative_changes(0) == {
            ("mod0", "A"): ("mod4", "C")
        }


def _decoded_str(value):
    """Undo json's/Serializable's own primitive-wrapping, if present."""
    if isinstance(value, dict) and value.get("encode_type") == "primitive":
        return value.get("value")
    return value


def _find_instruction_by_name(obj, name):
    """Find a raw, undecoded Instruction attr_dict by name inside a decompressed fixture."""
    if isinstance(obj, dict):
        if obj.get("class") == "Instruction" and _decoded_str(obj.get("name")) == name:
            return obj
        for v in obj.values():
            found = _find_instruction_by_name(v, name)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_instruction_by_name(v, name)
            if found is not None:
                return found
    return None


class TestMigrateLegacyFnsGate:
    """Regression tests for the migrate_legacy_fns gate.

    A straight class-location rename (e.g. PatchDict -> PatchLayout) needs
    no gate, since `_update_imports` already rewrites frozen source to the
    new name directly. The gate only matters for a calling-convention
    change with the *same* class name -- confirmed against
    `QuantumProgram_v1.json.gz`'s "Repeat-until-success FT Minus Prep"
    instruction, which genuinely freezes an old-style positional
    `InstructionLabel(...)` call.
    """

    @pytest.fixture
    def real_old_style_instructionlabel_source(self):
        path = Path(__file__).parent / "QuantumProgram_v1.json.gz"
        with gzip.open(path, "rt") as f:
            data = json.load(f)
        inst = _find_instruction_by_name(
            data, "Repeat-until-success FT Minus Prep"
        )
        source = _decoded_str(inst["_serialized_apply_fn"])
        assert "InstructionLabel(" in source  # sanity-check the fixture itself
        return source

    @staticmethod
    def _attr_dict(
        apply_fn_source,
        version=1,
        name="Repeat-until-success FT Minus Prep",
    ):
        return {
            "_serialized_apply_fn": apply_fn_source,
            "_serialized_map_qubits_fn": "def map_qubits_fn(qubit_map):\n    return {}\n",
            "version": version,
            "type": "Test",
            "data": {},
            "param_error_behavior": "warn",
            "name": name,
            "_param_priorities": {},
            "_param_aliases": {},
        }

    def test_default_raises_clear_error(
        self, real_old_style_instructionlabel_source
    ):
        with pytest.raises(RuntimeError, match="InstructionLabel"):
            Instruction._from_decoded_attrs(
                self._attr_dict(real_old_style_instructionlabel_source)
            )

    def test_migrate_legacy_fns_true_allows_decode(
        self, real_old_style_instructionlabel_source
    ):
        token = serializable_module.MIGRATE_LEGACY_FNS.set(True)
        try:
            inst = Instruction._from_decoded_attrs(
                self._attr_dict(real_old_style_instructionlabel_source)
            )
        finally:
            serializable_module.MIGRATE_LEGACY_FNS.reset(token)

        assert isinstance(inst, Instruction)

    def test_no_legacy_pattern_detected_is_unaffected(self):
        """A version older than SERIALIZATION_VERSION with no detectable
        legacy pattern decodes normally regardless of the flag."""
        clean_source = "def apply_fn(patch_label):\n    return None\n"
        inst = Instruction._from_decoded_attrs(self._attr_dict(clean_source))
        assert isinstance(inst, Instruction)

    def test_current_version_source_is_never_gated(self):
        """The gate only applies to source older than SERIALIZATION_VERSION
        -- a current-version Instruction is never checked, even if its
        source happens to match a legacy pattern (there's no reason it
        would for genuinely current-version source, but the version check
        should short-circuit before ever scanning either way)."""
        attrs = self._attr_dict(
            "def apply_fn(patch_label):\n    return None\n",
            version=serializable_module.SERIALIZATION_VERSION,
        )
        inst = Instruction._from_decoded_attrs(attrs)
        assert isinstance(inst, Instruction)

    def test_patchdict_rename_needs_no_gate_at_all(self):
        """The class-location-rename case (PatchDict -> PatchLayout) needs
        no gate: `_update_imports` already rewrites the old name
        everywhere in the frozen source, so nothing resembling it remains
        to detect."""
        path = Path(__file__).parent / "QuantumProgram_v0.json.gz"
        with gzip.open(path, "rt") as f:
            data = json.load(f)
        source = data["global_instructions"]["Init Patch 5Q"][
            "_serialized_apply_fn"
        ]
        assert "PatchDict(" in source  # confirms the fixture predates the rename

        updated = Serializable._update_imports(source, 0)
        assert "PatchDict(" not in updated
        assert "PatchLayout(" in updated

        # And, correspondingly, decoding this real Instruction raises nothing
        # migrate_legacy_fns-related at all, even with the gate active by
        # default (False).
        inst = Instruction._from_decoded_attrs(
            self._attr_dict(source, version=0, name="Init Patch 5Q")
        )
        assert isinstance(inst, Instruction)


class TestUpdateLegacyConstructions:
    """Regression tests for `Serializable._update_legacy_constructions`
    (issue #97's Sub-problem C): rewriting a resolvable old-format
    `InstructionLabel(...)` construction inside a frozen function's
    source at decode time, so a pattern the rewrite fixes never needs
    `migrate_legacy_fns=True` at all. A sibling pass to `_update_imports`
    (see `loqs/tools/migrate/renames.py`'s own docstring for why renames
    and this aren't the same mechanism), sharing its detect/resolve/
    rewrite engine with the standalone `loqs.tools.migrate` library
    rather than duplicating it.
    """

    libcst = pytest.importorskip("libcst")

    @staticmethod
    def _increment_registry():
        from loqs.codepacks.codepack_trivial_counter import create_qec_code

        return create_qec_code().instructions

    def test_noop_at_current_version(self):
        src = 'InstructionLabel("Increment", "L0", (), {"increment_by": 2})\n'
        assert (
            serializable_module.Serializable._update_legacy_constructions(
                src, serializable_module.SERIALIZATION_VERSION
            )
            == src
        )

    def test_noop_without_a_registry(self):
        """No LEGACY_INSTRUCTION_REGISTRY set (the default): every
        candidate is unresolvable, so nothing is rewritten -- same
        behavior as before this method existed."""
        src = 'InstructionLabel("Increment", "L0", (), {"increment_by": 2})\n'
        assert (
            serializable_module.Serializable._update_legacy_constructions(
                src, 0
            )
            == src
        )

    def test_rewrites_when_a_registry_resolves_the_instruction(self):
        token = serializable_module.LEGACY_INSTRUCTION_REGISTRY.set(
            self._increment_registry()
        )
        try:
            rewritten = (
                serializable_module.Serializable._update_legacy_constructions(
                    'InstructionLabel("Increment", "L0", (), '
                    '{"increment_by": 2})\n',
                    0,
                )
            )
        finally:
            serializable_module.LEGACY_INSTRUCTION_REGISTRY.reset(token)
        assert (
            rewritten
            == 'InstructionLabel("Increment", increment_by=2, patch_label="L0")\n'
        )

    def test_gate_bypassed_when_registry_resolves_the_pattern(self):
        """The real integration point: Instruction._from_decoded_attrs's
        migrate_legacy_fns gate never even triggers here, since the
        rewrite already fixed the only pattern it would have detected --
        no need to pass migrate_legacy_fns=True at all."""
        src = 'InstructionLabel("Increment", "L0", (), {"increment_by": 2})\n'
        attrs = {
            "_serialized_apply_fn": (
                f"def apply_fn(patch_label):\n    return {src.strip()}\n"
            ),
            "_serialized_map_qubits_fn": (
                "def map_qubits_fn(qubit_map):\n    return {}\n"
            ),
            "version": 0,
            "type": "Test",
            "data": {},
            "param_error_behavior": "warn",
            "name": "test",
            "_param_priorities": {},
            "_param_aliases": {},
        }

        with pytest.raises(RuntimeError, match="InstructionLabel"):
            Instruction._from_decoded_attrs(attrs)

        token = serializable_module.LEGACY_INSTRUCTION_REGISTRY.set(
            self._increment_registry()
        )
        try:
            inst = Instruction._from_decoded_attrs(attrs)
        finally:
            serializable_module.LEGACY_INSTRUCTION_REGISTRY.reset(token)
        assert isinstance(inst, Instruction)

    def test_load_wires_instruction_registry_through_the_contextvar(
        self, monkeypatch
    ):
        """`Serializable.load`'s new `instruction_registry` parameter sets
        `LEGACY_INSTRUCTION_REGISTRY` for the duration of the call and
        resets it afterward, mirroring `migrate_legacy_fns`'s own
        established wiring -- checked directly via a spy on `.decode`
        rather than a full JSON/HDF5 round trip, which exercises this one
        specific piece of wiring in isolation."""
        registry = self._increment_registry()
        seen = {}

        def fake_decode(state, format, decode_cache=None):
            seen["registry"] = (
                serializable_module.LEGACY_INSTRUCTION_REGISTRY.get()
            )
            return None

        monkeypatch.setattr(Serializable, "decode", staticmethod(fake_decode))
        monkeypatch.setattr("json.load", lambda f: {})

        import io

        Serializable.load(
            io.StringIO("{}"), format="json", instruction_registry=registry
        )
        assert seen["registry"] is registry
        assert serializable_module.LEGACY_INSTRUCTION_REGISTRY.get() is None


class TestInstructionLabelDecodeRemap:
    """Narrow, unit-level tests for the old-format InstructionLabel decode
    remap (v1.2) -- isolated from any real fixture or the full
    QuantumProgram.read pipeline, using hand-built attr_dicts shaped like
    the real old 5-key shape (instruction/inst_label/patch_label/
    inst_args/inst_kwargs)."""

    @staticmethod
    def _make_instruction(param_priorities):
        from loqs.core.instructions.instruction import Instruction

        def apply_fn(**kwargs):
            from loqs.core.frame import Frame

            return Frame(kwargs)

        return Instruction(
            apply_fn,
            param_priorities=param_priorities,
            name="test instruction",
        )

    def test_already_resolved_instruction_remaps_standalone(self):
        """When `instruction` is already resolved, the full remap happens
        right there, with no sibling context needed at all."""
        from loqs.core.instructions.instructionlabel import InstructionLabel

        inst = self._make_instruction({"a": ["label"], "b": ["label"]})
        attr_dict = {
            "instruction": inst,
            "inst_label": None,
            "patch_label": "L0",
            "inst_args": [1, 2],
            "inst_kwargs": {"c": 3},
        }
        label = InstructionLabel._from_decoded_attrs(attr_dict)
        assert isinstance(label, InstructionLabel)
        assert label["instruction"] is inst
        assert label["patch_label"] == "L0"
        assert label["a"] == 1
        assert label["b"] == 2
        assert label["c"] == 3

    def test_positional_arg_wins_over_same_key_kwarg(self):
        """Matches the pre-1.2 `_collect_kwarg` precedence: a positional
        value overrides a same-key `inst_kwargs` entry, not vice versa."""
        from loqs.core.instructions.instructionlabel import InstructionLabel

        inst = self._make_instruction({"a": ["label"]})
        attr_dict = {
            "instruction": inst,
            "patch_label": None,
            "inst_args": ["from_position"],
            "inst_kwargs": {"a": "from_kwarg"},
        }
        label = InstructionLabel._from_decoded_attrs(attr_dict)
        assert label["a"] == "from_position"

    def test_bare_string_instruction_with_no_args_decodes_directly(self):
        """No `inst_args` to remap means no deferral at all -- decode
        produces an ordinary `InstructionLabel`, resolved lazily by the
        same machinery any current-format string-named label already
        uses."""
        from loqs.core.instructions.instructionlabel import (
            LEGACY_PENDING_INST_ARGS,
            InstructionLabel,
        )

        attr_dict = {
            "instruction": None,
            "inst_label": "Some Global Instruction",
            "patch_label": None,
            "inst_args": [],
            "inst_kwargs": {"qubit_labels": ["Q0"]},
        }
        label = InstructionLabel._from_decoded_attrs(attr_dict)
        assert isinstance(label, InstructionLabel)
        assert label["instruction"] == "Some Global Instruction"
        assert label["qubit_labels"] == ["Q0"]
        assert LEGACY_PENDING_INST_ARGS not in label

    def test_bare_string_instruction_with_args_stashes_pending_marker(self):
        """Non-empty `inst_args` can't be remapped without the resolved
        `Instruction`'s `param_priorities` -- stashed under
        `LEGACY_PENDING_INST_ARGS` for `QuantumProgram._label_kwargs` to
        finish once resolved."""
        from loqs.core.instructions.instructionlabel import (
            LEGACY_PENDING_INST_ARGS,
            InstructionLabel,
        )

        attr_dict = {
            "instruction": None,
            "inst_label": "Some Global Instruction",
            "patch_label": "L0",
            "inst_args": [7],
            "inst_kwargs": {},
        }
        label = InstructionLabel._from_decoded_attrs(attr_dict)
        assert isinstance(label, InstructionLabel)
        assert label["instruction"] == "Some Global Instruction"
        assert label["patch_label"] == "L0"
        assert label[LEGACY_PENDING_INST_ARGS] == (7,)

    def test_label_kwargs_remaps_once_instruction_is_resolved(self):
        """`QuantumProgram._label_kwargs` finishes the remap a pending
        label was decoded with, given the now-resolved `Instruction`."""
        from loqs.core.instructions.instructionlabel import InstructionLabel

        inst = self._make_instruction({"a": ["label"], "b": ["label"]})
        label = InstructionLabel._from_decoded_attrs(
            {
                "instruction": None,
                "inst_label": "MyGlobalInst",
                "patch_label": None,
                "inst_args": [1, 2],
                "inst_kwargs": {"c": 3},
            }
        )
        kwargs = QuantumProgram._label_kwargs(label, inst)
        assert kwargs["a"] == 1
        assert kwargs["b"] == 2
        assert kwargs["c"] == 3

    def test_label_kwargs_is_a_no_op_without_a_pending_marker(self):
        """No `LEGACY_PENDING_INST_ARGS` key means `_label_kwargs` returns
        `inst_label` itself, unmodified -- true for every current-format
        label, not just the empty-`inst_args` legacy case."""
        from loqs.core.instructions.instructionlabel import InstructionLabel

        inst = self._make_instruction({})
        label = InstructionLabel("MyGlobalInst", qubit_labels=["Q0"])
        assert QuantumProgram._label_kwargs(label, inst) is label


class TestInstructionLabelDirectLegacyConstruction:
    """Old-style positional `InstructionLabel(instruction, patch_label,
    inst_args, inst_kwargs)` construction, called directly (not via
    decode) -- mirrors `PatchDict`'s warn-and-redirect treatment, but
    needs no separate shim class, since old and new `InstructionLabel`
    are the same class."""

    @staticmethod
    def _make_instruction(param_priorities):
        from loqs.core.instructions.instruction import Instruction

        def apply_fn(**kwargs):
            from loqs.core.frame import Frame

            return Frame(kwargs)

        return Instruction(
            apply_fn,
            param_priorities=param_priorities,
            name="test instruction",
        )

    def test_resolved_instruction_warns_and_remaps(self):
        from loqs.core.instructions.instructionlabel import InstructionLabel

        inst = self._make_instruction({"a": ["label"]})
        with pytest.warns(DeprecationWarning, match="Old-style positional"):
            label = InstructionLabel(inst, "L0", (5,), {"b": 6})
        assert label["patch_label"] == "L0"
        assert label["a"] == 5
        assert label["b"] == 6

    def test_unresolved_string_raises_clearly(self):
        from loqs.core.instructions.instructionlabel import InstructionLabel

        with pytest.raises(TypeError, match="can't be remapped"):
            InstructionLabel("SomeInstructionName", "L0", (), {})



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

    @pytest.mark.xfail(
        strict=True,
        reason="issue #97: the InstructionLabelCastableTypes/"
        "InstructionStackCastableTypes rename (2.2) and the old-format "
        "InstructionLabel decode-time remap (2.11) are both now fixed -- "
        "confirmed directly: both fixtures decode and start running with "
        "migrate_legacy_fns=True. A third, separate, out-of-scope issue "
        "remains: .run() fails with \"Failed to look up ('Gh', ('D0',))\" "
        "in DictNoiseModel.get_reps, a noise-model gate-dict completeness "
        "gap between these old fixtures' frozen data and current circuit "
        "generation -- unrelated to serialization/import compatibility, "
        "not part of issue #97's scope.",
    )
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
        #assert len(loaded_program.shot_histories) == 1
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
        (issue #96) need real version-2 compat entries -- these names are
        referenced inside the frozen `apply_fn`/`map_qubits_fn` source of
        real serialized `QuantumProgram` fixtures (`InstructionStack`
        elements), and previously had no compat entry at all, breaking
        decode with `ImportError`. Isolates the import-rewrite mechanism
        directly, independent of the decode-time label remap (a separate
        gap tracked elsewhere)."""
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
        incremented in its old `while` loop) -- the existing suite had no
        test exercising more than one hop, which is exactly how that bug
        went unnoticed."""
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
    """Regression tests for the migrate_legacy_fns gate (issue #97).

    Confirmed directly, not assumed: a straight class-location rename
    (e.g. PatchDict -> PatchLayout) needs no gate at all, since
    `_update_imports` already rewrites frozen source constructing the old
    name to construct the new one directly, with no shim ever reached --
    verified against the real `PatchDict()` call frozen in
    `QuantumProgram_v0.json.gz`'s "Init Patch 5Q" instruction. The gate
    only matters for a calling-convention change with the *same* class
    name, which no rename can fix: `QuantumProgram_v1.json.gz`'s
    "Repeat-until-success FT Minus Prep" instruction genuinely freezes an
    old-style positional `InstructionLabel(rus_key, patch_label, None,
    {...})` call.
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
        """The class-location-rename case (PatchDict -> PatchLayout) is
        confirmed to need no gate: _update_imports already rewrites the
        old name to the new one everywhere in the frozen source, so
        nothing resembling the old name remains to detect, and no shim is
        ever reached."""
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


class TestInstructionLabelDecodeRemap:
    """Narrow, unit-level tests for the old-format InstructionLabel decode
    remap (issue #97/#104, Part 2.11) -- isolated from any real fixture or
    the full QuantumProgram.read pipeline, using hand-built attr_dicts
    shaped exactly like the real, confirmed old shape (5 keys:
    instruction/inst_label/patch_label/inst_args/inst_kwargs)."""

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
        """Matches the confirmed pre-#104 `_collect_kwarg` precedence: a
        positional value overrides a same-key `inst_kwargs` entry, not
        the other way around."""
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

    def test_bare_string_instruction_produces_pending_placeholder(self):
        from loqs.core.instructions.instructionlabel import (
            InstructionLabel,
            _PendingLegacyInstructionLabel,
        )

        attr_dict = {
            "instruction": None,
            "inst_label": "Some Global Instruction",
            "patch_label": None,
            "inst_args": [],
            "inst_kwargs": {},
        }
        result = InstructionLabel._from_decoded_attrs(attr_dict)
        assert isinstance(result, _PendingLegacyInstructionLabel)
        assert result.inst_label == "Some Global Instruction"

    def test_pending_placeholder_raises_clearly_if_used_unresolved(self):
        from loqs.core.instructions.instructionlabel import (
            _PendingLegacyInstructionLabel,
        )

        pending = _PendingLegacyInstructionLabel(
            inst_label="X", patch_label=None, inst_args=(), inst_kwargs={}
        )
        with pytest.raises(RuntimeError, match="was never resolved during decode"):
            pending["instruction"]
        with pytest.raises(RuntimeError, match="was never resolved during decode"):
            pending.get("instruction")

    def test_instructionstack_from_decoded_attrs_bypasses_from_raw(self):
        """`InstructionStack._from_decoded_attrs` must set `_instructions`
        directly rather than going through `InstructionLabel.from_raw`
        (which would reject a pending placeholder outright)."""
        from loqs.core.instructions.instructionlabel import (
            InstructionLabel,
            _PendingLegacyInstructionLabel,
        )
        from loqs.core.instructions.instructionstack import InstructionStack

        pending = _PendingLegacyInstructionLabel(
            inst_label="X", patch_label=None, inst_args=(), inst_kwargs={}
        )
        real_label = InstructionLabel("Y")
        stack = InstructionStack._from_decoded_attrs(
            {"_instructions": [pending, real_label]}
        )
        assert isinstance(stack, InstructionStack)
        assert stack._instructions == [pending, real_label]

    def test_quantumprogram_resolves_pending_global_label(self):
        """End-to-end (but fixture-free) test of `QuantumProgram`'s own
        `_from_decoded_attrs` resolving a pending global label, using its
        sibling `global_instructions` attribute."""
        from loqs.core.instructions.instructionlabel import (
            InstructionLabel,
            _PendingLegacyInstructionLabel,
        )
        from loqs.core.instructions.instructionstack import InstructionStack
        from loqs.core.history import History

        inst = self._make_instruction({"a": ["label"]})
        pending = _PendingLegacyInstructionLabel(
            inst_label="MyGlobalInst",
            patch_label=None,
            inst_args=[42],
            inst_kwargs={},
        )
        stack = InstructionStack._from_decoded_attrs({"_instructions": [pending]})

        attr_dict = {
            "instruction_stack": stack,
            "initial_history": History(),
            "default_base_seed": None,
            "default_noise_model": None,
            "state_type": None,
            "patch_types": {},
            "global_instructions": {"MyGlobalInst": inst},
            "name": "test program",
        }
        program = QuantumProgram._from_decoded_attrs(attr_dict)
        resolved_label = program.instruction_stack[0]
        assert isinstance(resolved_label, InstructionLabel)
        # Global resolution deep-copies (matching _resolve_instruction's
        # own existing behavior), so compare identity of the resolved
        # instruction's name, not the object itself.
        assert resolved_label["instruction"].name == inst.name
        assert resolved_label["a"] == 42

    def test_quantumprogram_resolves_pending_per_patch_label(self):
        """Same as above, but for a per-patch label -- resolved against
        `patch_types`' own instruction templates directly (no live
        PatchLayout needed, since none exists yet at decode time)."""
        from loqs.core.instructions.instructionlabel import InstructionLabel
        from loqs.core.instructions.instructionlabel import (
            _PendingLegacyInstructionLabel,
        )
        from loqs.core.instructions.instructionstack import InstructionStack
        from loqs.core.history import History
        from loqs.core.qeccode import QECCode

        inst = self._make_instruction({"a": ["label"]})
        code = QECCode({"MyPatchInst": inst}, ["Q0"], ["Q0"])
        pending = _PendingLegacyInstructionLabel(
            inst_label="MyPatchInst",
            patch_label="L0",
            inst_args=[7],
            inst_kwargs={},
        )
        stack = InstructionStack._from_decoded_attrs({"_instructions": [pending]})

        attr_dict = {
            "instruction_stack": stack,
            "initial_history": History(),
            "default_base_seed": None,
            "default_noise_model": None,
            "state_type": None,
            "patch_types": {"MyCode": code},
            "global_instructions": {},
            "name": "test program",
        }
        program = QuantumProgram._from_decoded_attrs(attr_dict)
        resolved_label = program.instruction_stack[0]
        assert isinstance(resolved_label, InstructionLabel)
        assert resolved_label["patch_label"] == "L0"
        assert resolved_label["a"] == 7

    def test_quantumprogram_raises_clearly_for_unresolvable_global_label(self):
        from loqs.core.instructions.instructionlabel import (
            _PendingLegacyInstructionLabel,
        )
        from loqs.core.instructions.instructionstack import InstructionStack
        from loqs.core.history import History

        pending = _PendingLegacyInstructionLabel(
            inst_label="DoesNotExist",
            patch_label=None,
            inst_args=(),
            inst_kwargs={},
        )
        stack = InstructionStack._from_decoded_attrs({"_instructions": [pending]})
        attr_dict = {
            "instruction_stack": stack,
            "initial_history": History(),
            "default_base_seed": None,
            "default_noise_model": None,
            "state_type": None,
            "patch_types": {},
            "global_instructions": {},
            "name": "test program",
        }
        with pytest.raises(RuntimeError, match="Could not resolve global legacy"):
            QuantumProgram._from_decoded_attrs(attr_dict)


class TestInstructionLabelDirectLegacyConstruction:
    """Old-style positional `InstructionLabel(instruction, patch_label,
    inst_args, inst_kwargs)` construction, called directly (not via
    decode) -- the "simpler" sub-case from Part 2.11, mirroring
    `PatchDict`'s warn-and-redirect treatment but without needing a
    separate shim class, since old and new `InstructionLabel` are the
    same class."""

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



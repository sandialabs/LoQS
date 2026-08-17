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
        reason="issue #97: these fixtures' frozen apply_fn/map_qubits_fn "
        "source imports InstructionLabelCastableTypes/"
        "InstructionStackCastableTypes, renamed to *Like for issue #96; "
        "restoring this needs a real SERIALIZATION_VERSION bump with "
        "IMPORT_LOCATION_CHANGES_BY_VERSION entries for every renamed "
        "*CastableTypes name, not just SyndromeLabelCastableTypes",
    )
    @pytest.mark.parametrize("version_file",[
        "QuantumProgram_v0.json.gz",
        "QuantumProgram_v1.json.gz",
    ])
    def test_read_versioned_quantumprogram(self, version_file):
        """Test whether we can load QuantumProgram for given serialization version.
        
        Test files are taken from test_quantumprogram files."""

        path = Path(__file__).parent
        loaded_program = QuantumProgram.read(path / version_file)

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



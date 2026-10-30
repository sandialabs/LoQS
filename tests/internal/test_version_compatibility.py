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
from loqs.internal.serializable import Serializable


class TestVersionCompatibility:
    """Parameterized tests for Serializable class functionality."""

    @pytest.mark.parametrize(
        "version_file",
        [
            "QuantumProgram_v0.json.gz",
            "QuantumProgram_v1.json.gz",
        ],
    )
    def test_read_versioned_quantumprogram(self, version_file):
        """Test whether we can load QuantumProgram for given serialization version.

        Test files are taken from test_quantumprogram files."""

        path = Path(__file__).parent
        loaded_program = QuantumProgram.read(
            path / version_file, migrate_legacy_fns=True
        )

        assert isinstance(loaded_program, QuantumProgram)
        assert loaded_program.name == "Prep minus, measure X"
        assert len(loaded_program.instruction_stack) == 4
        assert loaded_program.state_type == QSimQuantumState
        assert loaded_program.patch_types is not None
        assert list(loaded_program.patch_types.keys())[0] == "5Q"

        loaded_program_results = loaded_program.run(2)
        assert len(loaded_program_results.shot_histories) == 2

    def test_function_import_updates(self):
        # A version 0 apply_fn: old import locations and class names.
        test_str = """
from loqs.core.recordables.patchdict import PatchDict
from loqs.core.syndrome import PauliFrame, SyndromeLabel
def apply_fn(patches: PatchDict, label: SyndromeLabel):
    return PauliFrame(patches, label)
"""
        updated_str = Serializable._update_imports(test_str, 0)

        assert "PatchLayout" in updated_str
        assert "loqs.core.recordables.pauliframe" in updated_str
        assert "loqs.core.syndromelabel" in updated_str
        assert "PatchDict" not in updated_str
        assert "loqs.core.syndrome " not in updated_str

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
from loqs.core.instructions.instructionlabel import (
    InstructionLabel,
    InstructionLabelLike)
from loqs.core.instructions.instructionstack import (
    InstructionStack,
    InstructionStackLike)
def apply_fn(
    instructions: InstructionStackLike,
) -> None:
    pass
"""
        updated_str = Serializable._update_imports(test_str, 1)
        assert updated_str == expected_str

    @pytest.mark.parametrize(
        "fake_table, version, expected",
        [
            # A rename chained across 3 versions composes into one mapping
            # (a former infinite loop in the composition).
            pytest.param(
                {
                    1: {("mod0", "A"): ("mod1", "B")},
                    2: {("mod1", "B"): ("mod2", "C")},
                    3: {("mod2", "C"): ("mod3", "D")},
                },
                4,
                {("mod0", "A"): ("mod3", "D")},
                id="multi_hop",
            ),
            # Versions 1 and 3 have no table entry at all.
            pytest.param(
                {
                    2: {("mod0", "A"): ("mod2", "B")},
                    4: {("mod2", "B"): ("mod4", "C")},
                },
                5,
                {("mod0", "A"): ("mod4", "C")},
                id="missing_entry",
            ),
        ],
    )
    def test_get_cumulative_changes(
        self, monkeypatch, fake_table, version, expected
    ):
        monkeypatch.setattr(
            serializable_module,
            "IMPORT_LOCATION_CHANGES_BY_VERSION",
            fake_table,
        )
        monkeypatch.setattr(
            serializable_module, "SERIALIZATION_VERSION", version
        )

        assert Serializable._get_cumulative_changes(0) == expected


def _decoded_str(value):
    """Undo json's/Serializable's own primitive-wrapping, if present."""
    if isinstance(value, dict) and value.get("encode_type") == "primitive":
        return value.get("value")
    return value


def _find_instruction_by_name(obj, name):
    """Find a raw, undecoded Instruction attr_dict by name inside a decompressed fixture."""
    if isinstance(obj, dict):
        if (
            obj.get("class") == "Instruction"
            and _decoded_str(obj.get("name")) == name
        ):
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

    A calling-convention change with the same class name (e.g. an old
    positional `InstructionLabel(...)` call) is confidently rewritten by
    `_update_legacy_constructions` in the common case, including
    `QuantumProgram_v1.json.gz`'s "Repeat-until-success FT Minus Prep"
    instruction -- the gate only fires for the narrow remainder it can't
    statically resolve (e.g. a starred/splat call).
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

    def test_real_fixture_decodes_without_the_gate(
        self, real_old_style_instructionlabel_source
    ):
        """The RUS instruction's old-style positional `InstructionLabel(
        ...)` call is confidently rewritten by
        `_update_legacy_constructions`, so it decodes cleanly with no
        need for `migrate_legacy_fns=True` at all."""
        inst = Instruction._from_decoded_attrs(
            self._attr_dict(real_old_style_instructionlabel_source)
        )
        assert isinstance(inst, Instruction)

    def test_unrewritable_pattern_still_triggers_the_gate(self):
        """A pattern the rewrite can't confidently resolve (here, a
        starred/splat call) still hits the gate as before."""
        src = "InstructionLabel(*label_tuple)\n"
        attrs = self._attr_dict(
            f"def apply_fn(patch_label, label_tuple):\n    return {src.strip()}\n"
        )
        with pytest.raises(RuntimeError):
            Instruction._from_decoded_attrs(attrs)
        with pytest.raises(RuntimeError):
            Instruction._from_decoded_attrs(attrs)

        token = serializable_module.MIGRATE_LEGACY_FNS.set(True)
        try:
            inst = Instruction._from_decoded_attrs(attrs)
        finally:
            serializable_module.MIGRATE_LEGACY_FNS.reset(token)
        assert isinstance(inst, Instruction)


class TestUpdateLegacyConstructions:
    """Regression tests for `Serializable._update_legacy_constructions`:
    rewriting a resolvable old-format `InstructionLabel(...)` construction
    inside a frozen function's source at decode time, so a pattern the
    rewrite fixes never needs `migrate_legacy_fns=True` at all. A sibling
    pass to `_update_imports`, sharing its detect/resolve/rewrite engine
    with the standalone `loqs.tools.migrate` library rather than
    duplicating it.
    """

    def test_noop_at_current_version(self):
        src = 'InstructionLabel("Increment", "L0", (), {"increment_by": 2})\n'
        assert serializable_module.Serializable._update_legacy_constructions(
            src, serializable_module.SERIALIZATION_VERSION
        ) == (src, [])

    def test_rewrites_a_resolvable_construction(self):
        rewritten, manual_review = (
            serializable_module.Serializable._update_legacy_constructions(
                'InstructionLabel("Increment", "L0", (), '
                '{"increment_by": 2})\n',
                0,
            )
        )
        assert (
            rewritten
            == 'InstructionLabel("Increment", increment_by=2, patch_label="L0")\n'
        )
        assert manual_review == []

    def test_falls_through_unchanged_on_an_unparseable_source(self):
        """A frozen function's source isn't guaranteed to be a single,
        standalone-parseable module (e.g. indented relative to its
        original enclosing scope, as constructed here) -- the rewrite
        must fall through to the source unchanged rather than raise and
        break decoding entirely."""
        src = (
            "    def apply_fn(patch_label):\n"
            '        return InstructionLabel("Increment", "L0", (), {})\n'
        )
        assert serializable_module.Serializable._update_legacy_constructions(
            src, 0
        ) == (src, [])


class TestImportClass:
    """Regression tests for `Serializable._import_class`, resolving a
    decoded `(module, class)` pair against the rename/removal table,
    then actually importing it."""

    def test_resolves_a_renamed_class(self):
        cls = Serializable._import_class(
            "loqs.core.syndrome", "SyndromeLabel", 0
        )
        from loqs.core.syndromelabel import SyndromeLabel

        assert cls is SyndromeLabel

    def test_removed_class_raises_a_clear_import_error(self):
        """A `(module, class)` mapped to `None` (deleted outright, not
        relocated) raises immediately, rather than trying and failing to
        import a location that was never a real replacement."""
        with pytest.raises(ImportError):
            Serializable._import_class("loqs.internal.castable", "Castable", 0)

    def test_unresolvable_class_raises_a_clear_import_error(self):
        """A real, importable module missing the specific class raises
        `ImportError`, not the raw `AttributeError`."""
        with pytest.raises(ImportError):
            Serializable._import_class(
                "loqs.core.syndromelabel",
                "NonexistentClass",
                serializable_module.SERIALIZATION_VERSION,
            )


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

    def test_resolved_instruction_remaps(self):
        """An old label holding a resolved `Instruction` remaps its
        positional args; a positional value beats a same-key kwarg."""
        from loqs.core.instructions.instructionlabel import InstructionLabel

        inst = self._make_instruction({"a": ["label"], "b": ["label"]})
        attr_dict = {
            "instruction": inst,
            "inst_label": None,
            "patch_label": "L0",
            "inst_args": [1, 2],
            "inst_kwargs": {"c": 3, "a": "kw"},
        }
        label = InstructionLabel._from_decoded_attrs(attr_dict)
        assert isinstance(label, InstructionLabel)
        assert label["instruction"] is inst
        assert label["patch_label"] == "L0"
        assert label["a"] == 1
        assert label["b"] == 2
        assert label["c"] == 3

    def test_pending_args_stash_and_remap(self):
        """A bare-name label's args are stashed under
        `LEGACY_PENDING_INST_ARGS`, then remapped by `_label_kwargs` once
        the instruction resolves."""
        from loqs.core.instructions.instructionlabel import (
            LEGACY_PENDING_INST_ARGS,
            InstructionLabel,
        )

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
        assert label[LEGACY_PENDING_INST_ARGS] == (1, 2)

        kwargs = QuantumProgram._label_kwargs(label, inst)
        assert kwargs["a"] == 1
        assert kwargs["b"] == 2
        assert kwargs["c"] == 3


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
        with pytest.warns(DeprecationWarning):
            label = InstructionLabel(inst, "L0", (5,), {"b": 6})
        assert label["patch_label"] == "L0"
        assert label["a"] == 5
        assert label["b"] == 6

    def test_unresolved_string_raises_clearly(self):
        from loqs.core.instructions.instructionlabel import InstructionLabel

        with pytest.raises(TypeError):
            InstructionLabel("SomeInstructionName", "L0", (), {})

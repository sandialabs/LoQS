"""Tester for loqs.tools.migrate"""

from pathlib import Path

import pytest

pytest.importorskip("libcst")

from loqs.codepacks.codepack_trivial_counter import create_qec_code
from loqs.tools.migrate import migrate_source
from loqs.tools.migrate.config import MigrationConfig, load_instruction_registry
from loqs.tools.migrate.flags import detect_flagged_patterns
from loqs.tools.migrate.labels import migrate_instruction_labels
from loqs.tools.migrate.notebook import migrate_notebook_source
from loqs.tools.migrate.renames import RENAMES, rewrite_renames

FIXTURES = Path(__file__).parent / "migrate_fixtures"


@pytest.fixture
def instructions():
    return create_qec_code().instructions


class TestRewriteRenames:
    def test_rewrites_import_and_usage(self):
        src = (
            "from loqs.core.recordables.patchdict import PatchDict\n\n"
            'patches = PatchDict({"L0": None})\n'
        )
        result = rewrite_renames(src)
        assert result.changed
        assert "PatchLayout" in result.source
        assert "PatchDict" not in result.source
        assert result.manual_review == []

    def test_does_not_corrupt_docstring_narrating_history(self):
        """A docstring describing the *history* of a rename must not be
        rewritten -- doing so would silently invert its meaning (see
        renames.py's module docstring for the real example this
        regression-tests)."""
        src = FIXTURES.joinpath("renames_before.py").read_text()
        result = rewrite_renames(src)
        assert "used to construct a `PatchDict` directly" in result.source

    def test_does_not_corrupt_unrelated_from_prefixed_identifier(self):
        """A local variable/method chain merely starting with the text
        "from" and spanning multiple lines must not be misread as a
        multi-line import statement."""
        src = (
            "from_prime_basis_circ = build()\n"
            "from_prime_basis_circ.pad_single_qubit_idles_by_duration_inplace(\n"
            "    idle_gates, gate_durations\n"
            ")\n"
        )
        result = rewrite_renames(src)
        assert result.source == src
        assert not result.changed

    def test_no_op_when_nothing_matches(self):
        src = "x = 1\n"
        result = rewrite_renames(src)
        assert result.source == src
        assert not result.changed
        assert result.manual_review == []

    def test_deleted_name_flagged_not_guessed(self):
        src = FIXTURES.joinpath("deleted_name_before.py").read_text()
        result = rewrite_renames(src)
        assert result.source == src  # never rewritten, only flagged
        lines = {item.line for item in result.manual_review}
        assert lines == {2, 5}  # the import line and the usage line

    def test_matches_golden_fixture(self):
        before = FIXTURES.joinpath("renames_before.py").read_text()
        after = FIXTURES.joinpath("renames_after.py").read_text()
        result = rewrite_renames(before)
        assert result.source == after


class TestMigrateInstructionLabels:
    def test_resolvable_kwargs_form(self, instructions):
        src = 'InstructionLabel("Increment", "L0", (), {"increment_by": 2})\n'
        result = migrate_instruction_labels(src, instructions)
        assert result.changed
        assert (
            result.source
            == 'InstructionLabel("Increment", increment_by=2, patch_label="L0")\n'
        )
        assert result.manual_review == []

    def test_resolvable_bare_tuple(self, instructions):
        src = 'tup = ("Increment", "L0", (), {"increment_by": 2})\n'
        result = migrate_instruction_labels(src, instructions)
        assert (
            result.source
            == 'tup = InstructionLabel("Increment", increment_by=2, patch_label="L0")\n'
        )

    def test_three_tuple_with_none_patch_label(self, instructions):
        src = 'tup = ("Init Counter", None, (7,))\n'
        result = migrate_instruction_labels(src, instructions)
        assert result.source == 'tup = InstructionLabel("Init Counter", initial_value=7)\n'

    def test_none_literal_inst_args_and_kwargs_treated_as_empty(self, instructions):
        """A real shape found in QuantumProgram_v1.json.gz's frozen
        source: `InstructionLabel(name, patch_label, None, {...})` uses a
        literal `None` for inst_args, not an empty tuple `()` --
        `_remap_legacy_positional_args`'s own runtime handling
        (`tuple(inst_args or ())`) treats both the same way."""
        src = 'InstructionLabel("Increment", "L0", None, None)\n'
        result = migrate_instruction_labels(src, instructions)
        assert result.source == 'InstructionLabel("Increment", patch_label="L0")\n'

    def test_three_tuple_with_string_second_element_is_not_a_candidate(
        self, instructions
    ):
        """A pyGSTi-style circuit-layer gate-label tuple, e.g.
        `("Gcphase", "A0", "D4")`, must not be treated as a candidate --
        confirmed via real testing against docs/notebooks/buildinstruction.md
        that this shape is far more common than the legacy label shape it
        could be confused with (see labels.py's module docstring)."""
        src = 'layer = ("Gcphase", "A0", "D4")\n'
        result = migrate_instruction_labels(src, instructions)
        assert result.source == src
        assert result.manual_review == []

    def test_already_modern_call_is_untouched(self, instructions):
        src = 'InstructionLabel("Increment", patch_label="L0", counter=5)\n'
        result = migrate_instruction_labels(src, instructions)
        assert not result.changed
        assert result.manual_review == []

    def test_resolved_instruction_object_is_flagged_not_rewritten(
        self, instructions
    ):
        """May already work today via a DeprecationWarning if it
        evaluates to a resolved Instruction (see
        InstructionLabel.__init__) -- still not silently rewritten either
        way, since the tool can't know a variable's runtime value."""
        src = 'InstructionLabel(some_instruction, "L0", (), {})\n'
        result = migrate_instruction_labels(src, instructions)
        assert not result.changed
        assert len(result.manual_review) == 1
        assert "DeprecationWarning" in result.manual_review[0].message
        assert "TypeError" in result.manual_review[0].message

    def test_unknown_instruction_name_is_flagged(self, instructions):
        src = 'InstructionLabel("Nonexistent", "L0", (), {})\n'
        result = migrate_instruction_labels(src, instructions)
        assert not result.changed
        assert "Could not resolve instruction" in result.manual_review[0].message

    def test_non_literal_kwargs_is_flagged(self, instructions):
        src = 'InstructionLabel("Increment", "L0", (), extra_kwargs)\n'
        result = migrate_instruction_labels(src, instructions)
        assert not result.changed
        assert "literal tuple/list/dict" in result.manual_review[0].message

    def test_splat_call_is_flagged(self, instructions):
        src = "InstructionLabel(*label_tuple)\n"
        result = migrate_instruction_labels(src, instructions)
        assert not result.changed
        assert "splat call" in result.manual_review[0].message

    def test_double_star_kwargs_unpack_is_already_modern(self, instructions):
        """A real false positive found by testing against this repo's own
        `builders.py`/`instructionlabel.py`: `InstructionLabel(instruction,
        **kwargs)` was misclassified as an unresolvable positional splat --
        `**kwargs` doesn't affect positional-arg counting at all."""
        src = "InstructionLabel(instruction, **kwargs)\n"
        result = migrate_instruction_labels(src, instructions)
        assert not result.changed
        assert result.manual_review == []

    def test_legacy_positional_with_trailing_double_star_is_rewritten(
        self, instructions
    ):
        """A legacy positional call combined with a `**kwargs` unpack is
        still rewritten -- the unpack is carried through verbatim, last,
        since its contents can't be known statically."""
        src = (
            'InstructionLabel("Increment", "L0", (), {"increment_by": 2}, '
            "**extra)\n"
        )
        result = migrate_instruction_labels(src, instructions)
        assert (
            result.source
            == 'InstructionLabel("Increment", increment_by=2, patch_label="L0", **extra)\n'
        )
        assert result.manual_review == []

    def test_no_registry_flags_every_string_named_candidate(self):
        src = 'InstructionLabel("Increment", "L0", (), {})\n'
        result = migrate_instruction_labels(src, instructions={})
        assert not result.changed
        assert "Could not resolve instruction" in result.manual_review[0].message

    def test_idempotent(self, instructions):
        src = 'InstructionLabel("Increment", "L0", (), {"increment_by": 2})\n'
        once = migrate_instruction_labels(src, instructions)
        twice = migrate_instruction_labels(once.source, instructions)
        assert twice.source == once.source
        assert not twice.changed

    def test_matches_golden_fixture(self, instructions):
        before = FIXTURES.joinpath("labels_before.py").read_text()
        after = FIXTURES.joinpath("labels_after.py").read_text()
        result = migrate_instruction_labels(before, instructions)
        assert result.source == after
        assert len(result.manual_review) == 4


class TestDetectFlaggedPatterns:
    def test_matches_golden_fixture(self):
        src = FIXTURES.joinpath("flags_before.py").read_text()
        items = detect_flagged_patterns(src)
        assert [item.line for item in items] == [3, 5, 4, 4]

    def test_no_false_positive_on_ordinary_dict_access(self):
        assert detect_flagged_patterns('x = d["key"]\n') == []

    def test_no_false_positive_on_unrelated_third_party_cast(self):
        """A real false positive found by testing against this repo's own
        pyGSTi-integration code: `pygsti`'s own `Circuit.cast`/`Evotype.cast`
        classmethods share the removed LoQS API's exact method name."""
        src = "return _Circuit.cast(obj)\n"
        assert detect_flagged_patterns(src) == []

    def test_no_false_positive_on_unrelated_receiver_cast(self):
        src = "stack = some_stack.cast(InstructionStack)\n"
        assert detect_flagged_patterns(src) == []

    def test_no_false_positive_on_unrelated_include_idles_kwarg(self):
        """A real false positive found by testing against this repo's own
        surf17 codepack helpers: several still-current, unrelated
        `include_idles: bool` parameters exist on lower-level circuit-
        building functions, never renamed by issue #108."""
        src = "circuit_inst = build_circuit_instruction(include_idles=True)\n"
        assert detect_flagged_patterns(src) == []


class TestMigrateSource:
    def test_matches_golden_fixture_labels(self, instructions):
        before = FIXTURES.joinpath("labels_before.py").read_text()
        after = FIXTURES.joinpath("labels_after.py").read_text()
        result = migrate_source(before, instructions=instructions)
        assert result.source == after

    def test_matches_golden_fixture_renames(self):
        before = FIXTURES.joinpath("renames_before.py").read_text()
        after = FIXTURES.joinpath("renames_after.py").read_text()
        result = migrate_source(before)
        assert result.source == after

    def test_idempotent_on_already_migrated_source(self, instructions):
        after = FIXTURES.joinpath("labels_after.py").read_text()
        result = migrate_source(after, instructions=instructions)
        assert result.source == after
        assert not result.changed


class TestMigrateNotebookSource:
    def test_matches_golden_fixture(self, instructions):
        before = FIXTURES.joinpath("notebook_before.md").read_text()
        after = FIXTURES.joinpath("notebook_after.md").read_text()
        result = migrate_notebook_source(before, instructions=instructions)
        assert result.source == after

    def test_note_fence_is_never_touched(self, instructions):
        before = FIXTURES.joinpath("notebook_before.md").read_text()
        result = migrate_notebook_source(before, instructions=instructions)
        assert (
            'mention of `PatchDict` or `InstructionLabel("Name", "L0", (), {})`'
            in result.source
        )

    def test_field_line_is_passed_through_and_not_treated_as_code(
        self, instructions
    ):
        """`docs/notebooks/workflow.md` has a real `:tags: [...]` field
        line; this used to raise a ParserSyntaxError instead of migrating
        the cell."""
        before = FIXTURES.joinpath("notebook_before.md").read_text()
        result = migrate_notebook_source(before, instructions=instructions)
        assert ":tags: [scroll-output]" in result.source

    def test_cross_cell_import_context_is_a_known_limitation(self, instructions):
        """Each cell is migrated independently, with no memory of an
        earlier cell's imports (unlike a real notebook kernel) -- so a
        bare `PatchDict()` relying on an import done in an earlier cell
        is neither rewritten nor flagged here. Not a problem for the real
        `docs/notebooks/*.md` today (already fully migrated), but locked
        in as documented, understood behavior rather than a silent gap."""
        before = FIXTURES.joinpath("notebook_before.md").read_text()
        result = migrate_notebook_source(before, instructions=instructions)
        assert "patches2 = PatchDict()" in result.source


class TestMigrationConfig:
    def test_load_instruction_registry(self):
        registry = load_instruction_registry(
            "loqs.codepacks.codepack_trivial_counter:create_qec_code"
        )
        assert set(registry) == {"Increment", "Init Counter"}

    def test_instructions_for_uses_configured_registry(self):
        config = MigrationConfig(
            {
                "some/file.py": "loqs.codepacks.codepack_trivial_counter:create_qec_code",
            }
        )
        assert set(config.instructions_for("some/file.py")) == {
            "Increment",
            "Init Counter",
        }

    def test_instructions_for_unconfigured_file_is_empty(self):
        config = MigrationConfig({})
        assert config.instructions_for("unknown.py") == {}

    def test_caches_registry_across_files(self):
        config = MigrationConfig(
            {
                "a.py": "loqs.codepacks.codepack_trivial_counter:create_qec_code",
                "b.py": "loqs.codepacks.codepack_trivial_counter:create_qec_code",
            }
        )
        first = config.instructions_for("a.py")
        second = config.instructions_for("b.py")
        # Same cached dict object reused, not rebuilt, for the second file.
        assert first is second

    def test_from_json(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(
            '{"some/file.py": '
            '"loqs.codepacks.codepack_trivial_counter:create_qec_code"}'
        )
        config = MigrationConfig.from_json(config_path)
        assert set(config.instructions_for("some/file.py")) == {
            "Increment",
            "Init Counter",
        }


class TestRenamesTableCoverage:
    def test_every_rename_target_is_importable(self):
        """A regression check for the table itself: every non-`None`
        destination in RENAMES should be a real, currently-importable
        `(module, name)` -- catching a typo before it silently produces
        broken rewrites."""
        import importlib

        for old_key, new_loc in RENAMES.items():
            if new_loc is None:
                continue
            new_module, new_name = new_loc
            mod = importlib.import_module(new_module)
            assert hasattr(mod, new_name), (
                f"{old_key} -> {new_loc}: {new_name!r} not found in "
                f"{new_module!r}"
            )

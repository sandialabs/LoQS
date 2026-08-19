"""Tester for loqs.tools.migrate"""

from pathlib import Path

from loqs.tools.migrate import migrate_source
from loqs.tools.migrate.flags import detect_flagged_patterns
from loqs.tools.migrate.labels import migrate_instruction_labels
from loqs.tools.migrate.notebook import migrate_notebook_source
from loqs.tools.migrate.renames import RENAMES, rewrite_renames

FIXTURES = Path(__file__).parent / "migrate_fixtures"


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
        src = FIXTURES.joinpath("renames_before.py").read_text(encoding="utf-8")
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
        src = FIXTURES.joinpath("deleted_name_before.py").read_text(encoding="utf-8")
        result = rewrite_renames(src)
        assert result.source == src  # never rewritten, only flagged
        lines = {item.line for item in result.manual_review}
        assert lines == {2, 5}  # the import line and the usage line

    def test_matches_golden_fixture(self):
        before = FIXTURES.joinpath("renames_before.py").read_text(encoding="utf-8")
        after = FIXTURES.joinpath("renames_after.py").read_text(encoding="utf-8")
        result = rewrite_renames(before)
        assert result.source == after


class TestMigrateInstructionLabels:
    def test_resolvable_kwargs_form(self):
        src = 'InstructionLabel("Increment", "L0", (), {"increment_by": 2})\n'
        result = migrate_instruction_labels(src)
        assert result.changed
        assert (
            result.source
            == 'InstructionLabel("Increment", increment_by=2, patch_label="L0")\n'
        )
        assert result.manual_review == []

    def test_resolvable_bare_tuple(self):
        """A bare tuple rewrites to a bare dict, not an InstructionLabel(...)
        call -- unlike an explicit call being rewritten, building a call
        here would introduce a new InstructionLabel reference the file
        may never have needed to import."""
        src = 'tup = ("Increment", "L0", (), {"increment_by": 2})\n'
        result = migrate_instruction_labels(src)
        assert (
            result.source
            == 'tup = {"instruction": "Increment", "increment_by": 2, "patch_label": "L0"}\n'
        )

    def test_three_tuple_with_none_patch_label(self):
        src = 'tup = ("Init Counter", None, (7,))\n'
        result = migrate_instruction_labels(src)
        assert (
            result.source
            == 'tup = {"instruction": "Init Counter", "_legacy_inst_args": (7,)}\n'
        )

    def test_bare_tuple_in_a_list_rewrites_to_a_dict(self):
        """A bare tuple embedded in a larger structure (e.g. a stack's
        own list of raw entries, alongside an already-modern bare
        string) rewrites the same way a standalone one does: to a bare
        dict, not an InstructionLabel(...) call, so the file never needs
        a new InstructionLabel import it may not have had before."""
        src = (
            'stack = ["Init State", '
            '("Increment", "L0", (), {"increment_by": 2})]\n'
        )
        result = migrate_instruction_labels(src)
        assert result.source == (
            'stack = ["Init State", '
            '{"instruction": "Increment", "increment_by": 2, "patch_label": "L0"}]\n'
        )

    def test_none_literal_inst_args_and_kwargs_treated_as_empty(self):
        """A real shape found in QuantumProgram_v1.json.gz's frozen
        source: `InstructionLabel(name, patch_label, None, {...})` uses a
        literal `None` for inst_args, not an empty tuple `()` --
        `_remap_legacy_positional_args`'s own runtime handling
        (`tuple(inst_args or ())`) treats both the same way."""
        src = 'InstructionLabel("Increment", "L0", None, None)\n'
        result = migrate_instruction_labels(src)
        assert result.source == 'InstructionLabel("Increment", patch_label="L0")\n'

    def test_three_tuple_with_string_second_element_is_not_a_candidate(self):
        """A pyGSTi-style circuit-layer gate-label tuple, e.g.
        `("Gcphase", "A0", "D4")`, must not be treated as a candidate --
        confirmed via real testing against docs/notebooks/buildinstruction.md
        that this shape is far more common than the legacy label shape it
        could be confused with (see labels.py's module docstring)."""
        src = 'layer = ("Gcphase", "A0", "D4")\n'
        result = migrate_instruction_labels(src)
        assert result.source == src
        assert result.manual_review == []

    def test_already_modern_call_is_untouched(self):
        src = 'InstructionLabel("Increment", patch_label="L0", counter=5)\n'
        result = migrate_instruction_labels(src)
        assert not result.changed
        assert result.manual_review == []

    def test_non_literal_instruction_and_name_are_rewritten(self):
        """Neither the instruction expression nor the instruction name
        needs to be a literal string, or resolvable against any known
        registry, to be rewritten -- `InstructionLabel`'s own
        `LEGACY_PENDING_INST_ARGS` stash is resolved lazily at runtime
        once the real instruction is available, so this tool never needs
        to know what instructions exist."""
        src = 'InstructionLabel(some_instruction, "L0", (), {})\n'
        result = migrate_instruction_labels(src)
        assert result.source == 'InstructionLabel(some_instruction, patch_label="L0")\n'
        assert result.manual_review == []

        src = 'InstructionLabel("Nonexistent Instruction", "L0", (), {})\n'
        result = migrate_instruction_labels(src)
        assert (
            result.source
            == 'InstructionLabel("Nonexistent Instruction", patch_label="L0")\n'
        )
        assert result.manual_review == []

    def test_non_literal_kwargs_is_flagged(self):
        src = 'InstructionLabel("Increment", "L0", (), extra_kwargs)\n'
        result = migrate_instruction_labels(src)
        assert not result.changed
        assert "literal dict" in result.manual_review[0].message

    def test_splat_call_is_flagged(self):
        src = "InstructionLabel(*label_tuple)\n"
        result = migrate_instruction_labels(src)
        assert not result.changed
        assert "splat call" in result.manual_review[0].message

    def test_double_star_kwargs_unpack_is_already_modern(self):
        """A real false positive found by testing against this repo's own
        `builders.py`/`instructionlabel.py`: `InstructionLabel(instruction,
        **kwargs)` was misclassified as an unresolvable positional splat --
        `**kwargs` doesn't affect positional-arg counting at all."""
        src = "InstructionLabel(instruction, **kwargs)\n"
        result = migrate_instruction_labels(src)
        assert not result.changed
        assert result.manual_review == []

    def test_legacy_positional_with_trailing_double_star_is_rewritten(self):
        """A legacy positional call combined with a `**kwargs` unpack is
        still rewritten -- the unpack is carried through verbatim, last,
        since its contents can't be known statically."""
        src = (
            'InstructionLabel("Increment", "L0", (), {"increment_by": 2}, '
            "**extra)\n"
        )
        result = migrate_instruction_labels(src)
        assert (
            result.source
            == 'InstructionLabel("Increment", increment_by=2, patch_label="L0", **extra)\n'
        )
        assert result.manual_review == []

    def test_idempotent(self):
        src = 'InstructionLabel("Increment", "L0", (), {"increment_by": 2})\n'
        once = migrate_instruction_labels(src)
        twice = migrate_instruction_labels(once.source)
        assert twice.source == once.source
        assert not twice.changed

    def test_matches_golden_fixture(self):
        before = FIXTURES.joinpath("labels_before.py").read_text(encoding="utf-8")
        after = FIXTURES.joinpath("labels_after.py").read_text(encoding="utf-8")
        result = migrate_instruction_labels(before)
        assert result.source == after
        assert len(result.manual_review) == 2


class TestDetectFlaggedPatterns:
    def test_matches_golden_fixture(self):
        src = FIXTURES.joinpath("flags_before.py").read_text(encoding="utf-8")
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
        building functions, never touched by this v1.2 rename."""
        src = "circuit_inst = build_circuit_instruction(include_idles=True)\n"
        assert detect_flagged_patterns(src) == []


class TestMigrateSource:
    def test_matches_golden_fixture_labels(self):
        before = FIXTURES.joinpath("labels_before.py").read_text(encoding="utf-8")
        after = FIXTURES.joinpath("labels_after.py").read_text(encoding="utf-8")
        result = migrate_source(before)
        assert result.source == after

    def test_matches_golden_fixture_renames(self):
        before = FIXTURES.joinpath("renames_before.py").read_text(encoding="utf-8")
        after = FIXTURES.joinpath("renames_after.py").read_text(encoding="utf-8")
        result = migrate_source(before)
        assert result.source == after

    def test_idempotent_on_already_migrated_source(self):
        after = FIXTURES.joinpath("labels_after.py").read_text(encoding="utf-8")
        result = migrate_source(after)
        assert result.source == after
        assert not result.changed


class TestMigrateNotebookSource:
    def test_matches_golden_fixture(self):
        before = FIXTURES.joinpath("notebook_before.md").read_text(encoding="utf-8")
        after = FIXTURES.joinpath("notebook_after.md").read_text(encoding="utf-8")
        result = migrate_notebook_source(before)
        assert result.source == after

    def test_note_fence_is_never_touched(self):
        before = FIXTURES.joinpath("notebook_before.md").read_text(encoding="utf-8")
        result = migrate_notebook_source(before)
        assert (
            'mention of `PatchDict` or `InstructionLabel("Name", "L0", (), {})`'
            in result.source
        )

    def test_field_line_is_passed_through_and_not_treated_as_code(self):
        """`docs/notebooks/workflow.md` has a real `:tags: [...]` field
        line; this used to raise a ParserSyntaxError instead of migrating
        the cell."""
        before = FIXTURES.joinpath("notebook_before.md").read_text(encoding="utf-8")
        result = migrate_notebook_source(before)
        assert ":tags: [scroll-output]" in result.source

    def test_cross_cell_import_context_is_a_known_limitation(self):
        """Each cell is migrated independently, with no memory of an
        earlier cell's imports (unlike a real notebook kernel) -- so a
        bare `PatchDict()` relying on an import done in an earlier cell
        is neither rewritten nor flagged here. Not a problem for the real
        `docs/notebooks/*.md` today (already fully migrated), but locked
        in as documented, understood behavior rather than a silent gap."""
        before = FIXTURES.joinpath("notebook_before.md").read_text(encoding="utf-8")
        result = migrate_notebook_source(before)
        assert "patches2 = PatchDict()" in result.source


class TestRenamesTableCoverage:
    def test_every_rename_target_is_importable(self):
        """A regression check for the table itself: every non-`None`
        destination in RENAMES should be a real, importable `(module,
        name)` -- catching a typo before it silently produces broken
        rewrites. A target module whose *file* genuinely exists but
        fails to actually execute because an optional third-party
        backend dependency (e.g. `pygsti`, `quantumsim`) isn't installed
        is skipped rather than failed, since that's a real environment
        gap, not a broken rename entry -- some backend modules raise a
        plain `ImportError` with a custom message here rather than
        letting the underlying `ModuleNotFoundError` propagate, so the
        check can't rely on inspecting the exception's own module name."""
        import importlib
        import importlib.util

        for old_key, new_loc in RENAMES.items():
            if new_loc is None:
                continue
            new_module, new_name = new_loc
            assert importlib.util.find_spec(new_module) is not None, (
                f"{old_key} -> {new_loc}: {new_module!r} has no importable spec"
            )
            try:
                mod = importlib.import_module(new_module)
            except ImportError:
                continue  # an optional third-party backend dependency isn't installed
            assert hasattr(mod, new_name), (
                f"{old_key} -> {new_loc}: {new_name!r} not found in "
                f"{new_module!r}"
            )

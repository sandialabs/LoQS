"""Tester for loqs.tools.migrate"""

import json
from pathlib import Path

import pytest

from loqs.tools.migrate import migrate_source
from loqs.tools.migrate.flags import (
    detect_flagged_patterns,
    rewrite_cast_to_from_raw,
)
from loqs.tools.migrate.ipynb import migrate_ipynb_source
from loqs.tools.migrate.labels import migrate_instruction_labels
from loqs.tools.migrate.notebook import migrate_notebook_source
from loqs.tools.migrate.renames import RENAMES, rewrite_renames

FIXTURES = Path(__file__).parent / "migrate_fixtures"


class TestRewriteRenames:
    @pytest.mark.parametrize(
        "old_module", ["loqs.core.recordables", "loqs.core"]
    )
    def test_rewrites_patchdict_reexport_imports(self, old_module):
        """`PatchDict` is overwhelmingly imported from one of its two
        convenience re-exports rather than its own defining submodule
        (which the golden fixture covers) -- both need their own table
        entry alongside it, since a rewrite keyed only on the defining
        submodule never matches either re-export's own qualified name."""
        src = (
            f"from {old_module} import PatchDict\n\n"
            'patches = PatchDict({"L0": None})\n'
        )
        result = rewrite_renames(src)
        assert result.changed
        assert f"from {old_module} import PatchLayout" in result.source
        assert "PatchDict" not in result.source

    def test_rewrites_attribute_access_reference(self):
        result = rewrite_renames(
            "import loqs.core\nx = loqs.core.PatchDict()\n"
        )
        assert result.changed
        assert result.source == (
            "import loqs.core\nx = loqs.core.PatchLayout()\n"
        )

    def test_deleted_name_flagged_not_guessed(self):
        src = FIXTURES.joinpath("deleted_name_before.py").read_text(
            encoding="utf-8"
        )
        result = rewrite_renames(src)
        assert result.source == src  # never rewritten, only flagged
        lines = {item.line for item in result.manual_review}
        assert lines == {2, 5}  # the import line and the usage line

    def test_matches_golden_fixture(self):
        before = FIXTURES.joinpath("renames_before.py").read_text(
            encoding="utf-8"
        )
        after = FIXTURES.joinpath("renames_after.py").read_text(
            encoding="utf-8"
        )
        result = rewrite_renames(before)
        assert result.source == after
        # RepTuple's import (line 3) and usage (line 6) are left untouched
        # but flagged, not rewritten to OperationRep (an abstract class).
        assert {item.line for item in result.manual_review} == {3, 6}

    def test_stimdictnoisemodel_is_flagged_not_blindly_rewritten(self):
        """`STIMDictNoiseModel`'s decode-time redirect target
        (`DictNoiseModel`) takes `(gate_dict, inst_dict)` as two separate
        positional arguments where `STIMDictNoiseModel` took one
        `(gate_dict, inst_dict)` tuple -- confirms a blind rewrite (which
        would silently pass the whole tuple as `gate_dict` alone) doesn't
        happen."""
        src = (
            "from loqs.backends.model.stimdictmodel import STIMDictNoiseModel\n\n"
            "model = STIMDictNoiseModel((gate_dict, inst_dict))\n"
        )
        result = rewrite_renames(src)
        assert result.source == src  # never rewritten, only flagged
        lines = {item.line for item in result.manual_review}
        assert lines == {1, 3}  # the import line and the usage line


class TestMigrateInstructionLabels:
    def test_bare_tuple_with_non_literal_kwargs_is_spliced_via_double_star(
        self,
    ):
        src = '("Increment", "L0", (), extra_kwargs)\n'
        result = migrate_instruction_labels(src)
        assert result.changed
        assert not result.manual_review
        assert result.source == (
            '{"instruction": "Increment", **extra_kwargs, "patch_label": "L0"}\n'
        )

    def test_none_literal_inst_args_and_kwargs_treated_as_empty(self):
        """A real shape found in QuantumProgram_v1.json.gz's frozen
        source: `InstructionLabel(name, patch_label, None, {...})` uses a
        literal `None` for inst_args, not an empty tuple `()` --
        `_remap_legacy_positional_args`'s own runtime handling
        (`tuple(inst_args or ())`) treats both the same way."""
        src = 'InstructionLabel("Increment", "L0", None, None)\n'
        result = migrate_instruction_labels(src)
        assert (
            result.source
            == 'InstructionLabel("Increment", patch_label="L0")\n'
        )

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

    def test_colliding_patch_label_kwarg_is_flagged(self):
        """Pre-1.2, a global instruction (positional patch_label is None)
        could still carry an unrelated `inst_kwargs["patch_label"]` apply_fn
        kwarg -- the modern single `patch_label` key can't represent both,
        so this must be flagged rather than silently resolved as per-patch."""
        src = 'InstructionLabel("Init Something", None, (), {"patch_label": "L0"})\n'
        result = migrate_instruction_labels(src)
        assert not result.changed
        assert result.manual_review[0].kind == "patch_label_kwarg"

    def test_rename_patch_label_renames_the_key_and_still_flags(self):
        """`rename_patch_label` fixes the collision by renaming the
        colliding inst_kwargs key, leaving the real positional
        patch_label (resolution scope) untouched, but a manual-review
        item is still emitted -- the corresponding Instruction's apply_fn
        parameter still needs the same rename by hand."""
        src = (
            'InstructionLabel("Increment", "L0", (), {"patch_label": "L1"})\n'
        )
        result = migrate_instruction_labels(
            src, rename_patch_label="new_patch_label"
        )
        assert result.changed
        assert (
            result.source
            == 'InstructionLabel("Increment", new_patch_label="L1", patch_label="L0")\n'
        )
        assert len(result.manual_review) == 1
        assert result.manual_review[0].kind == "patch_label_kwarg"

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

    def test_matches_golden_fixture(self):
        before = FIXTURES.joinpath("labels_before.py").read_text(
            encoding="utf-8"
        )
        after = FIXTURES.joinpath("labels_after.py").read_text(
            encoding="utf-8"
        )
        result = migrate_instruction_labels(before)
        assert result.source == after
        assert len(result.manual_review) == 1


class TestDetectFlaggedPatterns:
    def test_matches_golden_fixture(self):
        src = FIXTURES.joinpath("flags_before.py").read_text(encoding="utf-8")
        items = detect_flagged_patterns(src)
        assert [item.line for item in items] == [3, 4, 4, 5]

    @pytest.mark.parametrize(
        "src",
        [
            # pygsti's own Circuit.cast / Evotype.cast share the removed
            # LoQS API's method name
            "return _Circuit.cast(obj)\n",
            "stack = some_stack.cast(InstructionStack)\n",
            # several current, unrelated functions take include_idles
            "circuit_inst = build_circuit_instruction(include_idles=True)\n",
        ],
        ids=[
            "third_party_cast",
            "class_as_argument",
            "unrelated_include_idles",
        ],
    )
    def test_no_false_positive(self, src):
        assert detect_flagged_patterns(src) == []


class TestRewriteCastToFromRaw:
    @pytest.mark.parametrize("cls", ["InstructionLabel", "SyndromeLabel"])
    def test_rewrites_cast_to_from_raw(self, cls):
        result = rewrite_cast_to_from_raw(f"label = {cls}.cast(x)\n")
        assert result.changed
        assert result.source == f"label = {cls}.from_raw(x)\n"
        assert result.manual_review == []
        assert len(result.rewrites) == 1

    def test_no_false_positive_on_unrelated_receiver(self):
        src = "x = some_label.cast(InstructionLabel)\n"
        result = rewrite_cast_to_from_raw(src)
        assert not result.changed
        assert result.source == src


class TestMigrateSource:
    def test_matches_golden_fixture_labels(self):
        """`migrate_source`'s own output, unlike `migrate_instruction_labels`
        called directly, also has a `# LOQS-MIGRATE` comment inserted
        above each remaining flagged line (see `labels_after_annotated.py`
        vs. the lower-level `labels_after.py`), since a confident rewrite
        happened elsewhere in the same file."""
        before = FIXTURES.joinpath("labels_before.py").read_text(
            encoding="utf-8"
        )
        after = FIXTURES.joinpath("labels_after_annotated.py").read_text(
            encoding="utf-8"
        )
        result = migrate_source(before)
        assert result.source == after

    def test_matches_golden_fixture_renames(self):
        before = FIXTURES.joinpath("renames_before.py").read_text(
            encoding="utf-8"
        )
        after = FIXTURES.joinpath("renames_after_annotated.py").read_text(
            encoding="utf-8"
        )
        result = migrate_source(before)
        assert result.source == after

    def test_idempotent_on_already_migrated_source(self):
        after = FIXTURES.joinpath("labels_after.py").read_text(
            encoding="utf-8"
        )
        result = migrate_source(after)
        assert result.source == after
        assert not result.changed

    @pytest.mark.parametrize(
        "src, expected",
        [
            ('instrument_name = "Iz"\n', 'instrument_name = "Imrz"\n'),
            ("instrument_name = 'Iz'\n", "instrument_name = 'Imrz'\n"),
        ],
        ids=["double_quote", "single_quote"],
    )
    def test_rename_iz_rewrites_confidently(self, src, expected):
        result = migrate_source(src, rename_iz=True)
        assert result.changed
        assert result.source == expected
        assert not result.manual_review

    def test_iz_still_flagged_when_rename_iz_is_false(self):
        src = 'instrument_name = "Iz"\n'
        result = migrate_source(src)
        assert not result.changed
        assert any(item.kind == "iz" for item in result.manual_review)

    def test_manual_review_lines_stay_accurate_after_annotation(self):
        """Every reported `manual_review.line` must point at the exact
        line the message is actually about in the *final* `.source` --
        both the earlier rewrite that collapses a multi-line call onto
        one line, and this pass's own inline `# LOQS-MIGRATE` comments,
        push later lines down and have to be accounted for."""
        before = FIXTURES.joinpath("labels_before.py").read_text(
            encoding="utf-8"
        )
        result = migrate_source(before)
        lines = result.source.splitlines()
        for item in result.manual_review:
            assert 1 <= item.line <= len(lines)
            assert "InstructionLabel" in lines[item.line - 1]

    def test_cast_rewrite_coexists_with_remaining_flag(self):
        """A confident `.cast()` -> `.from_raw()` rewrite and a still-flagged
        `.cast()` call (a class with no `.from_raw()`) in the same file are
        both reported correctly, with the flag's line number accurate in
        the final (annotated) source."""
        src = (
            "label = InstructionLabel.cast(x)\n"
            "stack = InstructionStack.cast(y)\n"
        )
        result = migrate_source(src)
        assert result.changed
        assert "InstructionLabel.from_raw(x)" in result.source
        assert len(result.manual_review) == 1
        lines = result.source.splitlines()
        item = result.manual_review[0]
        assert "InstructionStack.cast(y)" in lines[item.line - 1]


class TestMigrateNotebookSource:
    def test_matches_golden_fixture(self):
        before = FIXTURES.joinpath("notebook_before.md").read_text(
            encoding="utf-8"
        )
        after = FIXTURES.joinpath("notebook_after.md").read_text(
            encoding="utf-8"
        )
        result = migrate_notebook_source(before)
        assert result.source == after


class TestMigrateIpynbSource:
    def test_matches_golden_fixture(self):
        before = FIXTURES.joinpath("notebook_before.ipynb").read_text(
            encoding="utf-8"
        )
        after = FIXTURES.joinpath("notebook_after.ipynb").read_text(
            encoding="utf-8"
        )
        result = migrate_ipynb_source(before)
        assert result.source == after
        assert result.changed

    def test_idempotent_on_already_migrated_source(self):
        after = FIXTURES.joinpath("notebook_after.ipynb").read_text(
            encoding="utf-8"
        )
        result = migrate_ipynb_source(after)
        assert result.source == after
        assert not result.changed

    def test_manual_review_items_are_labeled_by_cell(self):
        before = FIXTURES.joinpath("notebook_before.ipynb").read_text(
            encoding="utf-8"
        )
        result = migrate_ipynb_source(before)
        assert {item.cell for item in result.manual_review} == {2, 4}

    def test_flagged_cell_is_annotated_even_without_its_own_rewrite(self):
        """A cell with nothing confidently rewritten still gets its own
        `# LOQS-MIGRATE` comment once *some* cell in the notebook needs
        rewriting -- the whole file is being modified either way, so
        every remaining flagged spot should be documented in the file
        itself, not just the one cell that happened to trigger a rewrite."""
        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        "from loqs.backends.reps import RepTuple\n",
                        "rep_class = RepTuple\n",
                    ],
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        "from loqs.core.recordables.patchdict import PatchDict\n",
                        'patches = PatchDict({"L0": None})\n',
                    ],
                },
            ]
        }
        result = migrate_ipynb_source(json.dumps(notebook))
        assert result.changed
        cells = json.loads(result.source)["cells"]
        first_cell_source = "".join(cells[0]["source"])
        assert "# LOQS-MIGRATE" in first_cell_source
        assert len(result.manual_review) == 2
        for item in result.manual_review:
            assert item.cell == 1

    @pytest.mark.parametrize(
        "magic",
        ["%matplotlib inline", "%%time"],
        ids=["line_magic", "cell_magic"],
    )
    def test_magic_line_does_not_block_the_cell(self, magic):
        """A `%matplotlib inline` line or a `%%time` cell-magic first line
        isn't valid Python on its own, but the real code in the same
        cell should still migrate rather than the whole cell being left
        untouched."""
        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        f"{magic}\n",
                        "from loqs.core.recordables.patchdict import PatchDict\n",
                        'patches = PatchDict({"L0": None})\n',
                    ],
                }
            ]
        }
        result = migrate_ipynb_source(json.dumps(notebook))
        assert result.changed
        cell_source = "".join(json.loads(result.source)["cells"][0]["source"])
        assert magic in cell_source
        assert "PatchLayout" in cell_source
        assert "PatchDict" not in cell_source

    def test_wholly_non_python_cell_magic_is_flagged_not_mangled(self):
        """A `%%bash` (or `%%html`/`%%javascript`) cell's body isn't
        Python at all -- stripping every line down to something
        parseable would eventually succeed on an all-`pass` module, but
        the cell should be left completely untouched and flagged instead
        of silently reduced to that."""
        bash_script = '%%bash\necho hello\nfor f in *.txt; do cat "$f"; done\n'
        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": bash_script.splitlines(keepends=True),
                }
            ]
        }
        result = migrate_ipynb_source(json.dumps(notebook))
        assert not result.changed
        cell_source = "".join(json.loads(result.source)["cells"][0]["source"])
        assert cell_source == bash_script
        assert any(item.cell == 1 for item in result.manual_review)


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
            assert (
                importlib.util.find_spec(new_module) is not None
            ), f"{old_key} -> {new_loc}: {new_module!r} has no importable spec"
            try:
                mod = importlib.import_module(new_module)
            except ImportError:
                continue  # an optional third-party backend dependency isn't installed
            assert hasattr(mod, new_name), (
                f"{old_key} -> {new_loc}: {new_name!r} not found in "
                f"{new_module!r}"
            )

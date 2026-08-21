#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.2                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""`loqs-migrate` console script: a thin argument-parsing wrapper around
[](api:loqs.tools.migrate)'s library functions -- all the real detect/
resolve/rewrite logic lives there, reusable without going through a
subprocess at all.

```
loqs-migrate <path> [--dry-run] [--no-backup] [--rename_Iz] [--rename_patch_label NAME]
```

Rewriting in place is the default action; `--dry-run` instead only
reports what would happen, never touching a file -- useful as a CI/
pre-flight gate. `<path>` may be a single file or a directory (walked
recursively for `.py`/`.ipynb`/`.md` files). A `.py` file is migrated
with [](api:migrate_source); a `.md` file is treated as a MyST Markdown
notebook and migrated with [](api:migrate_notebook_source); a `.ipynb`
file is migrated cell-by-cell with [](api:migrate_ipynb_source). There
is deliberately no `loqs-migrate data <file>` option: migrating an
already-serialized file is free via [](api:Serializable)'s own decode
compatibility and an ordinary `program.write(path)` round-trip, needing
no bespoke tool.

`--rename_Iz` and `--rename_patch_label NAME` opt into two rewrites that
are otherwise only flagged (see [](api:migrate_source)'s own docstring
for why each defaults to off). Whenever a run finds something either
flag would have addressed, a hint suggesting the exact follow-up
invocation is printed after the summary line.

Every run ends with a one-line summary (files scanned, how many were
rewritten or would be, how many were flagged for manual review) even
when there's nothing to report -- otherwise a scan that finds nothing to
do is indistinguishable from one that silently failed to look at the
right files.

Rewriting a file backs it up to a sibling `<name>.bak` first, unless
`--no-backup` is given; `--dry-run` never writes anything, so no backup
is made either. An existing `.bak` from a previous run is silently
overwritten, since it's meant as a one-shot undo for the rewrite about to
happen, not a growing history.

FUTURE WORK: wiring `loqs-migrate --dry-run` into LoQS's own CI as a gate
against new legacy patterns creeping back in would be low-cost and high-
value once real-world use has proven the detection half solid, but isn't
done as part of this tool's initial version.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loqs.tools.migrate import MigrationResult, migrate_source
from loqs.tools.migrate.ipynb import migrate_ipynb_source
from loqs.tools.migrate.notebook import migrate_notebook_source
from loqs.tools.migrate.report import format_manual_review_block

_TARGET_SUFFIXES = (".py", ".ipynb", ".md")


def _iter_target_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        p
        for p in path.rglob("*")
        if p.suffix in _TARGET_SUFFIXES and p.is_file()
    )


def _migrate_file(
    path: Path, *, rename_iz: bool = False, rename_patch_label: str | None = None
) -> MigrationResult:
    source = path.read_text(encoding="utf-8")
    kwargs = {"rename_iz": rename_iz, "rename_patch_label": rename_patch_label}
    if path.suffix == ".md":
        return migrate_notebook_source(source, **kwargs)
    if path.suffix == ".ipynb":
        return migrate_ipynb_source(source, **kwargs)
    return migrate_source(source, **kwargs)


def _backup_path(path: Path) -> Path:
    return path.with_name(path.name + ".bak")


def _format_followup_suggestion(
    paths: list[Path], *, iz_found: bool, patch_label_found: bool, rewrote: bool
) -> str | None:
    """A suggested follow-up invocation using whichever of
    `--rename_Iz`/`--rename_patch_label` would address something this run
    only flagged, or `None` if neither applies.

    `--no-backup` is only appended when `rewrote` is true -- this run
    already wrote (and backed up) at least one file, so suggesting
    another backup pass is redundant; when nothing was actually written
    yet (a `--dry-run`, or nothing else needed a confident rewrite), the
    suggested command keeps the default backup protection for what would
    be its first real write.
    """
    if not (iz_found or patch_label_found):
        return None
    flags = []
    if iz_found:
        flags.append("--rename_Iz")
    if patch_label_found:
        flags.append("--rename_patch_label <new_patch_label>")
    if rewrote:
        flags.append("--no-backup")
    targets = " ".join(str(p) for p in paths)
    return (
        "Hint: some manual-review items above can be auto-migrated. Consider running:\n"
        f"  loqs-migrate {targets} {' '.join(flags)}"
    )


def _run(
    paths: list[Path],
    *,
    write: bool,
    backup: bool = True,
    rename_iz: bool = False,
    rename_patch_label: str | None = None,
) -> int:
    """The CLI's actual implementation, shared by both write and
    `--dry-run` modes. Returns a process exit code: 0 if nothing needed
    attention, 1 if anything was flagged for manual review (whether or
    not anything else was also rewritten), 2 on a file-level error."""
    any_manual_review = False
    any_error = False
    changed_files = 0
    flagged_files = 0
    iz_found = False
    patch_label_found = False

    files = [f for path in paths for f in _iter_target_files(path)]
    if not files:
        print("No .py/.ipynb/.md files found.", file=sys.stderr)
        return 2

    for file in files:
        try:
            result = _migrate_file(
                file, rename_iz=rename_iz, rename_patch_label=rename_patch_label
            )
        except Exception as exc:  # noqa: BLE001 -- report and keep going
            print(f"{file}: error: {exc}", file=sys.stderr)
            any_error = True
            continue

        if result.changed:
            changed_files += 1
            action = "would rewrite" if not write else "rewrote"
            print(f"{file}: {action}")
            if write:
                try:
                    if backup:
                        backup_file = _backup_path(file)
                        backup_file.write_bytes(file.read_bytes())
                        print(f"{file}: backed up to {backup_file}")
                    file.write_text(result.source, encoding="utf-8")
                except OSError as exc:
                    print(f"{file}: error: {exc}", file=sys.stderr)
                    any_error = True
                    continue

        if result.manual_review:
            flagged_files += 1
            any_manual_review = True
            print(format_manual_review_block(str(file), result.manual_review))
            iz_found = iz_found or any(item.kind == "iz" for item in result.manual_review)
            # Still flagged (as a reminder to update the matching
            # Instruction's apply_fn) even once --rename_patch_label is
            # already in use -- only suggest the flag while it hasn't
            # been supplied yet, or the hint would just loop.
            if rename_patch_label is None:
                patch_label_found = patch_label_found or any(
                    item.kind == "patch_label_kwarg" for item in result.manual_review
                )

    noun = "file" if len(files) == 1 else "files"
    rewrite_verb = "rewritten" if write else "would be rewritten"
    print(
        f"{len(files)} {noun} scanned: {changed_files} {rewrite_verb}, "
        f"{flagged_files} flagged for manual review."
    )

    suggestion = _format_followup_suggestion(
        paths,
        iz_found=iz_found,
        patch_label_found=patch_label_found,
        rewrote=write and changed_files > 0,
    )
    if suggestion:
        print(suggestion)

    if any_error:
        return 2
    if any_manual_review:
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loqs-migrate",
        description=(
            "Rewrite .py/.ipynb/MyST Markdown source still using pre-1.2 LoQS APIs."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="File(s) or directory/directories to scan.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Report what would be rewritten without touching any file. "
            "Useful as a CI/pre-flight gate."
        ),
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help=(
            "Don't back up a rewritten file to <name>.bak first. "
            "Backing up is the default."
        ),
    )
    parser.add_argument(
        "--rename_Iz",
        dest="rename_iz",
        action="store_true",
        help=(
            'Confidently rewrite a bare "Iz"/\'Iz\' string literal to '
            '"Imrz"/\'Imrz\' instead of only flagging it. Off by default, '
            "since this is a blind string match that could collide with "
            "unrelated text."
        ),
    )
    parser.add_argument(
        "--rename_patch_label",
        dest="rename_patch_label",
        metavar="NEW_NAME",
        default=None,
        help=(
            "Rewrite a legacy InstructionLabel's colliding "
            "inst_kwargs['patch_label'] key to NEW_NAME instead of only "
            "flagging it. Still flagged either way, as a reminder that "
            "the corresponding Instruction's apply_fn parameter needs "
            "the same rename by hand."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return _run(
        args.paths,
        write=not args.dry_run,
        backup=not args.no_backup,
        rename_iz=args.rename_iz,
        rename_patch_label=args.rename_patch_label,
    )


if __name__ == "__main__":
    sys.exit(main())

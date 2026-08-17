#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.1.1                                                                           #
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
loqs-migrate source <path> [--config <mapping-file>] [--dry-run]
loqs-migrate check <path>       # detect-only, no rewrite; for CI/pre-flight
```

`<path>` may be a single file or a directory (walked recursively for
`.py`/`.md` files). A `.py` file is migrated with
[](api:migrate_source); a `.md` file is treated as a MyST Markdown
notebook and migrated with [](api:migrate_notebook_source) instead. There
is deliberately no `loqs-migrate data <file>` subcommand: migrating an
already-serialized file is free via [](api:Serializable)'s own decode
compatibility and an ordinary `program.write(path)` round-trip, needing
no bespoke tool.

FUTURE WORK: wiring `loqs-migrate check` into LoQS's own CI as a gate
against new legacy patterns creeping back in would be low-cost and high-
value once real-world use has proven the detection half solid, but isn't
done as part of this tool's initial version.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loqs.tools.migrate import MigrationResult, migrate_source
from loqs.tools.migrate.config import MigrationConfig
from loqs.tools.migrate.notebook import migrate_notebook_source


def _iter_target_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        p
        for p in path.rglob("*")
        if p.suffix in (".py", ".md") and p.is_file()
    )


def _migrate_file(path: Path, config: MigrationConfig) -> MigrationResult:
    source = path.read_text()
    instructions = config.instructions_for(path)
    if path.suffix == ".md":
        return migrate_notebook_source(source, instructions=instructions)
    return migrate_source(source, instructions=instructions)


def _run(paths: list[Path], config: MigrationConfig, *, write: bool) -> int:
    """Shared implementation for both subcommands. Returns a process exit
    code: 0 if nothing needed attention, 1 if anything was flagged for
    manual review (whether or not anything else was also rewritten), 2 on
    a file-level error."""
    any_manual_review = False
    any_error = False

    files = [f for path in paths for f in _iter_target_files(path)]
    if not files:
        print("No .py/.md files found.", file=sys.stderr)
        return 2

    for file in files:
        try:
            result = _migrate_file(file, config)
        except Exception as exc:  # noqa: BLE001 -- report and keep going
            print(f"{file}: error: {exc}", file=sys.stderr)
            any_error = True
            continue

        if result.changed:
            action = "would rewrite" if not write else "rewrote"
            print(f"{file}: {action}")
            if write:
                file.write_text(result.source)

        for item in result.manual_review:
            any_manual_review = True
            print(f"{file}:{item.line}: {item.message}")

    if any_error:
        return 2
    if any_manual_review:
        return 1
    return 0


def _cmd_source(args: argparse.Namespace) -> int:
    config = (
        MigrationConfig.from_json(args.config)
        if args.config
        else MigrationConfig()
    )
    return _run(args.paths, config, write=not args.dry_run)


def _cmd_check(args: argparse.Namespace) -> int:
    config = (
        MigrationConfig.from_json(args.config)
        if args.config
        else MigrationConfig()
    )
    return _run(args.paths, config, write=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loqs-migrate",
        description=(
            "Rewrite .py/MyST Markdown source still using pre-1.2 LoQS APIs."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="File(s) or directory/directories to scan.",
    )
    common.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Path to a JSON file mapping source file paths to "
            "'module:function' instruction-registry references (see "
            "loqs.tools.migrate.config.MigrationConfig.from_json)."
        ),
    )

    source_parser = subparsers.add_parser(
        "source", parents=[common], help="Rewrite files in place."
    )
    source_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be rewritten without touching any file.",
    )
    source_parser.set_defaults(func=_cmd_source)

    check_parser = subparsers.add_parser(
        "check",
        parents=[common],
        help="Detect-only; never rewrites. Useful as a CI/pre-flight gate.",
    )
    check_parser.set_defaults(func=_cmd_check)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

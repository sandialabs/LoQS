#!/usr/bin/env python3
"""
Convert LoQS tutorial notebooks between the tracked .md (Jupytext MyST) source,
which uses `[](api:SomeTarget)` shorthand links, and a working .ipynb with real,
clickable absolute-URL links. Supports conversion in both directions.

Requires the `docs` extra (`pip install ".[docs]"`), which includes jupytext,
nbformat, and the API inventory build tools.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import nbformat

from docs_scripts.api_inventory import (
    ApiInventory,
    rewrite_api_links,
    unrewrite_api_links,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = REPO_ROOT / "docs" / "notebooks"
INVENTORY_PATH = REPO_ROOT / "docs" / "_api_inventory.json"
API_REF_CONFIG = REPO_ROOT / "mkdocs-api-ref.yml"
DEFAULT_BASE_URL = "https://loqs.readthedocs.io/en/latest/reference"


def _ensure_api_inventory() -> ApiInventory:
    """
    Load or build the API inventory. Caches on disk across invocations.
    """
    if INVENTORY_PATH.exists():
        return ApiInventory.load(INVENTORY_PATH)

    print("Building API inventory, this may take ~30-60s...")
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "mkdocs",
                "build",
                "-f",
                str(API_REF_CONFIG),
                "-d",
                tmpdir,
            ],
            cwd=REPO_ROOT,
            check=True,
        )

    if not INVENTORY_PATH.exists():
        raise RuntimeError(
            f"API inventory build failed: {INVENTORY_PATH} not found"
        )

    return ApiInventory.load(INVENTORY_PATH)


def _run_to_ipynb(paths: list[Path], base_url: str) -> int:
    """
    Convert .md notebooks to .ipynb with resolved api: links.
    """
    inv = _ensure_api_inventory()

    total_changed = 0
    for path in paths:
        subprocess.run(
            [sys.executable, "-m", "jupytext", "--to", "ipynb", str(path)],
            cwd=REPO_ROOT,
            check=True,
        )

        ipynb_path = path.with_suffix(".ipynb")
        nb = nbformat.read(str(ipynb_path), as_version=4)

        changed_count = 0
        for cell in nb.cells:
            if cell.cell_type == "markdown":
                original = cell.source
                cell.source = rewrite_api_links(
                    original, inv, url_prefix=base_url, page_src=str(path)
                )
                if cell.source != original:
                    changed_count += 1

        if changed_count > 0:
            nbformat.write(nb, str(ipynb_path))

        print(f"{path.stem}: {changed_count} cell(s) updated")
        total_changed += changed_count

    print(f"Total: {total_changed} cell(s) updated")
    return 0


def _run_to_md(paths: list[Path], base_url: str) -> int:
    """
    Convert .ipynb notebooks back to .md with api: shorthand restored.
    """
    inv = _ensure_api_inventory()

    total_changed = 0
    for path in paths:
        md_path = path.with_suffix(".md")

        subprocess.run(
            [
                sys.executable,
                "-m",
                "jupytext",
                "--to",
                "myst",
                str(path),
                "-o",
                str(md_path),
            ],
            cwd=REPO_ROOT,
            check=True,
        )

        original_text = md_path.read_text(encoding="utf-8")
        new_text = unrewrite_api_links(original_text, inv, url_prefix=base_url)

        if new_text != original_text:
            md_path.write_text(new_text, encoding="utf-8")
            print(f"{path.stem}: api: shorthand restored")
            total_changed += 1
        else:
            print(f"{path.stem}: no changes needed")

    print(f"Total: {total_changed} notebook(s) updated")
    return 0


def main(argv: list[str] | None = None) -> int:
    """
    Convert notebooks between .md and .ipynb with api: link handling.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    to_ipynb = subparsers.add_parser(
        "to-ipynb",
        help="Convert .md to .ipynb with resolved api: links",
    )
    to_ipynb.add_argument(
        "notebooks",
        nargs="*",
        type=Path,
        help="Notebook paths to convert (.md files)",
    )
    to_ipynb.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Base URL for api: link resolution (default: {DEFAULT_BASE_URL})",
    )

    to_md = subparsers.add_parser(
        "to-md",
        help="Convert .ipynb back to .md with api: shorthand",
    )
    to_md.add_argument(
        "notebooks",
        nargs="*",
        type=Path,
        help="Notebook paths to convert (.ipynb files)",
    )
    to_md.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Base URL for api: link recovery (default: {DEFAULT_BASE_URL})",
    )

    args = parser.parse_args(argv)

    if args.command == "to-ipynb":
        if not args.notebooks:
            args.notebooks = sorted(NOTEBOOKS_DIR.glob("*.md"))

        # Validate targets
        for path in args.notebooks:
            if not path.exists():
                print(f"Error: {path} does not exist", file=sys.stderr)
                return 2
            if path.suffix != ".md":
                print(
                    f"Error: {path} is not a .md file (use 'to-md' for .ipynb)",
                    file=sys.stderr,
                )
                return 2

        if not args.notebooks:
            print("No .md notebooks found", file=sys.stderr)
            return 2

        return _run_to_ipynb(args.notebooks, args.base_url)

    elif args.command == "to-md":
        if not args.notebooks:
            args.notebooks = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))

        # Validate targets
        for path in args.notebooks:
            if not path.exists():
                print(f"Error: {path} does not exist", file=sys.stderr)
                return 2
            if path.suffix != ".ipynb":
                print(
                    f"Error: {path} is not a .ipynb file (use 'to-ipynb' for .md)",
                    file=sys.stderr,
                )
                return 2

        if not args.notebooks:
            print("No .ipynb notebooks found", file=sys.stderr)
            return 2

        return _run_to_md(args.notebooks, args.base_url)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

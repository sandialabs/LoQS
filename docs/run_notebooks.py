#!/usr/bin/env python3
"""
Execute every tutorial notebook under docs/notebooks/*.md and report any
that fail. This is the same check the "notebook-tests" CI job in
.github/workflows/loqs.yml runs, so it doubles as a way to reproduce a CI
notebook failure locally.

Uses jupytext and nbclient directly, rather than the `jupytext --execute`
CLI, so a failing notebook's partial output (every cell up through the one
that raised, with its traceback) can still be written out for inspection --
the CLI discards the whole notebook on any unexpected error. A cell tagged
`raises-exception` (see docs/notebooks/timedepmodel.md) is treated as an
expected failure by nbclient and does not stop execution. Each notebook
runs in its own throwaway scratch working directory, since some tutorials
(e.g. workflow.md's checkpointing example) write relative-path scratch
files of their own that would otherwise collide across notebooks sharing
one directory.

Requires the `docs` extra (`pip install ".[docs]"` or `".[all]"`), which
includes jupytext, nbclient, and ipykernel.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
import time
from pathlib import Path

import jupytext
import nbformat
from nbclient import NotebookClient

NOTEBOOKS_DIR = Path(__file__).resolve().parent / "notebooks"


def discover_notebooks() -> list[Path]:
    """Return every notebook under docs/notebooks/, sorted by name."""
    return sorted(NOTEBOOKS_DIR.glob("*.md"))


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as a compact "1h02m03s"-style string."""
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def execute_notebook(notebook: Path, keep_dir: Path) -> Exception | None:
    """
    Execute *notebook* in an isolated scratch working directory, discarding that
    directory (including any files the notebook itself wrote) once done. On any
    failure -- a cell raising, or a kernel-level problem like a bad kernelspec
    or a crash -- write the notebook's outputs so far (including a failing
    cell's traceback, if it got that far) into *keep_dir*, for inspection.
    Return the exception on failure, or None on success.

    A Ctrl-C is handled specially, not reported as a notebook failure:
    nbclient's own SIGINT handling kills the kernel and cancels the in-flight
    cell wait, which surfaces here as `asyncio.CancelledError` (the client
    has no per-cell timeout); a plain `KeyboardInterrupt` is handled the same
    way in case one arrives outside that window. Either way, this confirms
    the kernel is dead and re-raises as `KeyboardInterrupt`, so main()'s
    caller aborts the whole run instead of moving on to the next notebook.
    """
    nb = jupytext.read(notebook)
    with tempfile.TemporaryDirectory(prefix=f"loqs-nb-{notebook.stem}-") as scratch:
        client = NotebookClient(
            nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": scratch}}
        )
        try:
            client.execute()
            return None
        except (KeyboardInterrupt, asyncio.CancelledError) as exc:
            if client.km is not None:
                try:
                    client.km.shutdown_kernel(now=True)
                except Exception:
                    pass
            raise KeyboardInterrupt from exc
        except Exception as exc:  # noqa: BLE001 -- any failure here should still yield a diagnosable per-notebook result
            keep_dir.mkdir(parents=True, exist_ok=True)
            nbformat.write(nb, keep_dir / f"{notebook.stem}.ipynb")
            return exc


def main() -> int:
    # A failing cell's exception text is arbitrary and can contain
    # characters outside a console's default encoding (e.g. Windows's
    # legacy code-page stdout/stderr) -- reconfigure both streams to UTF-8
    # with lossy fallback so reporting a failure can never itself crash
    # with a UnicodeEncodeError and hide the real error.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "notebooks",
        nargs="*",
        help=(
            "Notebook names to run, without the .md extension "
            "(e.g. 'workflow fttests'). Defaults to every notebook "
            "under docs/notebooks/."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("notebook-output"),
        help=(
            "Directory to copy a failed notebook's executed .ipynb into, "
            "for inspection (default: ./notebook-output). Notebooks that "
            "succeed leave nothing behind here."
        ),
    )
    args = parser.parse_args()

    if args.notebooks:
        candidates = [NOTEBOOKS_DIR / f"{name}.md" for name in args.notebooks]
        missing = [c for c in candidates if not c.exists()]
        if missing:
            print(
                f"Unknown notebook(s): {', '.join(m.stem for m in missing)}",
                file=sys.stderr,
            )
            return 2
    else:
        candidates = discover_notebooks()

    failures = []
    timings: list[tuple[str, float]] = []
    start_all = time.monotonic()
    try:
        for notebook in candidates:
            print(f"--- Executing {notebook.stem} ---")
            start = time.monotonic()
            error = execute_notebook(notebook, args.output_dir)
            elapsed = time.monotonic() - start
            timings.append((notebook.stem, elapsed))
            print(f"{notebook.stem}: {format_duration(elapsed)}")
            if error is not None:
                print(f"{notebook.stem} FAILED: {error}")
                failures.append(notebook.stem)
    except KeyboardInterrupt:
        print("\nInterrupted -- aborting remaining notebooks.", file=sys.stderr)
        return 130

    print("\nTimings:")
    for name, elapsed in timings:
        print(f"  {name}: {format_duration(elapsed)}")
    print(f"Total: {format_duration(time.monotonic() - start_all)}")

    if failures:
        print(f"\nFailed notebooks: {', '.join(failures)}", file=sys.stderr)
        return 1

    print(f"\nAll {len(candidates)} notebook(s) executed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

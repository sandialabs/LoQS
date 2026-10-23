"""Validation records: digests, the record decision and the writer."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from validation.bugs import KNOWN_BUGS
from validation.cases import declared_checks
from validation.declaration import CLASSES, CodepackDeclaration

ROOT = Path(__file__).resolve().parent.parent
RECORDS_DIR = Path(__file__).resolve().parent / "records"
ACCEPTED = "passed"


def digest(path: str | Path) -> str:
    """sha256 of a file's bytes with CRLF turned into LF."""
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def file_digests(decl: CodepackDeclaration) -> dict[str, str]:
    """Digests of a declaration's files and its own declaration module."""
    paths = list(decl.files) + [f"validation/declarations/{decl.name}.py"]
    return {p: digest(ROOT / p) for p in sorted(paths)}


def harness_digests() -> dict[str, str]:
    """Digests of every `validation/*.py`, which are recorded not enforced."""
    return {
        _rel(p): digest(p) for p in sorted((ROOT / "validation").glob("*.py"))
    }


def _class_of(case_id: str) -> str:
    return case_id.split(".")[2]


def _summary(bug: str) -> str:
    return KNOWN_BUGS.get(bug, "")


def decide_record(
    decl: CodepackDeclaration,
    expected_ids: Sequence[str],
    reports: Mapping[str, str],
    start_files: Mapping[str, str],
    meta: Mapping[str, Any] | None = None,
    big_configs: Sequence[str] = (),
) -> dict[str, Any] | str:
    """The record to write, or a refusal string.

    Refuses unless every expected case reported and none failed, errored or
    was skipped; a case whose declared check now passes reports as failed.
    Also refuses when the files' digests differ from `start_files`, taken at
    session start, so a record never certifies bytes that didn't run.
    """
    now = file_digests(decl)
    changed = sorted(
        p
        for p in set(now) | set(start_files)
        if now.get(p) != start_files.get(p)
    )
    if changed:
        return f"files changed during the run: {', '.join(changed)}"
    missing = [i for i in expected_ids if i not in reports]
    if missing:
        return f"{len(missing)} expected cases did not run, e.g. {missing[0]}"
    bad = [i for i in expected_ids if reports[i] != ACCEPTED]
    if bad:
        return (
            f"{len(bad)} cases did not pass, e.g. {bad[0]}: {reports[bad[0]]}"
        )
    cases: dict[str, Any] = {}
    known: dict[str, Any] = {}
    classes: dict[str, Any] = {}
    for cid in sorted(expected_ids):
        declared = declared_checks(decl, cid)
        pairs = sorted(f"{c}:{b}" for c, ids in declared.items() for b in ids)
        cases[cid] = pairs if pairs else "pass"
        for check, ids in declared.items():
            known[f"{cid}:{check}"] = {b: _summary(b) for b in ids}
        entry = classes.setdefault(_class_of(cid), {"passed": 0, "known": 0})
        entry["known" if pairs else "passed"] += 1
    for cls in CLASSES:
        classes.setdefault(cls, "not declared")
    record: dict[str, Any] = {
        "codepack": decl.name,
        "files": now,
        "harness": harness_digests(),
        "classes": classes,
        "known_failures": known,
        "cases": cases,
        "big_configs": sorted(big_configs),
    }
    record.update(meta or {})
    return record


def run_metadata() -> dict[str, Any]:
    """Commit, machine and date fields of a record."""

    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            return out.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain")
    versions = {"python": platform.python_version()}
    for mod in ("stim", "pygsti"):
        try:
            versions[mod] = __import__(mod).__version__
        except ImportError:
            versions[mod] = "not installed"
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "machine": {
            "platform": platform.platform(),
            "cpus": os.cpu_count(),
            **versions,
        },
    }


def write_record(name: str, record: Mapping[str, Any]) -> Path:
    """Write `validation/records/<name>.json` with sorted keys."""
    RECORDS_DIR.mkdir(exist_ok=True)
    path = RECORDS_DIR / f"{name}.json"
    path.write_text(
        json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return path


def command_line(args: Sequence[str]) -> str:
    """The pytest command a record was written by."""
    return " ".join([Path(sys.executable).name, "-m", "pytest", *args])

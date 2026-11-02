"""Validation records: digests, the record decision and the writer."""

from __future__ import annotations

import ast
import datetime
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from loqs.backends import ListPhysicalCircuit
from loqs.backends.circuit.basecircuit import BasePhysicalCircuit
from loqs.core import Instruction
from loqs.core.recordables.measurementoutcomes import MeasurementOutcomes

from validation.bugs import KNOWN_BUGS
from validation.cases import declared_checks
from validation import engine
from validation.declaration import CLASSES, Backend, CodepackDeclaration

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


def _drop_docstrings(source: str) -> str:
    """Source without docstring lines, blank lines and comment-only lines."""
    tree = ast.parse(source)
    doc_lines: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            )
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
            and body[0].lineno > node.lineno
        ):
            end = body[0].end_lineno or body[0].lineno
            doc_lines.update(range(body[0].lineno, end + 1))
    kept = []
    for number, line in enumerate(source.splitlines(), start=1):
        text = line.strip()
        if number in doc_lines or not text or text.startswith("#"):
            continue
        kept.append(line)
    return "\n".join(kept)


def _function_source(fn: Any) -> tuple[str, ast.AST]:
    try:
        raw = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError) as exc:
        raise ValueError(f"no source for {fn!r}: {exc}") from exc
    return _drop_docstrings(raw), ast.parse(raw)


def source_digest(fn: Any) -> str:
    """Digest of a function's behaviour-bearing source.

    Covers the source without docstrings, comments and blank lines, every
    same-module function it names (recursively) and the value of every
    other non-callable, non-module global it names.
    """
    parts: dict[str, str] = {}
    values: dict[str, Any] = {}
    queue = [fn]
    seen = {id(fn)}
    while queue:
        cur = queue.pop(0)
        text, tree = _function_source(cur)
        parts[f"{cur.__module__}.{cur.__qualname__}"] = text
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        for name in sorted(names):
            if name not in cur.__globals__:
                continue
            obj = cur.__globals__[name]
            if inspect.ismodule(obj):
                continue
            if inspect.isfunction(obj) and obj.__module__ == cur.__module__:
                if id(obj) not in seen:
                    seen.add(id(obj))
                    queue.append(obj)
            elif not callable(obj):
                values[f"{cur.__module__}.{name}"] = canonical(obj)
    blob = json.dumps(
        {"functions": parts, "globals": values},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def _dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def canonical(value: Any, memo: dict[int, Any] | None = None) -> Any:
    """A JSON-able, run-independent form of a rendered value.

    Raises `ValueError` for an object whose `repr` holds a memory address.
    """
    memo = {} if memo is None else memo
    if isinstance(value, Instruction):
        if id(value) not in memo:
            memo[id(value)] = {
                "type": value.type,
                "name": value.name,
                "data": canonical(dict(value.data), memo),
                "priorities": canonical(value._param_priorities, memo),
                "aliases": canonical(value._param_aliases, memo),
                "errors": value.param_error_behavior,
                "apply": source_digest(value.apply_fn),
                "map": source_digest(value.map_qubits_fn),
            }
        return memo[id(value)]
    if isinstance(value, BasePhysicalCircuit):
        layers = [
            sorted(_dumps(canonical(op, memo)) for op in layer)
            for layer in ListPhysicalCircuit(value).circuit
        ]
        return {
            "circuit": layers,
            "qubits": [canonical(q, memo) for q in value.qubit_labels],
        }
    if isinstance(value, MeasurementOutcomes):
        return {"outcomes": canonical(dict(value.outcomes), memo)}
    if isinstance(value, np.ndarray):
        return {"ndarray": [str(value.dtype), value.tolist()]}
    if isinstance(value, np.generic):
        return canonical(value.item(), memo)
    if isinstance(value, dict):
        items = {
            _dumps(canonical(k, memo)): canonical(v, memo)
            for k, v in value.items()
        }
        return dict(sorted(items.items()))
    if isinstance(value, (set, frozenset)):
        return {"set": sorted(_dumps(canonical(v, memo)) for v in value)}
    if isinstance(value, (list, tuple)):
        return [canonical(v, memo) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    text = repr(value)
    if " at 0x" in text:
        raise ValueError(
            f"cannot fingerprint {text}: its repr holds an address"
        )
    return {"repr": text}


def rendering_fingerprint(decl: CodepackDeclaration) -> str:
    """SHA-256 of what a declaration's configs and ops render.

    Covers each config's code (instructions, data, apply-function source)
    and each declared op's entries, on the List backend, so a renderer
    change shows without hashing the renderer's files.
    """
    listing = Backend("list", ListPhysicalCircuit)
    body: dict[str, Any] = {}
    for config in decl.configs:
        code = decl.build_code(config, listing)
        memo: dict[int, Any] = {}
        entry: dict[str, Any] = {
            "name": code.name,
            "qubits": canonical(list(code.template_qubits), memo),
            "data_qubits": canonical(list(code.template_data_qubits), memo),
            "instructions": {
                key: canonical(inst, memo)
                for key, inst in sorted(code.instructions.items())
            },
        }
        for op in decl.ops:
            if op.configs and config.name not in op.configs:
                continue
            roles = tuple(op.roles)
            patches, _ = engine.patch_labels(code, "renamed", len(roles))
            step = engine.Op("op", op.name, roles=roles)
            seams = engine.word_seams(decl, [step])
            entries = engine.op_entries(
                decl, config, step, dict(zip(roles, patches)), code, seams
            )
            entry[f"op:{op.name}"] = canonical(list(entries))
        body[config.name] = entry
    return hashlib.sha256(_dumps(body).encode()).hexdigest()


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
    start_rendering: str | None = None,
) -> dict[str, Any] | str:
    """The record to write, or a refusal string.

    Refuses unless every expected case reported and none failed, errored or
    was skipped; a case whose declared check now passes reports as failed.
    Also refuses when the files' digests differ from `start_files`, or the
    rendering fingerprint from `start_rendering`, both taken at session
    start, so a record never certifies bytes that didn't run.
    """
    now = file_digests(decl)
    changed = sorted(
        p
        for p in set(now) | set(start_files)
        if now.get(p) != start_files.get(p)
    )
    if changed:
        return f"files changed during the run: {', '.join(changed)}"
    rendering = rendering_fingerprint(decl)
    if start_rendering is not None and rendering != start_rendering:
        return "rendering changed during the run"
    missing = [i for i in expected_ids if i not in reports]
    if missing:
        return f"{len(missing)} expected cases did not run, e.g. {missing[0]}"
    bad = [i for i in expected_ids if reports[i] != ACCEPTED]
    if bad:
        return (
            f"{len(bad)} cases did not pass, e.g. {bad[0]}: {reports[bad[0]]}"
        )
    cases: dict[str, Any] = {}
    bugs: dict[str, str] = {}
    classes: dict[str, Any] = {}
    for cid in sorted(expected_ids):
        declared = declared_checks(decl, cid)
        pairs = sorted(f"{c}:{b}" for c, ids in declared.items() for b in ids)
        cases[cid] = pairs if pairs else "pass"
        for ids in declared.values():
            bugs.update({b: _summary(b) for b in ids})
        entry = classes.setdefault(_class_of(cid), {"passed": 0, "known": 0})
        entry["known" if pairs else "passed"] += 1
    for cls in CLASSES:
        classes.setdefault(cls, "not declared")
    record: dict[str, Any] = {
        "codepack": decl.name,
        "files": now,
        "harness": harness_digests(),
        "classes": classes,
        "bugs": dict(sorted(bugs.items())),
        "rendering": rendering,
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

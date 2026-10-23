"""Stale-record check for the validation harness's codepack records."""

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RECORDS = sorted((ROOT / "validation" / "records").glob("*.json"))


@pytest.mark.parametrize("record_path", RECORDS, ids=lambda p: p.stem)
def test_records_match_files(record_path):
    """A codepack file or declaration changed after its record was written."""
    files = json.loads(record_path.read_text(encoding="utf-8"))["files"]
    stale = []
    for rel, recorded in files.items():
        data = (ROOT / rel).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(data).hexdigest() != recorded:
            stale.append(rel)
    assert stale == []


def test_every_codepack_file_has_a_record():
    """A codepack file under loqs/codepacks that no validation record covers."""
    covered = set()
    for record_path in RECORDS:
        covered |= set(
            json.loads(record_path.read_text(encoding="utf-8"))["files"]
        )
    shipped = {
        p.relative_to(ROOT).as_posix()
        for p in (ROOT / "loqs" / "codepacks").rglob("*.py")
    }
    shipped.discard("loqs/codepacks/__init__.py")
    assert sorted(shipped - covered) == []

"""Counts a run's results per class and formats the verdict table."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from validation.declaration import CLASSES

ACCEPTED = "passed"


def _class_of(case_id: str) -> str:
    return case_id.split(".")[2]


def _ft_faults(tallies: Mapping[str, Mapping[str, Any]]) -> Any:
    """Fault counts over the ft cases' tallies, or None without any."""
    largest: dict[tuple[str, str], int] = {}
    runs = discarded = uncorrected = 0
    for cid, tally in tallies.items():
        if "faults" not in tally:
            continue
        pair = (cid.split(".")[1], tally["gadget"])
        largest[pair] = max(largest.get(pair, 0), tally["faults"])
        runs += tally["faults"]
        discarded += tally["discarded"]
        uncorrected += tally["uncorrected"]
    if not largest:
        return None
    return {
        "gadgets": len(largest),
        "distinct": sum(largest.values()),
        "runs": runs,
        "discarded": discarded,
        "uncorrected": uncorrected,
    }


def class_summary(
    declared: Mapping[str, Mapping[str, Sequence[str]]],
    outcomes: Mapping[str, str],
    seconds: Mapping[str, float],
    tallies: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Per-class counts of cases, passes, known failures and failures.

    `declared` maps every case id to its declared checks. A case whose
    outcome is `ACCEPTED` is known when it declares a check, else passed;
    any other outcome, or none, is failed. The ft block adds `faults`.
    """
    out: dict[str, dict[str, Any]] = {}
    for cls in CLASSES:
        ids = [i for i in declared if _class_of(i) == cls]
        if not ids:
            continue
        block: dict[str, Any] = {
            "cases": len(ids),
            "passed": 0,
            "known": 0,
            "failed": 0,
        }
        by_bug: dict[str, int] = {}
        for cid in ids:
            checks = declared[cid]
            if outcomes.get(cid) != ACCEPTED:
                block["failed"] += 1
            else:
                block["known" if checks else "passed"] += 1
            for bug in {b for bugs in checks.values() for b in bugs}:
                by_bug[bug] = by_bug.get(bug, 0) + 1
        block["by_bug"] = dict(sorted(by_bug.items()))
        timed = [seconds[i] for i in ids if i in seconds]
        block["seconds"] = round(sum(timed), 1) if timed else None
        if cls == "ft":
            block["faults"] = _ft_faults(
                {i: tallies[i] for i in ids if i in tallies}
            )
        out[cls] = block
    return out


def _table(rows: Sequence[Sequence[str]]) -> list[str]:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    fmt = ["  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows]
    fmt.insert(1, "  ".join("-" * w for w in widths))
    return [line.rstrip() for line in fmt]


def verdict_lines(codepack: str, classes: Mapping[str, Any]) -> list[str]:
    """The verdict table of one codepack, and its known bugs and faults."""
    rows = [(codepack, "cases", "passed", "known", "failed", "seconds")]
    notes: list[str] = []
    for cls in CLASSES:
        block = classes.get(cls)
        if block is None:
            continue
        if block == "not declared":
            rows.append((cls, "not declared", "", "", "", ""))
            continue
        sec = "--" if block["seconds"] is None else f"{block['seconds']:.1f}"
        rows.append(
            (
                cls,
                str(block["cases"]),
                str(block["passed"]),
                str(block["known"]),
                str(block["failed"]),
                sec,
            )
        )
        if block["by_bug"]:
            bugs = ", ".join(f"{b} ({n})" for b, n in block["by_bug"].items())
            notes.append(f"{cls} known failures: {bugs}")
        faults = block.get("faults")
        if faults:
            end = (
                "all corrected"
                if faults["uncorrected"] == 0
                else f"{faults['uncorrected']} uncorrected"
            )
            notes.append(
                f"ft faults: {faults['gadgets']} gadgets, "
                f"{faults['distinct']} distinct single faults, "
                f"{faults['runs']} runs, {faults['discarded']} discarded, "
                f"{end}"
            )
    return _table(rows) + notes

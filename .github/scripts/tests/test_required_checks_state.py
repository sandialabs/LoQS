"""Tests for required-checks-state.sh, run against JSON files."""

import json
from pathlib import Path

import pytest
from gitrepo import run_script

SCRIPT = Path(__file__).resolve().parents[1] / "required-checks-state.sh"

APP = 15368
RULES = [
    {"type": "deletion"},
    {"type": "pull_request", "parameters": {}},
    {
        "type": "required_status_checks",
        "parameters": {
            "required_status_checks": [
                {"context": "no-main-leak", "integration_id": APP},
                {"context": "all-tests-passed", "integration_id": APP},
            ]
        },
    },
]


def run(id, name, conclusion, app=APP, status="completed"):
    return {
        "id": id,
        "name": name,
        "status": status,
        "conclusion": conclusion,
        "app": {"id": app},
    }


def page(*runs):
    return {"total_count": len(runs), "check_runs": list(runs)}


LEAK = run(1, "no-main-leak", "success")
PASSED = run(2, "all-tests-passed", "success")


def decide(tmp_path, rules, *pages):
    (tmp_path / "rules.json").write_text(json.dumps(rules))
    (tmp_path / "runs.json").write_text("".join(json.dumps(p) for p in pages))
    return run_script(SCRIPT, tmp_path, "rules.json", "runs.json")


CASES = {
    "all_passed": ([page(LEAK, PASSED)], "green"),
    "not_reported": ([page(LEAK)], "pending"),
    "only_skipped": (
        [page(LEAK, run(3, "all-tests-passed", "skipped"))],
        "pending",
    ),
    "in_progress": (
        [page(LEAK, run(3, "all-tests-passed", None, status="in_progress"))],
        "pending",
    ),
    "failed_beats_pending": (
        [
            page(
                run(1, "no-main-leak", None, status="queued"),
                run(3, "all-tests-passed", "failure"),
            )
        ],
        "red",
    ),
    "cancelled": (
        [page(LEAK, run(3, "all-tests-passed", "cancelled"))],
        "red",
    ),
    "rerun_wins": (
        [
            page(
                LEAK,
                run(3, "all-tests-passed", "failure"),
                run(4, "all-tests-passed", "success"),
            )
        ],
        "green",
    ),
    "other_app": (
        [page(LEAK, run(3, "all-tests-passed", "success", app=1))],
        "pending",
    ),
    "two_pages": ([page(LEAK), page(PASSED)], "green"),
}


@pytest.mark.parametrize("pages, word", CASES.values(), ids=CASES.keys())
def test_states(tmp_path, pages, word):
    result = decide(tmp_path, RULES, *pages)
    assert (result.returncode, result.stdout.strip()) == (0, word)


UNDECIDABLE = {
    "no_required_check": (RULES[:2], page(LEAK, PASSED)),
    "not_api_json": (RULES, {"message": "Not Found"}),
}


@pytest.mark.parametrize(
    "rules, runs", UNDECIDABLE.values(), ids=UNDECIDABLE.keys()
)
def test_undecidable_is_an_error(tmp_path, rules, runs):
    result = decide(tmp_path, rules, runs)
    assert (result.returncode, result.stdout) == (2, "")

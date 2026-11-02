"""Self-tests of the harness: record decision and judging."""

import importlib.util
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from loqs.backends import ListPhysicalCircuit
from loqs.codepacks import codepack_surf17_multipatch as surf_multipatch
from loqs.codepacks.steane import SteaneCodepack
from loqs.core import Instruction
from loqs.core.frame import Frame
from loqs.core.instructions import builders

from validation import conftest, engine
from validation.cases import (
    declared_checks,
    generate,
    judge,
    validate_declaration,
)
from validation.bugs import KNOWN_BUGS
from validation.test_programs import RUNNERS, run_ft
from validation.declarations import DECLARATIONS
from validation.logical import Cond, Gate, Meas, Prep, check_shot
from validation import records
from validation.records import decide_record, file_digests, source_digest

DECL = DECLARATIONS["trivial_counter"]
IDS = ["a.default.static.x.1", "a.default.static.x.2"]


FILES = file_digests(DECL)


def _decide(reports, ids=IDS, start_files=FILES, decl=DECL, **kwargs):
    return decide_record(decl, ids, reports, start_files, **kwargs)


def test_record_decision():
    ok = {i: "passed" for i in IDS}
    record = _decide(ok)
    assert isinstance(record, dict)
    assert "cases" not in record
    assert record["classes"]["static"] == {
        "cases": 2,
        "passed": 2,
        "known": 0,
        "failed": 0,
        "by_bug": {},
        "seconds": None,
    }
    assert record["classes"]["ft"] == "not declared"
    assert set(record["files"]) >= set(DECL.files)
    # A missing case, or one that failed, errored or was skipped, refuses.
    assert isinstance(_decide({IDS[0]: "passed"}), str)
    for outcome in ("failed", "errored", "skipped"):
        assert isinstance(_decide({**ok, IDS[1]: outcome}), str)
    # A file edited during the run refuses.
    edited = {**FILES, DECL.files[0]: "0" * 64}
    assert isinstance(_decide(ok, start_files=edited), str)
    # A rendering that differs from the session start's refuses.
    assert isinstance(_decide(ok, start_rendering="0" * 64), str)
    assert isinstance(_decide(ok, start_rendering=record["rendering"]), dict)
    # A declared failure writes its bug's summary once, under `bugs`.
    known = replace(DECL, known_failures={"a.default.*:O1": "steane-1"})
    declared = _decide(ok, decl=known)
    assert isinstance(declared, dict)
    assert list(declared["bugs"]) == ["steane-1"]
    assert declared["bugs"]["steane-1"] == KNOWN_BUGS["steane-1"]
    assert "known_failures" not in declared
    assert declared["classes"]["static"]["known"] == 2
    assert declared["classes"]["static"]["by_bug"] == {"steane-1": 2}


def test_class_summary():
    s1, s2 = "a.default.static.x.1", "a.default.static.x.2"
    t1 = "a.default.truth.x.1"
    f1, f2, f3 = (
        f"a.{c}.ft.ft.g1-{i}-m-n-s"
        for c, i in (("c1", "i1"), ("c1", "i2"), ("c2", "i1"))
    )
    f4 = "a.c1.ft.ft.g2-i-m-n-s"
    declared = {
        s1: {"O1": ["b1", "b2"], "O2": ["b1"]},
        s2: {},
        t1: {},
        f1: {},
        f2: {},
        f3: {},
        f4: {},
    }
    outcomes = {
        s1: "passed",
        t1: "skipped",
        f1: "passed",
        f2: "passed",
        f3: "passed",
        f4: "passed",
    }
    tallies = {
        f1: {"gadget": "g1", "faults": 5, "discarded": 1, "uncorrected": 0},
        f2: {"gadget": "g1", "faults": 5, "discarded": 0, "uncorrected": 0},
        f3: {"gadget": "g1", "faults": 7, "discarded": 0, "uncorrected": 0},
        f4: {"gadget": "g2"},
    }
    out = records.class_summary(
        declared, outcomes, {s1: 1.04, s2: 2.0}, tallies
    )
    assert list(out) == ["static", "truth", "ft"]
    static = out["static"]
    assert {k: static[k] for k in ("cases", "passed", "known", "failed")} == {
        "cases": 2,
        "passed": 0,
        "known": 1,
        "failed": 1,
    }
    assert static["by_bug"] == {"b1": 1, "b2": 1}
    assert static["seconds"] == 3.0
    assert out["truth"]["failed"] == 1 and out["truth"]["seconds"] is None
    assert out["ft"]["faults"] == {
        "gadgets": 2,
        "distinct": 12,
        "runs": 17,
        "discarded": 1,
        "uncorrected": 0,
    }
    bare = records.class_summary(declared, outcomes, {}, {})
    assert bare["ft"]["faults"] is None
    clean = "\n".join(records.verdict_lines("a", out))
    assert "all corrected" in clean
    dirty = {**tallies, f2: {**tallies[f2], "uncorrected": 2}}
    bad = records.class_summary(declared, outcomes, {}, dirty)
    assert "all corrected" not in "\n".join(records.verdict_lines("a", bad))


def test_report_status(monkeypatch):
    # Phase reports in order: (when, outcome) per case.
    phases = [
        [("setup", "passed"), ("call", "passed"), ("teardown", "passed")],
        [("setup", "passed"), ("call", "failed"), ("teardown", "passed")],
        [("setup", "failed"), ("teardown", "passed")],
        [("setup", "passed"), ("call", "passed"), ("teardown", "failed")],
        [("setup", "skipped"), ("teardown", "passed")],
    ]
    ids = [f"a.default.static.x.{i}" for i in range(len(phases))]
    monkeypatch.setattr(conftest, "_REPORTS", {})
    monkeypatch.setattr(conftest, "_SECONDS", {})
    monkeypatch.setattr(conftest, "_TALLIES", {})
    by_id = {i: SimpleNamespace(codepack="a") for i in ids}
    monkeypatch.setattr(conftest, "_BY_ID", by_id)
    tally = {"gadget": "g", "faults": 3}
    for cid, reports in zip(ids, phases):
        for when, outcome in reports:
            conftest.pytest_runtest_logreport(
                SimpleNamespace(
                    nodeid=f"validation/test_x.py::test_x[{cid}]",
                    when=when,
                    failed=outcome == "failed",
                    skipped=outcome == "skipped",
                    duration=0.5,
                    user_properties=[("ft", tally)] if when == "call" else [],
                )
            )
    expected = ["passed", "failed", "errored", "errored", "skipped"]
    assert conftest._REPORTS == {"a": dict(zip(ids, expected))}
    # Every phase's duration sums; the tally comes from the call phase.
    assert conftest._SECONDS["a"][ids[0]] == 1.5
    assert conftest._SECONDS["a"][ids[2]] == 1.0
    assert conftest._TALLIES == {
        "a": {ids[0]: tally, ids[1]: tally, ids[3]: tally}
    }


CID = generate(DECL)[0].id


@pytest.mark.parametrize(
    "known, reproducers",
    [
        ({f"{CID}:O9": "steane-1"}, {}),
        ({"nomatch.*:outcome": "steane-1"}, {}),
        ({f"{CID}:O1": "no-such-bug"}, {}),
        ({}, {"r": ("no.such.case", ("outcome",))}),
        ({f"{CID}:outcome": "steane-1"}, {"r": (CID, ("outcome",))}),
    ],
)
def test_validate_declaration(known, reproducers):
    valid = replace(
        DECL,
        known_failures={f"{CID}:O1": "steane-1"},
        reproducers={"r": (CID, ("outcome",))},
    )
    assert validate_declaration(valid) == []
    bad = replace(DECL, known_failures=known, reproducers=reproducers)
    assert len(validate_declaration(bad)) == 1


def test_check_shot():
    prep_h = [Prep("a", "+Z"), Gate("H", ("a",))]
    z = (("a", "Z"),)
    twice = [*prep_h, Meas("m", z), Meas("m", z)]
    # A random outcome is postselected, then repeats must agree with it.
    assert not check_shot(twice, {"m": [1, 1]})
    assert check_shot(twice, {"m": [1, 0]})
    # A deterministic outcome must match, and a missing one is a problem.
    det = [Prep("a", "-Z"), Meas("m", z)]
    assert not check_shot(det, {"m": [1]})
    assert check_shot(det, {"m": [0]})
    assert check_shot(det, {})
    # A Pauli controlled by an earlier outcome corrects the state.
    cond = [
        Prep("a", "+X"),
        Meas("c", z),
        Cond("a", "X", ("c",)),
        Meas("m", z),
    ]
    assert not check_shot(cond, {"c": [1], "m": [0]})
    assert check_shot(cond, {"c": [1], "m": [1]})


@pytest.mark.parametrize(
    "cid, expected",
    [
        # H's circuit is 7 single-qubit gates in one layer: 7 locations,
        # 3 Paulis.
        (
            "steane.default.ft.ft.gate-H-FT_Zero_Prep-FT_Logical_X_Measure"
            "-none-clean",
            {"gadget": "gate-H", "faults": 21, "discarded": 0},
        ),
        # The FT prep discards the runs its verification flags, and they
        # still count as faults.
        (
            "steane.default.ft.ft.prep-own+none-FT_Logical_Z_Measure"
            "-none-clean",
            {"gadget": "prep", "faults": 78, "discarded": 36},
        ),
    ],
    ids=["gate-H", "prep"],
)
def test_sweep_tally(cid, expected):
    case = next(c for c in generate(DECLARATIONS["steane"]) if c.id == cid)
    tally: dict = {}
    assert run_ft(case, DECLARATIONS["steane"], tally) == {}
    assert tally == {**expected, "uncorrected": 0}


# Reintroduced historical bugs: each mutation edits the built code
# (`code_hook`) or an engine step (`op_hook`).


@dataclass(frozen=True)
class Mutation:
    """A mutation and, per case id, the checks that must fail on it."""

    cases: dict[str, tuple[str, ...]]
    codepack: str = "steane"
    code_hook: Callable[..., Any] | None = None
    op_hook: Callable[..., Any] | None = None


def _s_bare_circuit(code, config):
    """`S` and `Sdag` run only their physical circuit: no frame update."""
    ins = code.instructions
    ins["S"], ins["Sdag"] = ins["S Circuit"], ins["Sdag Circuit"]


def _h_no_syndrome_swap(code, config):
    """`H`'s frame update swaps the logical frame but not the syndrome."""
    ins = code.instructions

    def apply_fn(patches, patch_label):
        patch = patches[patch_label]
        frame = list(patch.data.get("logical_pauli_frame", [0, 0]))[::-1]
        new_patch = patch.copy()
        new_patch.data["logical_pauli_frame"] = frame
        patches[patch_label] = new_patch
        return Frame({"patches": patches})

    ins["H Frame Update"] = Instruction(apply_fn, name="H frame only")
    ins["H"] = builders.build_composite_instruction(
        [ins["H Circuit"], ins["H Frame Update"]], name="Logical H"
    )


def _x_measure_uses_h(code, config):
    """`FT Logical X Measure` is built with `H`, which also swaps the
    stored syndrome, in place of the bare `H Circuit`."""
    ins = code.instructions
    ins["FT Logical X Measure"] = builders.build_composite_instruction(
        [
            ins["H"],
            ins["Raw Z Data Measure"],
            ins["FT X logical parity calculation"],
        ],
        name="FT logical X measurement",
    )


def _s_sdag_swapped(code, config):
    """The `S` and `Sdag` instructions are exchanged."""
    ins = code.instructions
    ins["S"], ins["Sdag"] = ins["Sdag"], ins["S"]


def _x_wrong_support(code, config):
    """Logical `X` acts on the first three data qubits, not the last."""
    circuit = config.backends[0].circuit_backend(
        [[("Gxpi", q) for q in code.template_qubits[:3]]],
        qubit_labels=code.template_qubits,
    )
    code.instructions["X"] = builders.build_physical_circuit_instruction(
        circuit, pauli_frame_update="X", name="Logical X"
    )


def _cx_frame_only(op, entries, geometry):
    """The transversal CX's bookkeeping skips the syndrome conjugation."""
    if op.key != "transversal-cx":
        return entries
    cx = SteaneCodepack(circuit_backend=ListPhysicalCircuit).build_two_patch(
        "Transversal CX", geometry
    )
    circuit, book = cx.data["instructions"]
    n = len(book.data["syndrome_map"])
    book.data["syndrome_map"] = [
        [int(i == j) for j in range(n)] for i in range(n)
    ]
    book.name = "CX frame only"
    return [(circuit, None), (book, None)]


def _surface_cnot_xor_dropped(op, entries, geometry):
    """The transversal CNOT's bookkeeping conjugates the Pauli frames but
    leaves the syndrome histories unXORed."""
    if op.key != "transversal-cnot":
        return entries
    ctrl, tgt = geometry.label("ctrl"), geometry.label("tgt")

    def apply_fn(patches, ctrl_patch_label, tgt_patch_label):
        pc, pt = patches[ctrl_patch_label], patches[tgt_patch_label]
        fc, ft = surf_multipatch.pairwise_cnot_pauli_frames(
            pc.pauli_frame,
            pt.pauli_frame,
            pc.data_qubits,
            pt.data_qubits,
        )
        new_patches = patches.copy()
        new_patches[ctrl_patch_label] = pc.copy(pauli_frame=fc)
        new_patches[tgt_patch_label] = pt.copy(pauli_frame=ft)
        return Frame({"patches": new_patches})

    circuit = surf_multipatch.build_transversal_cnot_circuit_instruction(
        geometry.qubits("ctrl")[:9],
        geometry.qubits("tgt")[:9],
        circuit_backend=ListPhysicalCircuit,
    )
    book = Instruction(
        apply_fn,
        data={"ctrl_patch_label": ctrl, "tgt_patch_label": tgt},
        name="CNOT frame only",
    )
    return [(circuit, None), (book, None)]


P = "steane.default.tracking."
T = "steane.default.truth.truth."
_WORD = P + "word.FT_Zero_Prep-H-W{}-Adaptive_QEC-FT_Logical_Z_Measure"
_FT = "steane.default.ft.ft.check-S1-S5-S6-FT_Zero_Prep+H-"
S = "surface.surf10.tracking.mp-track.transversal-cnot-"
_MP = P + "mp-track.transversal-cx-FT_Zero_Prep-FT_Zero_Prep-Adaptive_QEC-"

MUTATIONS: dict[str, Mutation] = {
    "s-bare-circuit": Mutation(
        {
            _WORD.format("S"): ("outcome", "O3"),
            _WORD.format("Sdag"): ("outcome", "O3"),
            _FT + "FT_Logical_Z_Measure-WS-clean": ("outcome",),
        },
        code_hook=_s_bare_circuit,
    ),
    "h-no-syndrome-swap": Mutation(
        {
            P + "gate.FT_Zero_Prep-none-H-eqgq-Adaptive_QEC-"
            "FT_Logical_X_Measure": ("O2",),
            P + "gate.FT_Zero_Prep-H-H-eqgq-Adaptive_QEC-"
            "FT_Logical_Z_Measure": ("O2",),
        },
        code_hook=_h_no_syndrome_swap,
    ),
    "cx-frame-only": Mutation(
        {
            _MP + "pre": ("O2", "O3"),
            P + "mp-track.transversal-cx-FT_Zero_Prep+H-FT_Zero_Prep+H-"
            "Adaptive_QEC-twice": ("O2",),
        },
        op_hook=_cx_frame_only,
    ),
    "x-measure-uses-h": Mutation(
        {
            P
            + "word.FT_Zero_Prep-none-WS-Adaptive_QEC-FT_Logical_X_Measure": (
                "outcome",
            ),
            P + "word.Non-FT_Zero_Prep-none-WS-Adaptive_QEC-"
            "FT_Logical_X_Measure": ("outcome",),
        },
        code_hook=_x_measure_uses_h,
    ),
    "s-sdag-swapped": Mutation(
        {
            T + "FT_Zero_Prep-H+S-FT_Logical_Z_Measure-noqec": ("O3",),
            T + "FT_Zero_Prep-H+Sdag-FT_Logical_Z_Measure-noqec": ("O3",),
        },
        code_hook=_s_sdag_swapped,
    ),
    "x-wrong-support": Mutation(
        {
            T + "FT_Zero_Prep-X-FT_Logical_Z_Measure-noqec": ("outcome",),
            T + "Non-FT_Zero_Prep-X-FT_Logical_Z_Measure-noqec": ("outcome",),
        },
        code_hook=_x_wrong_support,
    ),
    "surface-cnot-xor-dropped": Mutation(
        {
            S + "Zero_Prep-Zero_Prep-QEC-pre": ("O2",),
            S + "Plus_Prep-Plus_Prep-QEC-pre": ("O2",),
        },
        codepack="surface",
        op_hook=_surface_cnot_xor_dropped,
    ),
}


def _failing(case, decl):
    try:
        return set(RUNNERS[case.cls](case, decl))
    except Exception:  # noqa: BLE001 -- a codepack error is an outcome failure
        return {"outcome"}


@pytest.mark.parametrize("mutation_id", sorted(MUTATIONS))
def test_mutation_caught(mutation_id):
    mutation = MUTATIONS[mutation_id]
    decl = DECLARATIONS[mutation.codepack]
    cases = {c.id: c for c in generate(decl)}
    hook_errors: list[Exception] = []

    def guard(hook):
        # A hook that raises would otherwise read as a caught `outcome`.
        if hook is None:
            return None

        def run(*args):
            try:
                return hook(*args)
            except Exception as exc:
                hook_errors.append(exc)
                raise

        return run

    for case_id, targets in mutation.cases.items():
        case = cases[case_id]
        assert not set(targets) & set(declared_checks(decl, case_id))
        assert not set(targets) & _failing(case, decl)
        with engine.hooked(guard(mutation.code_hook), guard(mutation.op_hook)):
            failing = _failing(case, decl)
        assert not hook_errors
        assert set(targets) <= failing


def test_judge():
    judge(set(), set())
    judge({"O4"}, {"O4"})
    with pytest.raises(AssertionError):
        judge({"O4"}, {"O4", "O3"})  # a second bug hides behind a known one
    with pytest.raises(AssertionError):
        judge({"O4"}, set())  # a fix that flips nothing


_MODULE = """
LIMIT = {limit}


def helper(x):
    {helper_doc}
    return x + {helper_k}


def apply(x):
    {apply_doc}
    # {comment}
    return helper(x) * LIMIT + {apply_k}
"""


def _digest(tmp_path, name, **changes):
    fields = dict(
        limit=2,
        helper_k=1,
        apply_k=0,
        helper_doc='"""Helper."""',
        apply_doc='"""Apply."""',
        comment="a comment",
    )
    fields.update(changes)
    path = tmp_path / f"{name}.py"
    path.write_text(_MODULE.format(**fields), encoding="utf-8")
    # One module name for every variant: the digest names functions by it.
    spec = importlib.util.spec_from_file_location("fp_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return source_digest(module.apply)


def test_fingerprint_sees_behaviour_not_docs(tmp_path):
    base = _digest(tmp_path, "m_base")
    # Docstrings and comments don't move the digest.
    assert base == _digest(
        tmp_path,
        "m_docs",
        helper_doc='"""Other help."""',
        apply_doc='"""Other."""',
        comment="another comment",
    )
    # The body, a same-module helper and a module constant do.
    for name, change in (
        ("m_body", {"apply_k": 1}),
        ("m_helper", {"helper_k": 2}),
        ("m_const", {"limit": 3}),
    ):
        assert base != _digest(tmp_path, name, **change)

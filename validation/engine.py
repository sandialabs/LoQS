"""Builds codes, models and programs from a declaration and runs them."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Sequence

from loqs.backends import ListPhysicalCircuit
from loqs.core import InstructionStack, PatchGeometry, QuantumProgram
from loqs.core.instructions import builders
from loqs.tools import fttools

from validation import logical, oracle
from validation.declaration import Backend, CodepackDeclaration, Config

MAX_FRAME_LIMIT = 100_000
OUTCOME_KEY = "logical_measurement"
LABEL_KINDS = ("default", "permuted", "renamed", "offset")
_PAD = 3
_MAX_PLACES = 3


@dataclass
class Hooks:
    """Mutation hooks of the harness's self-tests: `code_hook(code, config)`
    edits every built code in place, `op_hook(op, entries, geometry)`
    returns the replacement entries of an engine step (`geometry` is the
    `PatchGeometry` of an `op` or `sweep` step, else `None`). Both default
    to `None`."""

    code_hook: Any = None
    op_hook: Any = None


HOOKS = Hooks()


@contextmanager
def hooked(code_hook=None, op_hook=None):
    """Run the engine with a code hook and an op hook, then restore it."""
    saved = (HOOKS.code_hook, HOOKS.op_hook)
    HOOKS.code_hook, HOOKS.op_hook = code_hook, op_hook
    try:
        yield
    finally:
        HOOKS.code_hook, HOOKS.op_hook = saved


def build_code(decl: CodepackDeclaration, config: Config, backend: Backend):
    """Build the `QECCode` of a config on a backend."""
    code = decl.build_code(config, backend)
    if HOOKS.code_hook is not None:
        HOOKS.code_hook(code, config)
    return code


def build_model(
    decl: CodepackDeclaration,
    config: Config,
    qubits: Sequence[str],
    backend: Backend,
):
    """Build the ideal model over `qubits` on a backend."""
    return decl.build_model(config, qubits, backend)


def patch_labels(code, kind: str, n_patches: int = 1):
    """Per-patch qubit lists and the register they live in.

    `default` uses the template labels, `permuted` reverses them, `renamed`
    uses fresh pyGSTi-legal labels and `offset` places the patches after
    unused padding qubits. Patches get disjoint label sets.
    """
    template = list(code.template_qubits)
    n = len(template)
    patches: list[list[str]] = []
    pad: list[str] = []
    for i in range(n_patches):
        if kind == "default" and i == 0:
            patches.append(template)
        elif kind == "permuted" and i == 0:
            patches.append(template[::-1])
        elif kind == "renamed" or i > 0:
            patches.append([f"R{i}_{j}" for j in range(n)])
        elif kind == "offset":
            pad = [f"X{j}" for j in range(_PAD)]
            patches.append([f"O{j}" for j in range(n)])
        else:
            raise ValueError(f"unknown label kind {kind!r}")
    if kind == "offset" and n_patches > 1 and not pad:
        pad = [f"X{j}" for j in range(_PAD)]
    register = pad + [q for p in patches for q in p]
    return patches, register


def build_program(
    decl: CodepackDeclaration,
    code,
    model,
    backend: Backend,
    patches: dict[str, Sequence[str]],
    register: Sequence[str],
    entries: Sequence[Any],
    name: str = "validation",
) -> QuantumProgram:
    """Build a program that inits the state and patches, then `entries`."""
    stack: list[Any] = [
        {
            "instruction": "Init State",
            "state": len(register),
            "qubit_labels": list(register),
        }
    ]
    for label, qubits in patches.items():
        stack.append(
            {
                "instruction": f"Init Patch {decl.patch_type}",
                "new_patch_label": label,
                "qubits": list(qubits),
            }
        )
    stack += list(entries)
    return QuantumProgram(
        InstructionStack(stack),
        default_noise_model=model,
        state_type=backend.state_type,
        patch_types={decl.patch_type: code},
        name=name,
    )


def run_program(
    program: QuantumProgram,
    shots: int,
    expected: Sequence[str] = (),
    seed: int = 0,
):
    """Run a program; raise if a shot ends without an expected outcome key.

    The frame limit is far above the core default, which stops silently with
    only a warning and would leave long programs without their outcomes.
    """
    program.default_base_seed = seed
    results = program.run(
        shots, max_frame_limit=MAX_FRAME_LIMIT, verbose=False
    )
    for key in expected:
        for shot, history in results.shot_histories.items():
            data = history.collect_data(key, "all", strip_none_entries=True)
            if not data:
                raise RuntimeError(
                    f"shot {shot} ended without outcome key {key!r}"
                )
    return results


def script_keys(script) -> tuple[str, ...]:
    """The distinct measurement keys of a script, in order."""
    return tuple(
        dict.fromkeys(s.key for s in script if isinstance(s, logical.Meas))
    )


def shot_outcomes(history, script) -> dict[str, list]:
    """Each measurement key's reported values in a shot, in program order."""
    return {
        key: history.collect_data(key, "all", strip_none_entries=True)
        for key in script_keys(script)
    }


@dataclass(frozen=True)
class Op:
    """One engine step: prep, gate, qec, measure or error on a patch, or a
    declared multipatch `op` over the patch labels `roles` (in the order of
    the declared roles). `roles` default to the single patch `L0`."""

    kind: str
    key: str
    kwargs: tuple = ()
    roles: tuple = ("L0",)


@dataclass
class WordResult:
    """What running a word found: failing checks and its frame count."""

    failing: dict[str, str]
    frames: int


def _find(items, op: Op):
    return next(
        i
        for i in items
        if i.key == op.key
        and (
            op.kind != "measure"
            or tuple(sorted(i.kwargs.items())) == op.kwargs
        )
    )


def _declared(decl: CodepackDeclaration, op: Op):
    """The declared multipatch op or sweep an `op` or `sweep` step names."""
    items = decl.ops if op.kind == "op" else decl.sweeps
    return next(o for o in items if o.name == op.key)


def word_roles(ops: Sequence[Op]) -> list[str]:
    """Every patch label a word uses, sorted."""
    return sorted({r for op in ops for r in op.roles})


def plan_word(decl: CodepackDeclaration, ops: Sequence[Op]):
    """The logical script of a word, each step's expectations and windows.

    Returns `(script, expects, windows)`: `expects[i]` maps every logical
    product (`oracle.key_name`) over the word's patches to its expectation
    after op `i`, and `windows[i]` maps each patch to its `(O1, O2, O3)`
    flags: a patch is checked from its prep (O3) and from the first QEC
    after it (O1, O2) until its measure. O3 stops after a random measure,
    since later expectations then depend on the drawn outcome.
    `windows[i]` is `None` while an
    injected error has not yet met a QEC round on its patch (or on a patch
    an op coupled it to): no observation is placed there. Errors add no
    script step, since the tracked state must absorb them.
    """
    roles = word_roles(ops)
    keys = oracle.logical_keys(roles)
    script: list[logical.Step] = []
    sim = logical.LogicalSim(roles)
    expects, windows = [], []
    opened = {r: False for r in roles}
    qec = {r: False for r in roles}
    pending: set[str] = set()
    # Patches that took an error since their prep: a deferred decoder's
    # correction exists only at its final decode, so O3 skips them.
    injected: set[str] = set()
    deferred = decl.decoder == "deferred"
    random_branch = False
    for op in ops:
        steps: list[logical.Step] = []
        role = op.roles[0]
        if op.kind == "prep":
            steps = [logical.Prep(role, _find(decl.preps, op).state)]
            random_branch = random_branch and any(
                opened[r] for r in roles if r != role
            )
            opened[role], qec[role] = True, False
            pending.discard(role)
            injected.discard(role)
        elif op.kind == "gate":
            clifford = _find(decl.gates, op).clifford
            steps = [logical.Gate(clifford, (role,))]
        elif op.kind == "qec":
            qec[role] = True
            pending.discard(role)
        elif op.kind == "error":
            pending.add(role)
            injected.add(role)
        elif op.kind in ("op", "sweep"):
            declared = _declared(decl, op)
            names = dict(zip(declared.roles, op.roles))
            steps = [logical.rename(s, names) for s in declared.script]
            if pending & set(op.roles):
                pending |= set(op.roles)
            if injected & set(op.roles):
                injected |= set(op.roles)
        else:
            basis = _find(decl.measures, op).basis
            steps = [logical.Meas(OUTCOME_KEY, ((role, basis),))]
            opened[role] = False
        for step in steps:
            script.append(step)
            bit = None
            if isinstance(step, logical.Meas):
                bit = sim.deterministic_bit(step.paulis)
                random_branch = random_branch or bit is None
                bit = bit or 0
            sim.apply(step, bit)
        expects.append({oracle.key_name(k): sim.expectation(k) for k in keys})
        windows.append(
            None
            if pending
            else {
                r: (
                    opened[r] and qec[r],
                    opened[r] and qec[r],
                    opened[r]
                    and not random_branch
                    and not (deferred and r in injected),
                )
                for r in roles
            }
        )
    return script, expects, windows


def pauli_error(qubit: str, pauli: str, role: str = "L0"):
    """A global stack entry applying `Gxpi`, `Gypi` or `Gzpi` to a qubit."""
    circuit = ListPhysicalCircuit(
        [[(f"G{pauli.lower()}pi", qubit)]], qubit_labels=[qubit]
    )
    name = f"Injected {pauli} error on {qubit}"
    inst = builders.build_physical_circuit_instruction(circuit, name=name)
    return {"instruction": inst, "patch_label": role}


def word_seams(decl: CodepackDeclaration, ops: Sequence[Op]):
    """The bare qubit groups a word's ops need: name -> labels.

    Ops of one word share them: the largest size of each name is allocated.
    """
    sizes: dict[str, int] = {}
    for op in ops:
        if op.kind == "op":
            for name, n in _declared(decl, op).seams.items():
                sizes[name] = max(n, sizes.get(name, 0))
    return {
        name: [f"E_{name}{j}" for j in range(n)] for name, n in sizes.items()
    }


def with_seams(register: Sequence[str], seams) -> list[str]:
    """The register extended by every seam group's qubits."""
    return list(register) + [q for qs in seams.values() for q in qs]


def op_entries(
    decl: CodepackDeclaration,
    config: Config,
    op: Op,
    qubits: dict[str, Sequence[str]],
    code,
    seams: dict[str, Sequence[str]] | None = None,
) -> list[Any]:
    """The stack entries of an op; `qubits` maps patch labels to qubits."""
    geometry = None
    if op.kind in ("op", "sweep"):
        declared = _declared(decl, op)
        own = {
            name: list((seams or {})[name][:n])
            for name, n in getattr(declared, "seams", {}).items()
        }
        geometry = PatchGeometry(
            patches={
                name: (label, qubits[label])
                for name, label in zip(declared.roles, op.roles)
            },
            layout=config.options.get("layout", decl.patch_type),
            seams=own or None,
        )
    entries = _op_entries(decl, config, op, qubits, code, geometry)
    if HOOKS.op_hook is not None:
        entries = list(HOOKS.op_hook(op, entries, geometry))
    return entries


def _op_entries(decl, config, op, qubits, code, geometry) -> list[Any]:
    role = op.roles[0]
    if op.kind == "error":
        data = decl.data_qubits[dict(op.kwargs)["data"]]
        qubit = qubits[role][code.template_qubits.index(data)]
        return [pauli_error(qubit, op.key, role)]
    if op.kind == "measure":
        return [{"instruction": op.key, "patch_label": role} | dict(op.kwargs)]
    if op.kind in ("op", "sweep"):
        declared = _declared(decl, op)
        build = declared.build if op.kind == "op" else declared.entries
        return list(build(config, geometry) if callable(build) else build)
    return [(op.key, role)]


def run_word(
    decl: CodepackDeclaration,
    config: Config,
    backend: Backend,
    ops: Sequence[Op],
    *,
    shots: int = 8,
    seed: int = 0,
    observe: bool = False,
    kind: str = "default",
    code=None,
    model=None,
) -> WordResult:
    """Run a word noiselessly and judge it: outcomes, then O1-O4.

    Each shot's outcomes are judged against the logical script. With
    `observe` (STIM only) an observation follows every op except while an
    injected error awaits its QEC round. `code` and `model` may be given
    to reuse them across runs.
    """
    code = code if code is not None else build_code(decl, config, backend)
    roles = word_roles(ops)
    lists, register = patch_labels(code, kind, len(roles))
    qubits = dict(zip(roles, lists))
    seams = word_seams(decl, ops)
    register = with_seams(register, seams)
    if model is None:
        model = build_model(decl, config, register, backend)
    script, expects, windows = plan_word(decl, ops)
    entries: list[Any] = []
    for i, op in enumerate(ops):
        entries += op_entries(decl, config, op, qubits, code, seams)
        if observe and windows[i] is not None:
            label = f"after {op.key}"
            entries.append(
                oracle.observation(decl, i, label, windows[i], expects[i])
            )
    program = build_program(
        decl, code, model, backend, qubits, register, entries
    )
    results = run_program(program, shots, script_keys(script), seed)
    places: dict[str, list[str]] = {}
    for shot, history in results.shot_histories.items():
        problems = logical.check_shot(script, shot_outcomes(history, script))
        if problems:
            places.setdefault("outcome", []).append(
                f"shot {shot}: {problems[0]}"
            )
        if observe:
            oracle.evaluate(decl, history, shot, places)
    return WordResult(
        dict(oracle.summarize(places)), len(results.shot_histories[0])
    )


def run_placements(
    decl: CodepackDeclaration,
    config: Config,
    backend: Backend,
    runs: Sequence[tuple[str, Sequence[Op]]],
    *,
    shots: int = 2,
    seed: int = 0,
) -> dict[str, str]:
    """Run each `(placement, ops)` observed; union their failing checks.

    The result maps each failing check to a summary naming the placements
    that failed it. A codepack exception in a placement is an `outcome`
    failure of that placement.
    """
    code = build_code(decl, config, backend)
    n = max(len(word_roles(ops)) for _, ops in runs)
    _, register = patch_labels(code, "default", n)
    seams: dict[str, Sequence[str]] = {}
    for _, ops in runs:
        for name, qs in word_seams(decl, ops).items():
            if len(qs) > len(seams.get(name, ())):
                seams[name] = qs
    model = build_model(decl, config, with_seams(register, seams), backend)
    by_check: dict[str, list[str]] = {}
    for name, ops in runs:
        try:
            result = run_word(
                decl,
                config,
                backend,
                ops,
                shots=shots,
                seed=seed,
                observe=True,
                code=code,
                model=model,
            )
            failed = result.failing
        except Exception as exc:  # noqa: BLE001 -- any codepack error
            failed = {"outcome": f"{type(exc).__name__}: {exc}"}
        for check, detail in failed.items():
            by_check.setdefault(check, []).append(f"{name} ({detail[:150]})")
    return {
        check: f"{len(places)} of {len(runs)} placements: "
        + "; ".join(places[:_MAX_PLACES])
        for check, places in by_check.items()
    }


_ERROR_LABELS = ["Gxpi", "Gypi", "Gzpi"]
_MAX_FAILS = 3


def _target(entries, target: str, code):
    """`(index, instruction)` of the entry named `target`: a string entry by
    its key, an instruction entry by its name."""
    for i, entry in enumerate(entries):
        first = entry[0] if isinstance(entry, tuple) else entry["instruction"]
        if isinstance(first, str):
            if first == target:
                return i, code.instructions[first]
        elif first.name == target:
            return i, first
    raise ValueError(f"sweep target {target!r} is not among its entries")


def run_sweep(
    decl: CodepackDeclaration,
    config: Config,
    backend: Backend,
    ops: Sequence[Op],
    *,
    seed: int = 0,
) -> dict[str, str]:
    """Inject every fault of a word's one `sweep` step and judge each run.

    Each fault program runs for one shot and is judged by the word's
    logical script, unless the sweep's `discard` drops it. The `outcome`
    check fails when failures exceed the sweep's `allowed`.
    """
    code = build_code(decl, config, backend)
    roles = word_roles(ops)
    lists, register = patch_labels(code, "default", len(roles))
    qubits = dict(zip(roles, lists))
    seams = word_seams(decl, ops)
    register = with_seams(register, seams)
    model = build_model(decl, config, register, backend)
    script, _, _ = plan_word(decl, ops)
    entries: list[Any] = []
    target = None
    for op in ops:
        new = op_entries(decl, config, op, qubits, code, seams)
        if op.kind == "sweep":
            declared = _declared(decl, op)
            j, inst = _target(new, declared.target, code)
            target = (1 + len(qubits) + len(entries) + j, inst)
        entries += new
    assert target is not None, "a sweep word needs a sweep step"
    base = build_program(decl, code, model, backend, qubits, register, entries)
    programs = fttools.build_discrete_error_injection_programs(
        base,
        target[1],
        target[0],
        _ERROR_LABELS,
        post_twoq_gates=declared.post_twoq,
    )
    if not programs:
        raise RuntimeError("the sweep injected no faults")
    failures, discarded = [], 0
    for i, program in enumerate(programs):
        try:
            results = run_program(program, 1, script_keys(script), seed + i)
            history = results.shot_histories[0]
            if declared.discard is not None and declared.discard(history):
                discarded += 1
                continue
            problems = logical.check_shot(
                script, shot_outcomes(history, script)
            )
        except Exception as exc:  # noqa: BLE001 -- any codepack error
            problems = [f"{type(exc).__name__}: {exc}"]
        if problems:
            tag = program.name.partition(" + ")[2]
            failures.append(f"{tag} ({problems[0][:100]})")
    if discarded == len(programs):
        raise RuntimeError("the sweep discarded every fault")
    if len(failures) <= declared.allowed:
        return {}
    return {
        "outcome": f"{len(failures)} of {len(programs)} faults failed "
        f"({discarded} discarded, {declared.allowed} allowed): "
        + "; ".join(failures[:_MAX_FAILS])
    }

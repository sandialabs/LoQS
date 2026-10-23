"""Deterministic case generation, known-failure lookup and judging."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import Any, Mapping, Sequence

import stim

from validation import logical
from validation.bugs import KNOWN_BUGS
from validation.declaration import CHECKS, CodepackDeclaration
from validation.engine import LABEL_KINDS, OUTCOME_KEY, Op, plan_word
from validation.logical import has_check


@dataclass(frozen=True)
class Case:
    """One generated check: `<codepack>.<config>.<class>.<family>.<detail>`."""

    id: str
    codepack: str
    config: str
    cls: str
    family: str
    params: Mapping[str, Any] = field(default_factory=dict)
    big: bool = False


def _case(decl, config, cls, family, detail, big=False, **params):
    cid = f"{decl.name}.{config}.{cls}.{family}.{detail}"
    return Case(cid, decl.name, config, cls, family, params, big)


def _static_cases(decl: CodepackDeclaration) -> list[Case]:
    cases = [_case(decl, "all", "static", "completeness", "all")]
    for config in decl.configs:
        if "static" not in config.classes:
            continue
        big = config.big
        for backend in config.backends:
            for n in (1, 2):
                cases.append(
                    _case(
                        decl,
                        config.name,
                        "static",
                        "model-coverage",
                        f"{backend.name}-p{n}",
                        big,
                        backend=backend.name,
                        patches=n,
                    )
                )
        if decl.preps:
            for kind in LABEL_KINDS:
                cases.append(
                    _case(
                        decl,
                        config.name,
                        "static",
                        "labels",
                        kind,
                        big,
                        kind=kind,
                    )
                )
        for fmt in ("json", "hdf5"):
            cases.append(
                _case(
                    decl,
                    config.name,
                    "static",
                    "serialization",
                    fmt,
                    big,
                    format=fmt,
                )
            )
    return cases


def _slug(text: str) -> str:
    return text.replace(" ", "_")


def _mtag(measure) -> str:
    """A measure's id part: its key and kwargs, so variants stay distinct."""
    return _slug(measure.key) + "".join(
        f"-{k}_{v}" for k, v in measure.kwargs.items()
    )


def _words(decl: CodepackDeclaration, max_len: int):
    for n in range(max_len + 1):
        yield from itertools.product([g.key for g in decl.gates], repeat=n)


def _truth_cases(decl: CodepackDeclaration) -> list[Case]:
    cases = []
    for config in decl.configs:
        if "truth" not in config.classes:
            continue
        for family, backend, max_plain in [
            ("truth", config.backends[0], 2)
        ] + [("agreement", b, 1) for b in config.backends[1:]]:
            for prep in decl.preps:
                for m, measure in enumerate(decl.measures):
                    # Kwargs variants of a measure only get short words.
                    longest = 1 if measure.kwargs else max_plain
                    for word in _words(decl, min(longest, max_plain)):
                        for qec in (None, *decl.qec):
                            ops = truth_ops(decl, prep.key, word, m, qec)
                            if not logical_has_check(decl, ops):
                                continue
                            word_id = "+".join(word) or "none"
                            detail = "-".join(
                                [
                                    _slug(prep.key),
                                    word_id,
                                    _mtag(measure),
                                    "noqec" if qec is None else _slug(qec),
                                ]
                            )
                            if family == "agreement":
                                detail = f"{backend.name}-{detail}"
                            cases.append(
                                _case(
                                    decl,
                                    config.name,
                                    "truth",
                                    family,
                                    detail,
                                    config.big,
                                    backend=backend.name,
                                    prep=prep.key,
                                    word=word,
                                    measure=m,
                                    qec=qec,
                                )
                            )
    return cases


def truth_ops(decl, prep: str, word, measure: int, qec) -> list[Op]:
    """Prep, then each gate, then the measure, with QEC after prep and gates."""
    ops = [Op("prep", prep)]
    if qec is not None:
        ops.append(Op("qec", qec))
    for gate in word:
        ops.append(Op("gate", gate))
        if qec is not None:
            ops.append(Op("qec", qec))
    m = decl.measures[measure]
    ops.append(Op("measure", m.key, tuple(sorted(m.kwargs.items()))))
    return ops


def logical_has_check(decl, ops) -> bool:
    """Whether a word's logical script has something deterministic to judge."""
    script, _, _ = plan_word(decl, ops)
    return has_check(script)


PAULIS = "XYZ"
_ERR = Op("error", "")


def _basis(state: str) -> str:
    return state[-1]


def prep_inputs(decl: CodepackDeclaration):
    """`(prep key, basis, input word)`: a state in each basis, per prep.

    A basis a declared prep already prepares uses that prep alone; any
    other basis uses each prep followed by the shortest declared gate word
    that reaches it.
    """
    out = []
    for prep in decl.preps:
        for basis in "ZX":
            if _basis(prep.state) == basis:
                out.append((prep.key, basis, ()))
            elif not any(_basis(p.state) == basis for p in decl.preps):
                word = _shortest_word(decl, prep, basis)
                if word is not None:
                    out.append((prep.key, basis, word))
    return out


def _shortest_word(decl, prep, basis):
    for word in _words(decl, 2):
        script = [logical.Prep("L0", prep.state)] + [
            logical.Gate(_gate(decl, g).clifford, ("L0",)) for g in word
        ]
        sim = logical.LogicalSim(["L0"])
        for step in script:
            sim.apply(step)
        if sim.expectation([("L0", basis)]):
            return word
    return None


def _gate(decl, key):
    return next(g for g in decl.gates if g.key == key)


def _is_pauli(gate) -> bool:
    return gate.clifford in ("I", "X", "Y", "Z")


def _commute(a, b) -> bool:
    ta = stim.Tableau.from_named_gate(a.clifford)
    tb = stim.Tableau.from_named_gate(b.clifford)
    return ta.then(tb) == tb.then(ta)


def gate_word(decl, gate_key: str) -> tuple[str, ...]:
    """W(G) = G M G M G: M is the first declared non-Pauli gate that does
    not commute with G, else the first such Pauli."""
    g = _gate(decl, gate_key)
    others = [x for x in decl.gates if not _commute(g, x)]
    nonpauli = [x for x in others if not _is_pauli(x)]
    m = (nonpauli or others)[0]
    return (g.key, m.key, g.key, m.key, g.key)


def _gates_qec(gates, qec):
    out = []
    for gate in gates:
        out += [Op("gate", gate), Op("qec", qec)]
    return out


def _measures_for(decl, basis):
    return [i for i, m in enumerate(decl.measures) if m.basis == basis]


def _measure_op(decl, m: int, role: str = "L0") -> Op:
    meas = decl.measures[m]
    kwargs = tuple(sorted(meas.kwargs.items()))
    return Op("measure", meas.key, kwargs, (role,))


def _tracking_templates(decl: CodepackDeclaration):
    """`(family, detail, errors, template ops)`, error ops left blank.

    A template's error slots are `_ERR` and are filled from a placement.
    """
    out = []
    qecs = decl.qec
    for prep, basis, word in prep_inputs(decl):
        head = [Op("prep", prep)] + [Op("gate", g) for g in word]
        tag = f"{_slug(prep)}-{'+'.join(word) or 'none'}"
        for m in range(len(decl.measures)):
            tail = _measure_op(decl, m)
            mname = _mtag(decl.measures[m])
            for n in (1, 2):
                for keys in itertools.product(qecs, repeat=n):
                    ops = head + [_ERR]
                    ops += [Op("qec", k) for k in keys] + [tail]
                    detail = f"{tag}-{'+'.join(_slug(k) for k in keys)}"
                    out.append(("idle", f"{detail}-{mname}", 1, ops))
            for g in decl.gates:
                for q in qecs:
                    for pattern in ("eqgq", "egq"):
                        ops = head + [_ERR]
                        if pattern == "eqgq":
                            ops += [Op("qec", q)]
                        ops += [Op("gate", g.key), Op("qec", q), tail]
                        detail = f"{tag}-{g.key}-{pattern}-{_slug(q)}"
                        out.append(("gate", f"{detail}-{mname}", 1, ops))
    out += _word_templates(decl)
    out += _reuse_templates(decl)
    return out


def _word_templates(decl):
    out = []
    for prep in decl.preps:
        tag = _slug(prep.key)
        for m in range(len(decl.measures)):
            tail = _measure_op(decl, m)
            mname = _mtag(decl.measures[m])
            for q in decl.qec:
                for g in decl.gates:
                    if _is_pauli(g):
                        continue
                    w = gate_word(decl, g.key)
                    for inp in [()] + [(x.key,) for x in decl.gates]:
                        ops = [Op("prep", prep.key)]
                        ops += [Op("gate", x) for x in inp]
                        ops += [_ERR, Op("qec", q), _ERR, Op("qec", q)]
                        ops += _gates_qec(w, q) + [tail]
                        detail = (
                            f"{tag}-{'+'.join(inp) or 'none'}-W{g.key}-"
                            f"{_slug(q)}-{mname}"
                        )
                        out.append(("word", detail, 2, ops))
    return out


def _reuse_templates(decl):
    out = []
    for prep, basis, word in prep_inputs(decl):
        head = [Op("prep", prep)] + [Op("gate", x) for x in word]
        base = _slug(prep) + "-" + ("+".join(word) or "none")
        again = next(p for p in decl.preps if p.key == prep)
        for q in decl.qec:
            errs = [_ERR, Op("qec", q), _ERR, Op("qec", q)]
            for first in _measures_for(decl, basis):
                for last in _measures_for(decl, _basis(again.state)):
                    tail = [Op("prep", prep), _measure_op(decl, last)]
                    name = _mtag(decl.measures[first])
                    ops = head + errs + [_measure_op(decl, first)] + tail
                    detail = f"{base}-{_slug(q)}-meas-{name}-{last}"
                    out.append(("reuse", detail, 2, ops))
            if again.fresh_only:
                continue
            for last in _measures_for(decl, _basis(again.state)):
                ops = head + errs + [Op("prep", prep)]
                ops += [_measure_op(decl, last)]
                out.append(
                    ("reuse", f"{base}-{_slug(q)}-noread-{last}", 2, ops)
                )
    return out


def placements(
    decl: CodepackDeclaration,
    family: str,
    errors: int,
    roles: Sequence[str] = ("L0",),
):
    """Error placements: tuples of `(data index, Pauli, patch label)`.

    One error is placed on each of `roles`; two errors (same Pauli, two
    distinct data qubits) always on the first.
    """
    n = len(decl.data_qubits)
    if errors == 1:
        return [((d, p, r),) for r in roles for d in range(n) for p in PAULIS]
    paulis = "XZ" if family == "reuse" else PAULIS
    return [
        ((a, p, roles[0]), (b, p, roles[0]))
        for p in paulis
        for a in range(n)
        for b in range(n)
        if a != b
    ]


def fill(template: Sequence[Op], placement) -> list[Op]:
    """Replace a template's blank error ops with a placement's errors."""
    errs = iter(placement)
    out = []
    for op in template:
        if op is _ERR:
            d, p, role = next(errs)
            op = Op("error", p, (("data", d),), (role,))
        out.append(op)
    return out


def tracking_runs(decl: CodepackDeclaration, case: Case):
    """The `(placement name, ops)` runs of a tracking-style case."""
    p = case.params
    roles = p.get("error_roles", ("L0",))
    return [
        (
            "+".join(f"{pa}{d}@{r}" for d, pa, r in pl),
            fill(p["template"], pl),
        )
        for pl in placements(decl, case.family, p["errors"], roles)
    ]


def _tracking_cases(decl: CodepackDeclaration) -> list[Case]:
    cases = []
    for config in decl.configs:
        if "tracking" not in config.classes:
            continue
        for family, detail, errors, ops in _tracking_templates(decl):
            if not deterministic_outcome(decl, ops):
                continue
            cases.append(
                _case(
                    decl,
                    config.name,
                    "tracking",
                    family,
                    detail,
                    config.big,
                    template=tuple(ops),
                    errors=errors,
                )
            )
    return cases


def _agreement_error_cases(decl: CodepackDeclaration) -> list[Case]:
    """Each non-primary backend: an X or Z error on the first data qubit,
    one QEC round and a deterministic measure."""
    cases = []
    for config in decl.configs:
        if "truth" not in config.classes:
            continue
        for backend in config.backends[1:]:
            for prep, basis, word in prep_inputs(decl):
                head = [Op("prep", prep)] + [Op("gate", g) for g in word]
                for q in decl.qec:
                    for m in _measures_for(decl, basis):
                        for pauli in "XZ":
                            ops = head + [_ERR, Op("qec", q)]
                            ops.append(_measure_op(decl, m))
                            detail = (
                                f"{backend.name}-err{pauli}-{_slug(prep)}-"
                                f"{'+'.join(word) or 'none'}-{_slug(q)}-{m}"
                            )
                            cases.append(
                                _case(
                                    decl,
                                    config.name,
                                    "truth",
                                    "agreement",
                                    detail,
                                    config.big,
                                    backend=backend.name,
                                    template=tuple(ops),
                                    placement=((0, pauli, "L0"),),
                                )
                            )
    return cases


_LABELS = ("L0", "L1")


def _mp_ops(decl, config):
    """The declared ops (two or more roles) that run on a config."""
    return [
        op
        for op in decl.ops
        if len(op.roles) >= 2 and (not op.configs or config.name in op.configs)
    ]


def _op_labels(op) -> tuple[str, ...]:
    """The patch labels of an op's roles, in the order of the roles."""
    return tuple(f"L{i}" for i in range(len(op.roles)))


def _op_big(decl, config, op) -> bool:
    layout = config.options.get("layout", decl.patch_type)
    return config.big or layout in op.big_layouts


def _readout_measures(decl, op):
    """Per basis, the measure whose kwargs are the op's readout kwargs."""
    want = dict(op.readout_kwargs)
    return {
        b: next(
            i
            for i in _measures_for(decl, b)
            if decl.measures[i].kwargs == want
        )
        for b in "ZX"
    }


def _state_basis(decl, prep: str, word) -> str:
    """The Z or X basis a prep-and-gates state is definite in, else Z."""
    sim = logical.LogicalSim(["L0"])
    sim.apply(logical.Prep("L0", _prep(decl, prep).state))
    for g in word:
        sim.apply(logical.Gate(_gate(decl, g).clifford, ("L0",)))
    return next((b for b in "ZX" if sim.expectation([("L0", b)])), "Z")


def _prep(decl, key):
    return next(p for p in decl.preps if p.key == key)


def _input_ops(role: str, prep: str, word) -> list[Op]:
    return [Op("prep", prep, (), (role,))] + [
        Op("gate", g, (), (role,)) for g in word
    ]


def _qecs(decl, q: str) -> list[Op]:
    return [Op("qec", q, (), (r,)) for r in _LABELS]


def _op_step(op) -> Op:
    return Op("op", op.name, (), _op_labels(op))


def _measures(decl, bases, op=None) -> list[Op]:
    """A readout of each of the first two roles in its basis, with the
    measure kwargs the op asks for (the plain measure by default)."""
    by_basis = _readout_measures(decl, op) if op is not None else None
    return [
        _measure_op(
            decl,
            by_basis[b] if by_basis else _measures_for(decl, b)[0],
            role,
        )
        for b, role in zip(bases, _LABELS)
    ]


def _tag(prep: str, word) -> str:
    return "+".join([_slug(prep), *word])


def _mp_inputs(decl):
    """Per-role inputs of `mp-truth`: each prep, then the first prep that
    resets, followed by one gate: the first non-identity Pauli and the first
    two non-Pauli gates. Each is `(prep, gate word)`."""
    base = next(p for p in decl.preps if not p.fresh_only)
    paulis = [g for g in decl.gates if _is_pauli(g) and g.clifford != "I"]
    others = [g for g in decl.gates if not _is_pauli(g)]
    gates = paulis[:1] + others[:2]
    return [(p.key, ()) for p in decl.preps] + [
        (base.key, (g.key,)) for g in gates
    ]


def mp_truth_ops(decl, op, a, b, qec, reps: int) -> list[Op]:
    """Inputs on both roles, the op `reps` times, then a readout of each
    role in its input's basis; QEC rounds on both roles around the op."""
    ops = _input_ops("L0", *a) + _input_ops("L1", *b)
    if qec is not None:
        ops += _qecs(decl, qec)
    for _ in range(reps):
        ops.append(_op_step(op))
        if qec is not None:
            ops += _qecs(decl, qec)
    bases = (_state_basis(decl, *a), _state_basis(decl, *b))
    return ops + _measures(decl, bases, op)


def _mp_truth_cases(decl: CodepackDeclaration) -> list[Case]:
    cases = []
    for config in decl.configs:
        if "truth" not in config.classes:
            continue
        for op in _mp_ops(decl, config):
            for a in _mp_inputs(decl):
                for b in _mp_inputs(decl):
                    for qec, reps in [(None, 1)] * (not op.needs_qec) + [
                        (q, r) for q in decl.qec for r in (1, 2)
                    ]:
                        ops = mp_truth_ops(decl, op, a, b, qec, reps)
                        if not deterministic_outcome(decl, ops):
                            continue
                        detail = "-".join(
                            [
                                op.name,
                                _tag(*a),
                                _tag(*b),
                                "noqec" if qec is None else _slug(qec),
                                f"x{reps}",
                            ]
                        )
                        cases.append(
                            _case(
                                decl,
                                config.name,
                                "truth",
                                "mp-truth",
                                detail,
                                _op_big(decl, config, op),
                                backend=config.backends[0].name,
                                template=tuple(ops),
                                placement=(),
                            )
                        )
    return cases


def _mp_track_variants(decl, op, q: str):
    """`(name, ops between the inputs and the readout)`, one blank error."""
    opop = _op_step(op)
    qecs = _qecs(decl, q)
    out = [("pre", [_ERR] + qecs + [opop] + qecs)]
    if not op.needs_qec:
        out.append(("post", [opop, _ERR] + qecs))
    out += [
        ("twice", [_ERR] + qecs + [opop] + qecs + [opop] + qecs),
    ]
    for gate in (g for g in decl.gates if not _is_pauli(g)):
        for role in _LABELS:
            after = [opop, Op("gate", gate.key, (), (role,))]
            out.append(
                (f"then-{role}-{gate.key}", [_ERR] + qecs + after + qecs)
            )
    return out


def _mp_track_cases(decl: CodepackDeclaration) -> list[Case]:
    cases = []
    inputs = [x for x in prep_inputs(decl) if not _prep(decl, x[0]).fresh_only]
    for config in decl.configs:
        if "tracking" not in config.classes:
            continue
        for op in _mp_ops(decl, config):
            for a, b in itertools.product(inputs, repeat=2):
                head = _input_ops("L0", a[0], a[2])
                head += _input_ops("L1", b[0], b[2])
                tail = _measures(decl, (a[1], b[1]), op)
                for q in decl.qec:
                    for name, mid in _mp_track_variants(decl, op, q):
                        ops = head + mid + tail
                        if not deterministic_outcome(decl, ops):
                            continue
                        detail = "-".join(
                            [
                                op.name,
                                _tag(a[0], a[2]),
                                _tag(b[0], b[2]),
                                _slug(q),
                                name,
                            ]
                        )
                        cases.append(
                            _case(
                                decl,
                                config.name,
                                "tracking",
                                "mp-track",
                                detail,
                                _op_big(decl, config, op),
                                template=tuple(ops),
                                errors=1,
                                error_roles=_LABELS,
                            )
                        )
    return cases


def _measures_a_patch(op) -> bool:
    """Whether the op itself destructively measures one of its patches."""
    return any(
        isinstance(step, logical.Meas) and step.key == OUTCOME_KEY
        for step in op.script
    )


def _mp_reuse_cases(decl: CodepackDeclaration) -> list[Case]:
    """Ops that measure a patch: run the op, read both roles out, re-prep
    them and run it again, with one error before the first op."""
    cases = []
    inputs = [(p.key, ()) for p in decl.preps if not p.fresh_only]
    for config in decl.configs:
        if "tracking" not in config.classes:
            continue
        for op in (o for o in _mp_ops(decl, config) if _measures_a_patch(o)):
            for a, b in itertools.product(inputs, repeat=2):
                bases = (_state_basis(decl, *a), _state_basis(decl, *b))
                head = _input_ops("L0", *a) + _input_ops("L1", *b)
                for q in decl.qec:
                    qecs = _qecs(decl, q)
                    run = [_op_step(op)] + qecs
                    ops = head + [_ERR] + qecs + run
                    ops += _measures(decl, bases, op) + head + qecs + run
                    ops += _measures(decl, bases, op)
                    if not deterministic_outcome(decl, ops):
                        continue
                    detail = "-".join([op.name, _tag(*a), _tag(*b), _slug(q)])
                    cases.append(
                        _case(
                            decl,
                            config.name,
                            "tracking",
                            "mp-reuse",
                            detail,
                            _op_big(decl, config, op),
                            template=tuple(ops),
                            errors=1,
                            error_roles=_LABELS,
                        )
                    )
    return cases


def _independence_words(decl):
    """`(prep, input word, gate, readout basis)`: each resetting prep's
    inputs, then one gate (the first non-Pauli, the first non-identity
    Pauli) and a readout in the resulting basis."""
    resetting = {p.key for p in decl.preps if not p.fresh_only}
    paulis = [g for g in decl.gates if _is_pauli(g) and g.clifford != "I"]
    others = [g for g in decl.gates if not _is_pauli(g)]
    out = []
    for prep, basis, word in prep_inputs(decl):
        if prep not in resetting:
            continue
        for gate in others[:1] + paulis[:1]:
            full = word + (gate.key,)
            out.append((prep, word, gate.key, _state_basis(decl, prep, full)))
    return out


def _independence_cases(decl: CodepackDeclaration) -> list[Case]:
    cases: list[Case] = []
    if not decl.preps:
        return cases
    words = _independence_words(decl)
    for config in decl.configs:
        if "independence" not in config.classes:
            continue
        for wa, wb in itertools.product(words, repeat=2):
            if wa[:2] == wb[:2] or wa[2] == wb[2]:
                continue
            for ra, rb in (_LABELS, _LABELS[::-1]):
                q = decl.qec[0]
                a = _input_ops(ra, wa[0], wa[1]) + [_ERR]
                a += [Op("qec", q, (), (ra,)), Op("gate", wa[2], (), (ra,))]
                a += [Op("qec", q, (), (ra,))]
                a += [_measure_op(decl, _measures_for(decl, wa[3])[0], ra)]
                b = _input_ops(rb, wb[0], wb[1])
                b += [Op("qec", q, (), (rb,)), Op("gate", wb[2], (), (rb,))]
                b += [Op("qec", q, (), (rb,))]
                b += [_measure_op(decl, _measures_for(decl, wb[3])[0], rb)]
                woven = [
                    x
                    for pair in itertools.zip_longest(a, b)
                    for x in pair
                    if x is not None
                ]
                for sched, ops in (("interleaved", woven), ("blocks", a + b)):
                    detail = "-".join(
                        [
                            f"err{ra}",
                            f"{_tag(wa[0], wa[1])}+{wa[2]}",
                            f"{_tag(wb[0], wb[1])}+{wb[2]}",
                            sched,
                        ]
                    )
                    cases.append(
                        _case(
                            decl,
                            config.name,
                            "independence",
                            "independence",
                            detail,
                            config.big,
                            template=tuple(ops),
                            errors=1,
                            error_roles=(ra,),
                        )
                    )
    return cases


def _ft_inputs(decl, sweep, own: bool):
    """`(tag, head ops, ops after the sweep)` per input of a sweep.

    A sweep that is its own prep has no head; the gate word reaching each
    basis follows it. Otherwise inputs are each basis of every resetting
    prep (both roles in one basis for a two-role sweep).
    """
    if own:
        step = next(s for s in sweep.script if isinstance(s, logical.Prep))
        out: list[tuple[str, list[Op], list[Op]]] = []
        for basis in "ZX":
            word = _shortest_word(decl, step, basis)
            if word is not None:
                # A measure straight after the bare check circuit reads the
                # check's outcomes instead of the measure's own, so an
                # identity gate (when declared) stands between them.
                ident = [g.key for g in decl.gates if g.clifford == "I"]
                post = [Op("gate", g) for g in word or ident[:1]]
                out.append((f"own+{'+'.join(word) or 'none'}", [], post))
        return out
    inputs = [x for x in prep_inputs(decl) if not _prep(decl, x[0]).fresh_only]
    if len(sweep.roles) == 1:
        return [(_tag(p, w), _input_ops("L0", p, w), []) for p, _, w in inputs]
    return [
        (
            f"{_tag(a[0], a[2])}-{_tag(b[0], b[2])}",
            _input_ops("L0", a[0], a[2]) + _input_ops("L1", b[0], b[2]),
            [],
        )
        for a, b in itertools.product(inputs, repeat=2)
        if a[1] == b[1]
    ]


def _ft_tails(decl, sweep, has_meas: bool):
    """`(name, measure ops)`: none if the sweep measures itself."""
    if has_meas:
        return [("own", [])]
    if len(sweep.roles) == 1:
        return [
            (_mtag(m), [_measure_op(decl, i)])
            for i, m in enumerate(decl.measures)
        ]
    return [
        (f"readout{b}", _measures(decl, (b, b)))
        for b in "ZX"
        if _measures_for(decl, b)
    ]


def _ft_cases(decl: CodepackDeclaration) -> list[Case]:
    cases: list[Case] = []
    for config in decl.configs:
        if "ft" not in config.classes:
            continue
        for sweep in decl.sweeps:
            cases += _sweep_cases(decl, config, sweep)
    return cases


def _sweep_cases(decl, config, sweep) -> list[Case]:
    own = any(isinstance(s, logical.Prep) for s in sweep.script)
    has_meas = any(isinstance(s, logical.Meas) for s in sweep.script)
    q = decl.qec[0]
    roles = _LABELS[: len(sweep.roles)]
    step = Op("sweep", sweep.name, (), roles)
    suffixes: list[tuple[str, tuple]] = [("none", ())]
    if sweep.suffix:
        suffixes += [
            (f"W{g.key}", gate_word(decl, g.key))
            for g in decl.gates
            if not _is_pauli(g)
        ]
    seeds: list[tuple[str, list[Op]]] = [("clean", [])]
    if decl.error_pair and not own:
        errs = [
            Op("error", p, (("data", d),), (roles[0],))
            for d, p in decl.error_pair
        ]
        qecs = [Op("qec", q, (), (r,)) for r in roles]
        seeds.append(("tracked", errs[:1] + qecs + errs[1:] + qecs))
    cases = []
    for tag, head, post in _ft_inputs(decl, sweep, own):
        for tname, tail in _ft_tails(decl, sweep, has_meas):
            for sname, word in suffixes:
                for seed, seeding in seeds:
                    ops = head + seeding + [step] + post
                    ops += _gates_qec(word, q) + tail
                    if not deterministic_outcome(decl, ops):
                        continue
                    detail = "-".join([sweep.name, tag, tname, sname, seed])
                    cases.append(
                        _case(
                            decl,
                            config.name,
                            "ft",
                            "ft",
                            detail,
                            config.big,
                            ops=tuple(ops),
                        )
                    )
    return cases


def deterministic_outcome(decl, ops) -> bool:
    """Whether some measure of a word has a deterministic outcome.

    A tracking case needs one: a frame bit that is wrong shows in it, while
    a random outcome is only postselected.
    """
    script, _, _ = plan_word(decl, ops)
    sim = logical.LogicalSim(logical.roles_of(script))
    for step in script:
        bit = None
        if isinstance(step, logical.Meas):
            bit = sim.deterministic_bit(step.paulis)
            if bit is not None:
                return True
            bit = 0
        sim.apply(step, bit)
    return False


def generate(decl: CodepackDeclaration) -> list[Case]:
    """Every case of a declaration, in a deterministic order."""
    cases = _static_cases(decl) + _truth_cases(decl)
    cases += _agreement_error_cases(decl) + _tracking_cases(decl)
    cases += _mp_truth_cases(decl) + _mp_track_cases(decl)
    cases += _mp_reuse_cases(decl)
    cases += _independence_cases(decl) + _ft_cases(decl)
    return sorted(cases, key=lambda c: c.id)


def split_key(key: str) -> tuple[str, str]:
    """Split a `<case-id pattern>:<check>` known-failure key."""
    pattern, _, check = key.rpartition(":")
    return pattern, check


def _bug_ids(value) -> tuple[str, ...]:
    return (value,) if isinstance(value, str) else tuple(value)


def declared_checks(
    decl: CodepackDeclaration, case_id: str
) -> dict[str, list[str]]:
    """The checks a case is declared to fail, each with its bug ids."""
    out: dict[str, set[str]] = {}
    for key, value in decl.known_failures.items():
        pattern, check = split_key(key)
        if fnmatchcase(case_id, pattern):
            out.setdefault(check, set()).update(_bug_ids(value))
    return {check: sorted(ids) for check, ids in out.items()}


def validate_declaration(decl: CodepackDeclaration) -> list[str]:
    """Problems with a declaration's known failures and reproducers."""
    problems = []
    ids = [c.id for c in generate(decl)]
    if len(set(ids)) != len(ids):
        problems.append("duplicate case ids")
    for key, value in decl.known_failures.items():
        pattern, check = split_key(key)
        if check not in CHECKS:
            problems.append(f"{key}: unknown check {check!r}")
        if not any(fnmatchcase(i, pattern) for i in ids):
            problems.append(f"{key}: pattern matches no case")
        for bug in _bug_ids(value):
            if bug not in KNOWN_BUGS:
                problems.append(f"{key}: bug id {bug!r} is not in KNOWN_BUGS")
    for alias, (case_id, checks) in decl.reproducers.items():
        if case_id not in ids:
            problems.append(f"reproducer {alias}: no case {case_id}")
            continue
        declared = declared_checks(decl, case_id)
        for check in checks:
            if check in declared:
                problems.append(
                    f"reproducer {alias}: check {check} has a known failure"
                )
    return problems


def judge(declared, failing, details: Mapping[str, Any] | None = None):
    """Pass only when the failing checks equal the declared ones."""
    declared, failing = set(declared), set(failing)
    if declared == failing:
        return
    lines = [
        f"unexpected failing checks: {sorted(failing - declared)}",
        f"declared checks that now pass: {sorted(declared - failing)}",
    ]
    for check, detail in (details or {}).items():
        lines.append(f"{check}: {detail}")
    raise AssertionError("\n".join(lines))

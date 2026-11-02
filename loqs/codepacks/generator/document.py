#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.3                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""A codepack spec as a Markdown document with Stim circuit figures.

`document(cls, generated)` renders the page and the figures it names;
`write_document` writes them under a docs folder. Like `generate`, this
module imports Stim, so the package's `__init__` doesn't import it.
"""

from __future__ import annotations

import importlib
import inspect
import re
import textwrap
from pathlib import Path
from typing import Any

import stim

from loqs.codepacks.generator.codepack import GeneratedCodepack
from loqs.codepacks.generator.gates import GATES
from loqs.codepacks.generator.schema import (
    CheckCircuit,
    Circuit,
    CodeSpec,
    Composite,
    CustomApplyFn,
    Gate,
    Layer,
    Parallel,
    RepeatUntilSuccess,
    Src,
    resolve_circuits,
)

Figures = dict[str, tuple[str, list[str]]]


def _terms(matrix: Any, names: list[str]) -> list[tuple[str, list[str]]]:
    """`(new bit, [old bits XORed])` for each row of a bit-update matrix,
    leaving out the bits it doesn't change."""
    out = []
    for i, row in enumerate(matrix):
        olds = [names[j] for j, v in enumerate(row) if v]
        if olds != [names[i]]:
            out.append((names[i], olds))
    return out


def _layers(layers: list[Layer]) -> str:
    return f"{len(layers)} layer" + ("" if len(layers) == 1 else "s")


def _slug(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", key.lower()).strip("-")


def _stim_text(
    layers: list[Layer], qubits: list[str]
) -> tuple[str, list[str]]:
    """Stim circuit text of `layers` over the qubits they touch, in the
    order of `qubits`, and those qubits' labels."""
    touched = {q for layer in layers for _, qs in layer for q in qs}
    labels = [q for q in qubits if q in touched]
    index = {q: i for i, q in enumerate(labels)}
    lines: list[str] = []
    for n, layer in enumerate(layers):
        if n:
            lines.append("TICK")
        for name, qs in layer:
            gate = (
                "MR"
                if GATES[name].kind == "instrument"
                else str(GATES[name].unitary)
            )
            lines.append(gate + " " + " ".join(str(index[q]) for q in qs))
    return "\n".join(lines), labels


FIGURE_SCALE = 0.5


def figure_svg(text: str, labels: list[str]) -> str:
    """A Stim timeline diagram of `text` with each `q<i>` line label
    replaced by `labels[i]`, sized at `FIGURE_SCALE` of its viewBox so
    every figure draws its qubit lines the same distance apart."""
    svg = str(stim.Circuit(text).diagram("timeline-svg"))
    m = re.match(r'<svg viewBox="0 0 (\d+) (\d+)"', svg)
    if not m:
        raise RuntimeError("expected the diagram to open with its viewBox")
    w, h = (round(int(v) * FIGURE_SCALE) for v in m.groups())
    svg = svg.replace("<svg ", f'<svg width="{w}" height="{h}" ', 1)
    for i, label in enumerate(labels):
        old = f">q{i}<"
        if svg.count(old) != 1:
            raise RuntimeError(
                f"expected one {old!r} line label in the diagram, "
                f"found {svg.count(old)}"
            )
        svg = svg.replace(old, f">{label}<")
    return svg


def _src_line(src: Src) -> str:
    if src.kind == "paper":
        line = f"From the paper: {src.cite}"
        if src.reason:
            line += f". {src.reason[:1].upper()}{src.reason[1:]}"
        return line
    line = f"Ours: {src.reason}"
    if src.cite:
        line += f" (see {src.cite})"
    return line


def _paper_figure(src: Src, spec: CodeSpec) -> list[str]:
    """The paper's figures, if `src` names any."""
    if not src.figures:
        return []
    if not spec.figure_credit:
        raise ValueError(
            f"{src.figures!r} are paper figures, but the spec has no "
            "figure_credit"
        )
    alt = re.sub(r"\[@[^\]]*\]\s*", "", src.cite)
    out = ["", "**Paper:**"]
    for fig in src.figures:
        out += ["", f"![{alt}]({spec.name}/paper/{fig})"]
    return out


def _source(fn: str) -> list[str]:
    """A custom apply function's source, folded."""
    module, _, qual = fn.partition(":")
    code = inspect.getsource(getattr(importlib.import_module(module), qual))
    return [
        "",
        '<details markdown="1">',
        f"<summary><b>LoQS source:</b> <code>{qual}</code></summary>",
        "",
        "```python",
        textwrap.dedent(code).rstrip(),
        "```",
        "",
        "</details>",
    ]


def _bits(items: list[tuple[str, list[str]]]) -> list[str]:
    out = []
    for new, olds in items:
        rhs = "old " + " ^ ".join(f"`{o}`" for o in olds) if olds else "0"
        out.append(f"- new `{new}` = {rhs}")
    return out


def _fn_link(fn: str) -> str:
    module, _, qual = fn.partition(":")
    return f"[`{qual}`](api:{module}.{qual})"


def _ticks(names: Any) -> str:
    return ", ".join(f"`{n}`" for n in names)


def _runs(keys: Any) -> str:
    return "runs " + ", then ".join(f"`{k}`" for k in keys)


def _flowchart(it: Any, spec: CodeSpec) -> list[str]:
    """A Mermaid flowchart of a composite's parts, a repeat-until-success
    loop or a branching custom apply function, following the branches of
    the custom apply functions it reaches; nothing for other items. A
    composite or loop is drawn as a box, titled with its key, around the
    steps it is made of."""
    branches = {
        i.key: i.branches
        for i in spec.items
        if isinstance(i, CustomApplyFn) and i.branches
    }
    ids: dict[str, str] = {}
    edges: dict[tuple[str, str, str], None] = {}

    def node(key: str) -> str:
        if key not in ids:
            ids[key] = f"n{len(ids)}"
        return ids[key]

    def walk(seq: list[str], prev: str, label: str) -> None:
        for i, key in enumerate(seq):
            if prev:
                edges[(prev, node(key), label)] = None
            label = ""
            if key in branches:
                for cond, keys in branches[key]:
                    walk(list(keys) + seq[i + 1 :], node(key), cond)
                return
            prev = node(key)
        edges[(prev, "next", label)] = None

    inner: list[str] = []
    outer = ['  next(["next instruction"])']
    if isinstance(it, Composite):
        walk(list(it.parts), "", "")
    elif isinstance(it, CustomApplyFn) and it.branches:
        walk([it.key], "", "")
    elif isinstance(it, RepeatUntilSuccess):
        reset = node(f"{it.key} reset")
        walk(["clear tracking data", *it.body], reset, "")
        edges.pop(next(e for e in edges if e[1] == "next"))
        expected = ", ".join(
            f"{q} = {''.join(map(str, v))}" for q, v in it.expected
        )
        inner.append(f'  check{{"{expected}?"}}')
        outer.append('  fail(["error after max_repeats tries"])')
        edges[(ids[it.body[-1]], "check", "")] = None
        edges[("check", "next", "yes")] = None
        edges[("check", reset, "no, tries left")] = None
        edges[("check", "fail", "no, max_repeats tries")] = None
    else:
        return []
    inner[:0] = [
        f'  {nid}{{"{key}"}}' if key in branches else f'  {nid}["{key}"]'
        for key, nid in ids.items()
    ]
    if not isinstance(it, CustomApplyFn):
        inner = [f'  subgraph box["{it.key}"]', *inner, "  end"]
    out = ["", "**LoQS flowchart:**", "", "```mermaid", "flowchart TD"]
    out += inner + outer
    for a, b, label in edges:
        out.append(f'  {a} -->|"{label}"| {b}' if label else f"  {a} --> {b}")
    return out + ["```"]


def _value(v: Any) -> str:
    return ", ".join(map(str, v)) if isinstance(v, tuple) else str(v)


def _groups(spec: CodeSpec, tables: dict) -> list[list[Any]]:
    """The items in sections: a run of check circuits, or of gates whose
    maps are the identity, with one source and no role shares one."""
    roles = {r.key for r in spec.roles}

    def kind(it: Any) -> str:
        if it.key in roles:
            return ""
        if isinstance(it, CheckCircuit):
            return "check"
        if isinstance(it, Gate) and tables["gates"][it.key]["trivial"]:
            return "gate"
        return ""

    out: list[list[Any]] = []
    for it in spec.items:
        last = out[-1][-1] if out else None
        if last and kind(it) and kind(it) == kind(last) and it.src == last.src:
            out[-1].append(it)
        else:
            out.append([it])
    return out


def _describe(
    it: Any, circuits: dict[str, tuple[list[Layer], list[str]]], tables: dict
) -> tuple[list[str], tuple[str, list[Layer], list[str]] | None]:
    """An item's sentences, and its figure `(key, layers, qubits)` if it
    has a circuit."""
    if isinstance(it, Composite):
        runs = _runs(it.parts)
        return [f"{runs[:1].upper()}{runs[1:]}."], None
    if isinstance(it, CustomApplyFn):
        bullets = []
        if it.data:
            given = ", ".join(f"`{k}` = `{_value(v)}`" for k, v in it.data)
            bullets.append(f"- takes {given}")
        if it.derived:
            tables_read = _ticks(v for _, v in it.derived)
            bullets.append(f"- reads the generated tables {tables_read}")
        if it.scratch:
            bullets.append(
                f"- writes {_ticks(it.scratch)}, which a prep clears"
            )
        for cond, keys in it.branches:
            then = _runs(keys) if keys else "runs nothing more"
            bullets.append(f"- when {cond}: {then}")
        head = f"A hand-written apply function, {_fn_link(it.fn)}"
        return ([head + ":", "", *bullets] if bullets else [head + "."]), None
    key = it.key
    if isinstance(it, Gate):
        key = it.circuit_key or it.key
        parts = [f"The logical `{it.logical}` gate, as the circuit `{key}`."]
        if tables["gates"][it.key]["trivial"]:
            parts.append(
                "Its frame and syndrome maps are the identity, so no "
                "bookkeeping step runs."
            )
    elif isinstance(it, RepeatUntilSuccess):
        key = f"{it.key} reset"
        expected = ", ".join(
            f"`{q}` = {''.join(map(str, v))}" for q, v in it.expected
        )
        parts = [
            "Resets the patch with the circuit below and clears its "
            f"tracking data, then runs {_ticks(it.body)} until it reads "
            f"{expected}."
        ]
    elif isinstance(it, CheckCircuit):
        parts = [
            f"A check of stabilizer `{it.stabilizer}` with auxiliary qubit "
            f"`{it.aux}`, in {_layers(circuits[key][0])}."
        ]
    elif isinstance(it, Parallel):
        parts = [
            f"The circuits {_ticks(it.parts)} merged layer by layer from "
            "layer 0."
        ]
    else:
        assert isinstance(it, Circuit)
        parts = [f"A circuit of {_layers(circuits[key][0])}."]
    return parts, (key, *circuits[key])


def _group_section(
    group: list[Any],
    circuits: dict[str, tuple[list[Layer], list[str]]],
    figure: Any,
    reads: Any,
    spec: CodeSpec,
) -> list[str]:
    """One section for a run of like items, with a row each."""
    keys = [it.key for it in group]
    title = ", ".join(keys) if len(keys) <= 3 else f"{keys[0]} … {keys[-1]}"
    out = ["", f"### {title}", ""]
    if isinstance(group[0], Gate):
        names = [it.circuit_key or it.key for it in group]
        out.append(
            f"The logical {_ticks(it.logical for it in group)} gates, as the "
            f"circuits {_ticks(names)}. Their frame and syndrome maps are "
            "the identity, so no bookkeeping step runs."
        )
        rows = ["| Gate | Logical | LoQS circuit |", "|---|---|---|"]
        for it, name in zip(group, names):
            rows.append(
                f"| `{it.key}` | `{it.logical}` | "
                f"{figure(name, *circuits[name])} |"
            )
    else:
        out.append(
            "Each checks one stabilizer with one auxiliary qubit; the Reads "
            "column is derived."
        )
        rows = ["| Check | Reads | LoQS circuit |", "|---|---|---|"]
        for it in group:
            rows.append(
                f"| `{it.key}` | {reads(it.key)} | "
                f"{figure(it.key, *circuits[it.key])} |"
            )
    out += ["", _src_line(group[0].src), "", *rows]
    return out + _paper_figure(group[0].src, spec)


def document(
    cls: type[GeneratedCodepack], generated: dict
) -> tuple[str, dict[str, tuple[str, list[str]]]]:
    """The Markdown page of `cls.spec` and its figures, `{path: (Stim
    circuit text, qubit labels)}`, with `generated` as `generate(spec)`
    returns it."""
    spec = cls.spec
    f = spec.facts
    tables = generated["tables"]
    figures: Figures = {}
    slugs: dict[str, str] = {}

    def figure(key: str, layers: list[Layer], qubits: list[str]) -> str:
        slug = _slug(key)
        if slug in slugs:
            raise ValueError(
                f"figures {slugs[slug]!r} and {key!r} share the file name "
                f"{slug}.svg"
            )
        slugs[slug] = key
        path = f"{spec.name}/{slug}.svg"
        figures[path] = _stim_text(layers, qubits)
        return f"![{key}]({path})"

    circuits = resolve_circuits(spec)
    frame_names = [f"frame.{b}" for b in f.frame_layout]
    synd_names = list(f.stabilizer_names)
    roles = {r.key: r for r in spec.roles}
    out: list[str] = [f"# {spec.title}", ""]
    out.append(
        f"Generated from the spec `{spec.name}` of "
        f"[](api:{cls.__module__}.{cls.__name__}). Do not edit: "
        "regenerate with `python -m loqs.codepacks.generator "
        f"{cls.__module__}:{cls.__name__}`."
    )

    out += [
        "",
        "## How to read this page",
        "",
        "- Each circuit figure is a Stim timeline diagram: time runs left "
        "to right, each spec layer is one column (or several columns under a "
        "bracket when its gates overlap on the page), and each line is named "
        "by its qubit. `Imrz`, the measure-and-reset, is drawn as `MR`.",
        "- Each element shows what LoQS does (**LoQS**: its circuit, "
        "flowchart, rows, tracking updates or hand-written source) and, "
        "where the paper has one, the paper's own figure, table or "
        "pseudocode (**Paper**) to compare it with.",
        "- *From the paper* marks an element taken from the paper, with its "
        "citation. *Ours* marks our own choice, with its reason. *Derived* "
        "marks a statement the generator computed with Stim from the "
        "circuits and the declared logical action.",
        "- Tracking rules read `new bit = old bit ^ old bit ...`, `^` being "
        "XOR: a gate updates the logical frame, the stored syndrome and each "
        "data qubit's Pauli frame with them. Bits not listed are unchanged.",
        "- A flowchart shows where a hand-written apply function branches: "
        "each labelled arrow is a condition, and the instructions it leads "
        "to are the ones the function pushes onto the stack.",
        "- [Reading a codepack spec](../markdown/codepack-specs.md) explains "
        "each element and the generated artifact.",
    ]
    if spec.figure_credit:
        out.append(f"- {spec.figure_credit}")

    out += ["", "## The code", ""]
    out.append(f"- Data qubits: {', '.join(f'`{q}`' for q in f.data)}")
    out.append(f"- Auxiliary qubits: {', '.join(f'`{q}`' for q in f.aux)}")
    out += ["", "| Stabilizer | Pauli string |", "|---|---|"]
    out += [
        f"| `{n}` | `{s}` |" for n, s in zip(f.stabilizer_names, f.stabilizers)
    ]
    out += [""]
    for x, z in f.logicals:
        out.append(f"- Logical `X_L` = `{x}`, `Z_L` = `{z}`")
    out.append(f"- Source: {_src_line(f.src)}")
    out.append(
        f"- Tracking: the logical Pauli frame is `{f.tracking.logical_frame}`"
        f", holding the bits {', '.join(f'`{b}`' for b in f.frame_layout)} "
        "in that order; the stored syndrome is "
        f"`{', '.join(f.tracking.syndrome)}`, one bit per stabilizer in the "
        "order above."
    )
    if f.conventions_src:
        out.append(f"- Conventions: {_src_line(f.conventions_src)}")
    out += _paper_figure(f.src, spec)

    def reads(key: str) -> str:
        sidx = tables["syndrome_index"].get(key, {})
        return ", ".join(
            f"`{a}` as `{synd_names[i]}`" for a, i in sidx.items()
        )

    out += ["", "## Instructions"]
    for group in _groups(spec, tables):
        if len(group) > 1:
            out += _group_section(group, circuits, figure, reads, spec)
            continue
        it = group[0]
        out += ["", f"### {it.key}", ""]
        parts, drawn = _describe(it, circuits, tables)
        if it.key in roles:
            r = roles[it.key]
            detail = r.state or r.basis
            parts.append(
                f"Role for the validation harness: {r.role}"
                + (f" ({detail})" if detail else "")
                + "."
            )
        if reads(it.key):
            parts.append(f"Derived: it reads {reads(it.key)}.")
        if isinstance(it, CustomApplyFn):
            out += parts
        else:
            out.append(" ".join(parts))
        out += ["", _src_line(it.src)]
        if drawn:
            out += ["", "**LoQS circuit:**", "", figure(*drawn)]
        out += _flowchart(it, spec)
        if isinstance(it, CustomApplyFn):
            out += _source(it.fn)
        out += _paper_figure(it.src, spec)
        if isinstance(it, Gate) and not tables["gates"][it.key]["trivial"]:
            t = tables["gates"][it.key]
            out += ["", "**LoQS tracking updates** (derived):", ""]
            out += _bits(_terms(t["frame_map"], frame_names))
            out += _bits(_terms(t["syndrome_map"], synd_names))
            rule = ", ".join(f"`{a}`→`{b}`" for a, b in t["per_qubit"].items())
            out += [
                f"- each data qubit's Pauli frame entry maps {rule} "
                "(up to sign)"
            ]
            out += [f"- bookkeeping instruction: `{it.update_key}`"]

    out += ["", "## Lookup tables"]
    for tbl in spec.tables:
        out += ["", f"### {tbl.name}", ""]
        out.append(
            f"A {tbl.kind} table of {len(tbl.rows)} rows. "
            + _src_line(tbl.src)
        )
        out += ["", "**LoQS rows:**", ""]
        for row in tables["lut"][tbl.name]:
            out.append(f"- `{row}`")
        out += _paper_figure(tbl.src, spec)

    out += ["", "## Two-patch gates"]
    for g in spec.two_patch:
        t = tables["two_patch"][g.key]
        out += ["", f"### {g.key}", ""]
        sentence = (
            f"The logical `{g.logical}` gate across the patches "
            + ", ".join(f"`{r}`" for r in g.roles)
            + "."
        )
        out.append(sentence)
        out += ["", _src_line(g.src)]
        jq = [f"{r}.{q}" for r in g.roles for q in f.qubits]
        layers = [
            [(nm, tuple(qs)) for nm, qs in layer] for layer in t["layers"]
        ]
        out += ["", "**LoQS circuit:**", "", figure(g.key, layers, jq)]
        fn = [f"{r}.{b}" for r in g.roles for b in frame_names]
        sn = [f"{r}.{s}" for r in g.roles for s in synd_names]
        out += ["", "**LoQS joint tracking updates** (derived):", ""]
        out += _bits(_terms(t["frame_map"], fn))
        out += _bits(_terms(t["syndrome_map"], sn))

    return "\n".join(out) + "\n", figures


def write_document(
    cls: type[GeneratedCodepack], generated: dict, docs_dir: Path
) -> Path:
    """Write `cls`'s page to `<docs_dir>/codepacks/<name>.md` and its
    figures to `<docs_dir>/codepacks/<name>/`, replacing the folder's old
    `*.svg` files, and return the page's path."""
    page, figures = document(cls, generated)
    root = Path(docs_dir) / "codepacks"
    folder = root / cls.spec.name
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.svg"):
        old.unlink()
    for rel, (text, labels) in figures.items():
        (root / rel).write_text(figure_svg(text, labels))
    path = root / f"{cls.spec.name}.md"
    path.write_text(page)
    return path

"""Tester for loqs.codepacks.generator.document"""

import dataclasses
import re
from pathlib import Path

import pytest

import loqs

pytest.importorskip("stim")

from loqs.codepacks.generator.document import (  # noqa: E402
    FIGURE_SCALE,
    _groups,
    _terms,
    _value,
    document,
    figure_svg,
)
from loqs.codepacks.generator.generate import generate  # noqa: E402
from loqs.codepacks.generator.schema import (  # noqa: E402
    Composite,
    CustomApplyFn,
    RepeatUntilSuccess,
    Gate,
    resolve_circuits,
)
from loqs.codepacks.steane import SteaneCodepack  # noqa: E402


@pytest.fixture(scope="module")
def steane_page():
    spec = SteaneCodepack.spec
    generated = generate(spec)
    page, figures = document(SteaneCodepack, generated)
    return spec, generated, page, figures


def test_spec_document(steane_page):
    spec, generated, page, figures = steane_page
    # Every item, table and two-patch gate has a section or, in a shared
    # section, a row
    for key in (
        [it.key for it in spec.items]
        + [t.name for t in spec.tables]
        + [g.key for g in spec.two_patch]
    ):
        assert f"### {key}\n" in page or f"\n| `{key}` |" in page, key

    # Like items share a section; a gate with tracking updates doesn't
    groups = [[it.key for it in g] for g in _groups(spec, generated["tables"])]
    assert ["X", "Y", "Z"] in groups and ["H"] in groups
    assert [f"Unflagged S{i} Check" for i in range(1, 7)] in groups

    # Every circuit-valued item has a figure
    for key in resolve_circuits(spec):
        slug = re.sub(r"[^a-z0-9]+", "-", key.lower()).strip("-")
        assert f"{spec.name}/{slug}.svg" in figures, key
    assert f"{spec.name}/transversal-cx.svg" in figures

    # Each paper figure the spec names is drawn
    srcs = [spec.facts.src] + [it.src for it in spec.items]
    srcs += [t.src for t in spec.tables]
    named = {fig for src in srcs for fig in src.figures}
    assert named
    for fig in named:
        assert f"({spec.name}/paper/{fig})" in page, fig

    # Paper cites name entries of the bibliography
    bib = (Path(loqs.__file__).parents[1] / "docs/references.bib").read_text()
    keys = set(re.findall(r"\[@([^\]]+)\]", page))
    assert keys
    for key in keys:
        assert f"{{{key}," in bib, key

    # Custom apply functions link to their module.qualname
    for it in spec.items:
        if isinstance(it, CustomApplyFn):
            module, _, qual = it.fn.partition(":")
            assert f"(api:{module}.{qual})" in page, it.key
            assert f"\ndef {qual}(" in page, it.key
            section = page.split(f"\n### {it.key}\n", 1)[1].split("\n### ")[0]
            for k, v in it.data:
                assert f"`{k}` = `{_value(v)}`" in section, (it.key, k)

    # Figure line labels are the circuit's qubits
    text, labels = figures[f"{spec.name}/h-circuit.svg"]
    svg = figure_svg(text, labels)
    assert all(f">{q}<" in svg for q in spec.facts.data)
    assert not re.search(r">q\d+<", svg)
    with pytest.raises(RuntimeError):
        figure_svg(text, labels[:-1] + ["A9"] + ["extra"])

    # Every figure is drawn at the same scale of its viewBox
    for path in (f"{spec.name}/x.svg", f"{spec.name}/transversal-cx.svg"):
        svg = figure_svg(*figures[path])
        w, h, vw, vh = map(
            int,
            re.match(
                r'<svg width="(\d+)" height="(\d+)" viewBox="0 0 (\d+) (\d+)"',
                svg,
            ).groups(),
        )
        assert (w, h) == (round(vw * FIGURE_SCALE), round(vh * FIGURE_SCALE))

    # A paper figure needs the spec's credit
    bare = type(
        "Bare",
        (SteaneCodepack,),
        {"spec": dataclasses.replace(spec, figure_credit="")},
    )
    with pytest.raises(ValueError):
        document(bare, generated)

    # Gates with bookkeeping state it
    gates = [it for it in spec.items if isinstance(it, Gate)]
    assert gates and "LoQS tracking updates" in page


def test_update_terms(steane_page):
    _, generated, _, _ = steane_page
    gates = generated["tables"]["gates"]
    frame = ["fz", "fx"]
    synd = [f"s{i}" for i in range(6)]
    assert _terms(gates["H"]["frame_map"], frame) == [
        ("fz", ["fx"]),
        ("fx", ["fz"]),
    ]
    assert _terms(gates["S"]["frame_map"], frame) == [("fz", ["fz", "fx"])]
    assert _terms(gates["S"]["syndrome_map"], synd) == [
        (f"s{i}", [f"s{i}", f"s{i + 3}"]) for i in range(3)
    ]
    cx = generated["tables"]["two_patch"]["Transversal CX"]
    names = ["cz", "cx", "tz", "tx"]
    assert _terms(cx["frame_map"], names) == [
        ("cz", ["cz", "tz"]),
        ("tx", ["cx", "tx"]),
    ]


def _charts(page):
    """Each section's Mermaid chart, by its heading."""
    out = {}
    for section in page.split("\n### ")[1:]:
        key, _, body = section.partition("\n")
        if "```mermaid\n" in body:
            chart = body.split("```mermaid\n", 1)[1].split("```", 1)[0]
            out[key] = (
                chart,
                {k: n for n, k in re.findall(r'(n\d+)[\[{]"([^"]+)"', chart)},
            )
    return out


def test_flowcharts(steane_page):
    spec, _, page, _ = steane_page
    charts = _charts(page)
    # Each branching function's own chart, and every composite's chart,
    # draws each branch it reaches
    for it in spec.items:
        if isinstance(it, (Composite, RepeatUntilSuccess)):
            assert f'subgraph box["{it.key}"]' in charts[it.key][0], it.key
        if not (isinstance(it, CustomApplyFn) and it.branches):
            continue
        reaching = [it.key] + [
            k
            for k, (_, ids) in charts.items()
            if it.key in ids and k != it.key
        ]
        for key in reaching:
            chart, ids = charts[key]
            for cond, keys in it.branches:
                to = ids[keys[0]] if keys else "next"
                assert f'{ids[it.key]} -->|"{cond}"| {to}' in chart, (
                    key,
                    cond,
                )

    # The repeat-until-success loop retries from its reset
    rus = next(i for i in spec.items if isinstance(i, RepeatUntilSuccess))
    chart, ids = charts[rus.key]
    reset = ids[f"{rus.key} reset"]
    assert f"{ids[rus.body[-1]]} --> check" in chart
    assert 'check -->|"yes"| next' in chart
    assert re.search(rf'check -->\|"[^"]*"\| {reset}\n', chart)

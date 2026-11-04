"""Tester for loqs.codepacks.generator.codepack"""

import dataclasses
import importlib
import re
import sys
import types
from pathlib import Path

import pytest

import loqs
from loqs.backends.circuit.listcircuit import ListPhysicalCircuit
from loqs.codepacks.generator import StaleArtifactError
from loqs.codepacks.generator.codepack import shipped_codepacks, tables_hash
from loqs.codepacks.steane import SteaneCodepack
from loqs.codepacks.steane import _generated as steane_artifact


def _changed_spec(**changes):
    items = tuple(
        dataclasses.replace(it, **changes) if it.key == "S" else it
        for it in SteaneCodepack.spec.items
    )
    return dataclasses.replace(SteaneCodepack.spec, items=items)


@pytest.fixture
def fake_artifact(monkeypatch):
    """A copy of Steane's artifact as a module `fake_artifact`."""
    mod = types.ModuleType("fake_artifact")
    mod.INPUT_HASH = steane_artifact.INPUT_HASH
    mod.TABLES_HASH = steane_artifact.TABLES_HASH
    mod.GENERATED = steane_artifact.GENERATED
    monkeypatch.setitem(sys.modules, "fake_artifact", mod)
    # A re-import after the loader pops the entry must find it again
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name, _orig=importlib.import_module: (
            sys.modules.setdefault(name, mod)
            if name == "fake_artifact"
            else _orig(name)
        ),
    )
    return mod


def _locked(spec=None, artifact="fake_artifact"):
    class Locked(SteaneCodepack):
        pass

    Locked.artifact = artifact
    if spec is not None:
        Locked.spec = spec
    return Locked


def test_locked_construction_refuses(fake_artifact):
    # Control: the untouched copy builds
    _locked()(circuit_backend=ListPhysicalCircuit)

    with pytest.raises(StaleArtifactError):  # stale
        _locked(spec=_changed_spec(logical="S_DAG"))()

    fake_artifact.GENERATED = {"tables": {}, "checks": {}}  # edited
    with pytest.raises(StaleArtifactError):
        _locked()()

    with pytest.raises(StaleArtifactError):  # missing
        _locked(artifact="loqs.codepacks.steane._no_such_module")()


def test_unlocked_subclass_derives():
    pytest.importorskip("stim")
    spec = _changed_spec(name="Renamed S")

    class Unlocked(SteaneCodepack):
        locked = False
        artifact = "loqs.codepacks.steane._no_such_module"

    Unlocked.spec = spec
    code = Unlocked(circuit_backend=ListPhysicalCircuit).create_qec_code()
    names = {getattr(i, "name", None) for i in code.instructions.values()}
    assert "Renamed S" in names


def test_shipped_artifacts_current():
    pytest.importorskip("stim")
    from loqs.codepacks.generator.generate import generate

    shipped = shipped_codepacks()
    assert SteaneCodepack in shipped
    for cls in shipped:
        assert cls.locked, cls
        loaded = cls().generated
        assert tables_hash(loaded) == tables_hash(generate(cls.spec)), cls
    docs = Path(loqs.__file__).parents[1] / "docs"
    if not (docs.parent / "mkdocs.yml").exists():
        return
    from loqs.codepacks.generator.document import document

    index = (docs / "codepacks" / "index.md").read_text(encoding="utf-8")
    for cls in shipped:
        page, figures = document(cls, generate(cls.spec))
        committed = docs / "codepacks" / f"{cls.spec.name}.md"
        assert committed.read_text(encoding="utf-8") == page, cls
        assert all((docs / "codepacks" / p).exists() for p in figures), cls
        paper = re.findall(rf"\]\(({cls.spec.name}/paper/[^)]+)\)", page)
        assert all((docs / "codepacks" / p).exists() for p in paper), cls
        assert f"({cls.spec.name}.md)" in index, cls
        workflow = f"validation-{cls.spec.name}.yml"
        assert (
            f"actions/workflows/{workflow}/badge.svg?branch={{{{ binder_branch }}}}"
            in page
        ), cls
        assert (docs.parent / ".github" / "workflows" / workflow).exists(), cls


def test_cli_documents_only_shipped(monkeypatch, tmp_path):
    pytest.importorskip("stim")
    from loqs.codepacks.generator import __main__ as cli

    written = []
    monkeypatch.setattr(
        cli, "write_artifact", lambda cls: written.append(("a", cls)) or "a"
    )
    monkeypatch.setattr(
        cli,
        "write_document",
        lambda cls, generated, docs: written.append(("d", cls)) or "d",
    )
    monkeypatch.setattr(cli.loqs, "__file__", str(tmp_path / "loqs" / "x.py"))

    class Mine(SteaneCodepack):
        pass

    monkeypatch.setattr(sys.modules[__name__], "Mine", Mine, raising=False)
    assert cli.main([f"{__name__}:Mine"]) == 0
    assert written == [("a", Mine)]

    with pytest.raises(FileNotFoundError):
        cli.main(["loqs.codepacks.steane:SteaneCodepack"])
    assert written == [("a", Mine)]

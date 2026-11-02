"""Tester for loqs.codepacks.generator.codepack"""

import dataclasses
import importlib
import pkgutil
import sys
import types

import pytest

import loqs.codepacks
from loqs.backends.circuit.listcircuit import ListPhysicalCircuit
from loqs.codepacks.generator import (
    GeneratedCodepack,
    StaleArtifactError,
)
from loqs.codepacks.generator.codepack import tables_hash
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


def _subclasses(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from _subclasses(sub)


def test_shipped_artifacts_current():
    pytest.importorskip("stim")
    from loqs.codepacks.generator.generate import generate

    for info in pkgutil.walk_packages(
        loqs.codepacks.__path__, "loqs.codepacks."
    ):
        importlib.import_module(info.name)
    shipped = [
        c
        for c in set(_subclasses(GeneratedCodepack))
        if c.__module__.startswith("loqs.codepacks.")
    ]
    assert SteaneCodepack in shipped
    for cls in shipped:
        assert cls.locked, cls
        loaded = cls().generated
        assert tables_hash(loaded) == tables_hash(generate(cls.spec)), cls

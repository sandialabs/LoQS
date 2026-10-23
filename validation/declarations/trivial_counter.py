"""Declaration of the trivial counter codepack (no logical qubit)."""

from loqs.backends.model.dictmodel import DictNoiseModel
from loqs.codepacks import codepack_trivial_counter as module

from validation.declaration import (
    Backend,
    CodepackDeclaration,
    Config,
)

_WHY = "classical counter, no logical qubit"

BACKEND = Backend("dict", model_backend=DictNoiseModel)


def build_code(config, backend):
    return module.create_qec_code(**config.options)


def build_model(config, qubits, backend):
    return module.create_ideal_model(qubits)


DECLARATION = CodepackDeclaration(
    name="trivial_counter",
    files=("loqs/codepacks/codepack_trivial_counter.py",),
    patch_type="Counter",
    configs=(Config("default", classes=("static",), backends=(BACKEND,)),),
    build_code=build_code,
    build_model=build_model,
    internal={
        "Increment": _WHY,
        "Init Counter": _WHY,
        "Sleep": _WHY,
    },
)

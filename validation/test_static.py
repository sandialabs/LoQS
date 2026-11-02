"""Static checks: completeness, model coverage, labels, serialization."""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
import tempfile

from loqs.backends import PyGSTiNoiseModel
from loqs.core import QECCode
from loqs.codepacks.base import (
    walk_qec_code_circuits,
    walk_qec_code_gate_names,
)
from loqs.backends.circuit import ListPhysicalCircuit

from validation import engine
from validation.engine import Op

if sys.platform != "win32":
    import resource

CASE_CLASSES = ("static",)
_MODEL_BUILD_LIMIT = 6 * 1024**3


def _config(decl, case):
    return next(c for c in decl.configs if c.name == case.config)


def _is_builder(name: str) -> bool:
    return name.startswith("build_")


def completeness(case, decl):
    """Built keys equal declared keys; every public builder is declared."""
    config = decl.configs[0]
    code = engine.build_code(decl, config, config.backends[0])
    failing = {}
    declared = {k for k in decl.declared_keys if not _is_builder(k)}
    if set(code.instructions) != declared:
        failing["outcome"] = (
            f"undeclared keys {sorted(set(code.instructions) - declared)}, "
            f"declared but not built {sorted(declared - set(code.instructions))}"
        )
    if decl.data_qubits and tuple(decl.data_qubits) != tuple(
        code.template_data_qubits
    ):
        failing["data-qubits"] = (
            f"code data qubits {list(code.template_data_qubits)}, "
            f"declared {list(decl.data_qubits)}"
        )
    named = set(decl.internal)
    for op in decl.ops:
        named |= set(op.builders)
    for modname in decl.multipatch_modules:
        module = importlib.import_module(modname)
        shipped = {
            n
            for n, f in vars(module).items()
            if _is_builder(n)
            and callable(f)
            and getattr(f, "__module__", None) == module.__name__
        }
        if shipped - named:
            failing["outcome"] = (
                f"{failing.get('outcome', '')} unvalidated builders "
                f"{sorted(shipped - named)}"
            )
    return failing


@contextlib.contextmanager
def memory_cap(limit: int = _MODEL_BUILD_LIMIT):
    """Cap the address space so a runaway dense model build raises.

    A pyGSTi model without a `statevec` evotype builds dense operators over
    every qubit, which must fail with a `MemoryError` rather than exhaust
    the machine. Windows has no `resource`, so there the cap is skipped.
    """
    if sys.platform != "win32":
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
        try:
            yield
        finally:
            resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
    else:
        yield


def _backend(config, name):
    return next(b for b in config.backends if b.name == name)


def model_coverage(case, decl):
    """Walked gate names are in the ideal model, which is `statevec`."""
    config = _config(decl, case)
    backend = _backend(config, case.params["backend"])
    code = engine.build_code(decl, config, backend)
    _, register = engine.patch_labels(code, "default", case.params["patches"])
    with memory_cap():
        model = engine.build_model(decl, config, register, backend)
    # Model keys are `(name, qubits)` tuples.
    available = {k[0] for k in model.gate_keys}
    available |= {k[0] for k in model.instrument_keys}
    if isinstance(model, PyGSTiNoiseModel):
        # A Z measurement with reset needs no pyGSTi instrument
        available.add("Imrz")
    missing = walk_qec_code_gate_names(code) - available
    failing = {}
    if missing:
        failing["outcome"] = f"model lacks {sorted(missing)}"
    inner = getattr(model, "model", None)
    evotype = getattr(inner, "evotype", None)
    if evotype is not None and str(getattr(evotype, "name", evotype)) != (
        "statevec"
    ):
        failing["outcome"] = f"{failing.get('outcome', '')} evotype {evotype}"
    return failing


def smoke_ops(decl):
    """Each prep, gate and QEC key once, then each measure.

    Every measure after the first is preceded by a re-prep, so its outcome
    is judged on a freshly prepared patch.
    """
    ops = [Op("prep", p.key) for p in decl.preps]
    ops += [Op("gate", g.key) for g in decl.gates]
    ops += [Op("qec", k) for k in decl.qec]
    for i, m in enumerate(decl.measures):
        if i:
            ops.append(Op("prep", decl.preps[0].key))
        ops.append(Op("measure", m.key, tuple(sorted(m.kwargs.items()))))
    return ops


def _smoke(decl, config, kind="default", code=None):
    backend = config.backends[0]
    return engine.run_word(
        decl, config, backend, smoke_ops(decl), shots=2, kind=kind, code=code
    )


def labels(case, decl):
    """The smoke word passes under every label set, with the same frames."""
    config = _config(decl, case)
    kind = case.params["kind"]
    result = _smoke(decl, config, kind)
    failing = dict(result.failing)
    if result.frames != _smoke(decl, config).frames:
        failing["outcome"] = f"frame count differs under {kind} labels"
    return failing


def _json_text(code):
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        code.write(path, format="json")
        with open(path, encoding="utf-8") as f:
            return f.read()
    finally:
        os.unlink(path)


def serialization(case, decl):
    """A write/read round trip keeps the walked gate names."""
    config = _config(decl, case)
    code = engine.build_code(decl, config, config.backends[0])
    before = _json_text(code)
    if decl.preps:
        _smoke(decl, config, code=code)
    fmt = case.params["format"]
    fd, path = tempfile.mkstemp(suffix=f".{fmt}")
    os.close(fd)
    try:
        code.write(path, format=fmt)
        read = QECCode.read(path)
    finally:
        os.unlink(path)
    failing = {}
    if decl.preps:
        failing.update(_smoke(decl, config, code=read).failing)
    if walk_qec_code_gate_names(read) != walk_qec_code_gate_names(code):
        failing["outcome"] = "walked gate names changed in the round trip"
    if _json_text(code) != before:
        failing["outcome"] = f"{failing.get('outcome', '')} json form changed"
    return failing


def idle_padding(case, decl):
    """Every layer of every circuit leaf has an op on every qubit."""
    config = _config(decl, case)
    code = engine.build_code(decl, config, config.backends[0])
    offenders = []
    for circuit in walk_qec_code_circuits(code):
        layers = ListPhysicalCircuit(circuit).circuit
        qubits = list(circuit.qubit_labels)
        for i, layer in enumerate(layers):
            touched = {q for _, op_qubits in layer for q in op_qubits}
            missing = [q for q in qubits if q not in touched]
            if missing:
                offenders.append(f"layer {i} {layer} leaves {missing} idle")
    failing = {}
    if offenders:
        failing["outcome"] = "; ".join(offenders)
    return failing


FAMILIES = {
    "completeness": completeness,
    "idle-padding": idle_padding,
    "model-coverage": model_coverage,
    "labels": labels,
    "serialization": serialization,
}


def test_static(case, judged):
    judged(lambda c, decl: FAMILIES[c.family](c, decl))

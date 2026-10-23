"""Declarations: what a codepack ships, from which its checks are generated."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

CLASSES = ("static", "truth", "tracking", "independence", "ft")
CHECKS = ("outcome", "O1", "O2", "O3", "O4")


@dataclass(frozen=True)
class Backend:
    """One way to run a codepack: circuit and model backends and reps."""

    name: str
    circuit_backend: Any = None
    model_backend: Any = None
    gate_rep: Any = None
    state_type: Any = None


@dataclass(frozen=True)
class Config:
    """A `create_qec_code` configuration and the classes run on it.

    The first backend is the primary one and must be STIM in a config
    that runs programs beyond the static class.
    """

    name: str
    options: Mapping[str, Any] = field(default_factory=dict)
    classes: Sequence[str] = ("static",)
    backends: Sequence[Backend] = ()
    big: bool = False


@dataclass(frozen=True)
class Prep:
    """A logical prep key and the signed Pauli state it prepares.

    `fresh_only` marks a prep that assumes freshly reset data qubits, so a
    re-prep of a live patch without a measure in between is out of contract.
    """

    key: str
    state: str  # e.g. "+Z", "-X"
    fresh_only: bool = False


@dataclass(frozen=True)
class Measure:
    """A logical measure key, its basis and whether it is fault tolerant."""

    key: str
    basis: str
    ft: bool
    kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Gate:
    """A logical gate key and the stim Clifford name of its action."""

    key: str
    clifford: str


@dataclass(frozen=True)
class Tracked:
    """A patch's tracked state as read by a declaration; each may be `None`.

    `syndrome` is the stored syndrome bits in the order of `stabilizers`,
    `frame` a per-data-qubit Pauli string (in `data_qubits` order) and
    `logical` the logical Pauli frame as `(x, z)` bits.
    """

    syndrome: Sequence[int] | None = None
    frame: str | None = None
    logical: tuple[int, int] | None = None


@dataclass(frozen=True)
class MultipatchOp:
    """A multipatch operation over named patch roles.

    Roles past the first two are patches the op prepares and measures
    itself. `seams`: bare qubit groups (ancilla, seam) and their sizes;
    `configs`: configs it runs on (empty: all); `needs_qec`: the op refuses
    to run before a QEC round. A `Meas` key in `script` may name `{role}`.
    """

    name: str
    roles: Sequence[str]
    build: Callable[..., Any]
    builders: Sequence[str]
    script: Sequence[Any] = ()
    readout_kwargs: Mapping[str, Any] = field(default_factory=dict)
    big_layouts: Sequence[str] = ()
    seams: Mapping[str, int] = field(default_factory=dict)
    configs: Sequence[str] = ()
    needs_qec: bool = False


@dataclass(frozen=True)
class Sweep:
    """A fault-tolerance sweep over a leaf circuit of the stack.

    `entries` are stack entries (or `build(config, geometry)` returning
    them, as for ops) over the patch labels `roles`; `target` is the key or
    instruction name of the leaf circuit among them. `script` is their
    logical action (a `Prep` makes the sweep its own prep, a `Meas` its own
    measure), `discard(history)` drops a shot (a flagged attempt) and
    `suffix` lets gate words follow it. Faults run weight 1 before each
    gate, or weight 2 after each two-qubit gate with `post_twoq`.
    """

    name: str
    entries: Sequence[Any] | Callable[..., Sequence[Any]]
    target: str
    post_twoq: bool = False
    discard: Callable[..., bool] | None = None
    allowed: int = 0
    script: Sequence[Any] = ()
    roles: Sequence[str] = ("L0",)
    suffix: bool = False


@dataclass(frozen=True)
class CodepackDeclaration:
    """Everything the harness needs to generate a codepack's checks.

    `stabilizers`, `logical_x`, `logical_z` and the `Tracked.frame` string
    are Pauli strings over `data_qubits`, template labels of the code
    (default: the code's own template data qubits).
    """

    name: str
    files: Sequence[str]
    patch_type: str
    configs: Sequence[Config]
    build_code: Callable[[Config, Backend], Any]
    build_model: Callable[[Config, Sequence[str], Backend], Any]
    data_qubits: Sequence[str] = ()
    stabilizers: Sequence[str] = ()
    logical_x: str = ""
    logical_z: str = ""
    preps: Sequence[Prep] = ()
    measures: Sequence[Measure] = ()
    gates: Sequence[Gate] = ()
    qec: Sequence[str] = ()
    internal: Mapping[str, str] = field(default_factory=dict)
    tracked: Callable[..., Tracked] | None = None
    decoder: str = "eager"
    multipatch_modules: Sequence[str] = ()
    ops: Sequence[MultipatchOp] = ()
    sweeps: Sequence[Sweep] = ()
    error_pair: Sequence[Any] = ()
    known_failures: Mapping[str, str | Sequence[str]] = field(
        default_factory=dict
    )
    reproducers: Mapping[str, tuple[str, Sequence[str]]] = field(
        default_factory=dict
    )

    @property
    def declared_keys(self) -> set[str]:
        """Every instruction key the declaration accounts for."""
        keys = {p.key for p in self.preps} | {m.key for m in self.measures}
        keys |= {g.key for g in self.gates} | set(self.qec)
        return keys | set(self.internal)

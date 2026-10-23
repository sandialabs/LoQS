"""The logical script: what a program must do to its logical qubits.

A script is simulated by a `stim.TableauSimulator` with one qubit per patch
role. It is the oracle's reference, independent of how a codepack is built.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import stim

# Gates applied after a reset to reach each signed Pauli eigenstate.
_PREP_GATES = {
    "+Z": (),
    "-Z": ("X",),
    "+X": ("H",),
    "-X": ("H", "Z"),
    "+Y": ("H", "S"),
    "-Y": ("H", "S", "Z"),
}


@dataclass(frozen=True)
class Prep:
    """Reset a role and prepare the +1 eigenstate of a signed Pauli."""

    role: str
    state: str


@dataclass(frozen=True)
class Gate:
    """A Clifford gate by stim name on one role, or two for `CX`."""

    name: str
    roles: tuple[str, ...]


@dataclass(frozen=True)
class Meas:
    """A Pauli-product measurement; outcome 0 is the +1 eigenvalue."""

    key: str
    paulis: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Cond:
    """A Pauli on a role, applied when the XOR of earlier outcomes is 1."""

    role: str
    pauli: str
    keys: tuple[str, ...]


Step = Prep | Gate | Meas | Cond


class LogicalSim:
    """A tableau simulator over named roles."""

    def __init__(self, roles: Sequence[str], seed: int = 0):
        self.roles = list(roles)
        self.sim = stim.TableauSimulator(seed=seed)
        self.sim.set_num_qubits(len(self.roles))
        self.last: dict[str, int] = {}

    def _index(self, role: str) -> int:
        return self.roles.index(role)

    def observable(
        self, paulis: Sequence[tuple[str, str]]
    ) -> stim.PauliString:
        """The Pauli product over `(role, letter)` pairs."""
        obs = stim.PauliString(len(self.roles))
        for role, letter in paulis:
            obs[self._index(role)] = letter
        return obs

    def expectation(self, paulis: Sequence[tuple[str, str]]) -> int:
        """-1, 0 or 1: the expectation of a Pauli product."""
        return int(
            self.sim.peek_observable_expectation(self.observable(paulis))
        )

    def deterministic_bit(self, paulis: Sequence[tuple[str, str]]):
        """A measurement's outcome bit if deterministic, else `None`."""
        exp = self.expectation(paulis)
        return None if exp == 0 else (0 if exp == 1 else 1)

    def _gate(self, name: str, roles: Sequence[str]) -> None:
        tableau = stim.Tableau.from_named_gate(name)
        self.sim.do_tableau(tableau, [self._index(r) for r in roles])

    def apply(self, step: Step, bit: int | None = None) -> None:
        """Apply a step; a measurement is postselected on `bit`."""
        if isinstance(step, Prep):
            self.sim.reset_z(self._index(step.role))
            for name in _PREP_GATES[step.state]:
                self._gate(name, (step.role,))
        elif isinstance(step, Gate):
            self._gate(step.name, step.roles)
        elif isinstance(step, Meas):
            assert bit is not None
            self.sim.postselect_observable(
                self.observable(step.paulis), desired_value=bool(bit)
            )
            self.last[step.key] = bit
        else:
            parity = sum(self.last[k] for k in step.keys) % 2
            if parity:
                self._gate(step.pauli, (step.role,))


def rename(step: Step, mapping: Mapping[str, str]) -> Step:
    """The step with its roles renamed through `mapping`."""
    if isinstance(step, Prep):
        return Prep(mapping[step.role], step.state)
    if isinstance(step, Gate):
        return Gate(step.name, tuple(mapping[r] for r in step.roles))
    if isinstance(step, Meas):
        paulis = tuple((mapping[r], p) for r, p in step.paulis)
        return Meas(step.key.format(**mapping), paulis)
    keys = tuple(k.format(**mapping) for k in step.keys)
    return Cond(mapping[step.role], step.pauli, keys)


def roles_of(script: Sequence[Step]) -> list[str]:
    """Every role a script mentions, in order of first appearance."""
    roles: list[str] = []
    for step in script:
        if isinstance(step, (Prep, Cond)):
            names = [step.role]
        elif isinstance(step, Gate):
            names = list(step.roles)
        else:
            names = [r for r, _ in step.paulis]
        roles += [r for r in names if r not in roles]
    return roles


def check_shot(
    script: Sequence[Step], outcomes: Mapping[str, Sequence[int]]
) -> list[str]:
    """Problems found when walking a shot's outcomes through a script.

    `outcomes` maps each measurement key to its reported bits in program
    order. A deterministic outcome must match the script; a random one is
    postselected, so later deterministic outcomes are judged given it.
    """
    sim = LogicalSim(roles_of(script))
    used: dict[str, int] = {}
    for i, step in enumerate(script):
        bit = None
        if isinstance(step, Meas):
            n = used.get(step.key, 0)
            used[step.key] = n + 1
            reported = outcomes.get(step.key, ())
            if n >= len(reported):
                return [f"step {i}: outcome {step.key!r}#{n} not reported"]
            bit = int(reported[n])
            wanted = sim.deterministic_bit(step.paulis)
            if wanted is not None and wanted != bit:
                return [f"step {i}: outcome {bit}, script says {wanted}"]
        sim.apply(step, bit)
    return []


def has_check(script: Sequence[Step]) -> bool:
    """Whether a script has a deterministic outcome or logical expectation.

    The expectations are X, Y and Z of each role after each step, taken
    on the branch where every measurement reads 0.
    """
    sim = LogicalSim(roles_of(script))
    for step in script:
        bit = None
        if isinstance(step, Meas):
            if sim.deterministic_bit(step.paulis) is not None:
                return True
            bit = 0
        sim.apply(step, bit)
        for role in sim.roles:
            if any(sim.expectation([(role, p)]) for p in "XYZ"):
                return True
    return False

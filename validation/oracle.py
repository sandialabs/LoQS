"""The codespace oracle: global observations of a STIM state and checks O1-O4.

An observation is an `Instruction` that peeks at the live state between the
steps the engine emits and records raw numbers in its frame. After the run
`evaluate` turns the snapshots into the set of failing checks:

- O1: every declared stabilizer has expectation +1 or -1.
- O2: the codepack's stored syndrome equals the true stabilizer bits.
- O3: the logical X, Y and Z expectations, corrected by the tracked logical
  Pauli, equal the logical script's.
- O4: the tracked state re-read from the frame's `patches` after the run
  equals the snapshot taken during it.
"""

from __future__ import annotations

import itertools
from typing import Any, Mapping, Sequence

import stim

from loqs.core import Frame, Instruction

from validation.declaration import CodepackDeclaration, Tracked

OBS_KEY = "validation_obs"
_MAX_REPORTED = 3


def _normalized(tracked: Tracked) -> dict[str, Any]:
    """Plain, comparable copies of a `Tracked`'s fields."""
    return {
        "syndrome": (
            None
            if tracked.syndrome is None
            else tuple(int(b) for b in tracked.syndrome)
        ),
        "frame": tracked.frame,
        "logical": (
            None
            if tracked.logical is None
            else tuple(int(b) for b in tracked.logical)
        ),
    }


def read_tracked(decl: CodepackDeclaration, patch) -> dict[str, Any]:
    """The declaration's tracked reader applied to a patch, normalized."""
    if decl.tracked is None:
        return _normalized(Tracked())
    return _normalized(decl.tracked(patch))


def _data_labels(decl: CodepackDeclaration, patch) -> list:
    """The patch's qubit labels for the declaration's data qubits."""
    code = patch.code
    template = list(decl.data_qubits or code.template_data_qubits)
    return [patch.qubits[code.template_qubits.index(t)] for t in template]


def _pauli(pstr: str, indices: list[int], n: int) -> stim.PauliString:
    """The Pauli string `pstr` (optionally led by `-`) on qubit `indices`."""
    sign = -1 if pstr.startswith("-") else 1
    pstr = pstr.lstrip("+-")
    out = stim.PauliString(n) * sign
    for letter, index in zip(pstr, indices):
        if letter != "I":
            out[index] = letter
    return out


def logical_keys(roles: Sequence[str]) -> list[tuple[tuple[str, str], ...]]:
    """Every X/Y/Z product over a non-empty subset of the roles."""
    out = []
    for n in range(1, len(roles) + 1):
        for sub in itertools.combinations(roles, n):
            for letters in itertools.product("XYZ", repeat=n):
                out.append(tuple(zip(sub, letters)))
    return out


def key_name(key: Sequence[tuple[str, str]]) -> str:
    """A string name of a logical product, e.g. `L0.X,L1.Z`."""
    return ",".join(f"{r}.{p}" for r, p in key)


def key_roles(name: str) -> list[str]:
    """The roles a logical product's name mentions."""
    return [part.split(".")[0] for part in name.split(",")]


def observation(
    decl: CodepackDeclaration,
    index: int,
    label: str,
    flags: Mapping[str, tuple[bool, bool, bool]],
    expects: Mapping[str, int],
) -> tuple[Instruction, None]:
    """A global stack entry observing the patches named in `flags`.

    `flags[patch label]` says which of O1, O2 and O3 apply to that patch
    here, and `expects` maps each logical product name (`key_name`) to the
    script's expectation of it.
    """
    spec = {"flags": dict(flags), "expect": dict(expects)}
    roles = tuple(flags)

    def observe(state, patches, roles) -> Frame:
        n = state.state.num_qubits
        sim = state.state
        snap: dict[str, Any] = {
            "index": index,
            "label": label,
            "roles": list(roles),
            "spec": spec,
            "tracked": {},
            "stabilizers": {},
            "logical": {},
        }
        ops = {}
        for role in roles:
            patch = patches[role]
            qubits = _data_labels(decl, patch)
            idx = [state.qubit_labels.index(q) for q in qubits]
            snap["tracked"][role] = read_tracked(decl, patch)
            snap["stabilizers"][role] = [
                int(sim.peek_observable_expectation(_pauli(s, idx, n)))
                for s in decl.stabilizers
            ]
            lx = _pauli(decl.logical_x, idx, n)
            lz = _pauli(decl.logical_z, idx, n)
            ops[role] = {"X": lx, "Y": lx * lz * 1j, "Z": lz}
        for key in logical_keys(roles):
            pauli = stim.PauliString(n)
            for role, letter in key:
                pauli = pauli * ops[role][letter]
            snap["logical"][key_name(key)] = int(
                sim.peek_observable_expectation(pauli)
            )
        return Frame({OBS_KEY: snap})

    instruction = Instruction(
        observe, data={"roles": roles}, name=f"Observe {label}"
    )
    return (instruction, None)


def _flips(decl: CodepackDeclaration, tracked: Mapping[str, Any]):
    """Signs the tracked logical Pauli puts on the X, Y and Z expectations."""
    if tracked["logical"] is not None:
        x, z = tracked["logical"]
        return {
            "X": -1 if z else 1,
            "Y": -1 if x ^ z else 1,
            "Z": -1 if x else 1,
        }
    if tracked["frame"] is None:
        return None
    frame = stim.PauliString(tracked["frame"])
    lx = stim.PauliString(decl.logical_x)
    lz = stim.PauliString(decl.logical_z)
    sx = 1 if frame.commutes(lx) else -1
    sz = 1 if frame.commutes(lz) else -1
    return {"X": sx, "Y": sx * sz, "Z": sz}


def _record(failing: dict[str, list[str]], check: str, where: str) -> None:
    failing.setdefault(check, []).append(where)


def evaluate(decl: CodepackDeclaration, history, shot: int, failing) -> None:
    """Add this shot's O1-O4 failures to `failing` (check -> places)."""
    for frame in history:
        if OBS_KEY not in frame:
            continue
        snap = frame[OBS_KEY]
        spec = snap["spec"]
        where = f"shot {shot} obs {snap['index']} ({snap['label']})"
        signs = {}
        for role in snap["roles"]:
            o1, o2, _ = spec["flags"][role]
            exps = snap["stabilizers"][role]
            tracked = snap["tracked"][role]
            in_codespace = all(e in (-1, 1) for e in exps)
            if o1 and not in_codespace:
                _record(failing, "O1", f"{where} {role}: expectations {exps}")
            if o2 and in_codespace and tracked["syndrome"] is not None:
                true = tuple(1 if e == -1 else 0 for e in exps)
                if true != tracked["syndrome"]:
                    _record(
                        failing,
                        "O2",
                        f"{where} {role}: stored "
                        f"{tracked['syndrome']}, true {true}",
                    )
            signs[role] = _flips(decl, tracked)
            reread = read_tracked(decl, frame["patches"][role])
            if reread != tracked:
                _record(
                    failing,
                    "O4",
                    f"{where} {role}: re-read {reread}, "
                    f"during run {tracked}",
                )
        for name, want in spec["expect"].items():
            roles = key_roles(name)
            if not all(spec["flags"][r][2] and signs[r] for r in roles):
                continue
            got = snap["logical"][name]
            for part in name.split(","):
                role, letter = part.split(".")
                got *= signs[role][letter]
            if got != want:
                _record(
                    failing,
                    "O3",
                    f"{where}: {name} corrected {got}, script {want}",
                )


def summarize(failing: Mapping[str, list[str]]) -> dict[str, str]:
    """One detail string per failing check, listing a few places."""
    return {
        check: f"{len(places)} failures, e.g. "
        + "; ".join(places[:_MAX_REPORTED])
        for check, places in failing.items()
    }

#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.3                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Stim derivations from a spec's facts and circuits.

Everything here is a pure function of Pauli strings and circuit layers.
Bit conventions: a logical frame vector lists, per logical qubit, one bit
per entry of `Facts.frame_layout` (Steane: `[Z_bit, X_bit]`); a syndrome
vector has one bit per stabilizer. Maps are 0/1 matrices `M` with
`new = M @ old (mod 2)`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import stim

from loqs.codepacks.generator import gates
from loqs.codepacks.generator.schema import Facts, Layer, SpecError

# --- GF(2) and Pauli algebra ------------------------------------------------


def symp(p: stim.PauliString) -> np.ndarray:
    """The symplectic bits `[x..., z...]` of a Pauli string."""
    xs, zs = p.to_numpy()
    return np.concatenate([xs, zs]).astype(np.uint8)


def gf2_solve(rows: np.ndarray, v: np.ndarray) -> np.ndarray | None:
    """Bits b with `b @ rows == v` (mod 2), or None."""
    m, n = rows.shape
    aug = np.concatenate([rows.T, v.reshape(-1, 1)], axis=1) % 2  # n x (m+1)
    aug = aug.astype(np.uint8)
    piv_cols = []
    r = 0
    for c in range(m):
        hit = [i for i in range(r, n) if aug[i, c]]
        if not hit:
            continue
        aug[[r, hit[0]]] = aug[[hit[0], r]]
        for i in range(n):
            if i != r and aug[i, c]:
                aug[i] ^= aug[r]
        piv_cols.append(c)
        r += 1
        if r == n:
            break
    if any(aug[i, m] for i in range(r, n)):
        return None
    b = np.zeros(m, dtype=np.uint8)
    for i, c in enumerate(piv_cols):
        b[c] = aug[i, m]
    return b


class Algebra:
    """Stabilizers and paired logicals over `n` qubits, with decomposition."""

    def __init__(
        self,
        stabilizers: Any,
        logicals: Any,
    ) -> None:
        self.stabs = [stim.PauliString(s) for s in stabilizers]
        self.logicals = [
            (stim.PauliString(x), stim.PauliString(z)) for x, z in logicals
        ]
        self.n = len(self.stabs[0])
        for i, s in enumerate(self.stabs):
            for t in self.stabs[i + 1 :]:
                if not s.commutes(t):
                    raise SpecError("stabilizers don't commute")
        for k, (x, z) in enumerate(self.logicals):
            if x.commutes(z):
                raise SpecError(f"logical {k}: X_L and Z_L commute")
            for s in self.stabs:
                if not (s.commutes(x) and s.commutes(z)):
                    raise SpecError(f"logical {k} doesn't commute with {s}")
            for j, (x2, z2) in enumerate(self.logicals):
                if j != k and not (
                    x.commutes(x2) and x.commutes(z2) and z.commutes(z2)
                ):
                    raise SpecError(f"logicals {k} and {j} don't commute")
        gens = self.stabs + [p for xz in self.logicals for p in xz]
        self._gens = gens
        self._rows = np.array([symp(g) for g in gens])
        for i in range(len(gens)):
            others = np.delete(self._rows, i, axis=0)
            if gf2_solve(others, self._rows[i]) is not None:
                raise SpecError("stabilizers and logicals aren't independent")

    @property
    def k(self) -> int:
        """The number of logical qubits."""
        return len(self.logicals)

    def decompose(
        self, p: stim.PauliString
    ) -> tuple[np.ndarray, np.ndarray, complex] | None:
        """`(stab_bits, logical_bits [x0, z0, x1, z1, ...], sign)` or None.

        `sign` is p's sign relative to the product of the chosen generators
        (multiplied in generator order), so +1 means p is exactly that
        product."""
        b = gf2_solve(self._rows, symp(p))
        if b is None:
            return None
        q = stim.PauliString(self.n)
        for bit, g in zip(b, self._gens):
            if bit:
                q *= g
        ratio = complex(p.sign) / complex(q.sign)
        ns = len(self.stabs)
        return b[:ns], b[ns:], ratio

    def logical_rep(
        self, k: int, pauli: str, sign: complex = 1
    ) -> stim.PauliString:
        """The physical representative of a logical one-qubit Pauli on
        logical `k` (Y_L = i X_L Z_L)."""
        x, z = self.logicals[k]
        out = stim.PauliString(self.n)
        if pauli in "XY":
            out *= x
        if pauli in "ZY":
            out *= z
        if pauli == "Y":
            out *= 1j
        return out * sign


# --- circuits ---------------------------------------------------------------

_TABLEAU_CACHE: dict[str, stim.Tableau] = {}


def tableau(name: str) -> stim.Tableau:
    """The gate's Clifford, from its unitary (big-endian: the first target
    is the most significant qubit, as pyGSTi orders `Gcnot ctrl tgt`)."""
    if name not in _TABLEAU_CACHE:
        _TABLEAU_CACHE[name] = stim.Tableau.from_unitary_matrix(
            gates.unitary(name), endian="big"
        )
    return _TABLEAU_CACHE[name]


def unitary_tableau(
    layers: list[Layer], index: dict[str, int]
) -> stim.Tableau:
    """Clifford of unitary layers; `index` maps labels to tableau qubits.
    Idles are skipped; an instrument is refused."""
    t = stim.Tableau(len(index))
    for layer in layers:
        for name, qs in layer:
            e = gates.GATES[name]
            if e.kind == "instrument":
                raise SpecError(f"{name} inside a unitary part")
            if e.kind == "idle" or e.unitary == "I":
                continue
            t.append(tableau(name), [index[q] for q in qs])
    return t


def split_measured(layers: list[Layer]) -> tuple[list[Layer], list[str]]:
    """`(unitary layers, measured qubits)` for a circuit whose instruments
    all sit in its last layer, alone."""
    meas: list[str] = []
    body = list(layers)
    if body and all(gates.GATES[n].kind == "instrument" for n, _ in body[-1]):
        meas = [qs[0] for _, qs in body[-1]]
        body = body[:-1]
    for layer in body:
        for n, _ in layer:
            if gates.GATES[n].kind == "instrument":
                raise SpecError("a measurement before the last layer")
    return body, meas


def measured_observables(layers: list[Layer], facts: Facts) -> dict[str, str]:
    """For each measured aux, the data stabilizer its outcome reads (name,
    or `other:<pauli>`), from back-propagating Z_aux through the circuit
    with the aux in |0>."""
    qubits = facts.qubits
    index = {q: i for i, q in enumerate(qubits)}
    body, meas = split_measured(layers)
    inv = unitary_tableau(body, index).inverse()
    data_idx = [index[q] for q in facts.data]
    out = {}
    stabs = [
        (stim.PauliString(s), n)
        for n, s in zip(facts.stabilizer_names, facts.stabilizers)
    ]
    for a in meas:
        p = stim.PauliString(len(qubits))
        p[index[a]] = "Z"
        q = inv(p)
        for aq in facts.aux:
            if q[index[aq]] in (1, 2):  # X or Y on an aux in |0>: random
                raise SpecError(f"{a}'s outcome is random in the ideal case")
        dp = stim.PauliString([q[i] for i in data_idx]) * q.sign
        name = None
        for s, n in stabs:
            if dp == s:
                name = n
        out[a] = name if name is not None else f"other:{dp}"

    return out


# --- bookkeeping -------------------------------------------------------------


def _frame_basis(
    alg: Algebra, layout: tuple[str, ...]
) -> list[tuple[int, str, stim.PauliString]]:
    """Logical frame basis in vector order: per logical, one Pauli per
    `layout` letter."""
    out = []
    for k in range(alg.k):
        for letter in layout:
            out.append((k, letter, alg.logical_rep(k, letter)))
    return out


def _frame_bits(
    lbits: np.ndarray, layout: tuple[str, ...], k_total: int
) -> list[int]:
    """Frame-vector bits of a decomposition's logical bits [x0, z0, ...]."""
    v = []
    for k in range(k_total):
        x, z = lbits[2 * k], lbits[2 * k + 1]
        for letter in layout:
            v.append(int(z if letter == "Z" else x))
    return v


def gate_bookkeeping(
    t: stim.Tableau,
    alg: Algebra,
    layout: tuple[str, ...],
    declared: stim.Tableau,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive a code-preserving Clifford's bookkeeping and check it.

    `t` acts on the data qubits (in Facts order, joint for multi-patch).
    `declared` is a stim.Tableau of the logical action over `alg.k`
    logical qubits. Returns `(frame_map, syndrome_map)`.

    - Logical frame: a tracked error E becomes U E U^dag, so column j of
      the frame map is the logical part of U P_j U^dag.
    - Stored syndrome: a fixed check S_i measured after U reads
      U^dag S_i U on the state before it, so row i of the syndrome map is
      the stabilizer decomposition of U^dag S_i U (the inverse tableau).
    """
    inv = t.inverse()
    ns = len(alg.stabs)
    smap = np.zeros((ns, ns), dtype=np.uint8)
    for i, s in enumerate(alg.stabs):
        for tab, label in ((t, "U S U^dag"), (inv, "U^dag S U")):
            d = alg.decompose(tab(s))
            if d is None or d[1].any() or d[2] != 1:
                raise SpecError(
                    f"not code-preserving: {label} for stabilizer {i} "
                    + (
                        "leaves the group"
                        if d is None
                        else (
                            "has a logical part"
                            if d[1].any()
                            else "flips sign"
                        )
                    )
                )
        back = alg.decompose(inv(s))
        assert back is not None
        smap[i] = back[0]

    basis = _frame_basis(alg, layout)
    fmap = np.zeros((len(basis), len(basis)), dtype=np.uint8)
    for j, (_, _, p) in enumerate(basis):
        d = alg.decompose(t(p))
        if d is None:
            raise SpecError("a logical operator leaves the normalizer")
        fmap[:, j] = _frame_bits(d[1], layout, alg.k)

    # Declared logical action, with signs: U L(P) U^dag == L(T_decl(P)) * s
    # for a stabilizer element s with sign +1.
    for k in range(alg.k):
        for letter in "XZ":
            lp = stim.PauliString(alg.k)
            lp[k] = letter
            img = declared(lp)
            target = stim.PauliString(alg.n) * complex(img.sign)
            for kk in range(alg.k):
                if img[kk]:
                    target *= alg.logical_rep(kk, "_XYZ"[img[kk]])
            q = t(alg.logical_rep(k, letter)) * target
            d = alg.decompose(q)
            if d is None or d[1].any() or d[2] != 1:
                raise SpecError(
                    f"declared logical action is wrong on {letter}_L{k}: "
                    f"circuit gives {t(alg.logical_rep(k, letter))}"
                )
    return fmap, smap


def per_qubit_rules(
    layers: list[Layer], data: tuple[str, ...]
) -> dict[str, dict[str, str]] | None:
    """Sign-free per-qubit frame rule of a circuit of one-qubit gates on
    `data`, or None when a multi-qubit gate acts (no per-qubit rule)."""
    seqs: dict[str, list[str]] = {q: [] for q in data}
    for layer in layers:
        for name, qs in layer:
            if len(qs) != 1:
                return None
            if qs[0] in seqs:
                seqs[qs[0]].append(name)
    rules = {}
    for q, names in seqs.items():
        t = stim.Tableau(1)
        for n in names:
            if gates.GATES[n].kind == "gate":
                t.append(tableau(n), [0])
        rules[q] = {
            "I": "I",
            **{
                p: str(t(stim.PauliString(p)))[1:].replace("_", "I")
                for p in "XYZ"
            },
        }
    return rules

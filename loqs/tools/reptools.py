#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.1.1                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""A collection of tools for manipulating various backend representations."""

import functools
import itertools
from typing import List, Sequence, Tuple, TypeAlias

import numpy as np

from loqs.backends.reps import GateRep, KrausGateRep, UnitaryGateRep

Float: TypeAlias = float | np.floating

## KRAUS REPS


def pauli_sym_prod_phase(pstr1: str, pstr2: str) -> int:
    """Compute the phased symplectic product of two Pauli strings.

    Given two Pauli string `P_1` and `P_2`, this computes the quantity

    \[
    (-1)^<P_1, P_2>
    \]

    where <P_1, P_2> is the symplectic inner product as defined by
    Eqn. 2 of https://doi.org/10.1103/PRXQuantum.2.010322.
    Effectively, this phase factor will be 1 if `P_1` and `P_2`
    commute, and -1 if they do not commute.
    Commonly used in the Walsh-Hadamard transform from Pauli rates
    to Pauli eigenvalues.

    Parameters
    ----------
    pstr1:
        First Pauli string

    pstr2:
        Second Pauli string

    Returns
    -------
    int
        Phase as 1 if the two Paulis commute and -1 otherwise
    """
    assert all([c in "IXYZ" for c in pstr1.upper()])
    assert all([c in "IXYZ" for c in pstr2.upper()])

    phase = 1
    for c1, c2 in zip(pstr1.upper(), pstr2.upper()):
        if c1 == c2 or c1 == "I" or c2 == "I":
            # Commute
            continue

        # Don't commute
        phase *= -1
    return phase


def pauli_rates_to_eigvals(rates: Sequence[Float]) -> List[float]:
    r"""Convert Pauli error rates into Pauli eigenvalues.

    Given the Kraus representation of a Pauli-stochastic operator

    \[
    \rho \rightarrow \sum_i p_i P_i \rho P_i^\dagger,
    \]

    where `P_i` are the `i`-qubit Paulis, then the Pauli eigenvalues
    of this channel are given via the Walsh-Hadamard transform
    (Eqn. 5 of https://doi.org/10.1103/PRXQuantum.2.010322)

    \[
    \lambda_j = \sum_i p_i (-1)^{<P_i, P_j>}.
    \]

    The PTM representation of this channel is then just
    diag(`\vec{\lambda}`).

    Parameters
    ----------
    rates:
        Pauli rates

    Returns
    -------
    List[float]
        Pauli eigenvalues
    """
    N = int(np.log2(len(rates)) // 2)
    assert len(rates) == 4**N

    eigvals = []
    for pstr1 in itertools.product("IXYZ", repeat=N):
        ev1 = 0
        for pstr2, r2 in zip(itertools.product("IXYZ", repeat=N), rates):
            ev1 += r2 * pauli_sym_prod_phase("".join(pstr1), "".join(pstr2))
        eigvals.append(ev1)

    return eigvals


def pauli_eigvals_to_rates(eigvals: Sequence[Float]) -> List[float]:
    r"""Convert Pauli eigenvalues into Pauli error rates.

    Given the PTM representation of a Pauli-stochastic operator as
    diag(`\vec{\lambda}`), where `\lambda` are the Pauli eigenvalues,
    then applying the inverse Walsh-Hadamard transform
    (Eqn. 6 of https://doi.org/10.1103/PRXQuantum.2.010322)
    gives

    \[
    \p_j = \frac{1}{4**n} \sum_i \lambda_i (-1)^{<P_i, P_j>},
    \]

    where `P_i,P_j` are `n`-qubit Paulis.

    The Kraus representation of the operator is then

    \[
    \rho \rightarrow \sum_i p_i P_i \rho P_i^\dagger.
    \]


    Parameters
    ----------
    rates:
        Pauli eigenvalues

    Returns
    -------
    List[float]
        Pauli rates
    """
    N = int(np.log2(len(eigvals)) // 2)
    assert len(eigvals) == 4**N

    rates = []
    for pstr1 in itertools.product("IXYZ", repeat=N):
        r1 = 0
        for pstr2, ev2 in zip(itertools.product("IXYZ", repeat=N), eigvals):
            r1 += ev2 * pauli_sym_prod_phase("".join(pstr1), "".join(pstr2))
        rates.append(r1 / (4**N))

    return rates


def create_pauli_stochastic_kraus_rep(
    rates: Sequence[Float], qubits: Sequence[str | int]
) -> KrausGateRep:
    """Create a Pauli-stochastic [](api:KrausGateRep).

    Parameters
    ----------
    rates:
        The coefficients of the Kraus terms.

    qubits:
        The targeted qubits (needed for [](api:KrausGateRep) construction)

    Returns
    -------
    KrausGateRep
        The Pauli-stochasic Kraus representation
    """
    # Sanity checks
    assert all([p >= 0 and p <= 1 for p in rates])
    assert np.isclose(sum(rates), 1)

    # Number of stochastic rates = 4**N
    assert len(rates) == 4 ** len(qubits)

    Ps = {
        "I": np.eye(2),
        "X": np.array([[0, 1], [1, 0]]),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.array([[1, 0], [0, -1]]),
    }

    kraus_reps = []
    for prob, pauli_str in zip(
        rates, itertools.product("IXYZ", repeat=len(qubits))
    ):
        if prob < 1e-10:
            # Skip this term
            continue

        # Get all 1Q Paulis
        paulis_1Q = [Ps[pstr] for pstr in pauli_str]

        # Compute the multi-qubit Pauli
        pauli_NQ = functools.reduce(np.kron, paulis_1Q)

        # Add the Kraus matrix to the reps
        kraus_reps.append((np.sqrt(prob) * pauli_NQ, prob))

    return KrausGateRep(kraus_reps, qubits)


def create_depolarizing_kraus_rep(
    rate: Float, qubits: Sequence[str | int]
) -> KrausGateRep:
    """Create a depolarizing [](api:KrausGateRep).

    This is a convenience function that wraps
    [](api:create_pauli_stochastic_kraus_rep).

    Parameters
    ----------
    rate : Float
        The depolarizing rate.

    qubits : Sequence[str | int]
        The targeted qubits (needed for [](api:KrausGateRep) construction)

    Returns
    -------
    KrausGateRep
        A Pauli-stochasic Kraus representation
    """
    N = 4 ** len(qubits)
    return create_pauli_stochastic_kraus_rep(
        [1 - (N - 1) / N * rate] + [rate / N] * (N - 1), qubits
    )


def create_1Q_amp_damp_kraus_rep(prob: Float, qubit: str | int) -> KrausGateRep:
    """Create a 1-qubit amplitude damping channel.

    Parameters
    ----------
    prob : Float
        Probability of damping

    qubit : str | int
        Target qubit for [](api:KrausGateRep) construction)

    Returns
    -------
    KrausGateRep
        The amplitude damping channel
    """
    # Sanity checks
    assert prob >= 0 and prob <= 1

    A0 = np.array([[1, 0], [0, np.sqrt(1 - prob)]])
    A1 = np.array([[0, np.sqrt(prob)], [0, 0]])

    return KrausGateRep([(A0, None), (A1, None)], [qubit])


def _get_kraus_rep(rt: GateRep) -> List[Tuple[np.ndarray, float]]:
    assert isinstance(rt, (KrausGateRep, UnitaryGateRep))

    if isinstance(rt, UnitaryGateRep):
        assert isinstance(rt.unitary, np.ndarray)
        return [(rt.unitary, 1.0)]

    assert isinstance(rt.kraus_operators, (list, tuple))
    assert all([isinstance(r[0], np.ndarray) for r in rt.kraus_operators])
    return rt.kraus_operators


def dedup_kraus_reptuple(rt: KrausGateRep) -> KrausGateRep:
    """Deduplicate a [](api:KrausGateRep).

    The effectively normalizes all the Kraus operators
    and checks for any duplicates. If duplicates are found,
    the entries are consolidated into a single Kraus operator
    with the combined magnitude of all duplicates.

    Parameters
    ----------
    rt:
        The [](api:KrausGateRep) to deduplicate

    Returns
    -------
    KrausGateRep
        The deduplicated Kraus representation
    """
    Ks = _get_kraus_rep(rt)

    # Need to think about how to dedup non-unitaries with no fixed probabilities
    if not all([K[1] is not None for K in Ks]):
        raise ValueError(
            "Cannot deduplicate non-unital Kraus operators currently"
        )

    N = len(rt.qubits)
    # This will be a set of tuples of (normalized K, total sum of probability)
    # The last column is needed to unnormalized non-unital Kraus operators that do not have a state-independent probability
    # For unital Kraus operators, columns 2 and 3 should be the same
    normalized_Ks = []

    def dedup_K(Krep):
        Knormed = Krep[0] / np.sqrt(Krep[1])

        matched = False
        for i in range(len(normalized_Ks)):
            # Check the same up to phase
            if np.isclose(np.abs(np.vdot(Knormed, normalized_Ks[i][0])), 2**N):
                matched = True
                # We are the same an existing (normalized) Kraus matrix
                # Update the probability value
                normalized_Ks[i][1] += Krep[1]

                # Don't need to test any more
                break

        if not matched:
            # If we are here, didn't match any Kraus rep
            normalized_Ks.append([Knormed, Krep[1]])

    for Krep in Ks:
        dedup_K(Krep)

    # Unnormalize resulting unique Kraus matrices
    deduped_kraus_reps = [
        (Kn[0] * np.sqrt(Kn[1]), Kn[1]) for Kn in normalized_Ks
    ]

    return KrausGateRep(deduped_kraus_reps, rt.qubits)


def compose_kraus_reptuples(
    rt1: GateRep, rt2: GateRep, dedup: bool = True
) -> KrausGateRep:
    r"""Compose two Kraus/unitary gate representations together.

    Essentially just foils them out:

    .. math:

        M_{i,j} = \sum_i \sum_j c_i c_j K_i L_j

    where `K,L` are the Kraus operators for the two incoming channels
    and `M` is the combined output channel.

    It is possible for multiple `K_i L_j` terms to correspond to the same
    Kraus operator. The :param:`dedup` flag controls whether or not
    these duplicates are consolidated.

    Parameters
    ----------
    rt1 : GateRep
        The first [](api:KrausGateRep) or [](api:UnitaryGateRep).

    rt2 : GateRep
        The second [](api:KrausGateRep) or [](api:UnitaryGateRep).

    dedup : bool, optional
        Whether (True, default) or not (False) to deduplicate
        the output Kraus channel, by default True

    Returns
    -------
    KrausGateRep
        The output channel
    """
    assert rt1.qubits == rt2.qubits

    K1s = _get_kraus_rep(rt1)
    K2s = _get_kraus_rep(rt2)

    # We compose by foiling out the terms
    # Probabilities multiply, if available
    new_kraus_reps = []
    for K1 in K1s:
        for K2 in K2s:
            new_K = K2[0] @ K1[0]

            try:
                new_prob = K1[1] * K2[1]
            except TypeError:
                # One was None, we can't compute the probability
                new_prob = None

            new_kraus_reps.append((new_K, new_prob))

    new_rt = KrausGateRep(new_kraus_reps, rt1.qubits)

    if dedup is False:
        return new_rt

    # Make sure we only have unique Kraus operators
    try:
        deduped_rt = dedup_kraus_reptuple(new_rt)
    except ValueError:
        # Failed to dedup, just return undeduped version
        return new_rt

    return deduped_rt

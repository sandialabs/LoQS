#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.3                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""A collection of tools useful for [](api:QSimQuantumState) objects.

Primarily verbose printing for debugging currently.
"""

from typing import Any

# Conditional import for QSimQuantumState
try:
    from loqs.backends import QSimQuantumState
except ImportError:
    QSimQuantumState = Any  # type: ignore


def get_state_probs_phases(qsim_state: QSimQuantumState):
    """Compute computational basis probabilities and phases.

    This leaves the input state unchanged, but returns the probabilities
    of measuring each computational basis state and phase relative to
    the smallest bitstring with nonzero probability (the all 0 string
    when it has amplitude). If a basis state is missing, it is assumed to
    have no probability of being observed; `phases` has exactly the
    bitstrings with nonzero probability.

    Phases are found by a walk over single bit flips between nonzero
    bitstrings, with a Hadamard test on each flipped qubit. A `ValueError`
    is raised, rather than a wrong sign returned, when the support isn't
    linked by single flips (e.g. a GHZ state) or a phase is complex. It
    may also be raised, though this isn't guaranteed, for a mixed state.

    Parameters
    ----------
    qsim_state:
        State backend object to analyze

    Returns
    -------
    qubits:
        List of qubit labels

    probs:
        A dictionary of bitstring keys and probability values

    phases:
        A dictionary of bitstring keys and phase (relative to the
        smallest nonzero bitstring) values
    """
    qubits = qsim_state.state.names
    assert qubits is not None

    def get_bitstring_dict():
        probs = {}
        for bd, p in qsim_state.state.peak_multiple_measurements(qubits):
            bitstring = "".join([str(bd[q]) for q in qubits])
            probs[bitstring] = p if p > 1e-8 else 0
        return probs

    probs = get_bitstring_dict()

    # Walk the nonzero strings along single bit flips, in both directions,
    # starting from the smallest one (phase 1).
    nonzero = {bs for bs, p in probs.items() if p > 0}
    phases = {min(nonzero): 1}
    to_visit = list(phases)
    while to_visit:
        bs = to_visit.pop()
        for i in range(len(bs)):
            nb = bs[:i] + str(1 - int(bs[i])) + bs[i + 1 :]
            if nb in nonzero and nb not in phases:
                # Hadamard the flipped qubit and read the pair's 1-branch
                # string. For real amplitudes, q = (pa + pb)/2 - s*sqrt(pa*pb)
                # with s the relative sign, so c is +/-1; anything else
                # means a complex phase or a mixed state.
                qsim_state.state.hadamard(qubits[i])
                q = get_bitstring_dict().get(max(bs, nb), 0)
                qsim_state.state.hadamard(qubits[i])
                pa, pb = probs[bs], probs[nb]
                c = ((pa + pb) / 2 - q) / (pa * pb) ** 0.5
                if abs(abs(c) - 1) > 1e-6:
                    raise ValueError(
                        f"Cannot sign bitstring {nb}: its Hadamard test "
                        f"against {bs} fits neither a + nor a - phase"
                    )
                phases[nb] = phases[bs] * (1 if c > 0 else -1)
                to_visit.append(nb)

    if len(phases) < len(nonzero):
        raise ValueError(
            f"Cannot sign bitstring {min(nonzero - phases.keys())}: it is "
            "not linked by single bit flips through nonzero-probability "
            "strings to the smallest one"
        )

    return qubits, probs, phases


def print_state_probs_phases(qsim_state: QSimQuantumState):
    """Pretty-print the output of [](api:get_state_probs_phases).

    Refer to [](api:get_state_probs_phases) for documentation.
    """
    qubits, probs, phases = get_state_probs_phases(qsim_state)
    print(qubits)
    for bs, p in sorted(probs.items()):
        if p < 1e-8:
            continue
        phase = "+" if phases[bs] > 0 else "-"
        print(f"{bs}: {p} (Phase: {phase})")

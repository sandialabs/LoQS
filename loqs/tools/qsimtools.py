"""A collection of tools useful for :class:`.QSimQuantumState` objects.

Primarily verbose printing for debugging currently.
"""

from loqs.backends.state import QSimQuantumState


def get_state_probs_phases(qsim_state: QSimQuantumState):
    """Compute computational basis probabilities and phases.

    This leaves the input state unchanged, but returns the probabilities
    of measuring each computational basis state and phase relative to
    the all 0 state. If a basis state is missing, it is assumed to have
    no probability of being observed.

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
        all 0 state) values
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

    # Do graph traversal to compute relative phases
    phases = {}
    bitstrings_to_visit = set([("0" * 7, "None", -1)])
    seen_bitstrings = set()
    while len(bitstrings_to_visit):
        bs, off_by_one_bs, off_by_one_idx = bitstrings_to_visit.pop()
        if bs not in probs:
            # We don't care about this bitstring, continue
            pass
        elif off_by_one_bs == "None":
            # Must be all zeros, hard code this to + phase
            phases[bs] = 1
        else:
            # We can get our phase relative to the last bitstring
            # Hadamard the qubit where the bitflip is
            # If the 1 branch has the probability, then there is a -1 relative phase
            # Otherwise, they have the same relative phase
            qsim_state.state.hadamard(qubits[off_by_one_idx])
            phase_probs = get_bitstring_dict()
            qsim_state.state.hadamard(qubits[off_by_one_idx])

            # bs will always be the bitstring with the 1 branch,
            # so we can use that for a shorthand key
            factor = 1
            if phase_probs[bs] > 1e-8:
                factor = -1
            phases[bs] = factor * phases[off_by_one_bs]

        seen_bitstrings.add(bs)

        # Add all neighboring bitstrings that we haven't seen yet
        for i in range(len(bs)):
            new_bs = bs[:i] + "1" + bs[i + 1 :]
            if new_bs not in seen_bitstrings:
                bitstrings_to_visit.add((new_bs, bs, i))

    return qubits, probs, phases


def print_state_probs_phases(qsim_state: QSimQuantumState):
    """Pretty-print the output of :meth:`get_state_probs_phases`.

    Refer to :meth:`get_state_probs_phases` for documentation.
    """
    qubits, probs, phases = get_state_probs_phases(qsim_state)
    print(qubits)
    for bs, p in sorted(probs.items()):
        if p < 1e-8:
            continue
        phase = "+" if phases[bs] > 0 else "-"
        print(f"{bs}: {p} (Phase: {phase})")

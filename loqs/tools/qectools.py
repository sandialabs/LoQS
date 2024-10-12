"""A collection of tools useful for QEC.

Particularly for syndrome/Pauli frame/lookup table
manipulation and calculation.
"""

from collections import defaultdict
from collections.abc import Sequence
import itertools
from typing import Literal


def get_syndrome_from_stabilizers_and_pstr(
    stabilizers: Sequence[str], pstr: str
) -> str:
    """Compute a syndrome for a Pauli string given stabilizers.

    The computation here is as follows: for each stabilizer,
    check how many entries of ``pstr`` anticommute with the
    corresponding entry in the stabilizer. If there are an even
    number of anticommutations, the stabilizer will measure 0;
    otherwise, it will measure 1. Repeat for each stabilizer
    to build up the syndrome bitstring.

    Parameters
    ----------
    stabilizers:
        A sequence of Pauli strings representing the stabilizers

    pstr:
        The Pauli string to compute the syndrome for

    Returns
    -------
    str
        Syndrome bitstring as a string of ``"0"``s and ``"1"``s
    """
    assert all([len(s) == len(pstr) for s in stabilizers])
    for stab in stabilizers:
        assert all([c in "IXYZ" for c in stab])
    assert all([c in "IXYZ" for c in pstr])

    syndrome = ""
    for stab in stabilizers:
        non_commutes = 0
        for pstab, perr in zip(stab, pstr):
            if (
                (pstab == "X" and perr in "YZ")
                or (pstab == "Y" and perr != "I")
                or (pstab == "Z" and perr in "XY")
            ):
                non_commutes += 1
        # Our syndrome bit is the parity of noncommuting paulis
        syndrome += str(non_commutes % 2)

    return syndrome


def get_syndrome_dict_from_stabilizers_and_pstrs(
    stabilizers: Sequence[str],
    pstrs: Sequence[str],
    default_pstr: str | Literal["auto"] | None = "auto",
) -> dict[str, list[str]]:
    """Call :meth:`get_syndrome_from_stabilizers_and_pstr` for many Pauli strings.

    The output of this function can be used as a lookup table
    decoder if there is only a single entry -- i.e. data error --
    per syndrome key.

    Parameters
    ----------
    stabilizers:
        See :meth:`get_syndrome_from_stabilizers_and_pstr`.

    pstrs:
        List of Pauli strings, see
        :meth:`get_syndrome_from_stabilizers_and_pstr`.

    default_pstr:
        A default Pauli string to use for syndromes that do not
        have a corresponding entry in ``pstrs``. Can be a Pauli
        string, ``None`` to add no default, or ``"auto"``, where
        a Pauli string of all ``"I"`` of the correct length is
        used. Defaults to ``"auto"``.

    Returns
    -------
    dict[str, list[str]]
        A dictionary with syndrome string keys and a list of all
        corresponding Pauli strings as values.
    """
    raw_syndrome_dict = defaultdict(list)

    for pstr in pstrs:
        syndrome = get_syndrome_from_stabilizers_and_pstr(stabilizers, pstr)
        raw_syndrome_dict[syndrome].append(pstr)

    if default_pstr == "auto":
        num_data_qubits = len(stabilizers[0])
        default_pstr = "I" * num_data_qubits

    # Run through syndromes and add default if specified
    syndrome_dict = {}
    syndromes = [
        "".join(syndrome)
        for syndrome in itertools.product("01", repeat=len(stabilizers))
    ]
    for syndrome in syndromes:
        pstr_list = raw_syndrome_dict.get(syndrome, None)
        if pstr_list is None:
            if default_pstr is not None:
                pstr_list = [default_pstr]
            else:
                continue
        syndrome_dict[syndrome] = pstr_list

    return syndrome_dict


def get_weight_1_errors(num_qubits: int) -> list[str]:
    """Compute Pauli strings for weight-1 errors.

    The output of this can serve as the ``pstrs`` input to
    :meth:`.get_syndrome_dict_from_stabilizers_and_pstrs`
    for the purpose of computing lookup tables for correcting
    data errors.
    For an example, see the ``"Unflagged Decoder"`` instruction
    in :mod:`.codepack_5_1_3_quantinuum2022`.

    Parameters
    ----------
    num_qubits:
        The number of data qubits

    Returns
    -------
    list[str]
        All possible weight-1 Pauli strings
    """
    errors = []
    for i in range(num_qubits):
        for p in "XYZ":
            error = [
                "I",
            ] * num_qubits
            error[i] = p
            errors.append("".join(error))

    return errors


def get_hook_errors_in_flagged_check(
    stabilizer: str, check_order: Sequence[int] | None = None
) -> list[str]:
    """Compute Pauli strings for hook errors in flagged check circuits.

    This is an automated version of the calculation performed to get
    the data errors in Fig. 2d of arXiv:1705.02329.

    The output of this can serve as the ``pstrs`` input to
    :meth:`.get_syndrome_dict_from_stabilizers_and_pstrs`
    for the purpose of computing lookup tables for correcting
    measurement errors that result in hook errors.
    For an example, see the ``"Flagged <stab> Decoder"`` instructions
    in :mod:`.codepack_5_1_3_quantinuum2022`.

    Parameters
    ----------
    stabilizer:
        Pauli string of the stabilizer to check

    check_order:
        The order of qubits checked in the stabilizer.
        This is important because the first and last checks cannot spread,
        but that is not always done in ascending qubit order.
        Defaults to None, which does the checks in ascending qubit order.

    Returns
    -------
    list[str]
        All possible hook error Pauli strings
    """
    assert all([c in "IXYZ" for c in stabilizer])

    # Get weight and default check order if needed
    weight = sum([c != "I" for c in stabilizer])
    if check_order is None:
        check_order = [i for i, p in enumerate(stabilizer) if p != "I"]
    assert len(check_order) == weight

    # Run through checks and calculate hook errors
    hook_errors = []
    for i, stab_idx in enumerate(check_order):
        p = stabilizer[stab_idx]

        # Skip first and last check
        if i in [0, weight - 1]:
            continue

        # Any downstream checks will be triggered
        downstream_errors = ["I"] * len(stabilizer)
        for cidx in check_order[i + 1 :]:
            downstream_errors[cidx] = stabilizer[cidx]

        # On this qubit, we have all possible paulis
        for p in "IXYZ":
            hook_error = downstream_errors.copy()
            hook_error[stab_idx] = p
            hook_errors.append("".join(hook_error))

    return hook_errors


def compose_pstrs(pstr1: str, pstr2: str) -> str:
    """Multiply two Pauli strings.

    Among other uses, it can be used to apply Pauli string
    corrections to a frame. Mathematically, it is the same
    as :meth:`.PauliFrame.update_from_pauli_str`, but without
    requiring one of the Pauli strings to be wrapped up in
    a :class:`.PauliFrame`.

    Parameters
    ----------
    pstr1:
        First Pauli string

    pstr2:
        Second Pauli string

    Returns
    -------
    str
        Product of the two Pauli strings
    """
    assert len(pstr1) == len(pstr2)
    assert all([c in "IXYZ" for c in pstr1])
    assert all([c in "IXYZ" for c in pstr2])

    composed = ""
    for p1, p2 in zip(pstr1, pstr2):
        if p1 == p2:
            composed += "I"
        elif set((p1, p2)) in [set("YZ"), set("IX")]:
            composed += "X"
        elif set((p1, p2)) in [set("XZ"), set("IY")]:
            composed += "Y"
        else:
            composed += "Z"

    return composed


def compose_pstr_lists(
    pstr_list1: Sequence[str], pstr_list2: Sequence[str]
) -> list[str]:
    """Perform :meth:`compose_pstrs` on two sets of Pauli strings.

    Parameters
    ----------
    pstr_list1:
        First set of Pauli strings

    pstr_list2:
        Second set of Pauli strings

    Returns
    -------
    list[str]
        A list of Pauli products between every string in the first
        set with every string in the second set
    """
    composed_pstrs = []
    for pstr1 in pstr_list1:
        for pstr2 in pstr_list2:
            composed_pstrs.append(compose_pstrs(pstr1, pstr2))

    return composed_pstrs

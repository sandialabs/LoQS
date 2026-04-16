#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.0                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

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
    """Call (get_syndrome_from_stabilizers_and_pstr)[api:get_syndrome_from_stabilizers_and_pstr] for many Pauli strings.

    The output of this function can be used as a lookup table
    decoder if there is only a single entry -- i.e. data error --
    per syndrome key.

    Parameters
    ----------
    stabilizers : Sequence[str]
        See (get_syndrome_from_stabilizers_and_pstr)[api:get_syndrome_from_stabilizers_and_pstr].

    pstrs : Sequence[str]
        List of Pauli strings, see
        (get_syndrome_from_stabilizers_and_pstr)[api:get_syndrome_from_stabilizers_and_pstr].

    default_pstr : str | Literal["auto"] | None, optional
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

    REVIEW_SPHINX_REFERENCE
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
    (get_syndrome_dict_from_stabilizers_and_pstrs)[api:get_syndrome_dict_from_stabilizers_and_pstrs]
    for the purpose of computing lookup tables for correcting
    data errors.
    For an example, see the ``"Unflagged Decoder"`` instruction
    in (codepack_5_1_3_quantinuum2022)[api:codepack_5_1_3_quantinuum2022].

    Parameters
    ----------
    num_qubits : int
        The number of data qubits

    Returns
    -------
    list[str]
        All possible weight-1 Pauli strings

    REVIEW_SPHINX_REFERENCE
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
    (get_syndrome_dict_from_stabilizers_and_pstrs)[api:get_syndrome_dict_from_stabilizers_and_pstrs]
    for the purpose of computing lookup tables for correcting
    measurement errors that result in hook errors.
    For an example, see the ``"Flagged <stab> Decoder"`` instructions
    in (codepack_5_1_3_quantinuum2022)[api:codepack_5_1_3_quantinuum2022].

    Parameters
    ----------
    stabilizer : str
        Pauli string of the stabilizer to check

    check_order : Sequence[int] | None, optional
        The order of qubits checked in the stabilizer.
        This is important because the first and last checks cannot spread,
        but that is not always done in ascending qubit order.
        Defaults to None, which does the checks in ascending qubit order.

    Returns
    -------
    list[str]
        All possible hook error Pauli strings

    REVIEW_SPHINX_REFERENCE
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
    as (PauliFrame.update_from_pauli_str)[api:PauliFrame.update_from_pauli_str], but without
    requiring one of the Pauli strings to be wrapped up in
    a (PauliFrame)[api:PauliFrame].

    Parameters
    ----------
    pstr1 : str
        First Pauli string

    pstr2 : str
        Second Pauli string

    Returns
    -------
    str
        Product of the two Pauli strings

    REVIEW_SPHINX_REFERENCE
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
    """Perform (compose_pstrs)[api:compose_pstrs] on two sets of Pauli strings.

    Parameters
    ----------
    pstr_list1 : Sequence[str]
        First set of Pauli strings

    pstr_list2 : Sequence[str]
        Second set of Pauli strings

    Returns
    -------
    list[str]
        A list of Pauli products between every string in the first
        set with every string in the second set

    REVIEW_SPHINX_REFERENCE
    """
    composed_pstrs = []
    for pstr1 in pstr_list1:
        for pstr2 in pstr_list2:
            composed_pstrs.append(compose_pstrs(pstr1, pstr2))

    return composed_pstrs

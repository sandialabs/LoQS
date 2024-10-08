"""TODO"""

from collections import defaultdict
from collections.abc import Sequence
import itertools
from typing import Literal


def get_syndrome_from_stabilizers_and_pstr(
    stabilizers: Sequence[str], pstr: str
) -> str:
    """TODO"""
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
    """TODO"""
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
        pstr_list = raw_syndrome_dict.get(syndrome, [default_pstr])
        if pstr_list is None:
            continue
        syndrome_dict[syndrome] = pstr_list

    return syndrome_dict


def get_weight_1_errors(num_qubits: int) -> list[str]:
    """TODO"""
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
    """TODO

    This is an automated version of the calculation performed to get
    the data errors in Fig. 2d of arXiv:1705.02329.
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
    """TODO"""
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
    """TODO"""
    composed_pstrs = []
    for pstr1 in pstr_list1:
        for pstr2 in pstr_list2:
            composed_pstrs.append(compose_pstrs(pstr1, pstr2))

    return composed_pstrs

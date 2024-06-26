"""A LoQS QEC codepack for the 5-qubit code.

TODO
"""

from collections.abc import Sequence
import itertools
import numpy as np

from loqs.backends.circuit import PyGSTiPhysicalCircuit as PhysicalCircuit
from loqs.core import Instruction, QECCode
from loqs.core.instructions import common as ic


def create_qec_code():
    """TODO"""

    # Template qubits for defining one patch
    qubits = ["A0", "A1"] + [f"D{i}" for i in range(5)]

    operations: dict[str, Instruction] = {}

    # Non-FT |-> state prep
    # First gray box of Fig 3 of arxiv:2208.01863
    nonft_state_prep_circ = PhysicalCircuit(
        [
            [
                ("Gh", "D0"),
                ("Gh", "D1"),
                ("Gh", "D2"),
                ("Gh", "D3"),
                ("Gh", "D4"),
            ],
            [("Gcphase", "D0", "D1"), ("Gcphase", "D2", "D3")],
            [("Gcphase", "D1", "D2"), ("Gcphase", "D3", "D4")],
            [("Gcphase", "D0", "D4")],
        ],
        qubit_labels=qubits,
    )

    # TODO: Verify this is actually minus
    operations["Non-FT Minus Prep"] = ic.build_physical_circuit_instruction(
        nonft_state_prep_circ,
        include_outcomes=False,
        name="Non-fault-tolerant minus state prep",
        fault_tolerant=False,
    )

    # Try-until-success FT |-> state prep
    # First green box of Fig 3 of arxiv:2208.01863
    # ft_state_prep_checks_circ = PhysicalCircuit(
    #     [
    #         # FT check 1
    #         ("Gcnot", "A0", "D0"),
    #         ("Gcnot", "A0", "A1"),
    #         ("Gcphase", "A0", "D1"),
    #         ("Gcnot", "A0", "A1"),
    #         ("Gcphase", "A0", "D4"),
    #         ("Gh", "A0"),
    #         [("Iz", "A0"), ("Iz", "A1")],
    #         # FT check 2
    #         ("Gcphase", "A0", "D0"),
    #         ("Gcnot", "A0", "A1"),
    #         ("Gcnot", "A0", "D1"),
    #         ("Gcnot", "A0", "A1"),
    #         ("Gcphase", "A0", "D2"),
    #         ("Gh", "A0"),
    #         [("Iz", "A0"), ("Iz", "A1")],
    #         # FT check 3
    #         ("Gcphase", "A0", "D0"),
    #         ("Gcnot", "A0", "A1"),
    #         ("Gcnot", "A0", "D1"),
    #         ("Gcnot", "A0", "A1"),
    #         ("Gcphase", "A0", "D2"),
    #         ("Gh", "A0"),
    #         [("Iz", "A0"), ("Iz", "A1")],
    #     ],
    #     qubit_labels=qubits,
    # )
    # ft_state_prep_circ = nonft_state_prep_circ.append(
    #     ft_state_prep_checks_circ
    # )
    # ft_state_prep = ic.build_physical_circuit_instruction(
    #     ft_state_prep_circ,
    #     include_outcomes=True,
    #     reset_mcms=True,
    #     name="Non-fault-tolerant minus state prep",
    #     fault_tolerant=True,
    # )

    # operations["FT Minus Prep"] = RepeatUntilSuccess(
    #     ft_state_prep,
    #     name="Repeat-until-success fault-tolerant minus state prep",
    # )

    # Logical X (transversal)
    # Eqn B1 of arxiv:2208.01863
    logical_X_circ = PhysicalCircuit(
        [[("Gypi2", "D0"), ("Gxpi2", "D2"), ("Gypi2", "D4")]],
        qubit_labels=qubits,
    )
    operations["X"] = ic.build_physical_circuit_instruction(
        logical_X_circ,
        include_outcomes=False,
        name="Logical X",
        fault_tolerant=True,
    )

    # Logical Z (transversal)
    # Eqn B3 of arxiv:2208.01863
    logical_Z_circ = PhysicalCircuit(
        [[("Gxpi2", "D0"), ("Gzpi2", "D2"), ("Gxpi2", "D4")]],
        qubit_labels=qubits,
    )
    operations["Z"] = ic.build_physical_circuit_instruction(
        logical_Z_circ,
        include_outcomes=False,
        name="Logical Z",
        fault_tolerant=True,
    )

    # Logical H (transversal + permute)
    # Fig 2b of arxiv:1603.03948
    logical_H_circ = PhysicalCircuit(
        [[("Gh", q) for q in qubits[2:]]], qubit_labels=qubits
    )
    logical_H_circ_inst = ic.build_physical_circuit_instruction(
        logical_H_circ,
        include_outcomes=False,
        name="Logical H circuit",
        fault_tolerant=True,
    )
    logical_H_permutation = ic.build_patch_permute_instruction(
        {  # final: initial
            "D0": "D1",
            "D1": "D4",
            # D2 is unpermuted
            "D3": "D0",
            "D4": "D3",
        },
        name="Logical H permutation",
    )

    operations["H"] = ic.build_composite_instruction(
        [logical_H_circ_inst, logical_H_permutation], name="Logical H"
    )

    # Raw physical measurement
    operations["Measure Physical Qubits"] = (
        ic.build_physical_circuit_instruction(
            PhysicalCircuit([("Iz", q) for q in qubits], qubit_labels=qubits),
            include_outcomes=True,
            reset_mcms=False,
            name="Z-basis measurement for physical qubits",
            fault_tolerant=False,
        )
    )

    # Eqn B4-B7 of arxiv:2208.01863
    # stabilizers = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
    # syndrome_circuit = PhysicalCircuit([], qubit_labels=qubits)
    # syndrome_qubits = {}
    # for i, stab in enumerate(stabilizers):
    #     stabilizer_circuit = _create_syndrome_check_circuit(stab)
    #     syndrome_circuit.append_inplace(stabilizer_circuit)
    #     syndrome_qubits[stab] = ("A0", i)

    # TODO: Current SE is not FT
    # operations["SE"] = SyndromeExtraction(
    #     syndrome_circuit, syndrome_qubits, fault_tolerant=False
    # )

    # Decoder (computed from commutation relations to stabilizers)
    # lookup_table = _create_decoder_lookup(stabilizers)
    # operations["Decode"] = Decoder(lookup_table)

    # TODO: Combined QEC instruction (and two round version)

    # TODO: Logical CZ and CCZ

    return QECCode(operations, qubits)


## Helper functions
# def _create_adaptive_measure_instruction(qubits) -> Instruction:
#     # FT Adaptive Measurement Scheme
#     # Fig 13 of arxiv:2208.01863
#     measI_circ = PhysicalCircuit(
#         [
#             ("Gh", "A0"),
#             ("Gcphase", "A0", "D4"),
#             ("Gcnot", "A0", "A1"),
#             ("Gcnot", "A0", "D0"),
#             ("Gcnot", "A0", "A1"),
#             ("Gcphase", "A0", "D1"),
#             ("Gh", "A0"),
#             [("Iz", "A0"), ("Iz", "A1")],
#         ],
#         qubit_labels=qubits,
#     )
#     measI = QuantumClassicalLogicalOperation(
#         measI_circ, name="Part I of Adaptive Measure", reset_mcms=True
#     )

#     measII_circ = PhysicalCircuit(
#         [
#             ("Gh", "A0"),
#             ("Gcphase", "A0", "D0"),
#             ("Gcnot", "A0", "A1"),
#             ("Gcnot", "A0", "D1"),
#             ("Gcnot", "A0", "A1"),
#             ("Gcphase", "A0", "D2"),
#             ("Gh", "A0"),
#             [("Iz", "A0"), ("Iz", "A1")],
#         ],
#         qubit_labels=qubits,
#     )
#     measII = QuantumClassicalLogicalOperation(
#         measII_circ, name="Part II of Adaptive Measure", reset_mcms=True
#     )

#     measIII_circ = PhysicalCircuit(
#         [
#             ("Gh", "A0"),
#             ("Gcphase", "A0", "D2"),
#             ("Gcnot", "A0", "A1"),
#             ("Gcnot", "A0", "D3"),
#             ("Gcnot", "A0", "A1"),
#             ("Gcphase", "A0", "D4"),
#             ("Gh", "A0"),
#             [("Iz", "A0"), ("Iz", "A1")],
#         ],
#         qubit_labels=qubits,
#     )
#     measIII = QuantumClassicalLogicalOperation(
#         measIII_circ, name="Part III of Adaptive Measure", reset_mcms=True
#     )

#     # TODO: Really this is the nonft state prep reversed
#     state_decoder_circ = PhysicalCircuit(
#         [
#             [("Gcphase", "D0", "D4")],
#             [("Gcphase", "D1", "D2"), ("Gcphase", "D3", "D4")],
#             [("Gcphase", "D0", "D1"), ("Gcphase", "D2", "D3")],
#             [
#                 ("Gh", "D0"),
#                 ("Gh", "D1"),
#                 ("Gh", "D2"),
#                 ("Gh", "D3"),
#                 ("Gh", "D4"),
#             ],
#             [
#                 ("Iz", "D0"),
#                 ("Iz", "D1"),
#                 ("Iz", "D2"),
#                 ("Iz", "D3"),
#                 ("Iz", "D4"),
#             ],
#         ],
#         qubit_labels=qubits,
#     )

#     state_decoder = QuantumClassicalLogicalOperation(
#         state_decoder_circ,
#         name="Decoder circuit of Adaptive Measure",
#         reset_mcms=True,
#     )

#     # Flowchart for Fig 13
#     # Easiest to go from right to left when building up instructions
#     # In addition to all normal state, we will also store a "logical_outcome"
#     # which will correspond to M1 for most of the process

#     # Part III has two outcomes. If flag is 0 and measurement doesn't match,
#     # apply decoder. Otherwise, return original measurement
#     def partIII_fn(input: History) -> Frame:
#         pass

#     pass


# def _create_syndrome_check_circuit(stabilizer: str) -> PhysicalCircuit:
#     """TODO

#     Essentially the Wikipedia version of syndrome extraction.
#     TODO: Not FT to X errors on auxiliary?
#     Either use 2 qubits or use Las Vegas approach?
#     """
#     assert all(
#         [p in "IXZ" for p in stabilizer]
#     ), "Stabilizer must be Pauli string with only I, X, or Z"

#     layers = []

#     layers.append(("Gh", "A0"))

#     for Zloc in np.where([p == "Z" for p in stabilizer])[0]:
#         layers.append(("Gcphase", "A0", f"D{Zloc}"))

#     for Xloc in np.where([p == "X" for p in stabilizer])[0]:
#         layers.append(("Gcnot", "A0", f"D{Xloc}"))

#     layers.append(("Gh", "A0"))

#     layers.append(("Iz", "A0"))

#     qubits = ["A0"] + [f"D{i}" for i in range(len(stabilizer))]

#     return PhysicalCircuit(layers, qubit_labels=qubits)


# def _create_decoder_lookup(
#     stabilizers: Sequence[str],
# ) -> dict[tuple[int], tuple[str, str]]:
#     """TODO"""
#     qubits = [f"D{i}" for i in range(len(stabilizers[0]))]
#     reverse_lookup = {}
#     for p, qi in itertools.product("XYZ", range(len(qubits))):
#         syndrome = []
#         for stab in stabilizers:
#             if stab[qi] in (p, "I"):
#                 # Commute, should get 0
#                 syndrome.append(0)
#             else:
#                 # Don't commute, should get 1
#                 syndrome.append(1)

#         reverse_lookup[p, qubits[qi]] = tuple(syndrome)

#     lookup = {v: k for k, v in reverse_lookup.items()}
#     return lookup

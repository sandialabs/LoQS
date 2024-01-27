"""A LoQS QEC codepack for the surface code as implemented in Tomita & Svore 2014.
"""

from pygsti.circuits import Circuit

from loqs.core.syndromecircuit import (
    StabilizerPlaquetteFactory,
    SyndromeCircuit,
)


# TODO: Moving away from just including python files, we can do like pyGSTi modelpacks
# but what should be the objects/interface?

# One minor modification to prevent collisions with the alternate schedule
# The alt schedule swaps a/d checks, so we will insert spacers
# For the normal schedule, we will do a/d/space
# For the alt schedule, we will do space/d/a
# This spacer will be removed by delete_idle_layers=True (default) for the normal schedule
surface_factory = StabilizerPlaquetteFactory(
    {
        # Fig 2a
        "X": Circuit(
            [
                ("Gh", "aux"),
                ("Gcnot", "aux", "b"),
                ("Gcnot", "aux", "a"),
                ("Gcnot", "aux", "d"),
                [],
                ("Gcnot", "aux", "c"),
                ("Gh", "aux"),
                ("Iz", "aux"),
            ],
            line_labels=["aux", "a", "b", "c", "d"],
        ),
        # Fig 2b (including spacers for H layers)
        "Z": Circuit(
            [
                [],
                ("Gcnot", "b", "aux"),
                ("Gcnot", "a", "aux"),
                ("Gcnot", "d", "aux"),
                [],
                ("Gcnot", "c", "aux"),
                [],
                ("Iz", "aux"),
            ],
            line_labels=["aux", "a", "b", "c", "d"],
        ),
        # Fig 4b (including spacers for H layers)
        "Zalt": Circuit(
            [
                [],
                ("Gcnot", "b", "aux"),
                [],
                ("Gcnot", "d", "aux"),
                ("Gcnot", "a", "aux"),
                ("Gcnot", "c", "aux"),
                [],
                ("Iz", "aux"),
            ],
            line_labels=["aux", "a", "b", "c", "d"],
        ),
    }
)

##############
# SURFACE-25 #
##############
surface25_qubits = [f"D{i}" for i in range(12)] + [
    f"A{i}" for i in range(12, 15)
]

# For Surface-25, we can perform all stabilizers at once (Fig 1a)
# Note that for weight-3 stabilizers, we still use the weight-4 template
# but pass in no qubit.
# This will result in that CNOT being dropped when the template is filled out
# (which is what we want to maintain a proper schedule without collisions)
# Note that b = top, a = right, d = left, and c = bottom for the unrotated surface-25,
# i.e. data qubit order should start at the right and go clockwise
surface25_stabilizers = {
    "X": [
        ["A13", "D0", None, "D1", "D3"],
        ["A14", "D1", None, "D2", "D4"],
        ["A18", "D5", "D3", "D6", "D8"],
        ["A19", "D6", "D4", "D7", "D9"],
        ["A23", "D10", "D8", "D11", None],
        ["A24", "D11", "D9", "D12", None],
    ],
    "Z": [
        ["A15", None, "D0", "D3", "D5"],
        ["A16", "D3", "D1", "D4", "D6"],
        ["A17", "D4", "D2", None, "D7"],
        ["A20", None, "D5", "D8", "D10"],
        ["A21", "D8", "D6", "D9", "D11"],
        ["A22", "D9", "D7", None, "D12"],
    ],
}
surface25_syndrome = SyndromeCircuit(
    factory=surface_factory,
    stabilizers=surface25_stabilizers,
    qubit_labels=surface25_qubits,
)

# We can also easily use the alternate schedule by swapping out the stabilizer type
# since the factory knows how to make both checks already
surface25_syndrome_alt = surface25_syndrome.map_stabilizer_keys({"Z": "Zalt"})

##############
# SURFACE-17 #
##############
surface17_qubits = [f"D{i}" for i in range(9)] + [
    f"A{i}" for i in range(9, 17)
]

# For Surface-17, we can also perform all stabilizers at once (Fig 1b)
# This time, we have weight-2 checks instead of weight-3
surface17_stabilizers = {
    "X": [
        ["A9", None, None, "D1", "D2"],
        ["A11", "D0", "D1", "D3", "D4"],
        ["A14", "D4", "D5", "D7", "D8"],
        ["A16", "D6", "D7", None, None],
    ],
    "Z": [
        ["A10", None, "D0", None, "D3"],
        ["A12", "D1", "D2", "D4", "D5"],
        ["A13", "D3", "D4", "D6", "D7"],
        ["A15", "D5", None, "D8", None],
    ],
}
surface17_syndrome = SyndromeCircuit(
    factory=surface_factory,
    stabilizers=surface17_stabilizers,
    qubit_labels=surface17_qubits,
)

# Again, can use alternate Z stabilizer schedule
surface17_syndrome_alt = surface17_syndrome.map_stabilizer_keys({"Z": "Zalt"})

##############
# SURFACE-13 #
##############
surface13_qubits = [f"D{i}" for i in range(9)] + [
    f"A{i}" for i in range(9, 13)
]

# For Surface-13, we perform the stabilizers in two stages to reuse aux qubits (Fig 1c)
surface13_stabilizers = [
    {
        # First we do all weight-4 checks
        "X": [
            ["A9", "D0", "D1", "D3", "D4"],
            ["A12", "D4", "D5", "D7", "D8"],
        ],
        "Z": [
            ["A10", "D1", "D2", "D4", "D5"],
            ["A11", "D3", "D4", "D6", "D7"],
        ],
    },
    {
        # And then all weight-2 checks
        # (note that these plaquettes are flipped from Surface-17 weight-2 checks)
        "X": [
            ["A10", "D1", "D2", None, None],
            ["A11", None, None, "D6", "D7"],
        ],
        "Z": [["A9", "D0", None, "D3", None], ["A12", None, "D5", None, "D8"]],
    },
]

surface13_syndrome = SyndromeCircuit(
    factory=surface_factory,
    stabilizers=surface13_stabilizers,
    qubit_labels=surface13_qubits,
)

# Again, can use alternate Z stabilizer schedule
surface13_syndrome_alt = surface13_syndrome.map_stabilizer_keys({"Z": "Zalt"})

#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.3                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Steane [[7,1,3]] (Quantinuum 2021) as a spec.

Paper: Ryan-Anderson et al., [@ryananderson_realizing_2021] (PRX 11,
041058), cited by its own figure, table and section numbers
(arXiv:2107.07505v1 numbers them differently); its qubits 1-7 are D0-D6
here. The circuits are Fig 10's, gate for gate. Instruction names
(`name=`) are LoQS 1.3's verbatim, slips included, so existing scripts
find them; standardizing them is a non-goal.
"""

from loqs.codepacks.generator.schema import (
    CheckCircuit,
    Circuit,
    CodeSpec,
    Composite,
    CustomApplyFn,
    Facts,
    Gate,
    Item,
    LookupTable,
    Parallel,
    RepeatUntilSuccess,
    Role,
    TwoPatchGate,
    ours,
    paper,
)

D = tuple(f"D{i}" for i in range(7))
A = ("A0", "A1", "A2")
ALL_D = " ".join(D)
PRX = "[@ryananderson_realizing_2021]"
FIG1 = PRX + " Fig 1"
FIG10 = PRX + " Fig 10"
PACKED = "CNOTs on disjoint qubits in neighbouring columns share a layer"
FIG1_LOGICALS = ("fig1-logicals.png",)

FACTS = Facts(
    data=D,
    aux=A,
    stabilizer_names=("S1", "S2", "S3", "S4", "S5", "S6"),
    stabilizers=(
        "XXXXIII",
        "IXXIXXI",
        "IIXXIXX",
        "ZZZZIII",
        "IZZIZZI",
        "IIZZIZZ",
    ),
    logicals=(("IIIIXXX", "IIIIZZZ"),),
    src=paper(FIG1, figures=("fig1.png",)),
    frame_layout=("Z", "X"),
    conventions_src=ours(
        "the frame's bit order and the syndrome's stabilizer order are the "
        "ones existing LoQS scripts read"
    ),
)

_CUSTOM = "loqs.codepacks.steane.custom_apply_fns"
_FF = f"{_CUSTOM}:flagged_feedforward"
_CHANGED = "a flagged reading differs from latest_syndrome"
_UNCHANGED = "none differs"
_UNFLAGGED = (
    "Unflagged Parallel S1-S2-S3 Check",
    "Unflagged Parallel S4-S5-S6 Check",
    "QEC Decoder",
)

ITEMS: tuple[Item, ...] = (
    # --- prep ---------------------------------------------------------------
    Circuit(
        "Non-FT Zero Prep",
        """
        Gh D0 D4 D6
        Gcnot D0 D1 D4 D5 D6 D3
        Gcnot D0 D3 D4 D2 D6 D5
        Gcnot D4 D1 D3 D2
        """,
        paper(
            FIG10 + ", encoding circuit",
            "the paper leaves its wires unlabelled; ours are read top to "
            "bottom. " + PACKED,
            figures=("fig10-encoder.png",),
        ),
        name="Non-FT zero state prep",
    ),
    Circuit(
        "Non-FT Zero Prep + Checks",
        """
        include Non-FT Zero Prep
        Gcnot D1 A0
        Gcnot D3 A0
        Gcnot D5 A0
        Imrz A0
        """,
        paper(
            FIG10 + " and Sec II.A.1, encoding circuit and its Z_L check",
            "note: the check measures Z_L·S5·S6 (Z on D1, D3, D5)",
            figures=("fig10-encoder.png",),
        ),
        name="Non-FT Zero Prep + Checks",
    ),
    RepeatUntilSuccess(
        "FT Zero Prep",
        reset="Imrz " + ALL_D,
        body=("Non-FT Zero Prep + Checks",),
        expected=(("A0", (0,)),),
        src=ours(
            "the reset-then-retry loop and the default of 100 repeats; the "
            "paper repeats up to 3 times and then proceeds",
            cite=PRX + " Fig 10 caption, Fig 14",
            figures=("fig14-prep.png",),
        ),
        name="Repeat-until-success FT Zero Prep",
        reset_name="Reset to all 0 state",
    ),
    # --- gates (Fig 1 table) --------------------------------------------------
    Gate(
        "X",
        "Gxpi D4 D5 D6",
        "X",
        paper(FIG1, figures=FIG1_LOGICALS),
        name="Logical X",
    ),
    Gate(
        "Y",
        "Gypi D4 D5 D6",
        "Y",
        paper(FIG1, figures=FIG1_LOGICALS),
        name="Logical Y",
    ),
    Gate(
        "Z",
        "Gzpi D4 D5 D6",
        "Z",
        paper(FIG1, figures=FIG1_LOGICALS),
        name="Logical Z",
    ),
    Gate(
        "H",
        "Gh " + ALL_D,
        "H",
        paper(FIG1, figures=FIG1_LOGICALS),
        circuit_key="H Circuit",
        update_key="H Frame Update",
        name="Logical H",
        circuit_name="Logical H Circuit",
        update_name="Logical H Frame Update",
    ),
    Gate(
        "S",
        "Gzmpi2 " + ALL_D,
        "S",
        paper(
            FIG1
            + ": logical S is S-dagger on every data qubit; frame update "
            + PRX
            + " Fig 15 (sgate2pf), matched by the derivation",
            figures=FIG1_LOGICALS + ("fig15-sgate.png",),
        ),
        circuit_key="S Circuit",
        update_key="S Frame Update",
        name="Logical S",
        circuit_name="Logical S Circuit",
        update_name="Logical S Frame Update",
    ),
    Gate(
        "Sdag",
        "Gzpi2 " + ALL_D,
        "S_DAG",
        ours("the inverse of Fig 1's logical S; Fig 1 lists no S-dagger"),
        circuit_key="Sdag Circuit",
        update_key="S Frame Update",
        name="Logical Sdag",
        circuit_name="Logical Sdag Circuit",
        update_name="Logical S Frame Update",
    ),
    Gate(
        "I",
        "Gi " + ALL_D,
        "I",
        ours("an explicit logical idle, so a program can idle the patch"),
        name="Logical I",
    ),
    # --- QEC ------------------------------------------------------------------
    Circuit(
        "Flagged Parallel S1-S5-S6 Check",
        """
        Gh A0
        Gcnot A0 D3 D2 A2 D5 A1
        Gcnot A0 A1
        Gcnot A0 D0 D3 A2 D4 A1
        Gcnot A0 D1 D6 A2 D2 A1
        Gcnot A0 A2
        Gcnot A0 D2 D5 A2 D1 A1
        Gh A0
        Imrz A0 A1 A2
        """,
        paper(
            FIG10 + ", first flagged parallel circuit; auxes read S1, S5, S6 "
            "(Fig 11, Fig 17)",
            PACKED,
            figures=("fig10-flagged-1.png",),
        ),
        name="Flagged S1-S5-S6 Check",
    ),
    Circuit(
        "Flagged Parallel S2-S3-S4 Check",
        """
        Gh A1 A2
        Gcnot D3 A0 A2 D2 A1 D5
        Gcnot A1 A0
        Gcnot D0 A0 A2 D3 A1 D4
        Gcnot D1 A0 A2 D6 A1 D2
        Gcnot A2 A0
        Gcnot D2 A0 A2 D5 A1 D1
        Gh A1 A2
        Imrz A0 A1 A2
        """,
        paper(
            FIG10 + ", second flagged parallel circuit; auxes read S4, S2, S3 "
            "(Fig 17; the caption's {S1, S3, S4} is a typo)",
            PACKED,
            figures=("fig10-flagged-2.png",),
        ),
        name="Flagged S2-S3-S4 Check",
    ),
    *(
        CheckCircuit(
            f"Unflagged S{i} Check",
            f"S{i}",
            A[(i - 1) % 3],
            paper(
                FIG10 + ", unflagged circuits: data in support order",
                "the aux is ours: A0, A1, A2 for S1-S3 and again for S4-S6",
            ),
            # LoQS 1.3 names S4-S6 "S1"-"S3", kept verbatim.
            name=f"Unflagged S{(i - 1) % 3 + 1} Check",
        )
        for i in range(1, 7)
    ),
    Parallel(
        "Unflagged Parallel S1-S2-S3 Check",
        ("Unflagged S1 Check", "Unflagged S2 Check", "Unflagged S3 Check"),
        ours(
            "the three X checks run in parallel, one aux each",
            cite=FIG10 + ", unflagged circuits",
            figures=("fig10-unflagged.png",),
        ),
        name="Unflagged Parallel S1-S2-S3 Check",
    ),
    Parallel(
        "Unflagged Parallel S4-S5-S6 Check",
        ("Unflagged S4 Check", "Unflagged S5 Check", "Unflagged S6 Check"),
        ours(
            "the three Z checks run in parallel, one aux each",
            cite=FIG10 + ", unflagged circuits",
            figures=("fig10-unflagged.png",),
        ),
        name="Unflagged S4-S5-S6 Check",
    ),
    CustomApplyFn(
        "Flagged S1-S5-S6 Feed-Forward",
        _FF,
        paper(
            PRX + " Sec II.A.2, Fig 10 caption, Fig 17 (first branch)",
            figures=("fig17-flagged.png",),
        ),
        data=(("flag_qubits", A), ("first_check", True)),
        derived=(
            (
                "syndrome_index",
                "syndrome_index:Flagged Parallel S1-S5-S6 Check",
            ),
        ),
        qubit_data=("flag_qubits", "syndrome_index"),
        scratch=("flagged_syndrome_diff",),
        branches=(
            (_CHANGED, _UNFLAGGED),
            (
                _UNCHANGED,
                (
                    "Flagged Parallel S2-S3-S4 Check",
                    "Flagged S2-S3-S4 Feed-Forward",
                ),
            ),
        ),
        name="Flagged S1-S5-S6 Feed-Forward",
    ),
    CustomApplyFn(
        "Flagged S2-S3-S4 Feed-Forward",
        _FF,
        paper(
            PRX + " Sec II.A.2, Fig 10 caption, Fig 17 (second branch)",
            figures=("fig17-flagged.png",),
        ),
        data=(("flag_qubits", A), ("first_check", False)),
        derived=(
            (
                "syndrome_index",
                "syndrome_index:Flagged Parallel S2-S3-S4 Check",
            ),
        ),
        qubit_data=("flag_qubits", "syndrome_index"),
        scratch=("flagged_syndrome_diff",),
        branches=((_CHANGED, _UNFLAGGED), (_UNCHANGED, ())),
        name="Flagged S2-S3-S4 Feed-Forward",
    ),
    CustomApplyFn(
        "QEC Decoder",
        f"{_CUSTOM}:qec_decoder",
        paper(
            PRX + " Sec II.A.3, Tables I and II, Fig 17, Fig 11",
            figures=("fig17-decode.png", "fig11.png"),
        ),
        derived=(
            ("X_qubits", "aux_order:Unflagged Parallel S1-S2-S3 Check"),
            ("Z_qubits", "aux_order:Unflagged Parallel S4-S5-S6 Check"),
            ("data_lut", "lut:Table 1"),
            ("hook_lut", "lut:Table 2"),
        ),
        qubit_data=("X_qubits", "Z_qubits"),
        param_priorities=(
            ("X_outcomes", ("history[-2]",)),
            ("Z_outcomes", ("history[-1]",)),
        ),
        param_aliases=(
            ("X_outcomes", "measurement_outcomes"),
            ("Z_outcomes", "measurement_outcomes"),
        ),
        scratch=("flagged_syndrome_diff",),
        name="QEC Decoder",
    ),
    Composite(
        "Adaptive QEC",
        ("Flagged Parallel S1-S5-S6 Check", "Flagged S1-S5-S6 Feed-Forward"),
        paper(PRX + " Sec II.A.2, Fig 3, Fig 10", figures=("fig3.png",)),
        name="Start of Adaptive QEC",
    ),
    # --- measure ----------------------------------------------------------------
    Circuit(
        "Raw Z Data Measure",
        "Imrz " + ALL_D,
        paper(
            PRX + " Fig 10 final measurement, Fig 19 (meas_z_data)",
            figures=("fig19-measure.png",),
        ),
        name="Raw logical Z-basis measurement",
    ),
    Circuit(
        "Raw X Data Measure",
        "Gh " + ALL_D + "\nImrz " + ALL_D,
        ours(
            "every data qubit measured in X, for a program to call; the X "
            "measure uses the H Circuit and the Z readout instead"
        ),
        name="Raw logical X-basis measurement",
    ),
    *(
        CustomApplyFn(
            f"FT {b} logical parity calculation",
            f"{_CUSTOM}:logical_parity",
            ours(
                "decode the diff against latest_syndrome, then update the "
                "frame and infer again",
                cite=PRX + " Sec II.A.4, Fig 19",
                figures=("fig19-body.png",),
            ),
            data=(("data_qubits", D), ("measurement_basis", b)),
            derived=(
                ("checks", f"support:{b}-type"),
                ("readout", f"logical_support:{b}"),
                ("data_lut", "lut:Table 1"),
            ),
            qubit_data=("data_qubits",),
            name=f"FT {b} logical parity calculation",
        )
        for b in "ZX"
    ),
    Composite(
        "FT Logical Z Measure",
        ("Raw Z Data Measure", "FT Z logical parity calculation"),
        paper(PRX + " Sec II.A.4, Fig 19", figures=("fig19-body.png",)),
        name="FT logical Z measurement",
    ),
    Composite(
        "FT Logical X Measure",
        ("H Circuit", "Raw Z Data Measure", "FT X logical parity calculation"),
        paper(
            PRX + " Sec II.A.4, Fig 18 (rotate_meas_basis): logical H, then "
            "the Z readout",
            figures=("fig18.png",),
        ),
        name="FT logical X measurement",
    ),
)

TABLES = (
    LookupTable(
        "Table 1",
        "data",
        rows=(((0, 1, 0),), ((0, 1, 1),), ((0, 0, 1),)),
        src=paper(
            PRX + " Table I (= Fig 21 decoder_2d), its +1/-1 rows as 0/1 bits",
            figures=("table1.png", "fig21.png"),
        ),
    ),
    LookupTable(
        "Table 2",
        "flag",
        rows=(
            ((1, 0, 0), (0, 1, 0)),
            ((1, 0, 0), (0, 0, 1)),
            ((0, 1, 1), (0, 0, 1)),
        ),
        src=paper(
            PRX + " Table II (= Fig 22 decoder_flag_update), rows as bits",
            figures=("table2.png", "fig22.png"),
        ),
    ),
)

TWO_PATCH = (
    TwoPatchGate(
        "Transversal CX",
        ("ctrl", "tgt"),
        "Gcnot " + " ".join(f"ctrl.{q} tgt.{q}" for q in D),
        "CX",
        ours(
            "the textbook transversal CNOT; the paper has no logical "
            "two-qubit gate",
            cite=PRX + " Sec I",
        ),
        name="Transversal Logical CX",
    ),
)

ROLES = (
    Role("Non-FT Zero Prep", "prep", state="+Z", fresh_only=True),
    Role("FT Zero Prep", "prep", state="+Z"),
    Role("FT Logical Z Measure", "measure", basis="Z", ft=True),
    Role("FT Logical X Measure", "measure", basis="X", ft=True),
    Role("Adaptive QEC", "qec"),
)

SPEC = CodeSpec(
    name="steane",
    title="Quantinuum 2021 [[7,1,3]] color code",
    facts=FACTS,
    items=ITEMS,
    tables=TABLES,
    two_patch=TWO_PATCH,
    roles=ROLES,
    figure_credit=(
        "Paper figures adapted from " + PRX + ", Phys. Rev. X 11, 041058 "
        "(2021), under CC BY 4.0: cropped, with our qubit labels added in "
        "blue."
    ),
)

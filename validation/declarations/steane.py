"""Declaration of the Steane [[7,1,3]] codepack (Quantinuum 2021).

Operators and gate actions follow the paper (PhysRevX.11.041058, Fig. 1
and Fig. 10) as the codepack states them, never a run.
"""

from loqs.backends import (
    DictNoiseModel,
    ListPhysicalCircuit,
    PyGSTiNoiseModel,
    PyGSTiPhysicalCircuit,
    QSimQuantumState,
    STIMQuantumState,
)
from loqs.backends.reps import QSimSuperopGateRep, StimCircuitGateRep
from loqs.codepacks import codepack_7_1_3_multipatch as multipatch
from loqs.codepacks import codepack_7_1_3_quantinuum2021 as module

from validation import logical
from validation.declaration import (
    Backend,
    CodepackDeclaration,
    Config,
    Gate,
    Measure,
    MultipatchOp,
    Prep,
    Sweep,
    Tracked,
)
from validation.logical import Meas

OUTCOME_KEY = "logical_measurement"

STIM = Backend(
    "stim-dict",
    ListPhysicalCircuit,
    DictNoiseModel,
    StimCircuitGateRep,
    STIMQuantumState,
)
QSIM_DICT = Backend(
    "qsim-dict",
    ListPhysicalCircuit,
    DictNoiseModel,
    QSimSuperopGateRep,
    QSimQuantumState,
)
QSIM_PYGSTI = Backend(
    "qsim-pygsti",
    PyGSTiPhysicalCircuit,
    PyGSTiNoiseModel,
    QSimSuperopGateRep,
    QSimQuantumState,
)
STIM_PYGSTI_CIRCUITS = Backend(
    "stim-dict-pygsti",
    PyGSTiPhysicalCircuit,
    DictNoiseModel,
    StimCircuitGateRep,
    STIMQuantumState,
)

ALL = ("static", "truth", "tracking", "independence", "ft")

_COMPONENT = "component of {}, checked through it"


def build_code(config, backend):
    return module.create_qec_code(
        circuit_backend=backend.circuit_backend, **config.options
    )


def build_model(config, qubits, backend):
    return module.create_ideal_model(
        qubits,
        model_backend=backend.model_backend,
        gaterep=backend.gate_rep,
    )


def tracked(patch):
    """Stored syndrome and logical frame (`codepack_7_1_3_quantinuum2021.py`).

    `latest_syndrome` is `[S1..S3 (X-type), S4..S6 (Z-type)]` (`:895`,
    `:905-906`). `logical_pauli_frame` is `[Z_bit, X_bit]`: index 0 is
    corrected from the X-type checks (`:920`), index 1 from the Z-type
    ones (`:921`), so the Pauli is `(x, z) = (frame[1], frame[0])`.
    """
    z_bit, x_bit = patch.data.get("logical_pauli_frame", [0, 0])
    return Tracked(
        syndrome=list(patch.data.get("latest_syndrome", [0] * 6)),
        logical=(x_bit, z_bit),
    )


def build_cx(config, geometry):
    """The transversal CX entry, on the config's circuit backend."""
    inst = multipatch.build_transversal_cx_instruction(
        geometry,
        circuit_backend=config.backends[0].circuit_backend,
        include_idles=config.options.get("include_idles", False),
    )
    return [(inst, None)]


def a0_flag_fired(history) -> bool:
    """Whether the prep's check read 1 on `A0`: the RUS prep retries."""
    outcomes = history.collect_data(
        "measurement_outcomes", "all", strip_none_entries=True
    )
    return any(1 in o.outcomes.get("A0", ()) for o in outcomes)


def build_cx_sweep(config, geometry):
    """The CX's physical circuit, then its frame bookkeeping."""
    backend = config.backends[0].circuit_backend
    idles = config.options.get("include_idles", False)
    return [
        (
            multipatch.build_transversal_cx_circuit_instruction(
                geometry, circuit_backend=backend, include_idles=idles
            ),
            None,
        ),
        (
            multipatch.build_cx_bookkeeping_instruction(
                geometry.label("ctrl"), geometry.label("tgt")
            ),
            None,
        ),
    ]


def _sweeps():
    one = ("L0",)
    out = [
        # A check and the Non-FT prep before it: faults before the flag is
        # read, with the RUS retry modelled by dropping flagged runs.
        Sweep(
            "prep",
            (("Non-FT Zero Prep + Checks", "L0"),),
            "Non-FT Zero Prep + Checks",
            discard=a0_flag_fired,
            script=(logical.Prep("L0", "+Z"),),
        )
    ]
    for check, feed in (
        ("S1-S5-S6", "S1-S5-S6"),
        ("S2-S3-S4", "S2-S3-S4"),
    ):
        key = f"Flagged Parallel {check} Check"
        out.append(
            Sweep(
                f"check-{check}",
                ((key, "L0"), (f"Flagged {feed} Feed-Forward", "L0")),
                key,
                suffix=True,
            )
        )
    for basis in "ZX":
        raw = f"Raw {basis} Data Measure"
        out.append(
            Sweep(
                f"meas-{basis}",
                (
                    (raw, "L0"),
                    (f"FT {basis} logical parity calculation", "L0"),
                ),
                raw,
                script=(Meas(OUTCOME_KEY, (("L0", basis),)),),
            )
        )
    for gate, circuit, update in (
        ("H", "H Circuit", "H Frame Update"),
        ("S", "S Circuit", "S Frame Update"),
        ("Sdag", "Sdag Circuit", "S Frame Update"),
    ):
        clifford = "S_DAG" if gate == "Sdag" else gate
        out.append(
            Sweep(
                f"gate-{gate}",
                ((circuit, "L0"), (update, "L0")),
                circuit,
                script=(logical.Gate(clifford, one),),
            )
        )
    for name, twoq in (("cx-w1", False), ("cx-w2", True)):
        out.append(
            Sweep(
                name,
                build_cx_sweep,
                "Transversal CX physical circuit",
                post_twoq=twoq,
                script=(logical.Gate("CX", ("ctrl", "tgt")),),
                roles=("ctrl", "tgt"),
            )
        )
    return tuple(out)


_QEC = "Adaptive QEC"
_INTERNAL = {
    "Non-FT Zero Prep + Checks": _COMPONENT.format("FT Zero Prep"),
    "H Circuit": _COMPONENT.format("H"),
    "S Circuit": _COMPONENT.format("S"),
    "Sdag Circuit": _COMPONENT.format("Sdag"),
    "H Frame Update": _COMPONENT.format("H"),
    "S Frame Update": _COMPONENT.format("S and Sdag"),
    "Flagged Parallel S1-S5-S6 Check": _COMPONENT.format(_QEC),
    "Flagged Parallel S2-S3-S4 Check": _COMPONENT.format(_QEC),
    "Flagged S1-S5-S6 Feed-Forward": _COMPONENT.format(_QEC),
    "Flagged S2-S3-S4 Feed-Forward": _COMPONENT.format(_QEC),
    "QEC Decoder": _COMPONENT.format(_QEC),
    "Unflagged Parallel S1-S2-S3 Check": _COMPONENT.format(_QEC),
    "Unflagged Parallel S4-S5-S6 Check": _COMPONENT.format(_QEC),
    "Raw Z Data Measure": _COMPONENT.format("FT Logical Z Measure"),
    "Raw X Data Measure": _COMPONENT.format("FT Logical X Measure"),
    "FT Z logical parity calculation": _COMPONENT.format(
        "FT Logical Z Measure"
    ),
    "FT X logical parity calculation": _COMPONENT.format(
        "FT Logical X Measure"
    ),
    "build_transversal_cx_circuit_instruction": "multipatch builder",
    "build_cx_bookkeeping_instruction": "multipatch builder",
}
_INTERNAL.update(
    {
        f"Unflagged S{i} Check": "standalone single-stabilizer check, "
        "not reached by the QEC instruction"
        for i in range(1, 7)
    }
)

DECLARATION = CodepackDeclaration(
    name="steane",
    files=(
        "loqs/codepacks/codepack_7_1_3_quantinuum2021.py",
        "loqs/codepacks/codepack_7_1_3_multipatch.py",
    ),
    patch_type="Steane",
    configs=(
        Config(
            "default",
            classes=ALL,
            backends=(STIM, QSIM_DICT, QSIM_PYGSTI),
        ),
        Config(
            "idles",
            options={"include_idles": True},
            classes=("static", "truth"),
            backends=(STIM_PYGSTI_CIRCUITS,),
        ),
    ),
    build_code=build_code,
    build_model=build_model,
    # Template data qubits D0..D6; the code's own `template_data_qubits`
    # also lists A2 (`:94`), so the operators below name D0..D6 explicitly.
    data_qubits=tuple(f"D{i}" for i in range(7)),
    # Plaquettes of `mappings` (`:419-423`): {0,1,2,3}, {1,2,4,5}, {2,3,5,6};
    # S1-S3 are X-type and S4-S6 Z-type on them (`:550`, `:905-906`).
    stabilizers=(
        "XXXXIII",
        "IXXIXXI",
        "IIXXIXX",
        "ZZZZIII",
        "IZZIZZI",
        "IIZZIZZ",
    ),
    # The logical Paulis act on the bottom row D4-D6 (`:218`, `:547`).
    logical_x="IIIIXXX",
    logical_z="IIIIZZZ",
    preps=(
        # A bare circuit with no reset (`:150-156`); the FT prep resets the
        # data qubits first (`:158-163`, `:193-195`).
        Prep("Non-FT Zero Prep", "+Z", fresh_only=True),
        Prep("FT Zero Prep", "+Z"),
    ),
    measures=(
        Measure("FT Logical Z Measure", "Z", True),
        Measure("FT Logical X Measure", "X", True),
    ),
    # Logical S is physical S-dagger on every data qubit and the reverse
    # (`:213-214`), so key `S` acts as the stim `S` (X -> Y) on the logical.
    gates=(
        Gate("I", "I"),
        Gate("X", "X"),
        Gate("Y", "Y"),
        Gate("Z", "Z"),
        Gate("H", "H"),
        Gate("S", "S"),
        Gate("Sdag", "S_DAG"),
    ),
    qec=(_QEC,),
    internal=_INTERNAL,
    tracked=tracked,
    decoder="eager",
    multipatch_modules=("loqs.codepacks.codepack_7_1_3_multipatch",),
    # One physical `Gcnot` per data pair, control to target; the logical
    # action is a CX (module docstring, `build_transversal_cx_*`).
    ops=(
        MultipatchOp(
            "transversal-cx",
            ("ctrl", "tgt"),
            build_cx,
            ("build_transversal_cx_instruction",),
            script=(logical.Gate("CX", ("ctrl", "tgt")),),
        ),
    ),
    sweeps=_sweeps(),
    error_pair=((0, "X"), (1, "X")),
    # The two tracked-error cases of 549cb89: input H, X pairs, S or Sdag
    # word, measure Z; and its four fault cases: a flagged check, then the
    # W(S) or W(Sdag) word, measure Z.
    reproducers={
        f"ft-{c}-{g}": (
            f"steane.default.ft.ft.check-{c}-FT_Zero_Prep+H-"
            f"FT_Logical_Z_Measure-W{g}-clean",
            ("outcome",),
        )
        for c in ("S1-S5-S6", "S2-S3-S4")
        for g in ("S", "Sdag")
    }
    | {
        f"word-{g}": (
            f"steane.default.tracking.word.FT_Zero_Prep-H-W{g}-"
            "Adaptive_QEC-FT_Logical_Z_Measure",
            ("outcome", "O1", "O2", "O3"),
        )
        for g in ("S", "Sdag")
    },
    known_failures={
        "steane.default.static.model-coverage.qsim-pygsti-p*:outcome": (
            "steane-3"
        ),
        # Every case of these families fails O4 in the baseline run; the
        # `egq` gate cases (one QEC round) pass it, so they stay undeclared.
        "steane.default.tracking.idle.*:O4": "steane-2",
        "steane.default.tracking.gate.*-eqgq-*:O4": "steane-2",
        "steane.default.tracking.word.*:O4": "steane-2",
        "steane.default.tracking.reuse.*:O4": "steane-2",
        # Every `mp-track` case fails O4 in the baseline run; `independence`
        # only the block schedules (the interleaved ones pass it).
        "steane.default.tracking.mp-track.*:O4": "steane-2",
        "steane.default.independence.independence.*-blocks:O4": "steane-2",
        "steane.default.tracking.reuse.*:O3": "steane-1",
        "steane.default.tracking.reuse.*:outcome": "steane-1",
    },
)

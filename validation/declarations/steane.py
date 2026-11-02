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
from loqs.codepacks.steane import SteaneCodepack

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
    return SteaneCodepack(
        circuit_backend=backend.circuit_backend, **config.options
    ).create_qec_code()


def build_model(config, qubits, backend):
    return SteaneCodepack(
        circuit_backend=backend.circuit_backend, **config.options
    ).create_ideal_model(
        qubits,
        model_backend=backend.model_backend,
        gaterep=backend.gate_rep,
    )


def tracked(patch):
    """Stored syndrome and logical frame (`loqs/codepacks/steane/spec.py`).

    `latest_syndrome` is `[S1..S3 (X-type), S4..S6 (Z-type)]` (the spec's
    check circuits and syndrome order). `logical_pauli_frame` is
    `[Z_bit, X_bit]`: index 0 is corrected from the X-type checks, index 1
    from the Z-type ones, so the Pauli is `(x, z) = (frame[1], frame[0])`.
    """
    z_bit, x_bit = patch.data.get("logical_pauli_frame", [0, 0])
    return Tracked(
        syndrome=list(patch.data.get("latest_syndrome", [0] * 6)),
        logical=(x_bit, z_bit),
    )


def _build_cx_composite(config, geometry):
    cp = SteaneCodepack(
        circuit_backend=config.backends[0].circuit_backend, **config.options
    )
    return cp.build_two_patch("Transversal CX", geometry)


def build_cx(config, geometry):
    """The transversal CX entry, on the config's circuit backend."""
    return [(_build_cx_composite(config, geometry), None)]


def a0_flag_fired(history) -> bool:
    """Whether the prep's check read 1 on `A0`: the RUS prep retries."""
    outcomes = history.collect_data(
        "measurement_outcomes", "all", strip_none_entries=True
    )
    return any(1 in o.outcomes.get("A0", ()) for o in outcomes)


def build_cx_sweep(config, geometry):
    """The CX's physical circuit, then its frame bookkeeping."""
    composite = _build_cx_composite(config, geometry)
    return [(part, None) for part in composite.data["instructions"]]


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
                "Transversal Logical CX (physical circuit)",
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
        "loqs/codepacks/base.py",
        "loqs/codepacks/generator/__init__.py",
        "loqs/codepacks/generator/__main__.py",
        "loqs/codepacks/generator/codepack.py",
        "loqs/codepacks/generator/derive.py",
        "loqs/codepacks/generator/gates.py",
        "loqs/codepacks/generator/generate.py",
        "loqs/codepacks/generator/render.py",
        "loqs/codepacks/generator/schema.py",
        "loqs/codepacks/steane/__init__.py",
        "loqs/codepacks/steane/_generated.py",
        "loqs/codepacks/steane/custom_apply_fns.py",
        "loqs/codepacks/steane/spec.py",
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
    # Template data qubits D0..D6, as in `loqs/codepacks/steane/spec.py`.
    data_qubits=tuple(f"D{i}" for i in range(7)),
    # Plaquettes {0,1,2,3}, {1,2,4,5}, {2,3,5,6}; S1-S3 are X-type and
    # S4-S6 Z-type on them (`loqs/codepacks/steane/spec.py`).
    stabilizers=(
        "XXXXIII",
        "IXXIXXI",
        "IIXXIXX",
        "ZZZZIII",
        "IZZIZZI",
        "IIZZIZZ",
    ),
    # The logical Paulis act on the bottom row D4-D6
    # (`loqs/codepacks/steane/spec.py`).
    logical_x="IIIIXXX",
    logical_z="IIIIZZZ",
    preps=(
        # A bare circuit with no reset; the FT prep resets the data qubits
        # first (`loqs/codepacks/steane/spec.py`).
        Prep("Non-FT Zero Prep", "+Z", fresh_only=True),
        Prep("FT Zero Prep", "+Z"),
    ),
    measures=(
        Measure("FT Logical Z Measure", "Z", True),
        Measure("FT Logical X Measure", "X", True),
    ),
    # Logical S is physical S-dagger on each data qubit and the reverse
    # (`steane/spec.py`), so key `S` acts as the stim `S` (X -> Y).
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
    # action is a CX (`loqs/codepacks/steane/spec.py`).
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
)

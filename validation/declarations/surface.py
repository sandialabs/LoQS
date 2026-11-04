"""Declaration of the Surface-17 family codepack (Tomita and Svore 2014).

Operators follow the codepack's own check and logical definitions, never a
run. The multipatch and surgery modules are declared with their operations.
"""

from loqs.backends import (
    DictNoiseModel,
    ListPhysicalCircuit,
    NumpyStatevectorQuantumState,
    PyGSTiPhysicalCircuit,
    STIMQuantumState,
)
from loqs.backends.reps import StimCircuitGateRep, UnitaryGateRep
from loqs.codepacks import codepack_surf17_multipatch as multipatch
from loqs.codepacks import codepack_surf17_surgery as surgery
from loqs.codepacks import codepack_surf17_tomita2014 as module

from validation import logical
from validation.declaration import (
    Backend,
    CodepackDeclaration,
    Config,
    Gate,
    Measure,
    MultipatchOp,
    Prep,
    Tracked,
)

STIM = Backend(
    "stim-dict",
    ListPhysicalCircuit,
    DictNoiseModel,
    StimCircuitGateRep,
    STIMQuantumState,
)
NUMPY_DICT = Backend(
    "numpy-dict",
    ListPhysicalCircuit,
    DictNoiseModel,
    UnitaryGateRep,
    NumpyStatevectorQuantumState,
)
STIM_PYGSTI_CIRCUITS = Backend(
    "stim-dict-pygsti",
    PyGSTiPhysicalCircuit,
    DictNoiseModel,
    StimCircuitGateRep,
    STIMQuantumState,
)

_COMPONENT = "component of {}, checked through it"
_QEC = "QEC"

# The checks of `create_qec_code`, in the order of its syndrome labels
# (`codepack_surf17_tomita2014.py:573-613`): X checks first, then Z.
_X_CHECKS = ("XXIXXIIII", "IXXIIIIII", "IIIIXXIXX", "IIIIIIXXI")
_Z_CHECKS = ("ZIIZIIIII", "IZZIZZIII", "IIIZZIZZI", "IIIIIZIIZ")


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
    """The last stored syndrome round and the per-qubit Pauli frame.

    `syndrome_history_X` and `syndrome_history_Z` hold one 4-bit row per
    QEC round, in the order of `stabilizers` (`:664-671`). The
    frame stays uncorrected until the final decode (deferred decoder).
    """
    hx = patch.data.get("syndrome_history_X", [])
    hz = patch.data.get("syndrome_history_Z", [])
    syndrome = list(hx[-1]) + list(hz[-1]) if hx and hz else [0] * 8
    return Tracked(
        syndrome=syndrome,
        frame="".join(patch.pauli_frame.pauli_frame),
    )


_INTERNAL = {
    "H Circuit": _COMPONENT.format("H"),
    "H Permutation": _COMPONENT.format("H"),
    "Syndrome Extraction": _COMPONENT.format(_QEC),
    "Decoder": _COMPONENT.format(_QEC),
    "Raw Z Data Measure": _COMPONENT.format("FT Logical Z Measure"),
    "Raw X Data Measure": _COMPONENT.format("FT Logical X Measure"),
    "FT Z logical parity calculation": _COMPONENT.format(
        "FT Logical Z Measure"
    ),
    "FT X logical parity calculation": _COMPONENT.format(
        "FT Logical X Measure"
    ),
    "build_transversal_cnot_circuit_instruction": "multipatch builder",
    "build_cnot_bookkeeping_instruction": "multipatch builder",
    "build_merge_instruction": "half of the surgery parity builder",
    "build_split_instruction": "half of the surgery parity builder",
    "build_surgery_parity_instruction_sequence": "the surgery parity "
    "builder as a flat list, for error injection",
    "build_surgery_cnot_corrections_instruction": "part of the surgery CNOT",
    "build_split_byproduct_repair_instruction": "part of the surgery CNOT",
    "build_mzz_bell_corrections_instruction": "part of the Bell prep",
    "build_merged_check_matrices": "check-matrix helper, not an instruction",
}

# The diff modes keep the previous round as a reference for the final decode
# (`:768-1010`); both bases are set, as the surgery readouts do.
_MODES = ("clean_diff", "guarded_diff")


def _measures():
    out = []
    for basis in "ZX":
        key = f"FT Logical {basis} Measure"
        out.append(Measure(key, basis, True))
        for mode in _MODES:
            kwargs = {
                "reference_round_mode_X": mode,
                "reference_round_mode_Z": mode,
            }
            out.append(Measure(key, basis, True, kwargs))
    return tuple(out)


def _known_failures():
    """Baseline known failures per case pattern and check (see bugs.py).

    A case lists every bug that can fail a check in it; bugs sharing a
    check are attributed by which gate, measure or schedule the case
    exercises, not isolated per bug.
    """
    kf = {
        # Every QEC round's history frame re-reads later data.
        "surface.*.truth.truth.*-QEC:O4": "surface-3",
        "surface.*.tracking.*:O4": "surface-3",
        "surface.*.independence.independence.*:O4": "surface-3",
        # H after a QEC round leaves the checks and frame unrelabelled.
        "surface.*.truth.truth.*H*-QEC:O1": "surface-1",
        "surface.*.truth.truth.*H*-QEC:O3": "surface-1",
        "surface.*.tracking.gate.*-H-eqgq-*:O1": "surface-1",
        "surface.*.tracking.gate.*-H-eqgq-*:outcome": "surface-1",
        "surface.*.tracking.word.*-WH-*:O1": "surface-1",
        "surface.*.tracking.word.*-WH-*:outcome": "surface-1",
        # The diff readouts take round 0 as their reference.
        "surface.*.truth.agreement.numpy-dict-errX-Zero_Prep-none-QEC-[12]"
        ":outcome": "surface-6",
    }
    # A deterministic outcome after H and a QEC round: the measure basis
    # is the one the word leaves the state in (an odd number of H flips
    # the prep's basis).
    for prep, state in (("Zero_Prep", "Z"), ("Plus_Prep", "X")):
        flip = {"Z": "X", "X": "Z"}
        for word in ["H"] + [
            f"{a}+{b}" for a in "IXYZH" for b in "IXYZH" if "H" in a + b
        ]:
            basis = flip[state] if word.count("H") % 2 else state
            tails = [("truth.truth", "")]
            if word == "H":
                tails.append(("truth.agreement", "numpy-dict-"))
            for family, label in tails:
                kf[
                    f"surface.*.{family}.{label}{prep}-{word}-"
                    f"FT_Logical_{basis}_Measure*-QEC:outcome"
                ] = "surface-1"
    for fam in ("idle", "gate", "word"):
        kf[f"surface.*.tracking.{fam}.*Measure-reference_round*:outcome"] = (
            "surface-6"
        )
    # Re-prep keeps the previous run's history and frame.
    for config in _CONFIGS:
        if "tracking" not in config.classes:
            continue
        # One QEC round leaves the plain no-read cases correct.
        tail = "*-meas-*" if config.name == "surf17-r1" else "*"
        key = f"surface.{config.name}.tracking.reuse.{tail}:outcome"
        kf[key] = "surface-4"
    kf["surface.*.tracking.reuse.*-noread-[1245]:outcome"] = "surface-6"
    # Independence: H before a QEC round on either patch.
    ind = "surface.*.independence.independence"
    kf[f"{ind}.*H*:O1"] = "surface-1"
    kf[f"{ind}.err*-*_Prep+X-*_Prep+H-*:O3"] = "surface-1"
    kf[f"{ind}.err*-*_Prep+H-*_Prep+X-*:outcome"] = "surface-1"
    kf[f"{ind}.err*-Zero_Prep+X-Plus_Prep+H-*:outcome"] = "surface-1"
    kf[f"{ind}.err*-Plus_Prep+X-Zero_Prep+H-interleaved:outcome"] = "surface-1"
    kf.update(_multipatch_failures())
    return kf


def _multipatch_failures():
    """Known failures of the multipatch and surgery cases.

    The parity ops, Bell prep and surgery CNOT read a wrong parity when an
    error preceded the op, whether or not a QEC round saw it (surface-7);
    the transversal CNOT does not. A CNOT or surgery reused on the same
    patches reads stale state (surface-4).
    """
    truth = "surface.*.truth.mp-truth"
    track = "surface.*.tracking.mp-track"
    reuse = "surface.*.tracking.mp-reuse"
    kf: dict[str, str | tuple[str, ...]] = {
        f"{truth}.*-QEC-x[12]:O4": "surface-3",
        f"{track}.*-then-L[01]-H:O1": "surface-1",
        f"{reuse}.*:outcome": ("surface-4", "surface-7"),
        f"{reuse}.*-Plus_Prep-Plus_Prep-QEC:O3": "surface-4",
        f"{reuse}.*-Zero_Prep-Zero_Prep-QEC:O3": "surface-4",
        # The surgery CNOT whose control is in the X basis reads a stale
        # ancilla on its second run.
        f"{truth}.surgery-cnot-Plus_Prep-*-x2:outcome": "surface-4",
        f"{truth}.surgery-cnot-Zero_Prep+H-*-x2:outcome": "surface-4",
        f"{track}.surgery-cnot-*-QEC-twice:outcome": "surface-4",
        # The only transversal CNOT case an H breaks.
        f"{track}.transversal-cnot-Plus_Prep-Zero_Prep-QEC-then-L[01]-H:"
        "outcome": "surface-1",
    }
    for op in ("bell-prep", "surgery-cnot", "surgery-zz-*", "surgery-xx-*"):
        kf[f"{track}.{op}-*-then-L[01]-H:outcome"] = ("surface-1", "surface-7")
        for tail in ("pre", "twice"):
            kf[f"{track}.{op}-*-QEC-{tail}:outcome"] = "surface-7"
    for op, state in (("joint-zz", "Zero"), ("joint-xx", "Plus")):
        # Only the input whose parity is certain shows the wrong parity.
        for tail in ("pre", "twice", "then-L[01]-H"):
            kf[
                f"{track}.{op}-{state}_Prep-{state}_Prep-QEC-{tail}:outcome"
            ] = "surface-7"
        kf[f"{track}.{op}-*-then-L[01]-H:outcome"] = "surface-1"
    return kf


def _backend(config):
    return config.backends[0].circuit_backend


def _cnot(config, geometry):
    inst = multipatch.build_transversal_cnot_instruction(
        geometry, circuit_backend=_backend(config)
    )
    return [(inst, None)]


def _joint(builder):
    def build(config, geometry):
        inst = builder(
            geometry, geometry.seam[0], circuit_backend=_backend(config)
        )
        return [(inst, None)]

    return build


def _surgery(kind, mode):
    def build(config, geometry):
        inst = surgery.build_surgery_parity_instruction(
            kind, geometry, mode=mode, circuit_backend=_backend(config)
        )
        return [(inst, None)]

    return build


def _bell(config, geometry):
    return surgery.build_mzz_bell_prep_sequence(
        geometry, circuit_backend=_backend(config)
    )


def _surgery_cnot(config, geometry):
    return surgery.build_surgery_cnot_sequence(
        geometry, circuit_backend=_backend(config)
    )


_JOINT_ZZ = "joint_parity_zz_{a}_{b}"
_JOINT_XX = "joint_parity_xx_{a}_{b}"
_SURGERY_ZZ = "surgery_parity_zz_{a}_{b}"
_SURGERY_XX = "surgery_parity_xx_{a}_{b}"
_AB = ("a", "b")
_GUARDED = {
    "reference_round_mode_X": "guarded_diff",
    "reference_round_mode_Z": "guarded_diff",
}
# Two-patch ops run on `surf10` only, as `surf17` takes 34 qubits or more;
# the surgery CNOT, with no form at 32 or fewer, runs on every layout.
_ALL_LAYOUTS = ("surf10", "surf13", "surf17")


def _parity(key, letter):
    return logical.Meas(key, (("a", letter), ("b", letter)))


def _ops():
    """The multipatch and surgery operations, each with its logical action
    from its builder's docstring (`codepack_surf17_multipatch.py:21-39`,
    `codepack_surf17_surgery.py:2201-2208`, `:2461-2467`)."""
    surgeries = [
        MultipatchOp(
            f"surgery-{kind.lower()}-{mode}",
            _AB,
            _surgery(kind, mode),
            ("build_surgery_parity_instruction",),
            script=(
                _parity(_SURGERY_ZZ if kind == "ZZ" else _SURGERY_XX, kind[0]),
            ),
            readout_kwargs=_GUARDED,
            seams={"main": 3},
            configs=("surf10",),
            needs_qec=mode == "ft",
        )
        for kind in ("ZZ", "XX")
        for mode in ("simple", "ft")
    ]
    return (
        MultipatchOp(
            "transversal-cnot",
            ("ctrl", "tgt"),
            _cnot,
            ("build_transversal_cnot_instruction",),
            script=(logical.Gate("CX", ("ctrl", "tgt")),),
            configs=("surf10",),
        ),
        MultipatchOp(
            "joint-zz",
            _AB,
            _joint(multipatch.build_joint_parity_zz_instruction),
            ("build_joint_parity_zz_instruction",),
            script=(_parity(_JOINT_ZZ, "Z"),),
            seams={"main": 1},
            configs=("surf10",),
        ),
        MultipatchOp(
            "joint-xx",
            _AB,
            _joint(multipatch.build_joint_parity_xx_instruction),
            ("build_joint_parity_xx_instruction",),
            script=(_parity(_JOINT_XX, "X"),),
            seams={"main": 1},
            configs=("surf10",),
        ),
        *surgeries,
        MultipatchOp(
            "bell-prep",
            _AB,
            _bell,
            ("build_mzz_bell_prep_sequence",),
            script=(
                _parity(_SURGERY_ZZ, "Z"),
                logical.Cond("b", "X", (_SURGERY_ZZ,)),
            ),
            readout_kwargs=_GUARDED,
            seams={"main": 3},
            configs=("surf10",),
            needs_qec=True,
        ),
        MultipatchOp(
            "surgery-cnot",
            ("ctrl", "tgt", "anc"),
            _surgery_cnot,
            ("build_surgery_cnot_sequence",),
            script=(
                logical.Prep("anc", "+X"),
                logical.Meas(
                    "surgery_parity_zz_{ctrl}_{anc}",
                    (("ctrl", "Z"), ("anc", "Z")),
                ),
                logical.Meas(
                    "surgery_parity_xx_{anc}_{tgt}",
                    (("anc", "X"), ("tgt", "X")),
                ),
                logical.Meas("logical_measurement", (("anc", "Z"),)),
                logical.Cond("ctrl", "Z", ("surgery_parity_xx_{anc}_{tgt}",)),
                logical.Cond(
                    "tgt",
                    "X",
                    ("surgery_parity_zz_{ctrl}_{anc}", "logical_measurement"),
                ),
            ),
            readout_kwargs=_GUARDED,
            big_layouts=_ALL_LAYOUTS,
            seams={"zz": 3, "xx": 3},
            configs=_ALL_LAYOUTS,
            needs_qec=True,
        ),
    )


_ALL = ("static", "truth", "tracking", "independence")
_NO_PAIR = ("static", "truth", "tracking")

_CONFIGS = (
    Config(
        "surf17",
        options={"layout": "surf17"},
        classes=_NO_PAIR,
        backends=(STIM,),
    ),
    Config(
        "surf13",
        options={"layout": "surf13"},
        classes=_ALL,
        backends=(STIM,),
    ),
    Config(
        "surf10",
        options={"layout": "surf10"},
        classes=_ALL,
        backends=(STIM, NUMPY_DICT),
    ),
    Config(
        "surf13-idle17",
        options={"layout": "surf13", "idle_layout": "surf17"},
        classes=_NO_PAIR,
        backends=(STIM_PYGSTI_CIRCUITS,),
    ),
    Config(
        "surf10-idle17",
        options={"layout": "surf10", "idle_layout": "surf17"},
        classes=_NO_PAIR,
        backends=(STIM_PYGSTI_CIRCUITS,),
    ),
    Config(
        "surf17-r1",
        options={"num_qec_rounds": 1},
        classes=_NO_PAIR,
        backends=(STIM,),
    ),
)

DECLARATION = CodepackDeclaration(
    name="surface",
    files=(
        "loqs/codepacks/codepack_surf17_tomita2014.py",
        "loqs/codepacks/codepack_surf17_multipatch.py",
        "loqs/codepacks/codepack_surf17_surgery.py",
    ),
    patch_type="SURF",
    configs=_CONFIGS,
    build_code=build_code,
    build_model=build_model,
    data_qubits=tuple(f"D{i}" for i in range(9)),
    stabilizers=_X_CHECKS + _Z_CHECKS,
    # Logical X is X2 X4 X6 and logical Z is Z0 Z4 Z8 (`:395`, `:408`).
    logical_x="IIXIXIXII",
    logical_z="ZIIIZIIIZ",
    preps=(
        Prep("Zero Prep", "+Z"),
        Prep("Plus Prep", "+X"),
    ),
    measures=_measures(),
    # Logical H is the transversal H and a 90-degree rotation of the data
    # grid (`:439-500`), exchanging X and Z.
    gates=(
        Gate("I", "I"),
        Gate("X", "X"),
        Gate("Y", "Y"),
        Gate("Z", "Z"),
        Gate("H", "H"),
    ),
    qec=(_QEC,),
    internal=_INTERNAL,
    tracked=tracked,
    decoder="deferred",
    multipatch_modules=(
        "loqs.codepacks.codepack_surf17_multipatch",
        "loqs.codepacks.codepack_surf17_surgery",
    ),
    ops=_ops(),
    known_failures=_known_failures(),
)

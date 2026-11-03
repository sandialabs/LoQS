"""Declaration of the 5-qubit [[5,1,3]] codepack (Quantinuum 2022).

Operators and gate actions follow the paper (arXiv:2208.01863, App. B and
Fig. 2 of arXiv:1603.03948) as the codepack states them, never a run.
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
from loqs.codepacks import codepack_5_1_3_quantinuum2022 as module

from validation.declaration import (
    Backend,
    CodepackDeclaration,
    Config,
    Gate,
    Measure,
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

_COMPONENT = "component of {}, checked through it"
_DEBUG = "stabilizer debugging circuit, not reached by any instruction"


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
    """Stored raw syndrome and per-qubit Pauli frame.

    `raw_syndrome` holds the four stabilizer bits in the order of
    `stabilizers` (`codepack_5_1_3_quantinuum2022.py:1806-1850`) and is
    absent before the first QEC round. The Pauli frame lists one Pauli per
    current data qubit, in template order.
    """
    return Tracked(
        syndrome=list(patch.data.get("raw_syndrome", [0] * 4)),
        frame="".join(patch.pauli_frame.pauli_frame),
    )


_QEC = ("Unflagged QEC", "Flagged QEC")
_INTERNAL = {
    "Non-FT Minus Prep + Checks": _COMPONENT.format("FT Minus Prep"),
    "Logical H circuit": _COMPONENT.format("H"),
    "Logical H permutation": _COMPONENT.format("H"),
    "Logical Prime Basis Transform": _COMPONENT.format("the measures"),
    "Logical Prime Basis Inverse Transform": (
        "defined but not used by any instruction"
    ),
    "Non-FT Minus Unprep": _COMPONENT.format("FT Logical X Measure"),
    "FT Logical X Measure Part I Circuit": _COMPONENT.format(
        "FT Logical X Measure"
    ),
    "FT Logical X Measure Part I Feed-Forward": _COMPONENT.format(
        "FT Logical X Measure"
    ),
    "FT Logical X Measure Part II Circuit": _COMPONENT.format(
        "FT Logical X Measure"
    ),
    "FT Logical X Measure Part II Feed-Forward": _COMPONENT.format(
        "FT Logical X Measure"
    ),
    "FT Logical X Measure Part III Circuit": _COMPONENT.format(
        "FT Logical X Measure"
    ),
    "FT Logical X Measure Part III Feed-Forward": _COMPONENT.format(
        "FT Logical X Measure"
    ),
    "FT Logical X Measure Classical Decoder": _COMPONENT.format(
        "FT Logical X Measure"
    ),
    "Unflagged Decoder": _COMPONENT.format("Unflagged QEC"),
}
for _stab in ("XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"):
    _INTERNAL[f"{_stab} Stabilizer"] = _DEBUG
    _INTERNAL[f"Unflagged {_stab} Check"] = _COMPONENT.format("Unflagged QEC")
    for _part in ("Check", "Decoder", "Feed-Forward"):
        _INTERNAL[f"Flagged {_stab} {_part}"] = _COMPONENT.format(
            "Flagged QEC"
        )

_CP = "five_qubit.default"
# `[!n]` keeps the pattern off the `Non-FT` measures, whose ids also contain
# `-FT_Logical_<basis>_Measure`.
_FT_Z = "*[!n]-FT_Logical_Z_Measure*"
_NON_FT = "*Non-FT_Logical_*_Measure*"


def _known_failures():
    """Baseline known failures per case pattern and check (see bugs.py).

    A check lists every bug that can fail it in the matching cases; the
    bug of a case's `outcome` check is attributed by which tracked state
    or measure the case exercises, not isolated per bug.
    """
    kf = {
        "five_qubit.idles.static.idle-padding.all:outcome": "five-13",
        # Static: the unflagged decoder's closure and hardcoded `A0`, and
        # the 7-qubit assert and missing pyGSTi instrument.
        f"{_CP}.static.labels.permuted:outcome": "five-9",
        f"{_CP}.static.labels.renamed:outcome": "five-9",
        f"{_CP}.static.labels.offset:outcome": "five-8",
        f"{_CP}.static.model-coverage.*-p2:outcome": "five-8",
        f"{_CP}.static.model-coverage.qsim-pygsti-p1:outcome": "five-8",
        "five_qubit.idles.static.labels.permuted:outcome": "five-9",
        "five_qubit.idles.static.labels.renamed:outcome": "five-9",
        "five_qubit.idles.static.labels.offset:outcome": "five-8",
        "five_qubit.idles.static.model-coverage.*-p2:outcome": "five-8",
        # Independence runs two patches, so it meets the 7-qubit assert.
        f"{_CP}.independence.independence.*:outcome": "five-8",
        # FT Z measure after any deterministic Z state.
        f"{_CP}.truth.agreement.*-K-*_QEC-0:outcome": "five-12",
        f"{_CP}.truth.agreement.*-K-*_QEC-2:outcome": "five-5",
    }
    # Words that leave Z certain: exactly one of H and K, padded by Paulis.
    # (`-<word>-FT_` cannot match the `Non-FT_` measure.)
    words = ["H", "K"] + [
        w for a in "HK" for p in "IXZ" for w in (f"{a}+{p}", f"{p}+{a}")
    ]
    for cfg in ("default", "idles"):
        for w in words:
            kf[
                f"five_qubit.{cfg}.truth.truth.*-{w}-FT_Logical_Z_Measure-*:outcome"
            ] = "five-12"
    for w in "HK":
        kf[f"{_CP}.truth.agreement.*-{w}-FT_Logical_Z_Measure-*:outcome"] = (
            "five-12"
        )
    for fam in ("idle", "gate", "word", "reuse"):
        kf[f"{_CP}.tracking.{fam}.{_FT_Z}:outcome"] = "five-12"
        kf[f"{_CP}.tracking.{fam}.{_NON_FT}:outcome"] = "five-5"
    # Non-Pauli gates' tracked syndrome and frame.
    kf[f"{_CP}.tracking.gate.*-H-eqgq-*:O2"] = "five-1"
    kf[f"{_CP}.tracking.gate.*-H-eqgq-*:O3"] = "five-4"
    kf[f"{_CP}.tracking.gate.*-H-eqgq-*:outcome"] = ("five-1", "five-4")
    kf[f"{_CP}.tracking.gate.*-K-eqgq-*:O2"] = "five-2"
    kf[f"{_CP}.tracking.gate.*-K-eqgq-*:O3"] = "five-3"
    kf[f"{_CP}.tracking.gate.*-K-eqgq-*:outcome"] = ("five-2", "five-3")
    kf[f"{_CP}.tracking.word.*-W[HK]-*:O2"] = ("five-1", "five-2")
    kf[f"{_CP}.tracking.word.*-W[HK]-*:O3"] = ("five-3", "five-4")
    kf[f"{_CP}.tracking.word.*-W[HK]-*:outcome"] = ("five-1", "five-4")
    # Re-prep keeps the previous run's tracking state.
    # Its O3 holds only for `K` input read by the FT Z measure.
    for tail in (
        "FT_Minus_Prep-none-*",
        "Non-FT_Minus_Prep-*",
        "FT_Minus_Prep-K-*-noread-*",
        "FT_Minus_Prep-K-*-meas-Non-FT*",
    ):
        kf[f"{_CP}.tracking.reuse.{tail}:O3"] = "five-10"
    kf[f"{_CP}.tracking.reuse.*:outcome"] = "five-10"
    # Agreement runs with a Z error read by the non-FT X measure.
    kf[f"{_CP}.truth.agreement.*-errZ-*-3:outcome"] = "five-5"
    return kf


DECLARATION = CodepackDeclaration(
    name="five_qubit",
    files=("loqs/codepacks/codepack_5_1_3_quantinuum2022.py",),
    patch_type="FiveQubit",
    configs=(
        Config(
            "default",
            classes=("static", "truth", "tracking", "independence"),
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
    data_qubits=tuple(f"D{i}" for i in range(5)),
    # Eqns B4-B7: the four cyclic shifts of XZZXI, in the order of the
    # stored `raw_syndrome` (`:1806-1807`) and the debug circuits (`:1629-1680`).
    stabilizers=("XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"),
    # Eqns 18 and 20 of arXiv:2208.01863 (the codepack's B1 and B3,
    # `:233`, `:248`): logical X is -YIXIY and logical Z is -XIZIX.
    logical_x="-YIXIY",
    logical_z="-XIZIX",
    preps=(
        # The non-FT prep is a bare circuit with no reset (`:132-158`); the
        # FT prep resets the data qubits on each attempt (`:161-167`).
        Prep("Non-FT Minus Prep", "-X", fresh_only=True),
        Prep("FT Minus Prep", "-X"),
    ),
    measures=(
        Measure("FT Logical Z Measure", "Z", True),
        Measure("FT Logical X Measure", "X", True),
        Measure("Non-FT Logical Z Measure", "Z", False),
        Measure("Non-FT Logical X Measure", "X", False),
    ),
    # Logical K is physical K on every qubit (Fig. 2a of arXiv:1603.03948,
    # `:262`), which sends X -> Z -> Y -> X, the stim `C_ZYX`. Logical H is
    # physical H plus a permutation (Fig. 2b, `:277`).
    gates=(
        Gate("I", "I"),
        Gate("X", "X"),
        Gate("Z", "Z"),
        Gate("K", "C_ZYX"),
        Gate("H", "H"),
    ),
    qec=_QEC,
    internal=_INTERNAL,
    tracked=tracked,
    decoder="eager",
    known_failures=_known_failures(),
)

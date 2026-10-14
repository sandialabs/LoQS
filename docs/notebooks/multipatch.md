---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Multi-Patch Surface Code Operations

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sandialabs/LoQS/{{ binder_branch }}?filepath=docs/notebooks/multipatch.ipynb)

A single logical qubit is only half the story: real algorithms need logical qubits to *interact*. This tutorial walks through the two-qubit (2Q) operations LoQS's Surface-17/13/10 codepack provides, from simplest to most fault-tolerant, using two `"surf10"` patches throughout -- the smallest layout, chosen here purely to keep circuits small and simulation fast; every operation shown works identically on `"surf17"`/`"surf13"`. Every multi-patch instruction builder below takes a `PatchGeometry`, which bundles the patches' labels and qubit lists (and any seam qubits) into a single object -- built once per section and reused both for the builder call and for the `"Init Patch"` stack entries.

1. **Transversal logical CNOT**: the cheapest 2Q gate, but only fault-tolerant if the two patches are already adjacent on real hardware.
2. **Ancilla-mediated joint parity measurement**: a non-destructive `Z_L(A) Z_L(B)` (or `X_L(A) X_L(B)`) readout using one bare ancilla qubit -- cheap, but *not* fault-tolerant.
3. **Lattice-surgery joint parity measurement**: the same joint parity, made fault-tolerant by merging the two patches through a physical seam for several syndrome-extraction rounds instead of using a single ancilla.
4. **Bell pair preparation** via a single `M_ZZ` merge, as the simplest genuinely fault-tolerant lattice-surgery protocol.

+++

## Setup

```{code-cell} ipython3
from loqs.backends import STIMQuantumState
from loqs.backends.model.dictmodel import DictNoiseModel
from loqs.backends.reps import StimCircuitGateRep
from loqs.codepacks import codepack_surf17_tomita2014 as surf
from loqs.codepacks import codepack_surf17_multipatch as multipatch
from loqs.codepacks import codepack_surf17_surgery as surgery
from loqs.core import PatchGeometry, QuantumProgram

LAYOUT = "surf10"

def patch_qubits(suffix):
    """A surf10 patch's (data + auxiliary) qubit labels, suffixed to keep
    multiple patches' qubits distinct."""
    return [f"{q}{suffix}" for q in surf.layout_qubits(LAYOUT)]
```

We use the STIM backend throughout -- these are all Clifford circuits, and STIM's stabilizer simulation is both exact and dramatically faster than a dense statevector for this many qubits.

+++

## 1. Transversal logical CNOT

The cheapest possible 2Q gate: one physical `Gcnot` per aligned data-qubit pair (9 for surf10), all in a single layer, plus a classical bookkeeping step that exactly conjugates both patches' Pauli frames and keeps their syndrome histories consistent across the gate. It's fault-tolerant *if* the two patches are laid out so their data qubits can be paired by a real 2-qubit gate directly -- e.g. two adjacent patches on a fixed hardware grid.

```{code-cell} ipython3
q0 = patch_qubits("_0")
q1 = patch_qubits("_1")
all_qubits = q0 + q1

code = surf.create_qec_code(layout=LAYOUT, num_qec_rounds=3)
model = surf.create_ideal_model(all_qubits, gaterep=StimCircuitGateRep, model_backend=DictNoiseModel)

cnot_geometry = PatchGeometry(patches={"ctrl": ("L0", q0), "tgt": ("L1", q1)}, layout=LAYOUT)
cnot = multipatch.build_transversal_cnot_instruction(cnot_geometry)

stack = [
    {"instruction": "Init State", "state": len(all_qubits), "qubit_labels": all_qubits},
    *cnot_geometry.init_patch_entries("SURF"),
    ("Plus Prep", "L0"),
    ("Zero Prep", "L1"),
    ("QEC", "L0"),
    ("QEC", "L1"),
    (cnot, None),  # A global instruction: it acts on both patches, not one
    ("QEC", "L0"),
    ("QEC", "L1"),
    ("FT Logical Z Measure", "L0"),
    ("FT Logical Z Measure", "L1"),
]

program = QuantumProgram(
    stack,
    default_noise_model=model,
    state_type=STIMQuantumState,
    patch_types={"SURF": code},
    name="Transversal CNOT",
)
results = program.run(num_shots=20, verbose=False)
logicals = results.collect_shot_data("logical_measurement", "all", strip_none_entries=True)
logicals[:5]
```

`CNOT` applied to $\ket{+}_L \ket{0}_L$ leaves the two logical qubits perfectly correlated in the $Z$ basis every shot:

```{code-cell} ipython3
assert all(l0 == l1 for l0, l1 in logicals)
```

## 2. Ancilla-mediated joint parity (non-fault-tolerant)

Sometimes a full entangling gate is more than you need -- often a single joint *parity* measurement is enough (e.g. one round of a distance-3 repetition code across two logical qubits). `build_joint_parity_zz_instruction`/`build_joint_parity_xx_instruction` measure `Z_L(A) Z_L(B)` (or `X_L(A) X_L(B)`) directly with a single bare ancilla: six `Gcnot`s copy each patch's `Z_L` support onto the ancilla (prepared in $\ket{0}$), which is then measured and reset.

The result is stored under a frame key scoped by which two patches produced it (`joint_parity_zz_{{patch_a_label}}_{{patch_b_label}}`), so multiple simultaneous joint measurements between different patch pairs in a larger program can't collide.

```{code-cell} ipython3
ancilla = "Qanc"
all_qubits = q0 + q1 + [ancilla]
model = surf.create_ideal_model(all_qubits, gaterep=StimCircuitGateRep, model_backend=DictNoiseModel)

parity_geometry = PatchGeometry(patches={"a": ("L0", q0), "b": ("L1", q1)}, layout=LAYOUT)
zz = multipatch.build_joint_parity_zz_instruction(parity_geometry, ancilla)

stack = [
    {"instruction": "Init State", "state": len(all_qubits), "qubit_labels": all_qubits},
    *parity_geometry.init_patch_entries("SURF"),
    ("Zero Prep", "L0"),
    ("X", "L1"),  # flip L1 to |1>_L, so Z_L(L0) Z_L(L1) parity is odd
    ("QEC", "L0"),
    ("QEC", "L1"),
    (zz, None),
]
program = QuantumProgram(
    stack,
    default_noise_model=model,
    state_type=STIMQuantumState,
    patch_types={"SURF": code},
    name="Ancilla-mediated joint ZZ parity",
)
results = program.run(num_shots=20, verbose=False)
parity = results.collect_shot_data("joint_parity_zz_L0_L1", "all", strip_none_entries=True)
parity[:5]
```

$Z_L\ket{0} \cdot Z_L\ket{1}$ has odd parity every shot, as expected:

```{code-cell} ipython3
assert all(p[0] == 1 for p in parity)
```

The measured operator commutes with every stabilizer of both patches, so this is non-destructive -- QEC continues normally afterward, and both patches are still valid code states. What it is *not* is fault-tolerant: a single fault on the ancilla or on a data qubit during the six `Gcnot`s can flip the parity, or even spread an error onto a patch, with no protection at all. That's the gap lattice surgery closes.

+++

## 3. Lattice-surgery joint parity (fault-tolerant)

Instead of a single ancilla, lattice surgery temporarily fuses the two patches into one larger code block through a physical seam, and measures the joint parity as an ordinary syndrome-extraction check repeated over several rounds -- exactly like measuring any other stabilizer, just one that happens to span two patches. `build_surgery_parity_instruction` handles the whole merge-then-split sequence as a single composite instruction.

```{code-cell} ipython3
SEAMS = ["Qs0", "Qs1", "Qs2"]  # a d=3 seam always has exactly 3 qubits
all_qubits = q0 + q1 + SEAMS
model = surf.create_ideal_model(all_qubits, gaterep=StimCircuitGateRep, model_backend=DictNoiseModel)

surgery_geometry = PatchGeometry(patches={"a": ("L0", q0), "b": ("L1", q1)}, seam=SEAMS, layout=LAYOUT)
zz_surgery = surgery.build_surgery_parity_instruction("ZZ", surgery_geometry, mode="ft")

stack = [
    {"instruction": "Init State", "state": len(all_qubits), "qubit_labels": all_qubits},
    *surgery_geometry.init_patch_entries("SURF"),
    ("Zero Prep", "L0"),
    ("X", "L1"),
    ("QEC", "L0"),
    ("QEC", "L1"),
    (zz_surgery, None),
    ("QEC", "L0"),
    ("QEC", "L1"),
    ("FT Logical Z Measure", "L0"),
    ("FT Logical Z Measure", "L1"),
]
program = QuantumProgram(
    stack,
    default_noise_model=model,
    state_type=STIMQuantumState,
    patch_types={"SURF": code},
    name="Lattice-surgery ZZ parity",
)
results = program.run(num_shots=20, verbose=False)
parity = results.collect_shot_data("surgery_parity_zz_L0_L1", "all", strip_none_entries=True)
logicals = results.collect_shot_data("logical_measurement", "all", strip_none_entries=True)
parity[:5], logicals[:5]
```

Same odd parity as before, and the destructive per-patch readouts agree with it -- but now `mode="ft"` runs a real matching decode over the merge window, correcting for faults occurring anywhere during the several merge rounds, not just characterizing the ideal case. (`mode="simple"` is also available -- a cheaper majority-vote decode with no such protection, useful for quick sanity checks.)

```{code-cell} ipython3
assert all(p[0] == 1 for p in parity)
assert all(l0 ^ l1 == 1 for l0, l1 in logicals)
```

## 4. Bell pair preparation via a single M_ZZ merge

The simplest genuinely fault-tolerant lattice-surgery protocol: measuring $Z_L(A) Z_L(B)$ on the product state $\ket{+}_L \ket{+}_L$ projects onto a Bell pair, up to a classically-known correction. `build_mzz_bell_prep_sequence` bundles the merge and the outcome-conditioned frame correction together as a ready-to-splice list of stack entries.

```{code-cell} ipython3
bell_seq = surgery.build_mzz_bell_prep_sequence(surgery_geometry, mode="ft")

stack = [
    {"instruction": "Init State", "state": len(all_qubits), "qubit_labels": all_qubits},
    *surgery_geometry.init_patch_entries("SURF"),
    ("Plus Prep", "L0"),
    ("Plus Prep", "L1"),
    ("QEC", "L0"),
    ("QEC", "L1"),
    *bell_seq,
    ("QEC", "L0"),
    ("QEC", "L1"),
    {"instruction": "FT Logical Z Measure", "patch_label": "L0", "reference_round_mode_Z": "guarded_diff"},
    {"instruction": "FT Logical Z Measure", "patch_label": "L1", "reference_round_mode_Z": "guarded_diff"},
]
program = QuantumProgram(
    stack,
    default_noise_model=model,
    state_type=STIMQuantumState,
    patch_types={"SURF": code},
    name="Bell pair via lattice surgery",
)
results = program.run(num_shots=50, verbose=False)
logicals = results.collect_shot_data("logical_measurement", "all", strip_none_entries=True)
logicals[:8]
```

Both Bell branches, $\ket{00}$ and $\ket{11}$, appear at random, but the two logical qubits always agree with each other -- the signature of a genuine Bell pair, not just two independently-random bits:

```{code-cell} ipython3
assert all(l0 == l1 for l0, l1 in logicals)
assert {tuple(l) for l in logicals} == {(0, 0), (1, 1)}
```

`reference_round_mode_Z="guarded_diff"` is needed here for two independent reasons that happen to coincide: both patches are prepped in $\ket{+}_L$ (the conjugate basis to the $Z$ measurement, which makes the round-0 $Z$ syndrome layer genuinely random), and the merge itself rewrites each patch's grown boundary check, both of which the measurement decoder needs to be told to hedge against.


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

# Ancilla Qubits vs. a Joint Parity-Check Instrument

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sandialabs/LoQS/{{ binder_branch }}?filepath=docs/notebooks/paritycheckinstrument.ipynb)

The 3-qubit bit-flip repetition code protects against $X$ errors by encoding $|0\rangle_L = |000\rangle$ and $|1\rangle_L = |111\rangle$, and detects them via the two stabilizers $Z_0 Z_1$ and $Z_1 Z_2$.
There are two ways to measure a two-qubit parity like $Z_0 Z_1$ in LoQS:

1. **Ancilla-mediated**: entangle an ancilla qubit with the parity via two CNOTs, then perform an ordinary single-qubit ($1Q$) mid-circuit measurement (MCM) on the ancilla.
2. **Direct joint instrument**: apply a single two-qubit ($2Q$), two-outcome instrument straight to the data qubits, with no ancilla at all.

This notebook builds the same syndrome-extraction round both ways and confirms they agree, then compares the qubit cost.

+++

## Setup

We'll work directly with `DictNoiseModel`, `ListPhysicalCircuit`, and the statevector backend, rather than a full `QECCode`, to keep the comparison focused on the two measurement strategies themselves.

```{code-cell} ipython3
import numpy as np

from loqs.backends import ListPhysicalCircuit, NumpyStatevectorQuantumState as SVState
from loqs.backends.model.dictmodel import DictNoiseModel
from loqs.backends.reps import (
    UnitaryGateRep,
    ZBasisOutcomeOperationDictInstrumentRep,
    ZBasisProjectionInstrumentRep,
)

X = np.array([[0, 1], [1, 0]])
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

data_qubits = ["D0", "D1", "D2"]
```

## Approach 1: ancilla-mediated syndrome extraction

Two ancillas `A0`/`A1` each start in $|0\rangle$. `A0` picks up the parity of `D0`/`D1` via two CNOTs (data qubits as control, ancilla as target); `A1` does the same for `D1`/`D2`. A single-qubit Z-basis instrument (`MeasZ`) then reads each ancilla out.

```{code-cell} ipython3
anc_qubits = ["A0", "A1"]
qubits1 = data_qubits + anc_qubits

gate_dict1 = {"X": UnitaryGateRep(X, ()), "CNOT": UnitaryGateRep(CNOT, ())}
inst_dict1 = {"MeasZ": ZBasisProjectionInstrumentRep(None, True, ())}
model1 = DictNoiseModel(
    gate_dict1, inst_dict1, gatereps=[UnitaryGateRep], instreps=[ZBasisProjectionInstrumentRep]
)


def run_ancilla_syndrome(error_qubit=None):
    """Prepare |000>, optionally flip `error_qubit`, and extract the syndrome via ancillas."""
    layers = []
    if error_qubit is not None:
        layers.append([("X", (error_qubit,))])
    layers += [
        [("CNOT", ("D0", "A0"))],
        [("CNOT", ("D1", "A0"))],
        [("CNOT", ("D1", "A1"))],
        [("CNOT", ("D2", "A1"))],
        [("MeasZ", ("A0",)), ("MeasZ", ("A1",))],
    ]
    circuit = ListPhysicalCircuit(layers, qubits1)
    state = SVState(len(qubits1), qubits1)
    reps = model1.get_reps(circuit, [UnitaryGateRep], [ZBasisProjectionInstrumentRep])
    outcomes = state.apply_reps_inplace(reps)
    return outcomes["A0"][0], outcomes["A1"][0]


for error_qubit in [None, "D0", "D1", "D2"]:
    print(f"error on {error_qubit!s:>4} -> syndrome {run_ancilla_syndrome(error_qubit)}")
```

As expected: no syndrome fires with no error; an error on `D0` or `D2` only violates the stabilizer it's part of; an error on the shared qubit `D1` violates both.

+++

## Approach 2: a direct 2Q, single-outcome instrument

A genuine parity-check instrument has exactly **2 outcomes regardless of qubit count**: it isn't a $2^n$-outcome computational-basis measurement, but a single classical bit reporting which rank-$2^{n-1}$ parity subspace the state projected onto. We build one directly as a `ZBasisOutcomeOperationDictInstrumentRep`, with `outcome_qubits` naming a classical register (`"synd_01"`/`"synd_12"`) that isn't tied to any physical qubit -- because here, there is no physical ancilla qubit to tie it to.

```{code-cell} ipython3
even_proj = UnitaryGateRep(np.diag([1.0, 0, 0, 1.0]), ())  # projects onto {|00>, |11>}
odd_proj = UnitaryGateRep(np.diag([0, 1.0, 1.0, 0]), ())  # projects onto {|01>, |10>}

gate_dict2 = {"X": UnitaryGateRep(X, ())}
inst_dict2 = {
    "ParityZZ_01": ZBasisOutcomeOperationDictInstrumentRep(
        {"even": even_proj, "odd": odd_proj}, True, (), outcome_qubits="synd_01"
    ),
    "ParityZZ_12": ZBasisOutcomeOperationDictInstrumentRep(
        {"even": even_proj, "odd": odd_proj}, True, (), outcome_qubits="synd_12"
    ),
}
model2 = DictNoiseModel(
    gate_dict2,
    inst_dict2,
    gatereps=[UnitaryGateRep],
    instreps=[ZBasisOutcomeOperationDictInstrumentRep],
)


def run_instrument_syndrome(error_qubit=None):
    """Same experiment as above, but the parity is read out directly -- no ancillas."""
    layers = []
    if error_qubit is not None:
        layers.append([("X", (error_qubit,))])
    layers += [[("ParityZZ_01", ("D0", "D1")), ("ParityZZ_12", ("D1", "D2"))]]
    circuit = ListPhysicalCircuit(layers, data_qubits)
    state = SVState(len(data_qubits), data_qubits)
    reps = model2.get_reps(circuit, [UnitaryGateRep], [ZBasisOutcomeOperationDictInstrumentRep])
    outcomes = state.apply_reps_inplace(reps)
    return outcomes["synd_01"][0], outcomes["synd_12"][0]


for error_qubit in [None, "D0", "D1", "D2"]:
    print(f"error on {error_qubit!s:>4} -> syndrome {run_instrument_syndrome(error_qubit)}")
```

The syndromes match Approach 1 exactly, using only the 3 data qubits -- `A0`/`A1` never existed in this circuit at all. `"even"` and `"odd"` are recorded as `0`/`1` by their position in `outcome_ops`, the same convention an ordinary `{0, 1}`-keyed instrument uses.

## What's next?

This notebook built the instruments and circuits by hand to keep the comparison direct; see [Building a Complex Instruction](buildinstruction.md) for wrapping a syndrome-extraction round like this into a reusable `Instruction`, and [Building a QEC Code](buildqeccode.md) for assembling those into a full `QECCode`.

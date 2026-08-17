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

# Multi-Qubit Instruments

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sandialabs/LoQS/{{ binder_branch }}?filepath=docs/notebooks/multiqubitinstruments.ipynb)

`OutcomeOperationDictInstrumentRep` isn't limited to a single-qubit computational-basis measurement. It generalizes in two independent directions:

1. **Decomposable**: one classical label per physical qubit, reading multiple qubits' independent bits in a single instrument call.
2. **Joint**: a *single* classical channel that isn't owned by any one physical qubit, with an arbitrary number of outcomes (not just 2) -- e.g. a parity check, or a leakage-discriminating readout.

This notebook demonstrates both using a 3-qubit bit-flip repetition code's syndrome extraction as a running example, comparing them against the traditional ancilla-mediated approach. (For a worked example with more than 2 outcomes, see the [Leakage Modeling](leakage.md) tutorial's leakage-discriminating readout.)

+++

## Setup

We'll work directly with `DictNoiseModel`, `ListPhysicalCircuit`, and the statevector backend, rather than a full `QECCode`, to keep the comparison focused on the instrument-construction API itself.

```{code-cell} ipython3
import numpy as np

from loqs.backends import ListPhysicalCircuit, NumpyStatevectorQuantumState as SVState
from loqs.backends.model.dictmodel import DictNoiseModel
from loqs.backends.reps import (
    UnitaryGateRep,
    OutcomeOperationDictInstrumentRep,
    ZBasisProjectionInstrumentRep,
)

X = np.array([[0, 1], [1, 0]])
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

data_qubits = ["D0", "D1", "D2"]
```

The repetition code encodes $|0\rangle_L = |000\rangle$ and $|1\rangle_L = |111\rangle$, and detects $X$ errors via the two stabilizers $Z_0 Z_1$ and $Z_1 Z_2$.

+++

## Ancilla-mediated syndrome extraction

The traditional approach: two ancillas `A0`/`A1` each start in $|0\rangle$. `A0` picks up the parity of `D0`/`D1` via two CNOTs (data qubits as control, ancilla as target); `A1` does the same for `D1`/`D2`. An ordinary single-qubit Z-basis instrument (`MeasZ`) then reads each ancilla out.

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

## A joint instrument: direct parity-check readout

A genuine parity-check instrument has exactly **2 outcomes regardless of qubit count**: it isn't a $2^n$-outcome computational-basis measurement, but a single classical bit reporting which rank-$2^{n-1}$ parity subspace the state projected onto. We build one directly as a `OutcomeOperationDictInstrumentRep`, with `outcome_qubits` naming a classical register (`"synd_01"`/`"synd_12"`) that isn't tied to any physical qubit -- because here, there is no physical ancilla qubit to tie it to.

```{code-cell} ipython3
even_proj = UnitaryGateRep(np.diag([1.0, 0, 0, 1.0]), ())  # projects onto {|00>, |11>}
odd_proj = UnitaryGateRep(np.diag([0, 1.0, 1.0, 0]), ())  # projects onto {|01>, |10>}

gate_dict2 = {"X": UnitaryGateRep(X, ())}
inst_dict2 = {
    "ParityZZ_01": OutcomeOperationDictInstrumentRep(
        {"even": even_proj, "odd": odd_proj}, True, (), outcome_qubits="synd_01"
    ),
    "ParityZZ_12": OutcomeOperationDictInstrumentRep(
        {"even": even_proj, "odd": odd_proj}, True, (), outcome_qubits="synd_12"
    ),
}
model2 = DictNoiseModel(
    gate_dict2,
    inst_dict2,
    gatereps=[UnitaryGateRep],
    instreps=[OutcomeOperationDictInstrumentRep],
)


def run_instrument_syndrome(error_qubit=None):
    """Same experiment as above, but the parity is read out directly -- no ancillas."""
    layers = []
    if error_qubit is not None:
        layers.append([("X", (error_qubit,))])
    layers += [[("ParityZZ_01", ("D0", "D1")), ("ParityZZ_12", ("D1", "D2"))]]
    circuit = ListPhysicalCircuit(layers, data_qubits)
    state = SVState(len(data_qubits), data_qubits)
    reps = model2.get_reps(circuit, [UnitaryGateRep], [OutcomeOperationDictInstrumentRep])
    outcomes = state.apply_reps_inplace(reps)
    return outcomes["synd_01"][0], outcomes["synd_12"][0]


for error_qubit in [None, "D0", "D1", "D2"]:
    print(f"error on {error_qubit!s:>4} -> syndrome {run_instrument_syndrome(error_qubit)}")
```

The syndromes match the ancilla-mediated approach exactly, using only the 3 data qubits -- `A0`/`A1` never existed in this circuit at all. `"even"` and `"odd"` are recorded as `0`/`1` by their position in `outcome_ops`, the same convention an ordinary `{0, 1}`-keyed instrument uses.

+++

## A decomposable instrument: reading multiple qubits at once

The other generalization is the opposite of a joint channel: `outcome_qubits` can list *more than one* label, one per physical qubit, so a single instrument call reports each qubit's own independent bit. Unlike the parity check above, this is an ordinary (destructive) computational-basis readout -- just of several qubits at once instead of one at a time.

```{code-cell} ipython3
def basis_projector(bits):
    idx = bits[0] * 2 + bits[1]
    mat = np.zeros((4, 4))
    mat[idx, idx] = 1.0
    return UnitaryGateRep(mat, ())


read_both_ops = {
    (0, 0): basis_projector((0, 0)),
    (0, 1): basis_projector((0, 1)),
    (1, 0): basis_projector((1, 0)),
    (1, 1): basis_projector((1, 1)),
}
inst_dict3 = {
    "ReadBoth": OutcomeOperationDictInstrumentRep(
        read_both_ops, True, (), outcome_qubits=("D0", "D1")
    )
}
model3 = DictNoiseModel(
    {"X": UnitaryGateRep(X, ())},
    inst_dict3,
    gatereps=[UnitaryGateRep],
    instreps=[OutcomeOperationDictInstrumentRep],
)

for bits in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    circuit = ListPhysicalCircuit([[("ReadBoth", ("D0", "D1"))]], data_qubits)
    state = SVState(list(bits) + [0], data_qubits)
    reps = model3.get_reps(circuit, [UnitaryGateRep], [OutcomeOperationDictInstrumentRep])
    outcomes = state.apply_reps_inplace(reps)
    print(f"prepared {bits} -> read D0={outcomes['D0'][0]}, D1={outcomes['D1'][0]}")
```

One instrument, two classical outcomes, each independently reflecting the corresponding qubit's own bit -- as if `D0` and `D1` had each been measured with their own single-qubit instrument, but issued as a single call.

+++

## Resource cost

The ancilla-mediated approach needs 5 physical qubits (3 data + 2 ancilla) per syndrome round; the joint-instrument approach needs only the 3 data qubits. For a distance-$d$ repetition code this is the difference between $2d - 1$ and $d$ physical qubits -- a real savings whenever the hardware actually supports a native multi-qubit measurement.

+++

## Superposition is preserved by the joint instrument

A stabilizer measurement commutes with the logical operators, so it shouldn't disturb a logical superposition when no error has occurred. We check this directly for the joint parity-check instrument's logical $|+\rangle_L = (|000\rangle + |111\rangle)/\sqrt{2}$ -- note this is specific to the *joint* instrument; the decomposable `ReadBoth` instrument above is an ordinary destructive measurement and would collapse a superposition like any other Z-basis readout.

```{code-cell} ipython3
logical_plus = np.zeros(8)
logical_plus[0b000] = 1 / np.sqrt(2)
logical_plus[0b111] = 1 / np.sqrt(2)

circuit = ListPhysicalCircuit(
    [[("ParityZZ_01", ("D0", "D1")), ("ParityZZ_12", ("D1", "D2"))]], data_qubits
)
state = SVState(logical_plus.copy(), data_qubits)
reps = model2.get_reps(circuit, [UnitaryGateRep], [OutcomeOperationDictInstrumentRep])
outcomes = state.apply_reps_inplace(reps)

print("syndrome:", outcomes["synd_01"][0], outcomes["synd_12"][0])
print("state preserved:", np.allclose(np.abs(state.state.flatten()), logical_plus))
```

## What's next?

This notebook built the instruments and circuits by hand to keep the comparison direct; see [Building a Complex Instruction](buildinstruction.md) for wrapping a syndrome-extraction round like this into a reusable `Instruction`, [Building a QEC Code](buildqeccode.md) for assembling those into a full `QECCode`, and [Leakage Modeling](leakage.md) for a joint instrument with more than 2 outcomes.

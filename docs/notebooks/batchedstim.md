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

# Batched STIM Runs with StimEngine

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sandialabs/LoQS/{{ binder_branch }}?filepath=docs/notebooks/batchedstim.ipynb)

[StimEngine](api:StimEngine) runs many shots of a program on a
`STIMQuantumState` at close to stim's own speed. Where
[QuantumProgram.run](api:QuantumProgram.run) walks the instruction stack once
per shot and returns a `History` for each, [StimEngine.run](api:StimEngine.run)
returns only the outputs you declare, as one
[OutputResults](api:OutputResults) holding an array per output with a leading
shot axis.

This tutorial needs `stim` and `pymatching` (the `stim` and `pymatching`
extras), and the codepack section also needs `pygsti`. It builds a
surface-code memory program and runs it batched and per shot, then runs a
fault-tolerant Steane program on the engine's per-shot fast path, runs a
Pauli-product measurement with feed-forward and postselection, and finally
shows what the engine refuses.

```{code-cell} ipython3
import time

import numpy as np
import stim

from loqs.backends import (
    DictNoiseModel,
    ListPhysicalCircuit,
    STIMPhysicalCircuit,
    STIMQuantumState,
)
from loqs.backends.reps import StimCircuitGateRep, StimCircuitInstrumentRep
from loqs.codepacks import codepack_7_1_3_quantinuum2021
from loqs.core import (
    Frame,
    HistoryDataCollector,
    InstructionStack,
    OutputResults,
    QuantumProgram,
)
from loqs.core.instructions import Instruction, InstructionLabel
from loqs.core.instructions.builders import build_physical_circuit_instruction
from loqs.engines import StimEngine
from loqs.internal import Serializable
from loqs.tools.stimtools import build_stim_decoder_instruction, build_stim_model


def rate_and_error(x):
    """Mean of per-shot values and its standard error."""
    x = np.asarray(x, dtype=float).ravel()
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))
```

## A memory program

We take stim's generated distance-3 rotated surface code memory experiment with
five rounds, and a noisy twin of it with the same noise everywhere. The ideal
circuit becomes the LoQS program, and the noisy twin supplies the detector
error model that the decoder uses.

```{code-cell} ipython3
noise = dict(
    after_clifford_depolarization=0.005,
    after_reset_flip_probability=0.005,
    before_measure_flip_probability=0.005,
)
ideal = stim.Circuit.generated(
    "surface_code:rotated_memory_z", distance=3, rounds=5
)
noisy = stim.Circuit.generated(
    "surface_code:rotated_memory_z", distance=3, rounds=5, **noise
)
print(f"{ideal.num_qubits} qubits, {ideal.num_detectors} detectors")
```

The program is ordinary LoQS: circuit instructions followed by a classical
decoder instruction, so the same program also runs one shot at a time. For now
we split the circuit by hand at its `REPEAT` block, into a prep circuit (the
operations before it), one round and a readout circuit (the operations after
it). Each is a `STIMPhysicalCircuit` wrapped by
[build_physical_circuit_instruction](api:build_physical_circuit_instruction).
The readout gets a closing `TICK`, which keeps the notebook free of warnings.

```{code-cell} ipython3
prep, readout = stim.Circuit(), stim.Circuit()
round_body, repeat_count = None, 0
for op in ideal:
    if isinstance(op, stim.CircuitRepeatBlock):
        round_body, repeat_count = op.body_copy(), op.repeat_count
    elif round_body is None:
        prep.append(op)
    else:
        readout.append(op)
readout.append("TICK")

labels = list(range(ideal.num_qubits))


def circuit_instruction(circuit, name):
    physical = STIMPhysicalCircuit(
        circuit, qubit_labels=labels, suppress_tick_warning=True
    )
    return build_physical_circuit_instruction(physical, name=name)


prep_inst = circuit_instruction(prep, "Prep")
round_inst = circuit_instruction(round_body, "Round")
readout_inst = circuit_instruction(readout, "Readout")
print(f"{len(prep)} prep ops, a round repeated {repeat_count} times, "
      f"{len(readout)} readout ops")
```

The decoder instruction is built from the whole flattened ideal circuit, and
decodes with the noisy twin's detector error model. The model is built from the
ideal circuit and the noise arguments, and the program runs on a
`STIMQuantumState`.

```{code-cell} ipython3
decoder_inst = build_stim_decoder_instruction(
    STIMPhysicalCircuit(
        ideal.flattened(), qubit_labels=labels, suppress_tick_warning=True
    ),
    dem=noisy.detector_error_model(decompose_errors=True),
)

stack = (
    [{"instruction": "Init State", "state": len(labels), "qubit_labels": labels},
     InstructionLabel(prep_inst)]
    + [InstructionLabel(round_inst)] * repeat_count
    + [InstructionLabel(readout_inst), InstructionLabel(decoder_inst)]
)
program = QuantumProgram(
    stack,
    default_noise_model=build_stim_model(ideal, **noise),
    state_type=STIMQuantumState,
    default_base_seed=1,
)
```

## Run it batched

`outputs` takes the same collectors that LoQS's runners take; here the string
`"logical_error"` asks for the decoder's `logical_error` entry of every shot.
The measurement record, detectors and observables come with it. The DEM
decoder runs once, on the whole detector array.

```{code-cell} ipython3
shots = 100_000
start = time.perf_counter()
results = StimEngine().run(program, shots, outputs=["logical_error"])
batched_time = (time.perf_counter() - start) / shots

print("keys:", results.keys())
for key in ("measurements", "detectors", "logical_error"):
    print(f"{key}: shape {results[key].shape}")

det_rate = results["detectors"].mean()
ler, ler_err = rate_and_error(results["logical_error"])
print(f"detection-event rate {det_rate:.4f}")
print(f"logical error rate {ler:.4f} +/- {ler_err:.4f}")
print(f"wall time per shot {batched_time * 1e6:.2f} us")
```

## Compare with a per-shot run

The same program runs through `QuantumProgram.run`, and a
`HistoryDataCollector` pulls the same two outputs out of the histories.

```{code-cell} ipython3
per_shot_shots = 1000
start = time.perf_counter()
histories = program.run(per_shot_shots, verbose=False)
per_shot_time = (time.perf_counter() - start) / per_shot_shots

ps_ler = HistoryDataCollector.from_raw("logical_error").collect(histories)
ps_det = HistoryDataCollector.from_raw("detectors").collect(histories)

for label, d, l in [
    ("batched ", results["detectors"], results["logical_error"]),
    ("per shot", np.array(ps_det), np.array(ps_ler)),
]:
    (dr, de), (lr, le) = rate_and_error(d.mean(axis=1)), rate_and_error(l)
    print(f"{label}: detection events {dr:.4f} +/- {de:.4f}, "
          f"logical error {lr:.4f} +/- {le:.4f}")
print(f"per-shot time {per_shot_time * 1e3:.2f} ms, "
      f"batched speed-up {per_shot_time / batched_time:,.0f}x")
```

The two runs agree in distribution only, never shot for shot: they draw their
randomness differently. Within `StimEngine` the same seed gives the same
arrays.

```{code-cell} ipython3
again = StimEngine().run(program, 1000, outputs=["logical_error"])
once_more = StimEngine().run(program, 1000, outputs=["logical_error"])
print(np.array_equal(again["detectors"], once_more["detectors"]))
```

## Save the result

Results write to HDF5 or JSON by file extension. Large arrays belong in HDF5,
since the JSON file of the same result runs many times larger.

```{code-cell} ipython3
import os
import tempfile

with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "memory.h5")
    results.write(path)
    loaded = OutputResults.read(path)
    print(np.array_equal(loaded["detectors"], results["detectors"]))
    print(f"memory.h5 is {os.path.getsize(path) / 1e6:.1f} MB")
```

## The per-shot fast path

Some programs can't be traced once and replayed. The engine still runs them
without the per-shot executor's overhead, taking a per-shot fast path inside
the same `run` call. This is the fault-tolerant Steane program, with
repeat-until-success preparation and flag-driven adaptive error correction. We
build its code and ideal model from the codepack, and add depolarizing noise
at p = 0.01 by appending a depolarizing channel to each gate's STIM text and
a bit flip before each measurement.

```{code-cell} ipython3
p = 0.01
qubits = ["A0", "A1", "A2"] + [f"D{i}" for i in range(3, 10)]
code = codepack_7_1_3_quantinuum2021.create_qec_code(
    circuit_backend=ListPhysicalCircuit
)
steane_ideal = codepack_7_1_3_quantinuum2021.create_ideal_model(
    qubits, gaterep=StimCircuitGateRep, model_backend=DictNoiseModel
)

gates = {}
for (name, qs), rep in steane_ideal.gate_dict.items():
    channel = "DEPOLARIZE1" if len(qs) == 1 else "DEPOLARIZE2"
    targets = " ".join(str(i) for i in range(len(qs)))
    gates[(name, qs)] = StimCircuitGateRep(
        f"{rep.circuit_str}\n{channel}({p}) {targets}", qs
    )
instruments = {
    key: StimCircuitInstrumentRep(f"X_ERROR({p}) 0\nMR 0", key[1])
    for key in steane_ideal.inst_dict
}
steane_model = DictNoiseModel(
    gates,
    instruments,
    gatereps=[StimCircuitGateRep],
    instreps=[StimCircuitInstrumentRep],
)

steane_stack = [
    {"instruction": "Init State", "state": len(qubits), "qubit_labels": qubits},
    {"instruction": "Init Patch Steane", "new_patch_label": "L0", "qubits": qubits},
    ("FT Zero Prep", "L0"),
    ("Adaptive QEC", "L0"),
    ("Adaptive QEC", "L0"),
    ("FT Logical Z Measure", "L0"),
]
steane = QuantumProgram(
    InstructionStack(steane_stack),
    default_noise_model=steane_model,
    state_type=STIMQuantumState,
    patch_types={"Steane": code},
    default_base_seed=3,
)
```

The outputs are the last logical measurement and the repeat-until-success
counts, collected with the same specs as `HistoryDataCollector`.

```{code-cell} ipython3
outputs = [
    {"key": "logical_measurement", "indices": -1},
    {"key": "total_rus_count", "indices": "all", "strip_none_entries": True},
]
fast_shots = 2000
start = time.perf_counter()
fast = StimEngine().run(steane, fast_shots, outputs=outputs)
fast_time = (time.perf_counter() - start) / fast_shots

lengths = [len(m) for m in fast.records("measurements")]
repeats = [np.sum(r) for r in fast.records("total_rus_count")]
print("keys:", fast.keys(), "| detectors present:", "detectors" in fast)
print(f"measurement record length: min {min(lengths)}, max {max(lengths)}")
print("mean repeat count: %.3f +/- %.3f" % rate_and_error(repeats))
print("logical flip rate: %.4f +/- %.4f"
      % rate_and_error(fast["logical_measurement"]))
print(f"time per shot {fast_time * 1e3:.2f} ms")
```

Now the per-shot executor on the same program, for comparison.

```{code-cell} ipython3
slow_shots = 500
start = time.perf_counter()
slow = steane.run(slow_shots, verbose=False)
slow_time = (time.perf_counter() - start) / slow_shots

slow_flip = HistoryDataCollector.from_raw(outputs[0]).collect(slow)
slow_rus = [
    np.sum(r) for r in HistoryDataCollector.from_raw(outputs[1]).collect(slow)
]
print("per-shot flip rate: %.4f +/- %.4f" % rate_and_error(slow_flip))
print("per-shot repeat count: %.3f +/- %.3f" % rate_and_error(slow_rus))
print(f"time per shot {slow_time * 1e3:.2f} ms, "
      f"fast-path speed-up {slow_time / fast_time:.0f}x")
```

A program takes the fast path when the trace can't be replayed: an instruction
other than a composite reads the `stack` (as repeat-until-success and
flag-driven inserts do), or the program applies a `ProbabilisticStimGateRep`.
The engine chooses without any input from you, and
[StimEngine.compile](api:StimEngine.compile) refuses such programs, since
there's no single trace to compile. A fast-path result has a `measurements`
array whose length varies from shot to shot (use `records`), plus the outputs
you asked for, but no `detectors` or `observables`. Classical steps run once
per shot there, so it's slower than the batched path, but much faster than the
per-shot executor.

## Pauli-product measurements, feed-forward and postselection

Stim circuits can measure Pauli products with `MPP` and feed a measurement bit
forward as a Pauli gate controlled by a `rec` target. Both build in a
`STIMPhysicalCircuit` and run on the batched path as stim runs them. The
circuit below prepares `|++>`, measures `Z0*Z1`, corrects qubit 1 by the
outcome, adds a bit flip on both qubits, measures the parity again and flags a
detector if that second parity is odd.

```{code-cell} ipython3
parity_text = """
H 0 1
TICK
MPP Z0*Z1
CX rec[-1] 1
X_ERROR(0.05) 0 1
TICK
MPP Z0*Z1
DETECTOR rec[-1]
M 0 1
TICK
"""
parity = stim.Circuit(parity_text)
parity_labels = [0, 1]
parity_program = QuantumProgram(
    [
        {"instruction": "Init State", "state": 2, "qubit_labels": parity_labels},
        InstructionLabel(
            build_physical_circuit_instruction(
                STIMPhysicalCircuit(parity, qubit_labels=parity_labels),
                name="Parity checks",
            )
        ),
    ],
    default_noise_model=build_stim_model(parity),
    state_type=STIMQuantumState,
    default_base_seed=1,
)
parity_results = StimEngine().run(parity_program, 100_000)
```

The `UserWarning` above is expected. It says the circuit's own noise lines
apply on top of the model's, and here the model from `build_stim_model` adds
none, so the circuit's noise is all there is. The model supplies no noise for
`MPP` or for the `rec`-controlled gate either; noise comes from the circuit's
noise lines and an `MPP`'s own argument.

`measurements` is stim's whole record, four bits here: the two parities, then
the final `M 0 1`. The `MPP` bits have no qubit label, so they aren't in
`measurement_outcomes`, which holds only labelled measurements.
`results["kept"]` is true for the shots in which no detector fired. The
counts are its sum and the shots left over, so postselecting is indexing by it.

```{code-cell} ipython3
kept = parity_results["kept"]
print("measurements:", parity_results["measurements"].shape)
print("kept:", kept.dtype, kept.shape)
n_kept = int(kept.sum())
print(f"kept {n_kept}, discarded {len(kept) - n_kept}")
print(f"discard rate {1 - kept.mean():.4f} (expected 2 * 0.05 * 0.95 = 0.095)")
```

The second parity check is deterministic only because the feed-forward ran.
The first parity is a random bit, and without the correction the second would
repeat it; with it, the qubits agree whatever the first was. So the second
parity is odd only when exactly one of the two bit flips fired, which is the
discard rate above.

```{code-cell} ipython3
record = parity_results["measurements"]
print(f"first parity bit set in {record[:, 0].mean():.4f} of shots")
print(f"second parity bit set in {record[:, 1].mean():.4f} of shots")
```

Everything else refuses both instructions with `NotImplementedError`, and
refuses at once. `QuantumProgram.run` on the `STIMQuantumState` can't record
the outcome of an `MPP`:

```{code-cell} ipython3
:tags: [raises-exception]

parity_program.run(1, verbose=False)
```

The dense states and the engine's per-shot fast path refuse the same
instructions. A decoder from `build_stim_decoder_instruction` can't run on a
circuit with `MPP` yet (the engine refuses it), and `stim_record_from_outcomes`
refuses one, so a program that both postselects with `MPP` and decodes isn't
available.

## What it refuses

A program that neither path can reproduce raises an error and never returns
arrays. The `MPP` and `rec`-controlled instructions above are refused outside
the batched path in the same way. When the traced shot meets the problem, the
error comes before any shot runs; otherwise it comes when a shot first meets
it. The main cases are a state other than `STIMQuantumState`, a classical
instruction that reads the quantum state, a state rebuilt after the first
circuit, and a physical circuit not built by
`build_physical_circuit_instruction`.
[StimEngine.compile](api:StimEngine.compile) lists them all.

Here the memory program gets one more instruction after the readout, whose
`apply_fn` takes `state`:

```{code-cell} ipython3
:tags: [raises-exception]

def peek_at_state(state):
    return Frame({"n": len(state.qubit_labels)})


peek = Instruction(
    peek_at_state,
    name="Peek at the state",
    serialized_apply_fn=Serializable.serialize_function(peek_at_state),
)
stack_with_peek = list(program.instruction_stack)
stack_with_peek.insert(len(stack_with_peek) - 1, InstructionLabel(peek))
peeking = QuantumProgram.from_quantum_program(
    program, InstructionStack(stack_with_peek)
)
StimEngine().run(peeking, 10)
```

The engine can't replay an instruction that looks at the quantum state, since
the batched sampler never builds one per shot, so it says so instead of
guessing.

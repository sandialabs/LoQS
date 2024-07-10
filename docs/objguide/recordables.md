# Recordables

The `History` framework is designed so that users can store whatever metadata is needed for their QEC code implementation.
However, there are also some obvious objects which many QEC implementations will want.
These objects are stored in `loqs.core.recordable` and are overviewed here.

## MeasurementOutcomes

The `MeasurementOutcomes` object is a dict with qubit label keys and lists of measured bits as values.
It is not a single outcome because a qubit may have multiple measurements during the course of single `Instruction`.

This object is typically used as input for feed-forward instructions or decoding.

The default `MeasurementOutcomes` object is typically associated with the `"measurement_outcomes"` frame key.

See the [API Reference](/devguide/_autosummary/loqs.core.recordables.measurementoutcomes.MeasurementOutcomes) for more in-depth documentation.

(recordables-patchdict)=
## PatchDict

The `PatchDict` objeect is a dict with patch label keys and `QECCodePatch` objects.

This object serves as the main way to store information about the code patches (i.e. which QEC code is being run on which physical qubits), so is the main target for any operations that affect patch metadata (e.g. permutations, split, merge).
Additionally, the default `PatchDict` is heavily used by the `QuantumProgram` during `Instruction` resolution;
for more information on this, see the [QuantumProgram tutorial section on instruction resolution](quantumprogram-instruction-resolution).

The default `PatchDict` object is typically associated with the `"patches"` frame key.

See the [API Reference](/devguide/_autosummary/loqs.core.recordables.patchdict.PatchDict) for more in-depth documentation.

## StabilizerFrame

The `StabilizerFrame` is currently under construction, but is intended to store the current stabilizer frame of the simulation.

```{warning}
This section is incomplete and will be filled out as the `StabilizerFrame` implementation is solidified.
```

The default `StabilizerFrame` object is typically associated with the `"stabilizer_frame"` frame key.

See the [API Reference](/devguide/_autosummary/loqs.core.recordables.stabilizerframe.StabilizerFrame) for more in-depth documentation.

## What's next?

We have now covered how to store simulation data in `LoQS` via the `History`, `Frame`, and various "recordable" objects.

Next, we will cover the heart of `LoQS`: the `Instruction`, which will take past data and produce a new `Frame` moving the simulation forward.

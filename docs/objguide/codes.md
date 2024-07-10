# QEC Codes and Patches

Now that we have talked about [Instructions](/objguide/instructions), we can finally talk about the `QECCode` class, and its more concrete cousin the `QECCodePatch`.

## QEC Codes

The `QECCode` is probably one of the most important high-level objects in `LoQS` because it describes all the operations (i.e. `Instruction` objects) of a given QEC code.
It is an abstract implementation in that all the operations are given on a set of template qubits;
in this way, we can define the `QECCode` once and apply it across multiple regions -- i.e. "patches" -- of the physical quantum state in a modular way.

The `QECCode` has two main attributes: a dictionary of instruction label keys mapping to `Instruction` objects, and a list of template qubits that are being used by aforementioned `Instruction` objects.
The key member function is `QECCode.create_patch()`, which takes a set of physical qubits and returns a `QECCodePatch` object.

(qec-code-patches)=
## Code Patches

Whereas the `QECCode` was a "reference" implementation of a QEC code, the `QECCodePatch` is the concrete application of that QEC code to a set of physical qubits.
The `QECCodePath` is effectively a dictionary with instruction label keys and `Instruction` values that have been mapped to the relevant physical qubits.

It is this object that will be stored in the [`PatchDict` object](recordables-patchdict) that is stored in the standard `History` object, and this is also what will be used to do patch-specific [`Instruction` resolution](quantumprogram-instruction-resolution) as we will cover soon.

## What's next?

See the API Reference for [`QECCode`](/devguide/_autosummary/loqs.core.qeccode.QECCode) or [`QECCodePatch`](/devguide/_autosummary/loqs.core.qeccode.QECCodePatch) for more in-depth documentation of respective objects.

Next, we will talk about prebuilt `QECCode` objects called "codepacks".
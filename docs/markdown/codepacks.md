
# Codepacks

A "codepack" is to a `QECCode` as the instruction builders are to an `Instruction`;
they are utility functions that return predefined `QECCode` objects for several common QEC code implementations.
They are designed to give users an easy starting point for running logical qubit simulations with `LoQS`, and power users an example for pattern matching and/or a starting point for implementing their own QEC implementations.

```{warning}
Note that the provided codepacks are not necessarily optimized or even mapped to physical hardware (i.e. they may contain expanded gate sets, impose no connectivity constraints, etc.).
```

Each codepack module is generally defined by having a `create_qec_code()` function, which returns the desired `QECCode` object.
This function may take arguments that influence the returned `QECCode`;
for example, codepacks for scalable QEC codes may take a distance argument, QEC codes that have multiple auxiliary qubit reuse patterns or syndrome extraction schedules may provide those as options, etc.
These options should be documented in the docstring of the relevant `create_qec_code()` functions.

(codepacks-5qubit)=
## The 5-Qubit Code

Currently the only codepack provided is for the [[5,1,3]] perfect code, colloquially known as the [5-qubit code](https://errorcorrectionzoo.org/c/stab_5_1_3).
Our implementation is based heavily on {cite}`codepacks-ryananderson_implementing_2022`, which in turn uses piecewise fault tolerance from {cite}`codepacks-yoder_universal_2016` and flag fault-tolerance from {cite}`codepacks-chao_quantum_2018`.

This codepack is available [here](/_autosummary/loqs.codepacks.codepack_5_1_3_quantinuum2022).

This codepack is also the subject of the [Building a QECCode tutorial](/markdown/buildqeccode), so those interested in learning how to build a `QECCode` object or understanding this codepack in particular may want to check that out.

## References

```{bibliography}
:labelprefix: codepacks
:keyprefix: codepacks-
:filter: docname in docnames
:style: unsrt
```

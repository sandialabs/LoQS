#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.3                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""QEC codepacks

A "codepack" is a reference implementation of a [](api:QECCode).
Each codepack is a class that builds a predefined `QECCode` for a common QEC code.
They are designed to give users an easy starting point for running logical qubit simulations with `LoQS`,
and power users an example for pattern matching and/or a starting point for implementing their own QEC implementations.

!!! warning

    Note that the provided codepacks are not necessarily optimized or even mapped to physical hardware (i.e. they may contain expanded gate sets, impose no connectivity constraints, etc.).

To write a codepack, subclass [](api:BaseCodepack) and implement `create_qec_code()`, which takes no arguments and returns the [](api:QECCode).
A codepack's options (a distance, a syndrome extraction schedule, etc.) are arguments to its own `__init__`,
which stores them on the instance, and are documented in the class docstring.
Configuration must reach instructions through `Instruction.data`, never through a closure over `self`,
because instructions are serialized as source and re-run in a fresh scope.

The base class provides `create_ideal_model()`, which returns a [](api:BaseNoiseModel) representing a noiseless physical qubit model
that contains exactly the gates and instruments the code uses.
This is primarily for testing, such that users can immediately use this as the `default_noise_model` parameter of [](api:QuantumProgram.__init__).
Two class attributes customize it: `_nonstd_unitaries` maps non-standard gate names to unitaries,
and `_idle_gate_names` lists the names that are one-qubit identities.
Each shipped codepack should also have a validation declaration in `validation/declarations/`,
which states its stabilizers, logical operators and logical gate actions;
`validation/README.md` ("Where a codepack lives") maps all of a codepack's files.

A generated codepack, such as [](api:SteaneCodepack), is a spec plus a locked artifact:
the spec is pure data with each fact's source, and a generator derives the bookkeeping tables from it.
Each codepack lives in its own package, and not every codepack is exported from `loqs.codepacks`;
import a class from its package, for example `from loqs.codepacks.trivial_counter import TrivialCounterCodepack`.

Examples
--------
>>> from loqs.codepacks import TrivialCounterCodepack
>>> code = TrivialCounterCodepack().create_qec_code()
>>> code.name
'Trivial Counter Code'
"""

from loqs.codepacks.base import BaseCodepack
from loqs.codepacks.steane import SteaneCodepack
from loqs.codepacks.trivial_counter import TrivialCounterCodepack

__all__ = ["BaseCodepack", "SteaneCodepack", "TrivialCounterCodepack"]

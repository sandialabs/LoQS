#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.1.1                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""QEC codepacks

A "codepack" is a reference implementation of a [](api:QECCode).
Each codepack provides utility functions that return predefined `QECCode` objects for several common QEC code implementations.
They are designed to give users an easy starting point for running logical qubit simulations with `LoQS`,
and power users an example for pattern matching and/or a starting point for implementing their own QEC implementations.

!!! warning

    Note that the provided codepacks are not necessarily optimized or even mapped to physical hardware (i.e. they may contain expanded gate sets, impose no connectivity constraints, etc.).

Each codepack module is generally defined by having a `create_qec_code()` function, which returns the desired [](api:QECCode) object.
This function may take arguments that influence the returned [](api:QECCode);
for example, codepacks for scalable QEC codes may take a distance argument,
QEC codes that have multiple auxiliary qubit reuse patterns or syndrome extraction schedules may provide those as options, etc.
These options should be documented in the docstring of the relevant `create_qec_code()` functions.

Most codepacks also implement a `create_ideal_model()` function, which returns a [](api:BaseNoiseModel) that represents a noiseless physical qubit model.
This is primarily for testing, such that users can immediately use this as the `default_noise_model` parameter of [](api:QuantumProgram.__init__).

!!! note

    The codepack API is subject to change; or rather, there currently is no enforced API,
    and one may be created in the future.

Examples
--------
>>> from loqs.codepacks import codepack_trivial_counter as cp
>>> code = cp.create_qec_code()
>>> code.name
'Trivial Counter Code'
"""

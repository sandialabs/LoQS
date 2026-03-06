#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.0                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Circuit backend classes.

Below use syndrome extraction circuits for the surface code
as an example for how to generate complex tiled circuits
from simple templates.

Examples
--------

Here we generate the syndrome extraction circuit
for Surface-17 based on :cite:`tomita_lowdistance_2014`.

>>> from loqs.backends.circuit import PyGSTiPhysicalCircuit as PhysCirc
>>> X_template = PhysCirc([('Gh', 'aux'), ('Gcnot', 'aux', 'b'),
...     ('Gcnot', 'aux', 'a'), ('Gcnot', 'aux', 'd'),
...     ('Gcnot', 'aux', 'c'), ('Gh', 'aux'), ('Iz', 'aux')],
...     qubit_labels=['a', 'b', 'c', 'd', 'aux']
... ) # Fig 2a
>>> Z_template = PhysCirc([[], ('Gcnot', 'b', 'aux'),
...     ('Gcnot', 'a', 'aux'), ('Gcnot', 'd', 'aux'),
...     ('Gcnot', 'c', 'aux'), [], ('Iz','aux')],
...     qubit_labels=['a', 'b', 'c', 'd', 'aux']
... ) # Fig 2b (with idle layers to match X check H layers)
>>> qubits = [f"D{i}" for i in range(9)] + [f"A{i}" for i in range(9, 17)]
>>> X_syndrome = PhysCirc.from_circuit_tiling(
...     X_template,
...     qubits,
...     [
...         [None, None, "D1", "D2" , "A9"],
...         ["D0", "D1", "D3", "D4", "A11"],
...         ["D4", "D5", "D7", "D8", "A14"],
...         ["D6", "D7", None, None, "A16"],
...     ],
...     merge_offsets=0 # Can all overlap
... )
>>> Z_syndrome = PhysCirc.from_circuit_tiling(
...     Z_template,
...     qubits,
...     [
...             [None, "D0", None, "D3", "A10"],
...             ["D1", "D2", "D4", "D5", "A12"],
...             ["D3", "D4", "D6", "D7", "A13"],
...             ["D5", None, "D8", None, "A15"],
...     ],
...     merge_offsets=0 # Can all overlap
... )
>>> full_syndrome = X_syndrome.merge(Z_syndrome, 0)
>>> print(full_syndrome)
Physical pyGSTi circuit:
  Qubit D0  ---|  |-|CA10|-|TA11|-|    |-|    |-|  |-|  |---
  Qubit D1  ---|  |-|TA11|-|CA12|-|    |-|TA9 |-|  |-|  |---
  Qubit D2  ---|  |-|CA12|-|    |-|TA9 |-|    |-|  |-|  |---
  Qubit D3  ---|  |-|    |-|CA13|-|CA10|-|TA11|-|  |-|  |---
  Qubit D4  ---|  |-|CA13|-|TA14|-|TA11|-|CA12|-|  |-|  |---
  Qubit D5  ---|  |-|TA14|-|CA15|-|CA12|-|    |-|  |-|  |---
  Qubit D6  ---|  |-|    |-|TA16|-|    |-|CA13|-|  |-|  |---
  Qubit D7  ---|  |-|TA16|-|    |-|CA13|-|TA14|-|  |-|  |---
  Qubit D8  ---|  |-|    |-|    |-|TA14|-|CA15|-|  |-|  |---
  Qubit A9  ---|Gh|-|    |-|    |-|CD2 |-|CD1 |-|Gh|-|Iz|---
  Qubit A10 ---|  |-|TD0 |-|    |-|TD3 |-|    |-|  |-|Iz|---
  Qubit A11 ---|Gh|-|CD1 |-|CD0 |-|CD4 |-|CD3 |-|Gh|-|Iz|---
  Qubit A12 ---|  |-|TD2 |-|TD1 |-|TD5 |-|TD4 |-|  |-|Iz|---
  Qubit A13 ---|  |-|TD4 |-|TD3 |-|TD7 |-|TD6 |-|  |-|Iz|---
  Qubit A14 ---|Gh|-|CD5 |-|CD4 |-|CD8 |-|CD7 |-|Gh|-|Iz|---
  Qubit A15 ---|  |-|    |-|TD5 |-|    |-|TD8 |-|  |-|Iz|---
  Qubit A16 ---|Gh|-|CD7 |-|CD6 |-|    |-|    |-|Gh|-|Iz|---
<BLANKLINE>
>>> print(repr(full_syndrome))
Physical pyGSTi circuit: \
Circuit([Gh:A9Gh:A11Gh:A14Gh:A16][Gcnot:A11:D1Gcnot:A14:D5Gcnot:A16:D7\
Gcnot:D0:A10Gcnot:D2:A12Gcnot:D4:A13][Gcnot:A11:D0Gcnot:A14:D4Gcnot:A16:D6\
Gcnot:D1:A12Gcnot:D3:A13Gcnot:D5:A15][Gcnot:A9:D2Gcnot:A11:D4Gcnot:A14:D8\
Gcnot:D3:A10Gcnot:D5:A12Gcnot:D7:A13][Gcnot:A9:D1Gcnot:A11:D3Gcnot:A14:D7\
Gcnot:D4:A12Gcnot:D6:A13Gcnot:D8:A15][Gh:A9Gh:A11Gh:A14Gh:A16]\
[Iz:A9Iz:A11Iz:A14Iz:A16Iz:A10Iz:A12Iz:A13Iz:A15]\
@(D0,D1,D2,D3,D4,D5,D6,D7,D8,A9,A10,A11,A12,A13,A14,A15,A16))

.. bibliography::
    :filter: docname in docnames
"""

from .basecircuit import BasePhysicalCircuit
from .listcircuit import ListPhysicalCircuit

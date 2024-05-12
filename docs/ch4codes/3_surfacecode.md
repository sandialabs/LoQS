---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3.11 (loqs)
  language: python
  name: loqs
---

# Example: Distance-3 Surface Code

Working on making a nice interface for the d=3 surface codes from {cite}`tomita_lowdistance_2014`.

```{code-cell}
from loqs.codepacks import d3_surface_code as codepack
```

## Stabilizer Plaquette Template

One of the nice things about the surface code is that the stabilizers can be described in a tileable way.

```{image} ../images/TomitaSvoreFig2.png
:name: fig2
:width: 400px
```
(Reproduced from {cite}`tomita_lowdistance_2014`)

```{image} ../images/TomitaSvoreFig4b.png
:name: fig4b
:width: 400px
```
(Reproduced from {cite}`tomita_lowdistance_2014`)

We can take advantage of this by coding up a "template" circuit that we will use to build up the syndrome circuit.
Below we show the `CircuitPlaquetteFactory` object provided in the surface code codepack, which holds these template circuits and is responsible for substituting the correct qubit values into this template.

```{code-cell}
# X stabilizer, matching Fig 2a
print(str(codepack.surface_factory.circuit_templates['X']))
```

```{code-cell}
# Z stabilizer, matching Fig 2b
print(str(codepack.surface_factory.circuit_templates['Z']))
```

```{code-cell}
# Alternate Z stabilizer, matching Fig 4b
print(str(codepack.surface_factory.circuit_templates['Zalt']))
```

The `CircuitPlaquetteFactory` can be used to get the actual circuits by providing the template type and qubit labels. <font color="red"> Note that the qubit labels have to match the template line labels exactly. </font>

```{code-cell}
c = codepack.surface_factory.get_circuit("X", ["A0", "D0", "D1", "D2", "D3"])
print(str(c))
```

There are a few advanced modes that can be used. We can also drop certain CNOTs in the case of lower weight stabilizer checks that maintain the same schedule to avoid CNOT collisions, or the midcircuit measurement can be removed in case one wants to generate the syndrome preparation circuit rather than the syndrome extraction circuit.

```{code-cell}
# Pass in Nones to skip some checks
c = codepack.surface_factory.get_circuit("X", ["A0", "D0", None, "D2", None])
print(str(c))
```

```{code-cell}
# Can omit the instrument
c = codepack.surface_factory.get_circuit("X", ["A0", "D0", "D1", "D2", "D3"], omit_gates='Iz')
print(str(c))
```

## Syndrome Circuit Generation

A syndrome circuit can now be quickly specified as a `PlaquetteCircuit` by giving the factory (from above) and the stabilizer types and qubit lists, which is essentially the information provided in the figure below.

```{image} ../images/TomitaSvoreFig1.png
:name: fig1
:width: 400px
```
(Reproduced from {cite}`tomita_lowdistance_2014`)

Below we show examples of the stabilizer specifications given in Surface-17 and Surface-13.
In Surface-17, all stabilizers can be done at once because each stabilizer has a dedicated auxiliary qubit. In Surface-13, there are two phases: the weight-4 checks can be done in parallel, as can the weight-2, but they reuse the same auxiliary qubits. That is specified as well.

```{code-cell}
# All checks in one stage for Surface-17
syndrome_circ_surface17 = codepack.surface17_syndrome
syndrome_circ_surface17.spec.stage_specs
```

```{code-cell}
# Two stages for Surface-13
syndrome_circ_surface13 = codepack.surface13_syndrome
syndrome_circ_surface13.spec.stage_specs
```

Finally, we can generate the full syndrome extraction circuits.

```{code-cell}
# Single stage for Surface-17
c17 = syndrome_circ_surface17.get_circuit()
print(str(c17))
```

```{code-cell}
c13 = syndrome_circ_surface13.get_circuit()

# We can see the two phases of Surface-13
print(str(c13))
```

## References

```{bibliography}
```

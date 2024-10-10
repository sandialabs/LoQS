---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Building a QEC Code

In the previous tutorial, we saw how to create a multi-stage feed-forward instruction.
A single instruction does not make a QEC code, but luckily not all instructions will take quite as much effort.
In this tutorial, we show how you can build the majority of a QEC code using the [Instruction builders](/markdown/builders).

## The [[5,1,3]] Code

We will focus on a partial implementation of the [[5,1,3]] code based on our [codepack](codepacks-5qubit).
In particular, we will define:

- A non-fault-tolerant (non-FT) $\ket{-}$ prep and unprep
- A fault-tolerant (FT) $\ket{-}$ prep
- Logical $Z$, $X$, and $H$ gates
- Stabilizer code basis transformations
- Non-FT logical $Z$ and $X$ measurements
- TODO: QEC

Combined with the adaptive FT measure out from the previous tutorial, this gives a legitimately useful QEC code to start playing around with.

Like the previous tutorial, we will use the `PyGSTiPhysicalCircuit` [circuit backend](circuit-backends).

### Non-FT $\ket{-}$ Prep/Unprep

The non-fault-tolerant versions of operations are often much simpler

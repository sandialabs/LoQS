---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
---

# (Advanced) Instruction Builders

```{warning}
Although this is located in the Object Quickstart section, this should probably be considered an advanced topic.
It is not required to understand *how* the builders work in order to use them, although understanding them may avoid misuse/errors in some heretofore uncaught edge cases.

This section could still be helpful to power users who are trying to write complex `Instruction` objects themselves and want to see how almost every part of the `Instruction` API has been used in practice.

If this power user description does not apply to you and/or the wall of text below seems intimidating, you have two options:

1. For the more-technically inclined, there is frankly more text in these explanations than lines of code. If you are very comfortable with the [Instruction guide](/markdown/instructions), it may be useful to read the source code directly by following the links to the API reference and then come back to this page if/when the inline comments leave you puzzled.
2. For the more-application inclined, a showcase of how to use many of these builders can be found in the [Building a Codepack tutorial](/markdown/buildqeccode). Feel free to assume the black box works, skip all of this, and go do some actual science.
```

The downside of flexibility is complexity, and that is certainly true of constructing an `Instruction` from scratch.
Luckily, there are several kinds of `Instruction` types that are used ubiquitously throughout QEC implementations that have been provided
in the `loqs.core.instructions.builders` module.

Someone looking for a more pedagogical/simple-to-complex ordering may consider starting with: the physical circuit instruction (which is not the simplest but probably the most familiar to most users); then the patch instructions (which are the simplest); moving on to the lookup decoder instruction; and finally the object builder, composite, and repeat-until-success instructions.

```{note}
Demonstrating these functions is difficult to do without talking about [circuit backends](circuit-backends) or [code patches](qec-code-patches) first.

Rather than have constant forward references, we will only talk about the builders at a high-level here and leave demonstrations to the [Building a QEC Code tutorial](/markdown/buildqeccode).
```

The documentation for the builders can be found in the [API documentation](/_autosummary/loqs.core.instructions.builders), and includes both the parameters to the builder function, as well as a short overview of the returned `Instruction`, including the apply function, the map qubits function (if needed), and any parameter aliases/priorities that need to be set.

+++

## What's next?

As mentioned above, consider seeing many of the builders in action in the ["Building a QEC Code" tutorial](/markdown/buildqeccode).

Before jumping to that tutorial, it may be worth looking at our next section where we explore the `QECCode` and `QECCodePatch` objects (which directly precedes the section on codepacks themselves).

# Codepacks

A generated codepack is a spec plus a locked artifact. The spec holds the
code's facts, circuits and declared logical gates; the generator derives the
tracking tables from it with Stim, and a locked class builds its `QECCode`
from the committed artifact without needing Stim. Each page below is
rendered from a spec and committed with it.

Each codepack is certified by the validation harness: each codepack's page
shows its validation badge, and the harness
[README](https://github.com/sandialabs/LoQS/tree/{{ binder_branch }}/validation)
describes the checks.

- [Steane [[7,1,3]]](steane.md)

The other codepacks are in the [API reference](/reference).

## Validation status

Each badge shows whether a codepack's latest validation run in CI passed on
the branch these docs were built from; a known bug left counts as failing.
The trivial counter's run also covers the harness's own tests.

- [![steane validation](https://github.com/sandialabs/LoQS/actions/workflows/validation-steane.yml/badge.svg?branch={{ binder_branch }})](https://github.com/sandialabs/LoQS/actions/workflows/validation-steane.yml?query=branch%3A{{ binder_branch }})
- [![five_qubit validation](https://github.com/sandialabs/LoQS/actions/workflows/validation-five_qubit.yml/badge.svg?branch={{ binder_branch }})](https://github.com/sandialabs/LoQS/actions/workflows/validation-five_qubit.yml?query=branch%3A{{ binder_branch }})
- [![surface validation](https://github.com/sandialabs/LoQS/actions/workflows/validation-surface.yml/badge.svg?branch={{ binder_branch }})](https://github.com/sandialabs/LoQS/actions/workflows/validation-surface.yml?query=branch%3A{{ binder_branch }})
- [![trivial_counter validation](https://github.com/sandialabs/LoQS/actions/workflows/validation-trivial_counter.yml/badge.svg?branch={{ binder_branch }})](https://github.com/sandialabs/LoQS/actions/workflows/validation-trivial_counter.yml?query=branch%3A{{ binder_branch }})

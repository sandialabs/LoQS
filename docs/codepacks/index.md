# Codepacks

A generated codepack is a spec plus a locked artifact. The spec holds the
code's facts, circuits and declared logical gates; the generator derives the
tracking tables from it with Stim, and a locked class builds its `QECCode`
from the committed artifact without needing Stim. Each page below is
rendered from a spec and committed with it.

Each codepack is certified by the validation harness: see its
[README](https://github.com/sandialabs/LoQS/tree/main/validation) and its
[records](https://github.com/sandialabs/LoQS/tree/main/validation/records).

- [Steane [[7,1,3]]](steane.md)

The other codepacks are in the [API reference](/reference).

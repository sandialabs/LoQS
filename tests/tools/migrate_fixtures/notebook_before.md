---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
---

# A Test Notebook

```{note}
This fence is not a code cell and must be left untouched, including any
mention of `PatchDict` or `InstructionLabel("Name", "L0", (), {})` inside
it.
```

```{code-cell} ipython3
from loqs.core.recordables.patchdict import PatchDict
from loqs.core.instructions import InstructionLabel

patches = PatchDict({"L0": None})
label = InstructionLabel("Increment", "L0", (), {"increment_by": 2})
```

Some prose in between two cells.

```{code-cell} ipython3
x = 1 + 1
```

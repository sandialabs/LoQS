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
from loqs.core.recordables.patchlayout import PatchLayout
from loqs.core.instructions import InstructionLabel

patches = PatchLayout({"L0": None})
label = InstructionLabel("Increment", increment_by=2, patch_label="L0")
```

Some prose in between two cells.

```{code-cell} ipython3
x = 1 + 1
```

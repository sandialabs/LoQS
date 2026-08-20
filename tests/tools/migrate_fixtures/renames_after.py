"""Historical note: this file used to construct a `PatchDict` directly."""
from loqs.core.recordables.patchlayout import PatchLayout
from loqs.backends.reps import RepTuple

patches = PatchLayout({"L0": None})
rep = RepTuple(1, 2, 3)

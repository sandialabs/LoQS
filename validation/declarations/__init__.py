"""Registry of the shipped codepacks' declarations."""

from validation.declaration import CodepackDeclaration
from validation.declarations.five_qubit import DECLARATION as _five_qubit
from validation.declarations.steane import DECLARATION as _steane
from validation.declarations.surface import DECLARATION as _surface
from validation.declarations.trivial_counter import (
    DECLARATION as _trivial_counter,
)

DECLARATIONS: dict[str, CodepackDeclaration] = {
    d.name: d for d in (_five_qubit, _steane, _surface, _trivial_counter)
}

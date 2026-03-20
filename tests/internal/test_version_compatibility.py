"""Test Serializable version compatibility."""

from pathlib import Path

import pytest

from loqs.backends.state.qsimstate import QSimQuantumState
from loqs.core.quantumprogram import QuantumProgram
from loqs.internal.serializable import IMPORT_LOCATION_CHANGES_BY_VERSION, Serializable


class TestVersionCompatibility:
    """Parameterized tests for Serializable class functionality."""

    @pytest.mark.parametrize("version_file",[
        "QuantumProgram_v0.json.gz",
        "QuantumProgram_v1.json.gz",
    ])
    def test_read_versioned_quantumprogram(self, version_file):
        """Test whether we can load QuantumProgram for given serialization version.
        
        Test files are taken from test_quantumprogram files."""

        path = Path(__file__).parent
        loaded_program = QuantumProgram.read(path / version_file)

        assert isinstance(loaded_program, QuantumProgram)
        #assert len(loaded_program.shot_histories) == 1
        assert loaded_program.name == "Prep minus, measure X"
        assert len(loaded_program.instruction_stack) == 4
        assert loaded_program.state_type == QSimQuantumState
        assert loaded_program.patch_types is not None
        assert list(loaded_program.patch_types.keys())[0] == "5Q"

        loaded_program.run(2)
        assert len(loaded_program.shot_histories) == 2
    
    def test_function_import_updates(self):
        # This is a real physical circuit instruction apply_fn from version 0
        test_str="""
from collections.abc import Mapping, Sequence
import inspect as ins
import numpy as np
from loqs.backends import propagate_state
from loqs.backends.circuit import BasePhysicalCircuit
from loqs.backends.model import (
    BaseNoiseModel,
    TimeDependentBaseNoiseModel,
)
from loqs.backends.state import BaseQuantumState
from loqs.core.frame import Frame
from loqs.core.qeccode import QECCode, QECCodePatch
from loqs.core.recordables.measurementoutcomes import MeasurementOutcomes
from loqs.core.recordables.patchdict import PatchDict
from loqs.core.syndrome import (
    PauliFrame,
    SyndromeLabel,
    SyndromeLabelCastableTypes
)
from loqs.backends import (
    STIMQuantumState,
    STIMPhysicalCircuit,
    PyGSTiNoiseModel,
)
def apply_fn(
    model: BaseNoiseModel,
    circuit: BasePhysicalCircuit,
    state: BaseQuantumState,
    inplace: bool,
    error_injections: list[tuple[int, str, int]] | None,
    pauli_frame_update: str | list[str] | dict[str, str] | None,
    patch_label: str,
    patches: PatchDict,
) -> Frame:

    [physical circuit apply function]
    [talks about PauliFrame]

    return Frame(data)
"""
        expected_str="""
from collections.abc import Mapping, Sequence
import inspect as ins
import numpy as np
from loqs.backends import propagate_state
from loqs.backends.circuit import BasePhysicalCircuit
from loqs.backends.model import BaseNoiseModel, TimeDependentBaseNoiseModel
from loqs.backends.state import BaseQuantumState
from loqs.core.frame import Frame
from loqs.core.qeccode import QECCode, QECCodePatch
from loqs.core.recordables.measurementoutcomes import MeasurementOutcomes
from loqs.core.recordables.patchdict import PatchDict
from loqs.core.recordables.pauliframe import PauliFrame
from loqs.core.syndromelabel import SyndromeLabel
from loqs.core.syndromelabel import SyndromeLabelCastableTypes
from loqs.backends import STIMQuantumState, STIMPhysicalCircuit, PyGSTiNoiseModel
def apply_fn(
    model: BaseNoiseModel,
    circuit: BasePhysicalCircuit,
    state: BaseQuantumState,
    inplace: bool,
    error_injections: list[tuple[int, str, int]] | None,
    pauli_frame_update: str | list[str] | dict[str, str] | None,
    patch_label: str,
    patches: PatchDict,
) -> Frame:

    [physical circuit apply function]
    [talks about PauliFrame]

    return Frame(data)
"""

        updated_str = Serializable._update_imports(test_str, 0)
        assert updated_str == expected_str

        # Also try one where a module name changes
        renamed_loc_change = IMPORT_LOCATION_CHANGES_BY_VERSION[1].copy()
        renamed_loc_change[("loqs.core.syndrome", "PauliFrame")] = ("loqs.core.recordables.pauliframe", "PauliFrameRenamed")

        expected_str2 = """
from collections.abc import Mapping, Sequence
import inspect as ins
import numpy as np
from loqs.backends import propagate_state
from loqs.backends.circuit import BasePhysicalCircuit
from loqs.backends.model import BaseNoiseModel, TimeDependentBaseNoiseModel
from loqs.backends.state import BaseQuantumState
from loqs.core.frame import Frame
from loqs.core.qeccode import QECCode, QECCodePatch
from loqs.core.recordables.measurementoutcomes import MeasurementOutcomes
from loqs.core.recordables.patchdict import PatchDict
from loqs.core.recordables.pauliframe import PauliFrameRenamed
from loqs.core.syndromelabel import SyndromeLabel
from loqs.core.syndromelabel import SyndromeLabelCastableTypes
from loqs.backends import STIMQuantumState, STIMPhysicalCircuit, PyGSTiNoiseModel
def apply_fn(
    model: BaseNoiseModel,
    circuit: BasePhysicalCircuit,
    state: BaseQuantumState,
    inplace: bool,
    error_injections: list[tuple[int, str, int]] | None,
    pauli_frame_update: str | list[str] | dict[str, str] | None,
    patch_label: str,
    patches: PatchDict,
) -> Frame:

    [physical circuit apply function]
    [talks about PauliFrameRenamed]

    return Frame(data)
"""

        updated_str2 = Serializable._update_imports(test_str, loc_change=renamed_loc_change)
        assert updated_str2 == expected_str2



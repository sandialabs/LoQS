"""Tester for loqs.backends.circuit.pygsti.PyGSTiPhysicalCircuit"""

import mock
import pytest

pygsti = pytest.importorskip("pygsti")
from pygsti.baseobjs import Label
from pygsti.circuits import Circuit

import loqs.backends as backends_module
from loqs.backends import ListPhysicalCircuit, PyGSTiPhysicalCircuit as PhysCirc


class TestPyGSTiPhysicalCircuit:

    @classmethod
    def setup_class(cls):
        # Testing all possibilities in LayerTypes
        cls.test_circ = Circuit([
            "Gidle", ('Gxpi2', 'Q0'), ('Gypi2', "Q1"), ('Gcnot', 'Q0', "Q1"),
            [('Gxpi2', 'Q0'), ('Gypi2', "Q1")], Label('Gxpi2', ("Q0",))],
            line_labels=["Q0", "Q1"]) # type: ignore
        cls.test_circ_intlbls = Circuit([
            "Gidle", ('Gxpi2', 0), ('Gypi2', 1), ('Gcnot', 0, 1),
            [('Gxpi2', 0), ('Gypi2', 1)], Label('Gxpi2', (0,))],
            line_labels=[0, 1]) # type: ignore

    def _check(self, circ, expected_circ):
        assert circ.circuit == expected_circ
        assert list(circ.qubit_labels) == list(expected_circ.line_labels)

    def test_init(self):
        # Base initializer
        pc = PhysCirc(self.test_circ, self.test_circ.line_labels)
        self._check(pc, self.test_circ)

        # Test implicit qubit label logic
        pc = PhysCirc(self.test_circ)
        self._check(pc, self.test_circ)

        # Test copy
        pc2 = PhysCirc(pc)
        self._check(pc2, self.test_circ)

        # Direct construction handles the same shapes .cast() used to.
        pc = PhysCirc(self.test_circ)
        self._check(pc, self.test_circ)

        pc2 = PhysCirc(pc)
        self._check(pc2, self.test_circ)

        # We should also be able to do string versions and just layers
        pc = PhysCirc(repr(self.test_circ)[8:-2])
        self._check(pc, self.test_circ)

        pc = PhysCirc(self.test_circ.layertup)
        self._check(pc, self.test_circ)

        # Test failure raises error
        with pytest.raises(ValueError):
            PhysCirc(None) # type: ignore

    def test_init_raises_import_error_when_unavailable(self):
        original = backends_module._backend_availability["pygsti_circuit"]
        backends_module._backend_availability["pygsti_circuit"] = (
            backends_module.BackendAvailability("pygsti_circuit", False)
        )
        try:
            with pytest.raises(ImportError, match="PyGSTi backend is not available"):
                PhysCirc([("Gxpi2", "Q0")], ["Q0"])
        finally:
            backends_module._backend_availability["pygsti_circuit"] = original

    def test_init_from_list_circuit_cast_failure_wrapped_as_valueerror(self):
        lc = ListPhysicalCircuit([[("Gxpi2", ("Q0",))]], ["Q0"])
        with mock.patch(
            "loqs.backends.circuit.pygsticircuit._Circuit.cast",
            side_effect=Exception("boom"),
        ):
            with pytest.raises(
                ValueError, match="Failed to cast list circuit to pyGSTi circuit"
            ):
                PhysCirc(lc)

    def test_from_tiling(self):
        template = PhysCirc([("Gxpi2", "A")], ["A"])
        tiled = PhysCirc.from_circuit_tiling(
            template,
            qubit_labels=["Q0", "Q1"],
            tile_qubits=[["Q0"], {"A": "Q1"}],
        )
        expected = Circuit(
            [[("Gxpi2", "Q0"), ("Gxpi2", "Q1")]], line_labels=["Q0", "Q1"]
        )
        self._check(tiled, expected)
    
    def test_append(self):
        circ1 = Circuit([('Gxpi2', 'Q0'), ('Gypi2', 'Q1')])
        expected_circ = circ1.append_circuit(circ1)

        pc = PhysCirc(circ1)

        pc2 = pc.append(pc)
        self._check(pc2, expected_circ)

        pc.append_inplace(pc)
        self._check(pc, expected_circ)
    
    def test_pad(self):
        padded_circ = Circuit([
            "Gidle", [('Gxpi2', 'Q0'), ('Gi', 'Q1')], [('Gypi2', "Q1"), ('Gi', "Q0")],
            ('Gcnot', 'Q0', "Q1"),
            [('Gxpi2', 'Q0'), ('Gypi2', "Q1")], [Label('Gxpi2', ("Q0",)), ('Gi', "Q1")]],
            line_labels=["Q0", "Q1"]) #type: ignore
    
        pc = PhysCirc(self.test_circ)
        pc2 = pc.pad_single_qubit_idles("Gi")
        self._check(pc2, padded_circ)

        pc.pad_single_qubit_idles_inplace("Gi")
        self._check(pc, padded_circ)

    def test_qubits(self):
        test_circ2 = self.test_circ.copy(editable=True) # type: ignore
        test_circ2.line_labels = ["Q0", "Q1", "Q2"]

        # Set qubits
        pc = PhysCirc(self.test_circ)
        pc2 = pc.set_qubit_labels(test_circ2.line_labels)
        assert list(pc2.qubit_labels) == list(test_circ2.line_labels)

        pc.set_qubit_labels_inplace(self.test_circ.line_labels)
        assert list(pc.qubit_labels) == list(self.test_circ.line_labels)
        
        # Delete qubits
        pc3 = PhysCirc(test_circ2)
        pc4 = pc3.delete_qubits(["Q2"])
        self._check(pc4, self.test_circ)

        pc3.delete_qubits_inplace(["Q1", "Q2"])
        test_circ2.delete_lines(["Q1", "Q2"], delete_straddlers=True)
        self._check(pc3, test_circ2)

        # Map qubits
        pc5 = PhysCirc(self.test_circ)
        pc6 = pc5.map_qubit_labels({"Q0": 0, "Q1": 1})
        self._check(pc6, self.test_circ_intlbls)

        pc5.map_qubit_labels_inplace({"Q0": 0, "Q1": 1})
        self._check(pc5, self.test_circ_intlbls)

    def test_get_possible_discrete_error_locations(self):
        pc = PhysCirc(
            [("Gxpi2", "Q0"), ("Gcnot", "Q0", "Q1")], ["Q0", "Q1"]
        )

        default_locs = pc.get_possible_discrete_error_locations()
        assert sorted(default_locs) == [(0, 0), (1, 0), (1, 1)]

        post_twoq_locs = pc.get_possible_discrete_error_locations(
            post_twoq_gates=True
        )
        assert post_twoq_locs == [(2, (0, 1))]

    def test_pad_by_duration_missing_duration_raises(self):
        pc = PhysCirc([("Gxpi2", "Q0")], ["Q0"])
        with pytest.raises(KeyError, match="No duration for Gxpi2"):
            pc.pad_single_qubit_idles_by_duration_inplace(
                idle_names={1: "Gi"}, durations={}
            )

    def test_pad_by_duration_empty_layer_idle(self):
        pc = PhysCirc([[], []], ["Q0"])
        pc.pad_single_qubit_idles_by_duration_inplace(
            idle_names={1: "Gi"}, durations={}, empty_layer_idle="GEmptyIdle"
        )
        expected = Circuit(
            [[("GEmptyIdle", "Q0")], [("GEmptyIdle", "Q0")]],
            line_labels=["Q0"],
        )
        self._check(pc, expected)

    def test_serialization(self, make_temp_path):
        pc = PhysCirc(self.test_circ, self.test_circ.line_labels)

        with make_temp_path(suffix=".json") as tmp_path:
            pc.write(tmp_path)
            pc2 = PhysCirc.read(tmp_path)

        assert isinstance(pc2, PhysCirc)
        self._check(pc2, self.test_circ)


# class TestPyGSTiPhysicalCircuitFailedImport:
#         # Mock not having the pygsti available
#         def test_failed_import(self):
#             with mock.patch.dict('sys.modules', {
#                     'pygsti.circuits': None,
#                     'pygsti.baseobjs': None,
#                 }):

#                 with pytest.raises(ImportError):
#                     import importlib
#                     import sys

#                     mod = sys.modules['loqs.backends.circuit.pygsticircuit']
#                     importlib.reload(mod)
                    

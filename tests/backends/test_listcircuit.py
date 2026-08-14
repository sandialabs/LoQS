"""Tester for loqs.backends.circuit.pygsti.PyGSTiPhysicalCircuit"""

from unittest import mock

import pytest

from loqs.backends import ListPhysicalCircuit as PhysCirc
from loqs.internal.serializable import Serializable


class TestListPhysicalCircuit:

    @classmethod
    def setup_class(cls):
        # Testing all possibilities in LayerTypes
        cls.test_circ = [
            ('Gxpi2', 'Q0'), ('Gypi2', "Q1"), ('Gcnot', ['Q0', "Q1"]),
            [('Gxpi2', 'Q0'), ('Gypi2', "Q1")]
        ]
        cls.expected_circ = [
            [('Gxpi2', ('Q0',))], [('Gypi2', ("Q1",))], [('Gcnot', ('Q0', "Q1"))],
            [('Gxpi2', ('Q0',)), ('Gypi2', ("Q1",))]
        ]
        cls.test_labels = ("Q0", "Q1")
        cls.expected_circ_intlbls = [
            [('Gxpi2', (0,))], [('Gypi2', (1,))], [('Gcnot', (0, 1))],
            [('Gxpi2', (0,)), ('Gypi2', (1,))]
        ]

    def _check(self, circ, expected_circ, expected_labels):
        for l1, l2 in zip(circ.circuit, expected_circ):
            set1 = set(l1) if len(l1) else set()
            set2 = set(l2) if len(l2) else set()
            assert set1 == set2
        assert set(circ.qubit_labels) == set(expected_labels)

    def test_init(self):
        # Base initializer
        pc = PhysCirc(self.test_circ, self.test_labels)
        self._check(pc, self.expected_circ, self.test_labels)

        # Test implicit qubit label logic
        pc = PhysCirc(self.test_circ)
        self._check(pc, self.expected_circ, self.test_labels)

        # Test copy
        pc2 = PhysCirc(pc)
        self._check(pc2, self.expected_circ, self.test_labels)

        # Test failure raises error
        with pytest.raises(ValueError):
            PhysCirc(None) # type: ignore

    def test_init_from_pygsti_circuit_does_not_reconstruct_it(self):
        """An existing `PyGSTiPhysicalCircuit` is read directly, without
        constructing or copying a second one."""
        pytest.importorskip("pygsti")
        from loqs.backends import PyGSTiPhysicalCircuit

        pgc = PyGSTiPhysicalCircuit([("Gxpi2", "Q0")], ["Q0"])
        original_init = PyGSTiPhysicalCircuit.__init__
        call_count = [0]

        def counting_init(self, *args, **kwargs):
            call_count[0] += 1
            return original_init(self, *args, **kwargs)

        with mock.patch.object(PyGSTiPhysicalCircuit, "__init__", counting_init):
            PhysCirc(pgc)
        assert call_count[0] == 0

    def test_append(self):
        circ1 = [[('Gxpi2', ('Q0',)), ('Gypi2', ('Q1',))]]
        expected_circ = circ1 + circ1

        pc = PhysCirc(circ1)

        pc2 = pc.append(pc)
        self._check(pc2, expected_circ, self.test_labels)

        pc.append_inplace(pc)
        self._check(pc, expected_circ, self.test_labels)
    
    def test_pad(self):
        padded_circ = [
            [('Gxpi2', ('Q0',)), ('Gi', ("Q1",))], [('Gypi2', ("Q1",)), ('Gi', ("Q0",))],
            [('Gcnot', ('Q0', "Q1"))], [('Gxpi2', ('Q0',)), ('Gypi2', ("Q1",))]
        ]
    
        pc = PhysCirc(self.test_circ, self.test_labels)
        pc2 = pc.pad_single_qubit_idles("Gi")
        self._check(pc2, padded_circ, self.test_labels)

        pc.pad_single_qubit_idles_inplace("Gi")
        self._check(pc, padded_circ, self.test_labels)

    def test_qubits(self):
        new_labels = ["Q0", "Q1", "Q2"]

        # Set qubits
        pc = PhysCirc(self.test_circ)
        pc2 = pc.set_qubit_labels(new_labels)
        self._check(pc2, self.expected_circ, new_labels)

        pc.set_qubit_labels_inplace(new_labels)
        self._check(pc2, self.expected_circ, new_labels)
        
        # Delete qubits
        pc3 = PhysCirc(self.test_circ, new_labels)
        pc4 = pc3.delete_qubits(["Q2"])
        self._check(pc4, self.expected_circ, self.test_labels)

        pc3.delete_qubits_inplace(["Q1", "Q2"])
        expected_circ = [[('Gxpi2', ('Q0',))],[],[],[('Gxpi2', ('Q0',))]]
        self._check(pc3, expected_circ, ["Q0"])

        # Map qubits
        pc5 = PhysCirc(self.test_circ)
        pc6 = pc5.map_qubit_labels({"Q0": 0, "Q1": 1})
        self._check(pc6, self.expected_circ_intlbls,[0,1])

        pc5.map_qubit_labels_inplace({"Q0": 0, "Q1": 1})
        self._check(pc5, self.expected_circ_intlbls, [0,1])

    def test_str_and_repr(self):
        pc = PhysCirc([[("Gxpi2", ("Q0",))]], ["Q0"])
        assert "Built-in list circuit" in str(pc)
        assert "[('Gxpi2', ('Q0',))]" in str(pc)
        assert "Built-in list circuit" in repr(pc)

    def test_add_and_iadd(self):
        """`circuit + other` must return a new (unmodified-original)
        circuit; `circuit += other` must mutate and rebind to the same
        object, not silently become `None` (`__iadd__` must `return
        self`, not the `None` returned by `append_inplace`)."""
        c1 = PhysCirc([[("Gx", ("Q0",))]], ["Q0"])
        c2 = PhysCirc([[("Gy", ("Q0",))]], ["Q0"])

        c3 = c1 + c2
        assert c3.circuit == [[("Gx", ("Q0",))], [("Gy", ("Q0",))]]
        # Original must be untouched by __add__
        assert c1.circuit == [[("Gx", ("Q0",))]]

        original_id = id(c1)
        c1 += c2
        assert c1 is not None
        assert id(c1) == original_id
        assert c1.circuit == [[("Gx", ("Q0",))], [("Gy", ("Q0",))]]

    def test_from_circuit_tiling_sequence_and_mapping(self):
        """`tile_qubits` entries may be given either as a plain sequence
        (zipped against the template's own qubit labels) or as an
        explicit `Mapping`; both must produce the same qubit relabeling."""
        template = PhysCirc([[("Gx", ("A",))]], ["A"])
        tiled = PhysCirc.from_circuit_tiling(
            template,
            qubit_labels=["Q0", "Q1"],
            tile_qubits=[["Q0"], {"A": "Q1"}],
        )
        assert tiled.depth == 1
        self._check(
            tiled,
            [[("Gx", ("Q0",)), ("Gx", ("Q1",))]],
            ["Q0", "Q1"],
        )

    def test_from_circuit_tiling_dropped_qubit(self):
        """Mapping a tile qubit to `None` must drop operations on that
        qubit from the tile rather than erroring."""
        template = PhysCirc(
            [[("Gx", ("A",)), ("Gy", ("B",))]], ["A", "B"]
        )
        tiled = PhysCirc.from_circuit_tiling(
            template,
            qubit_labels=["Q0"],
            tile_qubits=[{"A": "Q0", "B": None}],
        )
        self._check(tiled, [[("Gx", ("Q0",))]], ["Q0"])

    def test_from_circuit_tiling_positive_offset(self):
        """A positive `merge_offsets` staggers each successive tile by
        that many layers, rather than overlapping them at layer 0."""
        template = PhysCirc(
            [[("Gx", ("A",))], [("Gy", ("A",))]], ["A"]
        )
        tiled = PhysCirc.from_circuit_tiling(
            template,
            qubit_labels=["Q0", "Q1"],
            tile_qubits=[["Q0"], ["Q1"]],
            merge_offsets=2,
        )
        assert tiled.depth == 4
        self._check(
            tiled,
            [
                [("Gx", ("Q0",))],
                [("Gy", ("Q0",))],
                [("Gx", ("Q1",))],
                [("Gy", ("Q1",))],
            ],
            ["Q0", "Q1"],
        )

    def test_from_circuit_tiling_negative_offset(self):
        """A negative `merge_offsets` is computed relative to the tiled
        circuit's current depth at the time each tile is merged (per the
        docstring: "an offset of -1 is equivalent to circuit.depth-1"),
        not relative to the end of the whole final circuit."""
        template = PhysCirc([[("Gx", ("A",))]], ["A"])
        tiled = PhysCirc.from_circuit_tiling(
            template,
            qubit_labels=["Q0"],
            tile_qubits=[["Q0"]],
            merge_offsets=-1,
        )
        # tiled_circuit starts empty (depth 0), so offset = 0 - (-1) = 1:
        # the single tile is merged starting at layer 1, leaving layer 0
        # empty.
        assert tiled.depth == 2
        self._check(tiled, [[], [("Gx", ("Q0",))]], ["Q0"])

    def test_insert(self):
        """`.insert()` (non-inplace) must return a new circuit with the
        other circuit's layers inserted at `idx`, leaving the original
        untouched (unlike `.insert_inplace()`, which is already tested
        via `.append_inplace()`)."""
        c1 = PhysCirc([[("Gx", ("Q0",))]], ["Q0"])
        c2 = PhysCirc([[("Gy", ("Q0",))]], ["Q0"])
        c3 = c1.insert(c2, 0)
        self._check(
            c3, [[("Gy", ("Q0",))], [("Gx", ("Q0",))]], ["Q0"]
        )
        # Original must be untouched
        self._check(c1, [[("Gx", ("Q0",))]], ["Q0"])

    def test_pad_single_qubit_idles_by_duration_not_inplace(self):
        pc = PhysCirc([[("Gx", ("Q0",))], []], ["Q0"])
        pc2 = pc.pad_single_qubit_idles_by_duration(
            idle_names={1: "Gi"}, durations={"Gx": 1}, empty_layer_idle="Gi"
        )
        self._check(
            pc2,
            [[("Gx", ("Q0",))], [("Gi", ("Q0",))]],
            ["Q0"],
        )
        # Original must be untouched
        self._check(pc, [[("Gx", ("Q0",))], []], ["Q0"])

    def test_from_circuit_tiling_merge_offsets_as_sequence(self):
        """`merge_offsets` may be given as a per-tile sequence (not just a
        single int broadcast to every tile)."""
        template = PhysCirc([[("Gx", ("A",))]], ["A"])
        tiled = PhysCirc.from_circuit_tiling(
            template,
            qubit_labels=["Q0", "Q1"],
            tile_qubits=[["Q0"], ["Q1"]],
            merge_offsets=[0, 1],
        )
        assert tiled.depth == 2
        self._check(
            tiled,
            [[("Gx", ("Q0",))], [("Gx", ("Q1",))]],
            ["Q0", "Q1"],
        )

    def test_from_circuit_tiling_does_not_reconstruct_already_correct_template_type(self):
        """A `template_circuit` that's already the target class is used
        directly, without constructing or copying a second instance of it."""
        template = PhysCirc([[("Gx", ("A",))]], ["A"])
        original_init = PhysCirc.__init__
        seen_circuits = []

        def recording_init(self, *args, **kwargs):
            seen_circuits.append(args[0] if args else kwargs.get("circuit"))
            return original_init(self, *args, **kwargs)

        with mock.patch.object(PhysCirc, "__init__", recording_init):
            PhysCirc.from_circuit_tiling(
                template,
                qubit_labels=["Q0", "Q1"],
                tile_qubits=[["Q0"], ["Q1"]],
            )
        assert all(circ is not template for circ in seen_circuits)

    def test_get_possible_discrete_error_locations(self):
        pc = PhysCirc(
            [[("Gxpi2", ("Q0",)), ("Gcnot", ("Q0", "Q1"))]], ["Q0", "Q1"]
        )

        default_locs = pc.get_possible_discrete_error_locations()
        # Single-qubit gate contributes one location; two-qubit gate
        # contributes one location per qubit, all at the gate's own layer.
        assert sorted(default_locs) == [(0, 0), (0, 0), (0, 1)]

        post_twoq_locs = pc.get_possible_discrete_error_locations(
            post_twoq_gates=True
        )
        # Only the two-qubit gate is reported, at layer index + 1, as a
        # single combined (qubit-index-tuple) location.
        assert post_twoq_locs == [(1, (0, 1))]

    def test_merge_inplace_adds_new_qubit_labels(self):
        """Merging in a circuit with qubits the base circuit doesn't
        already have must extend `qubit_labels`, not just the layers."""
        pc = PhysCirc([[("Gx", ("Q0",))]], ["Q0"])
        other = PhysCirc([[("Gy", ("Q1",))]], ["Q1"])
        pc.merge_inplace(other, 0)
        assert pc.qubit_labels == ["Q0", "Q1"]
        self._check(pc, [[("Gx", ("Q0",)), ("Gy", ("Q1",))]], ["Q0", "Q1"])

    def test_pad_by_duration_missing_duration_raises(self):
        pc = PhysCirc([[("Gx", ("Q0",))]], ["Q0"])
        with pytest.raises(KeyError, match="No duration for Gx"):
            pc.pad_single_qubit_idles_by_duration_inplace(
                idle_names={1: "Gi"}, durations={}
            )

    def test_pad_by_duration_empty_layer_idle(self):
        """An entirely empty layer has no `layer_duration`, so the idle
        name must come from `empty_layer_idle` (if given) instead of
        `idle_names`; if not given, the layer is left untouched."""
        pc = PhysCirc([[], []], ["Q0"])
        pc2 = pc.copy()

        pc.pad_single_qubit_idles_by_duration_inplace(
            idle_names={1: "Gi"}, durations={}, empty_layer_idle="GEmptyIdle"
        )
        self._check(
            pc,
            [[("GEmptyIdle", ("Q0",))], [("GEmptyIdle", ("Q0",))]],
            ["Q0"],
        )

        # Without empty_layer_idle, empty layers are left as-is
        pc2.pad_single_qubit_idles_by_duration_inplace(
            idle_names={1: "Gi"}, durations={}
        )
        self._check(pc2, [[], []], ["Q0"])

    def test_serialization(self, make_temp_path):
        pc = PhysCirc(self.test_circ, self.test_labels)

        with make_temp_path(suffix=".json") as tmp_path:
            pc.write(tmp_path)
            pc2 = PhysCirc.read(tmp_path)

        assert isinstance(pc2, PhysCirc)
        self._check(pc2, self.expected_circ, self.test_labels)
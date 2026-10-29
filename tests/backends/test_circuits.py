"""Shared tests for the physical circuit backends (List, PyGSTi and STIM)."""

from types import SimpleNamespace

import pytest

from loqs.backends import ListPhysicalCircuit

BACKENDS = ["list", "pygsti", "stim"]


def _trim(layers):
    """Drop empty layers after the last operation."""
    layers = list(layers)
    while layers and not layers[-1]:
        layers.pop()
    return layers


def _build_adapter(name):
    """Return the helpers that build and read circuits of backend `name`."""
    vocab = {"h": "Gh", "x": "Gx", "cnot": "Gcnot", "i": "Gi"}

    if name == "list":
        cls = ListPhysicalCircuit

        def make(layers, labels):
            return ListPhysicalCircuit(
                [list(layer) for layer in layers], list(labels)
            )

        def layers_of(circ):
            return _trim(set(layer) for layer in circ.circuit)

    elif name in ("stim", "stim-open"):
        pytest.importorskip("stim")
        from loqs.backends.circuit.stimcircuit import STIMPhysicalCircuit

        cls = STIMPhysicalCircuit
        vocab = {"h": "H", "x": "X", "cnot": "CX", "i": "I"}
        arity = {"CX": 2}

        def make(layers, labels):
            # A TICK ends every layer; the open form drops the last line,
            # which is the final TICK (or an empty last layer).
            lines = []
            for layer in layers:
                lines += [f"{gate} {' '.join(qs)}" for gate, qs in layer]
                lines.append("TICK")
            if name == "stim-open":
                lines.pop()
            return STIMPhysicalCircuit("\n".join(lines), list(labels))

        def layers_of(circ):
            labels = circ.qubit_labels
            layers = [set()]
            for instr in circ.circuit:
                if instr.name == "TICK":
                    layers.append(set())
                    continue
                targets = [labels[t.value] for t in instr.targets_copy()]
                n = arity.get(instr.name, 1)
                for k in range(0, len(targets), n):
                    layers[-1].add((instr.name, tuple(targets[k : k + n])))
            return _trim(layers)

    else:
        pytest.importorskip("pygsti")
        from loqs.backends import PyGSTiPhysicalCircuit

        cls = PyGSTiPhysicalCircuit

        def make(layers, labels):
            return PyGSTiPhysicalCircuit(
                ListPhysicalCircuit(
                    [list(layer) for layer in layers], list(labels)
                ),
                list(labels),
            )

        def layers_of(circ):
            return _trim(
                set(layer) for layer in ListPhysicalCircuit(circ).circuit
            )

    def ends_with_tick(circ):
        if name in ("stim", "stim-open"):
            return str(circ.circuit).endswith("TICK")
        return None

    return SimpleNamespace(
        name=name,
        ends_with_tick=ends_with_tick,
        cls=cls,
        make=make,
        layers_of=layers_of,
        depth_offset=1 if name == "stim" else 0,
        v=vocab,
    )


@pytest.fixture(params=BACKENDS)
def adapter(request):
    return _build_adapter(request.param)


@pytest.fixture(params=[*BACKENDS, "stim-open"])
def layer_adapter(request):
    """Every backend, with STIM both with and without a final TICK."""
    return _build_adapter(request.param)


@pytest.fixture(params=["list", "pygsti"])
def tile_adapter(request):
    """Backends that can hold idle qubit labels, empty layers and tiles."""
    return _build_adapter(request.param)


def _q(*labels):
    return tuple(labels)


def _expect(*layers):
    return [set(layer) for layer in layers]


def test_construct_and_copy(adapter):
    a = adapter
    layers = [[(a.v["h"], _q("Q0"))], [(a.v["cnot"], _q("Q0", "Q1"))]]
    circ = a.make(layers, ["Q0", "Q1"])
    assert circ.depth == len(layers) + a.depth_offset
    assert a.layers_of(circ) == _expect(*layers)
    assert list(circ.qubit_labels) == ["Q0", "Q1"]

    copies = [circ.copy(), a.cls(circ)]
    for copied in copies:
        assert a.layers_of(copied) == _expect(*layers)
        assert list(copied.qubit_labels) == ["Q0", "Q1"]

    circ.append_inplace(circ)
    for copied in copies:
        assert a.layers_of(copied) == _expect(*layers)


def test_invalid_input_raises(adapter):
    with pytest.raises(ValueError):
        adapter.cls(None)


def test_append_and_add(layer_adapter):
    a = layer_adapter
    first = [[(a.v["h"], _q("Q0"))]]
    second = [[(a.v["x"], _q("Q0"))]]
    left = a.make(first, ["Q0"])
    right = a.make(second, ["Q0"])
    tick = a.ends_with_tick(left)

    out = left.append(right)
    assert a.layers_of(out) == _expect(*first, *second)
    assert a.layers_of(left) == _expect(*first)
    assert a.ends_with_tick(out) == tick

    left.append_inplace(right)
    assert a.layers_of(left) == _expect(*first, *second)
    assert a.ends_with_tick(left) == tick

    summed = right + right
    assert a.layers_of(summed) == _expect(*second, *second)
    assert a.layers_of(right) == _expect(*second)

    before = right
    right += right
    assert right is before
    assert a.layers_of(right) == _expect(*second, *second)


def test_insert(layer_adapter):
    a = layer_adapter
    base_layers = [[(a.v["h"], _q("Q0"))], [(a.v["cnot"], _q("Q0", "Q1"))]]
    other_layers = [[(a.v["x"], _q("Q0")), (a.v["x"], _q("Q1"))]]
    other = a.make(other_layers, ["Q0", "Q1"])

    for idx, expected in [
        (0, [*other_layers, *base_layers]),
        (1, [base_layers[0], *other_layers, base_layers[1]]),
        (2, [*base_layers, *other_layers]),
    ]:
        base = a.make(base_layers, ["Q0", "Q1"])
        tick = a.ends_with_tick(base)
        out = base.insert(other, idx)
        assert a.layers_of(out) == _expect(*expected)
        assert a.layers_of(base) == _expect(*base_layers)
        assert a.ends_with_tick(out) == tick

        base.insert_inplace(other, idx)
        assert a.layers_of(base) == _expect(*expected)
        assert a.ends_with_tick(base) == tick


def test_merge(layer_adapter):
    a = layer_adapter
    circ = a.make(
        [[(a.v["x"], _q("Q0"))], [(a.v["h"], _q("Q1"))]], ["Q0", "Q1"]
    )
    other = a.make(
        [[(a.v["x"], _q("Q0"))], [(a.v["h"], _q("Q2"))]], ["Q2", "Q0"]
    )
    tick = a.ends_with_tick(circ)
    circ.merge_inplace(other, 1)
    assert list(circ.qubit_labels) == ["Q0", "Q1", "Q2"]
    assert a.layers_of(circ) == _expect(
        [(a.v["x"], _q("Q0"))],
        [(a.v["h"], _q("Q1")), (a.v["x"], _q("Q0"))],
        [(a.v["h"], _q("Q2"))],
    )
    assert a.ends_with_tick(circ) == tick


def test_pad_by_duration(layer_adapter):
    a = layer_adapter
    layers = [
        [(a.v["h"], _q("Q0")), (a.v["x"], _q("Q1"))],
        [],
        [(a.v["x"], _q("Q1"))],
    ]
    expected = _expect(
        [(a.v["h"], _q("Q0")), (a.v["x"], _q("Q1")), (a.v["i"], _q("Q2"))],
        [],
        [(a.v["x"], _q("Q1")), (a.v["i"], _q("Q0")), (a.v["i"], _q("Q2"))],
    )
    durations = {a.v["h"]: 1, a.v["x"]: 1}
    circ = a.make(layers, ["Q0", "Q1", "Q2"])
    tick = a.ends_with_tick(circ)

    out = circ.pad_single_qubit_idles_by_duration({1: a.v["i"]}, durations)
    assert a.layers_of(out) == expected
    assert a.layers_of(circ) == _expect(*layers)
    assert a.ends_with_tick(out) == tick

    circ.pad_single_qubit_idles_by_duration_inplace({1: a.v["i"]}, durations)
    assert a.layers_of(circ) == expected
    assert a.ends_with_tick(circ) == tick


def test_pad_single_qubit_idles(adapter):
    a = adapter
    layers = [[(a.v["h"], _q("Q0"))], [(a.v["cnot"], _q("Q0", "Q1"))]]
    expected = _expect(
        [(a.v["h"], _q("Q0")), (a.v["i"], _q("Q1"))],
        [(a.v["cnot"], _q("Q0", "Q1"))],
    )
    circ = a.make(layers, ["Q0", "Q1"])

    out = circ.pad_single_qubit_idles(a.v["i"])
    assert a.layers_of(out) == expected
    assert a.layers_of(circ) == _expect(*layers)

    circ.pad_single_qubit_idles_inplace(a.v["i"])
    assert a.layers_of(circ) == expected


def test_pad_empty_layer_idle(adapter):
    a = adapter
    circ = a.make([[], []], ["Q0"])
    circ.pad_single_qubit_idles_by_duration_inplace(
        idle_names={1: a.v["i"]},
        durations={},
        empty_layer_idle=a.v["i"],
    )
    assert a.layers_of(circ) == _expect(
        [(a.v["i"], _q("Q0"))], [(a.v["i"], _q("Q0"))]
    )


def test_pad_missing_duration_raises(adapter):
    a = adapter
    circ = a.make([[(a.v["x"], _q("Q0"))]], ["Q0"])
    with pytest.raises(KeyError):
        circ.pad_single_qubit_idles_by_duration_inplace(
            idle_names={1: a.v["i"]}, durations={}
        )


def test_qubit_labels(adapter):
    a = adapter
    layers = [
        [(a.v["h"], _q("Q0")), (a.v["x"], _q("Q1"))],
        [(a.v["cnot"], _q("Q0", "Q1"))],
    ]

    # Delete a used label: its operations go with it, and the labels
    # after it shift down
    circ = a.make(layers, ["Q0", "Q1"])
    out = circ.delete_qubits(["Q0"])
    assert list(out.qubit_labels) == ["Q1"]
    assert a.layers_of(out) == _expect([(a.v["x"], _q("Q1"))])
    assert list(circ.qubit_labels) == ["Q0", "Q1"]
    circ.delete_qubits_inplace(["Q0"])
    assert list(circ.qubit_labels) == ["Q1"]
    assert a.layers_of(circ) == _expect([(a.v["x"], _q("Q1"))])

    # Set and delete an idle label
    three = ["Q0", "Q1", "Q2"]
    circ = a.make(layers, ["Q0", "Q1"])
    out = circ.set_qubit_labels(three)
    assert list(out.qubit_labels) == three
    assert a.layers_of(out) == _expect(*layers)
    assert list(circ.qubit_labels) == ["Q0", "Q1"]
    circ.set_qubit_labels_inplace(three)
    assert list(circ.qubit_labels) == three
    out = circ.delete_qubits(["Q2"])
    assert list(out.qubit_labels) == ["Q0", "Q1"]
    assert a.layers_of(out) == _expect(*layers)
    assert list(circ.qubit_labels) == three

    # Map labels, leaving the unmapped label alone
    circ = a.make(layers, ["Q0", "Q1"])
    expected = _expect(
        [(a.v["h"], _q("Q3")), (a.v["x"], _q("Q1"))],
        [(a.v["cnot"], _q("Q3", "Q1"))],
    )
    out = circ.map_qubit_labels({"Q0": "Q3"})
    assert list(out.qubit_labels) == ["Q3", "Q1"]
    assert a.layers_of(out) == expected
    assert list(circ.qubit_labels) == ["Q0", "Q1"]
    circ.map_qubit_labels_inplace({"Q0": "Q3"})
    assert list(circ.qubit_labels) == ["Q3", "Q1"]
    assert a.layers_of(circ) == expected


def _idle_pair(a):
    return [(a.v["i"], _q("Q0")), (a.v["i"], _q("Q1"))]


def test_transplant_idle_schedule(layer_adapter):
    a = layer_adapter
    cnot = (a.v["cnot"], _q("Q0", "Q1"))
    reference = a.make([_idle_pair(a), [cnot], _idle_pair(a)], ["Q0", "Q1"])
    target = a.make([[], [], [cnot], [], []], ["Q0", "Q1"])
    tick = a.ends_with_tick(target)
    target.transplant_idle_schedule_inplace(
        reference, ["Q0", "Q1"], [a.v["i"]]
    )
    assert a.layers_of(target) == _expect(
        _idle_pair(a), [], [cnot], _idle_pair(a)
    )
    assert a.ends_with_tick(target) == tick


@pytest.mark.parametrize("case", ["mismatch", "insufficient"])
def test_transplant_idle_schedule_raises(adapter, case):
    a = adapter
    x = (a.v["x"], _q("Q0"))
    idle = (a.v["i"], _q("Q0"))
    if case == "mismatch":
        reference = a.make([[idle], [x], [idle]], ["Q0"])
        target = a.make([[x], [x]], ["Q0"])
    else:
        reference = a.make([[x], [idle], [idle]], ["Q0"])
        target = a.make([[x]], ["Q0"])
    with pytest.raises(ValueError):
        target.transplant_idle_schedule_inplace(reference, ["Q0"], [a.v["i"]])


@pytest.mark.parametrize("case", ["sequence", "mapping", "dropped"])
def test_tiling(tile_adapter, case):
    a = tile_adapter
    gx = a.v["x"]
    gh = a.v["h"]
    if case == "sequence":
        template = a.make([[(gx, _q("A"))]], ["A"])
        labels = ["Q0", "Q1"]
        tile_qubits = [["Q0"], ["Q1"]]
        expected = _expect([(gx, _q("Q0")), (gx, _q("Q1"))])
    elif case == "mapping":
        template = a.make([[(gx, _q("A"))]], ["A"])
        labels = ["Q0", "Q1"]
        tile_qubits = [{"A": "Q0"}, {"A": "Q1"}]
        expected = _expect([(gx, _q("Q0")), (gx, _q("Q1"))])
    else:
        template = a.make([[(gx, _q("A")), (gh, _q("B"))]], ["A", "B"])
        labels = ["Q0"]
        tile_qubits = [{"A": "Q0", "B": None}]
        expected = _expect([(gx, _q("Q0"))])

    tiled = a.cls.from_circuit_tiling(template, labels, tile_qubits)
    assert a.layers_of(tiled) == expected
    assert list(tiled.qubit_labels) == labels


def test_discrete_error_locations(adapter):
    a = adapter
    circ = a.make(
        [[(a.v["x"], _q("Q0"))], [(a.v["cnot"], _q("Q0", "Q1"))]],
        ["Q0", "Q1"],
    )
    assert sorted(circ.get_possible_discrete_error_locations()) == [
        (0, 0),
        (1, 0),
        (1, 1),
    ]
    assert circ.get_possible_discrete_error_locations(
        post_twoq_gates=True
    ) == [(2, (0, 1))]


@pytest.mark.parametrize("suffix", [".json", ".h5"])
def test_serialization_round_trip(adapter, make_temp_path, suffix):
    a = adapter
    layers = [
        [(a.v["h"], _q("Q0")), (a.v["x"], _q("Q1"))],
        [(a.v["cnot"], _q("Q0", "Q1"))],
    ]
    # Q2 is unused and sits between the used labels
    labels = ["Q0", "Q2", "Q1"]
    circ = a.make(layers, labels)
    with make_temp_path(suffix=suffix) as tmp_path:
        circ.write(tmp_path)
        loaded = a.cls.read(tmp_path)

    assert isinstance(loaded, a.cls)
    assert a.layers_of(loaded) == _expect(*layers)
    assert list(loaded.qubit_labels) == labels

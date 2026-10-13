"""Tests for `loqs.internal.checkpointstore`: layouts, naming, atomic
writes, snapshots, payload puts and the in-memory twin."""

from __future__ import annotations

import json
import os
import pickle
import shutil
from pathlib import Path
from typing import cast

import h5py
import pytest

from loqs.core import Instruction, QuantumProgram
from loqs.core.history import History
from loqs.core.programresults import ProgramResults
from loqs.internal import checkpointstore
from loqs.internal.checkpointstore import (
    COMMIT_MARKER_ATTR,
    ITEM_LAYOUT,
    SHOT_LAYOUT,
    BatchReceipt,
    CheckpointReadError,
    CheckpointStore,
    MemoryCheckpointStore,
    StoreLayout,
    StreamedAttr,
    atomic_h5_file,
    side_length,
)
from loqs.internal.encoder import HDF5Encoder
from loqs.internal.serializable import ResolvingDecodeCache, Serializable
from loqs.internal.swmrledger import open_swmr_writer
from loqs.internal.streamingmerge import (
    _resolve_checkpoint_object_group,
    iter_dict_attr_entries,
    merge_dict_attr,
)
from _shared_checkpoint_test_helpers import (
    BASE_SEED,
    _final_frame_signature,
    build_shot_program,
    build_sweep_runner,
    injection_for,
    record_h5_writes,
    record_seed_apply,
    sessions_in,
    simulate_process_death,
)

STREAMED = ("shot_histories", "shot_wall_clock_times")


@pytest.fixture(scope="module")
def histories() -> dict[int, History]:
    """Six real shot histories, each recording its own seed."""
    results = build_shot_program().run(num_shots=6, verbose=False)
    return {i: results.get_shot_history(i) for i in range(6)}


def _shot_results(histories, keys, **kwargs) -> ProgramResults:
    return ProgramResults(
        shot_histories={k: histories[k] for k in keys},
        shot_wall_clock_times={k: 0.5 + k for k in keys},
        num_shots=6,
        **kwargs,
    )


def _shot_entries(histories, keys) -> dict[int, dict]:
    return {
        k: {"shot_histories": histories[k], "shot_wall_clock_times": 0.5 + k}
        for k in keys
    }


def _read_marker(path: Path) -> dict[str, int]:
    with h5py.File(path, "r") as f:
        return json.loads(f.attrs[COMMIT_MARKER_ATTR])


def _payload_entries(path, attr):
    """(key, value) pairs of a payload attr, read through streamingmerge."""
    with h5py.File(path, "r") as f:
        cache = ResolvingDecodeCache(root=f, format="hdf5")
        return list(iter_dict_attr_entries(f, attr, decode_cache=cache))


# -- atomic writes ---------------------------------------------------------


def test_atomic_h5_file_commits_and_cleans_up(tmp_path):
    final = tmp_path / "a.h5"
    with atomic_h5_file(final) as f:
        f.attrs["x"] = 1
        assert (tmp_path / "a.h5.tmp").exists()
    assert final.exists() and not (tmp_path / "a.h5.tmp").exists()

    with pytest.raises(RuntimeError):
        with atomic_h5_file(final) as f:
            f.attrs["x"] = 2
            raise RuntimeError("boom")
    assert not (tmp_path / "a.h5.tmp").exists()
    with h5py.File(final, "r") as f:
        assert f.attrs["x"] == 1


def test_payload_put_interrupt_sweep(tmp_path, histories):
    """An interrupt anywhere in a payload write never leaves a payload
    under its final name or a surviving temp file."""
    entries = _shot_entries(histories, [2, 3])
    probe = CheckpointStore(tmp_path / "probe", SHOT_LAYOUT)
    with record_h5_writes() as tap:
        probe.put_batch(entries, worker_id="w_1")
    session = sessions_in(tap.sessions, "shot_payload")[0]
    assert len(session.events) > 3

    failures = []
    for event in range(len(session.events)):
        store = CheckpointStore(tmp_path / f"e{event}", SHOT_LAYOUT)
        injection = injection_for(
            tap.sessions, "shot_payload", 1, event_index=event
        )
        with record_h5_writes(injection) as run:
            with pytest.raises(KeyboardInterrupt):
                store.put_batch(entries, worker_id="w_1")
        assert run.fired
        leftover = sorted(p.name for p in store.directory.glob("*"))
        if leftover:
            failures.append((event, session.events[event], leftover))
    assert not failures, failures


def test_snapshot_interrupt_sweep(tmp_path, histories):
    """An interrupt anywhere in `write_snapshot` leaves the old canonical
    file readable and unchanged."""
    old = _shot_results(histories, [0])
    new = _shot_results(histories, [0, 1])
    probe = CheckpointStore(tmp_path / "probe", SHOT_LAYOUT)
    with record_h5_writes() as tap:
        probe.write_snapshot(new)
    session = sessions_in(tap.sessions, "results")[0]

    failures = []
    for event in range(len(session.events)):
        store = CheckpointStore(tmp_path / f"e{event}", SHOT_LAYOUT)
        store.write_snapshot(old)
        injection = injection_for(
            tap.sessions, "results", 1, event_index=event
        )
        with record_h5_writes(injection):
            with pytest.raises(KeyboardInterrupt):
                store.write_snapshot(new)
        try:
            read = ProgramResults.read(store.canonical_path)
            ok = sorted(read.shot_histories) == [0]
            ok = ok and _read_marker(store.canonical_path) == {
                n: 1 for n in STREAMED
            }
        except Exception as exc:  # noqa: BLE001
            ok = False
            session_event = repr(exc)
        else:
            session_event = session.events[event]
        names = sorted(p.name for p in store.directory.glob("*"))
        ok = ok and names == ["results.h5"]
        if not ok:
            failures.append((event, session_event, names))
    assert not failures, failures


# -- snapshots -------------------------------------------------------------


def test_snapshot_of_empty_results(tmp_path):
    store = CheckpointStore(tmp_path, SHOT_LAYOUT)
    store.write_snapshot(ProgramResults(num_shots=6))
    loaded = ProgramResults.read(store.canonical_path)
    assert dict(loaded.shot_histories) == {}
    assert _read_marker(store.canonical_path) == {n: 0 for n in STREAMED}


def test_snapshot_marker_chunks_and_decode(tmp_path, histories):
    store = CheckpointStore(tmp_path, SHOT_LAYOUT)
    store.write_snapshot(_shot_results(histories, [0, 1]))
    assert _read_marker(store.canonical_path) == {n: 2 for n in STREAMED}
    with h5py.File(store.canonical_path, "r") as f:
        container = _resolve_checkpoint_object_group(f)
        checked = 0
        for name in STREAMED:
            sides = container[name]["dict"]
            for side in ("keys", "values"):
                iterable = sides[side]["iterable"]
                if iterable.attrs["storage_format"] == "dataset":
                    assert iterable["data"].chunks is not None
                    assert iterable["data"].maxshape == (None,)
                    checked += 1
        assert checked == 3
        cache = ResolvingDecodeCache(root=f, format="hdf5")
        decoded = Serializable.decode(f["root"], "hdf5", decode_cache=cache)
    assert isinstance(decoded, ProgramResults)
    assert sorted(decoded.shot_histories) == [0, 1]
    for key, history in decoded.shot_histories.items():
        assert _final_frame_signature(history)[0] == BASE_SEED + key


# -- payload puts ----------------------------------------------------------


def test_payload_put_contents_and_names(tmp_path, histories):
    store = CheckpointStore(tmp_path, SHOT_LAYOUT)
    receipt = store.put_batch(
        _shot_entries(histories, [3, 2]), worker_id="w_1"
    )
    expected = tmp_path / "worker_w_1_shots_2_payload.h5"
    assert receipt == BatchReceipt((2, 3), str(expected))
    assert store.payload_paths() == [expected]
    assert store.ledger_paths() == []
    pairs = _payload_entries(expected, "shot_histories")
    assert [k for k, _ in pairs] == [2, 3]
    for key, history in pairs:
        assert _final_frame_signature(history)[0] == BASE_SEED + key
    times = _payload_entries(expected, "shot_wall_clock_times")
    assert times == [(2, 2.5), (3, 3.5)]


def test_payload_put_item_layout(tmp_path, histories):
    store = CheckpointStore(tmp_path, ITEM_LAYOUT)
    kept = _shot_results(histories, [0, 1])
    entries = {
        0: {
            "results": histories[0],
            "item_wall_clock_times": 1.5,
            "shot_wall_clock_times": {0: 0.5, 1: 1.5},
            "_program_results": kept,
        },
        1: {"results": histories[1]},
    }
    receipt = store.put_batch(entries, worker_id="w_2")
    assert receipt.keys == (0, 1)
    path = tmp_path / "worker_w_2_item_0_payload.h5"
    assert store.payload_paths() == [path]
    assert [k for k, _ in _payload_entries(path, "results")] == [0, 1]
    assert _payload_entries(path, "item_wall_clock_times") == [(0, 1.5)]
    ((key, program_results),) = _payload_entries(path, "_program_results")
    assert key == 0 and sorted(program_results.shot_histories) == [0, 1]


def test_empty_batch_writes_nothing(tmp_path):
    store = CheckpointStore(tmp_path / "d", SHOT_LAYOUT)
    receipt = store.put_batch({}, worker_id="w")
    assert receipt == BatchReceipt((), None)
    assert not (tmp_path / "d").exists() or not any((tmp_path / "d").iterdir())


@pytest.mark.parametrize(
    "entries, kwargs",
    [
        ({0: {"shot_histories": 1, "bogus": 2}}, {}),
        ({0: {"shot_wall_clock_times": 1.0}}, {}),
        ({0: {"shot_histories": 1}, 1: {"shot_wall_clock_times": 1.0}}, {}),
        ({0: {"shot_histories": 1}}, {"encode_cache": {}}),
    ],
)
def test_bad_batches_write_no_file(tmp_path, entries, kwargs):
    store = CheckpointStore(tmp_path, SHOT_LAYOUT)
    with pytest.raises(ValueError):
        store.put_batch(entries, worker_id="w", **kwargs)
    assert list(tmp_path.iterdir()) == []


# -- naming and pickling ---------------------------------------------------


@pytest.mark.parametrize("layout", [SHOT_LAYOUT, ITEM_LAYOUT])
@pytest.mark.parametrize("worker", ["a", "host_12_pid_3", "x_shots_7"])
def test_worker_id_of_round_trip(tmp_path, layout, worker):
    store = CheckpointStore(tmp_path, layout)
    assert store.worker_id_of(store.ledger_path(worker)) == worker
    assert store.worker_id_of(store.payload_path(worker, 12)) == worker
    for bad in ("results.h5", "runner.h5", "worker_a_other.h5"):
        with pytest.raises(ValueError):
            store.worker_id_of(bad)
    with pytest.raises(ValueError):
        store.worker_id_of(str(store.payload_path(worker, 1)) + ".tmp")


def test_path_listing_excludes_tmp_and_sorts(tmp_path):
    store = CheckpointStore(tmp_path, SHOT_LAYOUT)
    names = [
        "worker_b_shots_0_payload.h5",
        "worker_a_shots_4_payload.h5",
        "worker_a_shots_4_payload.h5.tmp",
        "worker_a_checkpoint.h5",
        "worker_a_checkpoint.h5.tmp",
        "worker_a_runner.h5",
        "other.h5",
    ]
    for name in names:
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in store.payload_paths()] == [
        "worker_a_shots_4_payload.h5",
        "worker_b_shots_0_payload.h5",
    ]
    assert [p.name for p in store.ledger_paths()] == ["worker_a_checkpoint.h5"]


def test_canonical_filename_override_survives_pickle(tmp_path, histories):
    store = CheckpointStore(
        tmp_path, SHOT_LAYOUT, canonical_filename="custom.h5"
    )
    assert store.canonical_path == tmp_path / "custom.h5"
    clone = pickle.loads(pickle.dumps(store))
    assert clone.canonical_path == tmp_path / "custom.h5"
    assert clone.layout == store.layout
    receipt = clone.put_batch(_shot_entries(histories, [0]), worker_id="w")
    assert receipt.path is not None and Path(receipt.path).exists()


# -- the in-memory twin ----------------------------------------------------


def test_memory_store_basics():
    store = MemoryCheckpointStore(ITEM_LAYOUT)
    receipt = store.put_batch(
        {
            3: {"results": "c"},
            1: {"results": "a", "item_wall_clock_times": 1.0},
        },
        worker_id="w",
    )
    assert receipt == BatchReceipt((1, 3), None)
    store.put_batch({1: {"results": "other"}, 2: {"results": "b"}})
    assert store.keys() == [1, 2, 3]
    assert store.get(1) == "a"
    assert store.keys("item_wall_clock_times") == [1]
    assert list(store.iter_items(keys=[3, 1, 9])) == [(1, "a"), (3, "c")]
    assert store.missing(range(5)) == [0, 4]
    with pytest.raises(KeyError):
        store.get(0)
    with pytest.raises(ValueError):
        store.put_batch({0: {"results": 1}}, worker_id="w", encode_cache={})
    with pytest.raises(ValueError):
        store.put_batch({0: {"bogus": 1}})
    assert store.consolidate() == ()
    assert store.write_snapshot(ProgramResults()) is None
    with pytest.raises(TypeError):
        store.nested(1)


# -- the read API ----------------------------------------------------------


def _seed(history) -> int:
    return _final_frame_signature(history)[0]


def _make_store(kind, tmp_path):
    if kind == "memory":
        return MemoryCheckpointStore(SHOT_LAYOUT)
    return CheckpointStore(tmp_path, SHOT_LAYOUT)


@pytest.mark.parametrize("kind", ["memory", "durable"])
def test_contract_round_trip(tmp_path, histories, kind):
    store = _make_store(kind, tmp_path)
    store.put_batch(_shot_entries(histories, [4, 2]), worker_id="w_a")
    store.put_batch(_shot_entries(histories, [0, 1]), worker_id="w_b")
    assert store.keys() == [0, 1, 2, 4]
    assert store.keys("shot_wall_clock_times") == [0, 1, 2, 4]
    assert store.missing(range(6)) == [3, 5]
    assert store.missing([2, 4]) == []
    for key in (0, 4):
        assert _seed(store.get(key)) == BASE_SEED + key
        assert store.get(key, "shot_wall_clock_times") == 0.5 + key
    with pytest.raises(KeyError):
        store.get(3)
    items = list(store.iter_items())
    assert [k for k, _ in items] == [0, 1, 2, 4]
    assert all(_seed(h) == BASE_SEED + k for k, h in items)
    subset = list(store.iter_items("shot_wall_clock_times", keys=[4, 0, 9]))
    assert subset == [(0, 0.5), (4, 4.5)]
    with pytest.raises(ValueError):
        store.keys("bogus")


def test_collision_rule_first_committed_wins(tmp_path, histories):
    """The same key in the canonical file and two payloads is always read
    from the canonical copy, else from the lower-named payload."""
    store = CheckpointStore(tmp_path, SHOT_LAYOUT)
    canonical = ProgramResults(
        shot_histories={0: histories[0]},
        shot_wall_clock_times={0: 10.0},
        num_shots=6,
    )
    store.write_snapshot(canonical)

    def entry(history, wall):
        return {0: {"shot_histories": history, "shot_wall_clock_times": wall}}

    # Written out of name order, with an extra key in the later payload.
    later = entry(histories[2], 30.0)
    later[5] = {
        "shot_histories": histories[5],
        "shot_wall_clock_times": 35.0,
    }
    store.put_batch(later, worker_id="b")
    store.put_batch(entry(histories[1], 20.0), worker_id="a")

    def check(expected_seed, expected_wall):
        assert store.keys("shot_wall_clock_times") == store.keys() == [0, 5]
        assert _seed(store.get(0)) == BASE_SEED + expected_seed
        assert store.get(0, "shot_wall_clock_times") == expected_wall
        items = dict(store.iter_items())
        assert _seed(items[0]) == BASE_SEED + expected_seed
        assert dict(store.iter_items("shot_wall_clock_times"))[0] == (
            expected_wall
        )
        assert _seed(items[5]) == BASE_SEED + 5

    check(0, 10.0)
    store.canonical_path.unlink()
    check(1, 20.0)


def test_truncated_payload_is_a_strict_error(tmp_path, histories):
    store = CheckpointStore(tmp_path, SHOT_LAYOUT)
    store.put_batch(_shot_entries(histories, [0, 1]), worker_id="a")
    bad = store.put_batch(_shot_entries(histories, [2, 3]), worker_id="b")
    path = Path(bad.path)
    path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])
    with pytest.raises(CheckpointReadError) as info:
        store.keys(strict=True)
    assert info.value.path == str(path) and str(path) in str(info.value)
    assert info.value.__cause__ is not None
    with pytest.raises(CheckpointReadError):
        store.missing(range(4))
    assert store.keys(strict=False) == [0, 1]


def test_payload_without_primary_attr_is_a_strict_error(tmp_path):
    path = tmp_path / "worker_a_shots_0_payload.h5"
    with h5py.File(path, "w") as f:
        merge_dict_attr(
            f,
            "shot_wall_clock_times",
            [(0, 1.0)],
            encode_cache={},
            key_use_dataset=True,
            value_use_dataset=True,
        )
    store = CheckpointStore(tmp_path, SHOT_LAYOUT)
    with pytest.raises(CheckpointReadError, match="primary"):
        store.keys("shot_wall_clock_times")
    assert store.keys("shot_wall_clock_times", strict=False) == []


def test_unpaired_payload_sides_are_a_strict_error(tmp_path, histories):
    store = CheckpointStore(tmp_path, SHOT_LAYOUT)
    bad = store.put_batch(_shot_entries(histories, [0, 1]), worker_id="a")
    with h5py.File(bad.path, "a") as f:
        del f["shot_histories"]["dict"]["values"]["iterable"]["1"]
    with pytest.raises(CheckpointReadError) as info:
        store.keys()
    assert info.value.path == bad.path
    with pytest.raises(CheckpointReadError):
        store.missing(range(2))
    assert store.keys(strict=False) == []


def _set_marker(path, **lengths):
    with h5py.File(path, "a") as f:
        marker = json.loads(f.attrs[COMMIT_MARKER_ATTR])
        marker.update(lengths)
        f.attrs.modify(COMMIT_MARKER_ATTR, json.dumps(marker))


def test_marker_beyond_physical_length_is_a_strict_error(tmp_path, histories):
    store = CheckpointStore(tmp_path, SHOT_LAYOUT)
    store.write_snapshot(_shot_results(histories, [0, 1]))
    assert store.keys() == [0, 1]
    _set_marker(store.canonical_path, shot_histories=5)
    with pytest.raises(CheckpointReadError) as info:
        store.keys()
    assert info.value.path == str(store.canonical_path)
    assert store.keys(strict=False) == []


def test_canonical_without_marker_is_a_strict_error(tmp_path, histories):
    store = CheckpointStore(tmp_path, SHOT_LAYOUT)
    store.write_snapshot(_shot_results(histories, [0]))
    with h5py.File(store.canonical_path, "a") as f:
        del f.attrs[COMMIT_MARKER_ATTR]
    with pytest.raises(CheckpointReadError, match="marker"):
        store.keys()


def test_missing_canonical_file_holds_nothing(tmp_path):
    store = CheckpointStore(tmp_path / "absent", SHOT_LAYOUT)
    assert store.keys() == []
    assert store.missing([1, 0]) == [0, 1]
    assert list(store.iter_items()) == []


def test_tail_beyond_marker_is_invisible(tmp_path, histories):
    store = CheckpointStore(tmp_path, SHOT_LAYOUT)
    store.write_snapshot(_shot_results(histories, [0, 1]))
    # Add an unpaired tail: key 5 lands in both attrs, beyond the marker.
    with h5py.File(store.canonical_path, "a") as f:
        container = _resolve_checkpoint_object_group(f)
        merge_dict_attr(
            container,
            "shot_histories",
            [(5, histories[5])],
            encode_cache={},
            key_use_dataset=True,
            value_use_dataset=False,
        )
        merge_dict_attr(
            container,
            "shot_wall_clock_times",
            [(5, 9.0)],
            encode_cache={},
            key_use_dataset=True,
            value_use_dataset=True,
        )
        sides = container["shot_histories"]["dict"]
        assert side_length(sides["keys"]["iterable"]) == 3
    for attr in STREAMED:
        assert store.keys(attr) == [0, 1]
        assert [k for k, _ in store.iter_items(attr)] == [0, 1]
        with pytest.raises(KeyError):
            store.get(5, attr)
        assert list(store.iter_items(attr, keys=[5])) == []
    assert store.missing([5]) == [5]


def _named_histories(name) -> dict[int, History]:
    """Two shots of a program whose one instruction is called `name`, so
    different names give different objects under the same cache IDs."""
    instruction = Instruction(apply_fn=record_seed_apply, name=name)
    program = QuantumProgram(
        [{"instruction": name, "fail_prob": 0.5}],
        global_instructions={name: instruction},
        default_base_seed=BASE_SEED,
    )
    results = program.run(num_shots=2, verbose=False)
    return {i: results.get_shot_history(i) for i in range(2)}


def _write_as_fresh_process(store, histories) -> None:
    """A canonical file whose shot 1 references sources in shot 0, with
    cache IDs restarting at 0 as in a new process."""
    HDF5Encoder.ENCODE_ID = 0
    store.write_snapshot(ProgramResults(num_shots=6))
    store.put_batch(_shot_entries(histories, [0, 1]), encode_cache={})


def _digest(obj) -> int:
    return Serializable._serial_hash(obj)


def test_get_sees_replaced_canonical_file(tmp_path, monkeypatch):
    """A replacement reusing the old file's cache IDs for other objects
    must not resolve through the old file's decode cache."""
    monkeypatch.setattr(HDF5Encoder, "ENCODE_ID", 0)
    versions = [_named_histories(name) for name in ("A", "B", "C")]
    store = CheckpointStore(tmp_path, SHOT_LAYOUT)
    _write_as_fresh_process(store, versions[0])
    assert _digest(store.get(1)) == _digest(versions[0][1])
    # Replaced by the store itself, then by another store object.
    _write_as_fresh_process(store, versions[1])
    assert _digest(store.get(1)) == _digest(versions[1][1])
    other = CheckpointStore(tmp_path, SHOT_LAYOUT)
    _write_as_fresh_process(other, versions[2])
    assert _digest(store.get(1)) == _digest(versions[2][1])


def test_replacement_just_after_open_is_seen(tmp_path, monkeypatch):
    """A replacement landing between a read's open and its identity check
    must not keep the old file's cache under the new file's identity."""
    monkeypatch.setattr(HDF5Encoder, "ENCODE_ID", 0)
    old, new = _named_histories("A"), _named_histories("B")
    store = CheckpointStore(tmp_path, SHOT_LAYOUT)
    _write_as_fresh_process(store, old)
    staged = CheckpointStore(tmp_path / "staged", SHOT_LAYOUT)
    _write_as_fresh_process(staged, new)
    real_open = checkpointstore.open_h5
    opens = []

    def open_then_replace(path, mode):
        f = real_open(path, mode)
        opens.append(path)
        if len(opens) == 2:
            os.replace(staged.canonical_path, path)
        return f

    monkeypatch.setattr(checkpointstore, "open_h5", open_then_replace)
    # `get` opens twice: once for the key index, then to decode, which
    # reads the old file while the path already names the new one.
    assert _digest(store.get(1)) == _digest(old[1])
    assert _digest(store.get(1)) == _digest(new[1])


def test_interrupted_read_leaves_no_placeholders(
    tmp_path, histories, monkeypatch
):
    """Shot 1 shares sources with shot 0. An interrupt at any point of
    decoding shot 0 must not leave shot 1 resolving them to the
    placeholders the cut-short decode left in the store's cache."""
    store = CheckpointStore(tmp_path, SHOT_LAYOUT)
    store.write_snapshot(ProgramResults(num_shots=6))
    store.put_batch(_shot_entries(histories, [0, 1]), encode_cache={})
    expected = Serializable._serial_hash(store.get(1))
    real_decode = Serializable.decode
    state = {"calls": 0, "stop": None}

    def decode(*args, **kwargs):
        state["calls"] += 1
        if state["calls"] == state["stop"]:
            raise KeyboardInterrupt
        return real_decode(*args, **kwargs)

    monkeypatch.setattr(Serializable, "decode", staticmethod(decode))
    CheckpointStore(tmp_path, SHOT_LAYOUT).get(0)
    failures = []
    for stop in range(1, state["calls"] + 1):
        reader = CheckpointStore(tmp_path, SHOT_LAYOUT)
        state.update(calls=0, stop=stop)
        with pytest.raises(KeyboardInterrupt):
            reader.get(0)
        state["stop"] = None
        if Serializable._serial_hash(reader.get(1)) != expected:
            failures.append(stop)
    assert failures == []


def test_open_retries_only_lock_contention(tmp_path, histories, monkeypatch):
    store = CheckpointStore(tmp_path, SHOT_LAYOUT)
    store.put_batch(_shot_entries(histories, [0, 1]), worker_id="w")
    real_file = h5py.File
    monkeypatch.setattr(checkpointstore.time, "sleep", lambda _s: None)
    calls = []

    def flaky(failures):
        def opener(*args, **kwargs):
            calls.append(args)
            if len(calls) <= failures:
                raise BlockingIOError("unable to lock file")
            return real_file(*args, **kwargs)

        return opener

    monkeypatch.setattr(h5py, "File", flaky(2))
    assert store.keys() == [0, 1]
    assert len(calls) == 3

    calls.clear()
    monkeypatch.setattr(h5py, "File", flaky(10**6))
    with pytest.raises(CheckpointReadError) as info:
        store.keys(strict=True)
    assert isinstance(info.value.__cause__, BlockingIOError)
    assert len(calls) == 5
    assert store.keys(strict=False) == []

    # A different open error is not retried.
    calls.clear()

    def broken(*args, **kwargs):
        calls.append(args)
        raise OSError("not an HDF5 file")

    monkeypatch.setattr(h5py, "File", broken)
    with pytest.raises(CheckpointReadError):
        store.keys()
    assert len(calls) == 1


# -- the nested view -------------------------------------------------------


def test_nested_view_reads_kept_results(tmp_path, histories):
    store = CheckpointStore(tmp_path, ITEM_LAYOUT)
    kept = {
        0: _shot_results(histories, [0, 1]),
        1: _shot_results(histories, [2, 3, 4]),
    }
    for worker, key in (("a", 0), ("b", 1)):
        store.put_batch(
            {
                key: {
                    "results": histories[key],
                    "_program_results": kept[key],
                }
            },
            worker_id=worker,
        )
    view = store.nested(1)
    assert view.keys() == [2, 3, 4]
    assert view.keys("shot_wall_clock_times") == [2, 3, 4]
    assert _seed(view.get(3)) == BASE_SEED + 3
    assert view.get(3, "shot_wall_clock_times") == 3.5
    assert [_seed(h) for _, h in view.iter_items()] == [
        BASE_SEED + k for k in (2, 3, 4)
    ]
    assert view.missing(range(5)) == [0, 1]
    assert store.nested(0).keys() == [0, 1]
    with pytest.raises(CheckpointReadError):
        store.nested(7).keys()
    for call in (
        lambda: view.put_batch({}),
        lambda: view.consolidate(),
        lambda: view.write_snapshot(ProgramResults()),
        lambda: view.nested(0),
    ):
        with pytest.raises(TypeError):
            call()


def test_nested_view_refuses_a_cache_stub(tmp_path, histories):
    """The same kept object put twice is stored once, then as a stub,
    which must raise rather than read as empty."""
    store = CheckpointStore(tmp_path, ITEM_LAYOUT)
    kept = _shot_results(histories, [0, 1])
    receipt = store.put_batch(
        {
            k: {"results": histories[k], "_program_results": kept}
            for k in (0, 1)
        },
        worker_id="a",
    )
    assert store.nested(0).keys() == [0, 1]
    with pytest.raises(CheckpointReadError) as info:
        store.nested(1).keys()
    assert info.value.path == receipt.path


# -- canonical append and consolidation ------------------------------------


def _physical_lengths(path: Path, layout=SHOT_LAYOUT) -> dict[str, int]:
    with h5py.File(path, "r") as f:
        container = _resolve_checkpoint_object_group(f)
        out = {}
        for attr in layout.attrs:
            name = attr.canonical
            if name not in container:
                out[name] = 0
                continue
            sides = container[name]["dict"]
            lengths = {
                side_length(sides[s]["iterable"]) for s in ("keys", "values")
            }
            out[name] = lengths.pop() if len(lengths) == 1 else -1
        return out


def _state_problems(store, histories, count) -> list[str]:
    """Why `store` does not hold exactly shots `0..count-1`, each once."""
    problems = []
    expected = list(range(count))
    try:
        if store.keys() != expected:
            problems.append(f"keys {store.keys()}")
        for key, history in store.iter_items():
            if _seed(history) != BASE_SEED + key:
                problems.append(f"wrong value at {key}")
        if _physical_lengths(store.canonical_path) != {
            n: count for n in STREAMED
        }:
            problems.append(
                f"lengths {_physical_lengths(store.canonical_path)}"
            )
        read = cast(ProgramResults, ProgramResults.read(store.canonical_path))
        if sorted(read.shot_histories) != expected:
            problems.append(f"read keys {sorted(read.shot_histories)}")
        for key, history in read.shot_histories.items():
            if _seed(history) != BASE_SEED + key:
                problems.append(f"read wrong value at {key}")
    except Exception as exc:  # noqa: BLE001
        problems.append(repr(exc))
    return problems


def test_canonical_append_basics(tmp_path, histories):
    store = CheckpointStore(tmp_path, SHOT_LAYOUT)
    with pytest.raises(FileNotFoundError, match="results.h5"):
        store.put_batch(_shot_entries(histories, [0]))
    store.write_snapshot(_shot_results(histories, [0, 1]))
    cache: dict = {}
    receipt = store.put_batch(
        _shot_entries(histories, [3, 1, 2]), encode_cache=cache
    )
    # The receipt lists every requested key, including a committed one.
    assert receipt.keys == (1, 2, 3)
    assert _state_problems(store, histories, 4) == []
    assert store.get(3, "shot_wall_clock_times") == 3.5
    assert cache
    assert _read_marker(store.canonical_path) == {n: 4 for n in STREAMED}


@pytest.mark.parametrize("exception", [KeyboardInterrupt, OSError])
def test_canonical_append_interrupt_sweep(tmp_path, histories, exception):
    """After an interrupt or error at any write of an append, appending the
    same shots again leaves each shot once, with its own value."""
    template = tmp_path / "template"
    CheckpointStore(template, SHOT_LAYOUT).write_snapshot(
        _shot_results(histories, [0, 1])
    )
    entries = _shot_entries(histories, [2, 3])
    probe_dir = tmp_path / "probe"
    shutil.copytree(template, probe_dir)
    with record_h5_writes() as tap:
        CheckpointStore(probe_dir, SHOT_LAYOUT).put_batch(entries)
    session = sessions_in(tap.sessions, "results")[0]
    assert len(session.events) > 5

    failures = []
    for event in range(len(session.events)):
        directory = tmp_path / f"e{event}"
        shutil.copytree(template, directory)
        store = CheckpointStore(directory, SHOT_LAYOUT)
        cache: dict = {}
        injection = injection_for(
            tap.sessions, "results", 1, event_index=event, exception=exception
        )
        with record_h5_writes(injection) as run:
            with pytest.raises(exception):
                store.put_batch(entries, encode_cache=cache)
        assert run.fired
        if exception is KeyboardInterrupt:
            simulate_process_death()
        store.put_batch(entries, encode_cache=cache)
        problems = _state_problems(store, histories, 4)
        if problems:
            failures.append((event, session.events[event], problems))
    assert not failures, failures


def _consolidation_template(tmp_path, histories) -> Path:
    template = tmp_path / "template"
    store = CheckpointStore(template, SHOT_LAYOUT)
    store.write_snapshot(_shot_results(histories, [0, 1]))
    store.put_batch(_shot_entries(histories, [2, 3]), worker_id="a")
    store.put_batch(_shot_entries(histories, [4, 5]), worker_id="b")
    return template


def test_consolidate_interrupt_sweep(tmp_path, histories):
    """After an interrupt at any write of `consolidate`, a second
    `consolidate` leaves every shot once and no payload behind."""
    template = _consolidation_template(tmp_path, histories)
    probe_dir = tmp_path / "probe"
    shutil.copytree(template, probe_dir)
    with record_h5_writes() as tap:
        merged = CheckpointStore(probe_dir, SHOT_LAYOUT).consolidate()
    assert merged == (2, 3, 4, 5)
    session = sessions_in(tap.sessions, "results")[0]
    assert len(session.events) > 5

    failures = []
    for event in range(len(session.events)):
        directory = tmp_path / f"e{event}"
        shutil.copytree(template, directory)
        store = CheckpointStore(directory, SHOT_LAYOUT)
        injection = injection_for(
            tap.sessions, "results", 1, event_index=event
        )
        with record_h5_writes(injection) as run:
            with pytest.raises(KeyboardInterrupt):
                store.consolidate()
        assert run.fired
        simulate_process_death()
        store.consolidate()
        problems = _state_problems(store, histories, 6)
        if store.payload_paths():
            problems.append("payloads remain")
        if problems:
            failures.append((event, session.events[event], problems))
    assert not failures, failures


def test_consolidate_returns_only_new_keys(tmp_path, histories):
    template = _consolidation_template(tmp_path, histories)
    store = CheckpointStore(template, SHOT_LAYOUT)
    store.put_batch(_shot_entries(histories, [1, 2]), worker_id="c")
    assert store.consolidate() == (2, 3, 4, 5)
    assert _state_problems(store, histories, 6) == []
    assert store.consolidate() == ()


def test_consolidate_requires_canonical_file(tmp_path, histories):
    store = CheckpointStore(tmp_path, SHOT_LAYOUT)
    store.put_batch(_shot_entries(histories, [0]), worker_id="a")
    with pytest.raises(FileNotFoundError, match="results.h5"):
        store.consolidate()
    assert len(store.payload_paths()) == 1


_WIDTH_LAYOUT = StoreLayout(
    "t.h5",
    "t_ledger",
    "t",
    (StreamedAttr("hist", False), StreamedAttr("label", True)),
)


def test_error_partway_through_a_batch_then_good_batch(tmp_path, histories):
    """A batch that fails on a too-wide string leaves nothing visible, and
    a later batch sharing the encode cache still decodes."""
    store = CheckpointStore(tmp_path, _WIDTH_LAYOUT)
    store.write_snapshot(ProgramResults())
    cache: dict = {}
    store.put_batch(
        {0: {"hist": histories[0], "label": "ab"}}, encode_cache=cache
    )
    before = {k: list(v) for k, v in cache.items()}
    bad = {
        1: {"hist": histories[1], "label": "cd"},
        2: {"hist": histories[2], "label": "far too wide"},
    }
    with pytest.raises(ValueError):
        store.put_batch(bad, encode_cache=cache)
    assert cache == before
    assert store.keys() == [0]
    good = {
        3: {"hist": histories[1], "label": "ef"},
        4: {"hist": histories[4], "label": "gh"},
    }
    store.put_batch(good, encode_cache=cache)
    assert store.keys() == store.keys("label") == [0, 3, 4]
    for key, source in ((0, 0), (3, 1), (4, 4)):
        assert _seed(store.get(key)) == BASE_SEED + source
    assert store.get(3, "label") == "ef"
    assert [v for _, v in store.iter_items("label")] == ["ab", "ef", "gh"]


def test_consolidate_dedups_per_attr(tmp_path, histories):
    store = CheckpointStore(tmp_path, ITEM_LAYOUT)
    store.write_snapshot(ProgramResults())
    # The runner puts a reduced result in before its other attrs arrive.
    store.put_batch({0: {"results": histories[0]}})
    store.put_batch(
        {
            0: {
                "results": histories[1],
                "item_wall_clock_times": 1.5,
                "shot_wall_clock_times": {0: 0.5},
                "_program_results": _shot_results(histories, [0, 1]),
            }
        },
        worker_id="w",
    )
    assert store.consolidate() == (0,)
    for attr in ITEM_LAYOUT.attrs:
        assert store.keys(attr.name) == [0], attr.name
    assert _physical_lengths(store.canonical_path, ITEM_LAYOUT) == {
        a.canonical: 1 for a in ITEM_LAYOUT.attrs
    }
    assert _seed(store.get(0)) == BASE_SEED
    assert store.get(0, "item_wall_clock_times") == 1.5
    assert store.nested(0).keys() == [0, 1]


def test_consolidate_removes_only_store_leftovers(tmp_path, histories):
    store = CheckpointStore(tmp_path, SHOT_LAYOUT)
    store.write_snapshot(_shot_results(histories, [0]))
    store.put_batch(_shot_entries(histories, [1]), worker_id="w")
    ledger = store.ledger_path("x")
    file, _ = open_swmr_writer(ledger, ["done"], capacity=2)
    file.close()
    orphans = [
        "worker_x_runner.h5.tmp",
        "worker_x_shots_0_payload.h5.tmp",
        "results.h5.tmp",
    ]
    for name in orphans:
        (tmp_path / name).write_bytes(b"partial")
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("keep")
    # Ledger-named, but not a ledger: left alone.
    imposter = store.ledger_path("y")
    with h5py.File(imposter, "w") as f:
        f.attrs["x"] = 1

    assert store.consolidate() == (1,)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        ["results.h5", "notes.txt", imposter.name]
    )
    assert store.keys() == [0, 1]


def test_truncated_payload_stops_consolidation(tmp_path, histories):
    store = CheckpointStore(tmp_path, SHOT_LAYOUT)
    store.write_snapshot(_shot_results(histories, [0]))
    store.put_batch(_shot_entries(histories, [1, 2]), worker_id="a")
    bad = store.put_batch(_shot_entries(histories, [3, 4]), worker_id="b")
    path = Path(bad.path)
    path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])
    payloads = store.payload_paths()
    with pytest.raises(CheckpointReadError) as info:
        store.consolidate()
    assert info.value.path == str(path)
    assert store.payload_paths() == payloads
    # The good payload merged before the failure is committed once.
    assert store.keys(strict=False) == [0, 1, 2]


def test_append_keeps_decode_cache_and_pickles(tmp_path, histories):
    store = CheckpointStore(tmp_path, SHOT_LAYOUT)
    store.write_snapshot(_shot_results(histories, [0]))
    assert _seed(store.get(0)) == BASE_SEED
    cache = store._canonical_decode_cache
    store.put_batch(_shot_entries(histories, [1]))
    assert _seed(store.get(1)) == BASE_SEED + 1
    assert store._canonical_decode_cache is cache
    clone = pickle.loads(pickle.dumps(store))
    clone.put_batch(_shot_entries(histories, [2]))
    assert store.keys() == [0, 1, 2]


# -- round trips through the real formats ----------------------------------


def test_shot_level_round_trip(tmp_path, histories):
    store = CheckpointStore(tmp_path, SHOT_LAYOUT)
    store.write_snapshot(ProgramResults(num_shots=6))
    store.put_batch(_shot_entries(histories, [0, 1]))
    store.put_batch(_shot_entries(histories, [2, 3]), worker_id="host_1")
    store.put_batch(_shot_entries(histories, [4, 5]), worker_id="host_2")
    assert store.consolidate() == (2, 3, 4, 5)
    assert store.payload_paths() == []

    read = ProgramResults.read(store.canonical_path)
    assert sorted(read.shot_histories) == list(range(6))
    assert sorted(read.shot_wall_clock_times) == list(range(6))
    for key in range(6):
        assert _seed(read.shot_histories[key]) == BASE_SEED + key
        assert read.shot_wall_clock_times[key] == 0.5 + key
    assert store.missing(range(6)) == []


def test_item_level_round_trip(tmp_path, histories):
    runner = build_sweep_runner(tmp_path / "items", tmp_path / "shots")
    store = CheckpointStore(tmp_path, ITEM_LAYOUT)
    store.write_snapshot(runner)
    kept = {
        0: _shot_results(histories, [0, 1]),
        1: _shot_results(histories, [2, 3]),
    }
    for key in (0, 1):
        store.put_batch(
            {
                key: {
                    "results": histories[key],
                    "item_wall_clock_times": 1.5 + key,
                    "shot_wall_clock_times": {key: 0.25},
                    "_program_results": kept[key],
                }
            },
            worker_id=f"w_{key}",
        )
    assert store.consolidate() == (0, 1)

    read = type(runner).read(store.canonical_path)
    assert sorted(read._reduced_results) == [0, 1]
    assert sorted(read._program_results) == [0, 1]
    for key in (0, 1):
        assert _seed(read._reduced_results[key]) == BASE_SEED + key
        assert read.item_wall_clock_times[key] == 1.5 + key
        assert read.shot_wall_clock_times[key] == {key: 0.25}
        assert sorted(read._program_results[key].shot_histories) == [
            2 * key,
            2 * key + 1,
        ]
    view = store.nested(1)
    assert view.keys() == [2, 3]
    for key, history in view.iter_items():
        assert _seed(history) == BASE_SEED + key

"""
Performance tests comparing JSON vs HDF5 serialization.

Tests file size and compute time for various scenarios:
- Different numbers of Frames
- Different array sizes
- Object caching efficiency
"""

import copy
import tempfile
import os
import time
import numpy as np
import pytest
import h5py

from loqs.core import Frame, History
from loqs.core.instructions.instruction import Instruction


class PerformanceTestConfig:
    """Configuration for performance tests."""

    def __init__(self, name, num_frames, array_size, use_caching=True, repeat_every=3):
        self.name = name
        self.num_frames = num_frames
        self.array_size = array_size
        self.use_caching = use_caching
        self.repeat_every = repeat_every

    def __str__(self):
        return (f"{self.name}: {self.num_frames} frames, array_size={self.array_size}")

@pytest.mark.parametrize("config", [
    # Small dataset
    PerformanceTestConfig("small", num_frames=5, array_size=50, use_caching=True),
    # Medium dataset
    PerformanceTestConfig("medium", num_frames=10, array_size=100, use_caching=True),
    # Large dataset (but keep runtime reasonable)
    PerformanceTestConfig("large", num_frames=20, array_size=200, use_caching=True),
    # Extra-large dataset
    PerformanceTestConfig("x-large", num_frames=40, array_size=300, use_caching=True),
])
def test_serialization_performance(config):
    """Test serialization performance for different dataset sizes."""

    print(f"\n=== Testing {config} ===")

    # Create test History with Frames containing arrays
    history = create_test_history(config.num_frames, config.array_size)

    # Test JSON serialization with caching
    json_time, json_size = _test_json_serialization(history, config)

    # Test HDF5 serialization with caching
    hdf5_time, hdf5_size = _test_hdf5_serialization(history, config)

    json_factor, json_unit = _get_print_factor_unit(json_size)
    hdf5_factor, hdf5_unit = _get_print_factor_unit(hdf5_size)

    # Report results
    print(f"JSON:  {json_time:.3f}s, {json_size/json_factor:.1f} {json_unit}")
    print(f"HDF5: {hdf5_time:.3f}s, {hdf5_size/hdf5_factor:.1f} {hdf5_unit}")
    print(f"HDF5 benefits: {json_size/hdf5_size:.1f}x size reduction, {json_time/hdf5_time:.1f}x faster")
    
    # Verify deserialization works correctly
    if config.name == "small":
        verify_deserialization(history)

@pytest.mark.parametrize("config", [
    PerformanceTestConfig(r"4% repeated", 50, 200, repeat_every=25),
    PerformanceTestConfig(r"10% repeated", 50, 200, repeat_every=5),
    PerformanceTestConfig(r"50% repeated", 50, 200, repeat_every=2),
    PerformanceTestConfig(r"100% repeated", 50, 200, repeat_every=1)
])
def test_caching_performance(config):
    """Test object caching efficiency by comparing with and without caching."""

    print(f"\n=== Testing repeat density: {config.name} ===")

    cache_config = copy.deepcopy(config)
    cache_config.use_caching = True

    no_cache_config = copy.deepcopy(config)
    no_cache_config.use_caching = False

    # Create history with repeated objects (good for testing caching)
    history = create_history_with_repeated_objects(config)

    # Test JSON serialization with caching
    json_time, json_size = _test_json_serialization(history, cache_config)

    # Test JSON serialization without caching
    json_no_cache_time, json_no_cache_size = _test_json_serialization(history, no_cache_config)

    # Test HDF5 serialization with caching
    hdf5_time, hdf5_size = _test_hdf5_serialization(history, cache_config)

    # Test HDF5 serialization without caching
    hdf5_no_cache_time, hdf5_no_cache_size = _test_hdf5_serialization(history, no_cache_config)


    json_factor, json_unit = _get_print_factor_unit(json_size)
    json_no_cache_factor, json_no_cache_unit = _get_print_factor_unit(json_no_cache_size)
    hdf5_factor, hdf5_unit = _get_print_factor_unit(hdf5_size)
    hdf5_no_cache_factor, hdf5_no_cache_unit = _get_print_factor_unit(hdf5_no_cache_size)

    print(f"JSON (with caching):  {json_time:.3f}s, {json_size/json_factor:.1f} {json_unit}")
    print(f"JSON (no caching):   {json_no_cache_time:.3f}s, {json_no_cache_size/json_no_cache_factor:.1f} {json_no_cache_unit}")
    print(f"HDF5 (with caching): {hdf5_time:.3f}s, {hdf5_size/hdf5_factor:.1f} {hdf5_unit}")
    print(f"HDF5 (no caching):  {hdf5_no_cache_time:.3f}s, {hdf5_no_cache_size/hdf5_no_cache_factor:.1f} {hdf5_no_cache_unit}")

    # Calculate caching benefits
    json_caching_benefit = json_no_cache_size / json_size if json_size > 0 else 1.0
    hdf5_caching_benefit = hdf5_no_cache_size / hdf5_size if hdf5_size > 0 else 1.0
    json_speedup = json_no_cache_time / json_time if json_time > 0 else 1.0
    hdf5_speedup = hdf5_no_cache_time / hdf5_time if hdf5_time > 0 else 1.0

    print(f"JSON caching stats: {json_caching_benefit:.1f}x size reduction, {json_speedup:.1f}x faster")
    print(f"HDF5 caching stats: {hdf5_caching_benefit:.1f}x size reduction, {hdf5_speedup:.1f}x faster")

    # Verify deserialization works correctly
    if config.name == r"4% repeated":
        verify_deserialization(history)


def create_test_history(num_frames, array_size):
    """Create a test History with specified number of Frames and array size."""

    frames = []
    for i in range(num_frames):
        # Create frame with test data
        data = {
            "array": np.random.random((array_size, array_size)),
            "metadata": {
                "frame_id": i,
                "timestamp": time.time(),
                "description": f"Test frame {i}"
            }
        }
        frames.append(Frame(data))

    return History(frames)


def create_history_with_repeated_objects(config):
    """Create history with repeated objects to test caching efficiency."""

    def apply_fn(x):
        return Frame()

    inst = Instruction(apply_fn, data={"array": np.random.random((config.array_size, config.array_size))}, name="Test")

    # Create some reusable objects
    shared_frame = Frame({"shared": "data", "inst": inst})

    frames = []
    for i in range(config.num_frames):
        # Mix of unique and repeated frames
        if i % config.repeat_every == 0:
            frames.append(shared_frame)  # Repeat this frame
        else:
            frames.append(Frame({
                "unique_id": i,
                "unique_inst": inst.copy()
            }))

    return History(frames)


def _test_json_serialization(history, config):
    """Test JSON serialization performance (helper function)."""

    with make_temp_path(suffix=".json") as temp_file:
        # Time the serialization
        start_time = time.time()

        history.write(temp_file, format="json", use_caching=config.use_caching)

        json_time = time.time() - start_time

        # Get file size
        json_size = os.path.getsize(temp_file)

        return json_time, json_size

    finally:
        #print(f"Leaving file {temp_file}")
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def _test_hdf5_serialization(history, config):
    """Test HDF5 serialization performance (helper function)."""

    with make_temp_path(suffix=".h5") as temp_file:
        # Time the serialization
        start_time = time.time()
        
        history.write(temp_file, format="hdf5", use_caching=config.use_caching)
        
        hdf5_time = time.time() - start_time

        # Get file size
        hdf5_size = os.path.getsize(temp_file)

        return hdf5_time, hdf5_size

    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)

def _get_print_factor_unit(size):
    factor = 1024
    unit = "KB"
    if size/factor > 1024:
        factor *= 1024
        unit = "MB"
    if size/factor > 1024:
        factor *= 1024
        unit = "GB"
    return factor, unit

def verify_deserialization(history):
    """Verify that deserialization works correctly for JSON format."""

    # Test JSON deserialization
    with make_temp_path(suffix=".json") as temp_file:
        history.write(temp_file, format="json")
        loaded_json = History.read(temp_file)
        assert isinstance(loaded_json, History)

        # Basic sanity checks
        assert len(loaded_json) == len(history)
        # Frame uses _data attribute
        assert str(loaded_json[0]._data.keys()) == str(history[0]._data.keys())
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)

    with make_temp_path(suffix=".h5") as temp_file:
        history.write(temp_file, format="hdf5")
        loaded_hdf5 = History.read(temp_file)
        assert isinstance(loaded_hdf5, History)

        # Basic sanity checks
        assert len(loaded_hdf5) == len(history)
        # Frame uses _data attribute
        assert str(loaded_hdf5[0]._data.keys()) == str(history[0]._data.keys())
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)




if __name__ == "__main__":
    # Run tests manually for debugging
    test_configs = [
        PerformanceTestConfig("small", num_frames=50, array_size=50),
        PerformanceTestConfig("medium", num_frames=100, array_size=100),
        PerformanceTestConfig("large", num_frames=200, array_size=200),
        PerformanceTestConfig("x-large", num_frames=400, array_size=300),
    ]

    print(" == PERFORMANCE TESTS ==")

    for config in test_configs:
        test_serialization_performance(config)

    print("\n == CACHING TESTS ==")

    cache_test_configs = [
        PerformanceTestConfig(r"4% repeated", 50, 200, repeat_every=25),
        PerformanceTestConfig(r"10% repeated", 50, 200, repeat_every=5),
        PerformanceTestConfig(r"50% repeated", 50, 200, repeat_every=2),
        PerformanceTestConfig(r"100% repeated", 50, 200, repeat_every=1)
    ]
    for config in cache_test_configs:
        test_caching_performance(config)
"""
Test performance using Histories with Frames that store Instructions.
"""

import tempfile
import os
import time
import numpy as np
import h5py
from pathlib import Path

# Add project root to Python path
import sys
sys.path.append('/projects')

from loqs.core import Frame, History, Instruction
from loqs.internal.serializable import Serializable


def get_file_size(file_path):
    """Get file size in bytes."""
    return Path(file_path).stat().st_size


def create_test_history(num_frames=10, array_size=50):
    """Create a test History with Frames containing MockSerializable objects."""
    from tests.internal.test_serializable import MockSerializable
    
    # Create a simple mock object
    mock_obj = MockSerializable(name="test_object", value=42, data={"key": "value"})
    
        # Create frames with mock objects
    frames = []
    for i in range(num_frames):
        # Create some test data
        if i % 2 == 0:
            # Use the same object
            mock_instance = mock_obj
        else:
            # Create a new object with identical content
            mock_instance = MockSerializable(name="test_object", value=42, data={"key": "value"})
        
        data = {
            'iteration': i,
            'array_data': np.random.random((array_size, array_size)),
            'mock_object': mock_instance,
            'metadata': {'frame_id': i, 'timestamp': time.time()}
        }
        
        frame = Frame(data=data)
        frames.append(frame)
    
    # Create history
    history = History(frames=frames)
    return history


def test_history_serialization():
    """Test serialization performance with Histories containing Frames."""
    print("=== History Serialization Performance Test ===")
    
    # Test different configurations
    configs = [
        ("Small", 5, 20),
        ("Medium", 10, 50),
        ("Large", 20, 100),
    ]
    
    for name, num_frames, array_size in configs:
        print(f"\nTesting {name} history ({num_frames} frames, {array_size}x{array_size} arrays):")
        
        # Create test history
        history = create_test_history(num_frames, array_size)
        
        # Test JSON serialization
        start_time = time.time()
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_file_json = f.name
        
        try:
            history.write(temp_file_json, format="json")
            json_time = time.time() - start_time
            json_size = get_file_size(temp_file_json)
            
            print(f"  JSON: {json_time:.3f}s, {json_size:,} bytes")
            
        finally:
            if os.path.exists(temp_file_json):
                os.unlink(temp_file_json)
        
        # Test HDF5 serialization with caching
        start_time = time.time()
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_file_hdf5 = f.name
        
        try:
            history.write(temp_file_hdf5, format="hdf5")
            hdf5_time = time.time() - start_time
            hdf5_size = get_file_size(temp_file_hdf5)
            
            print(f"  HDF5: {hdf5_time:.3f}s, {hdf5_size:,} bytes")
            print(f"  HDF5 benefits: {json_size/hdf5_size:.1f}x smaller, {json_time/hdf5_time:.1f}x faster")
            
        finally:
            if os.path.exists(temp_file_hdf5):
                os.unlink(temp_file_hdf5)


def test_mock_object_caching():
    """Test the new caching mechanism with MockSerializable objects."""
    print("\n=== Mock Object Caching Test ===")
    
    from tests.internal.test_serializable import MockSerializable
    
    # Create mock objects with identical content
    mock_obj1 = MockSerializable(name="identical_object", value=42, data={"key": "value"})
    mock_obj2 = MockSerializable(name="identical_object", value=42, data={"key": "value"})
    
    # Test serialization with caching
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        temp_file = f.name
    
    try:
        with h5py.File(temp_file, 'w') as h5_file:
            root_group = h5_file.create_group('root')
            cache = {}
            
            # First object should be source
            result1 = Serializable.encode(mock_obj1, format="hdf5", h5_group=root_group, encode_cache=cache, reset_encode_id=True)
            print(f"Object 1 cache type: {result1.attrs.get('cache_type', 'none')}")
            
            # Second object (identical content) should be copy
            result2 = Serializable.encode(mock_obj2, format="hdf5", h5_group=root_group, encode_cache=cache)
            print(f"Object 2 cache type: {result2.attrs.get('cache_type', 'none')}")
            
            # Same object again should be reference
            result3 = Serializable.encode(mock_obj1, format="hdf5", h5_group=root_group, encode_cache=cache)
            print(f"Object 1 (again) cache type: {result3.attrs.get('cache_type', 'none')}")
        
        file_size = get_file_size(temp_file)
        print(f"Total file size: {file_size:,} bytes")
        
        # Test deserialization
        with h5py.File(temp_file, 'r') as h5_file:
            root_read = h5_file['root']
            decoded_objects = []
            
            # Decode all objects
            for key in root_read.keys():
                object_group = root_read[key]
                decoded_object = Serializable.decode(object_group, format="hdf5")
                decoded_objects.append(decoded_object)
            
            # Verify we got the right number of objects
            assert len(decoded_objects) == 3
            
            # First and third should be the same object (reference)
            assert decoded_objects[0] is decoded_objects[2]
            
            # Second should be equal but different object (copy)
            assert decoded_objects[0] == decoded_objects[1]
            assert decoded_objects[0] is not decoded_objects[1]
            
            print("✓ Mock object caching and deserialization working correctly")
        
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def main():
    """Run all History performance tests."""
    print("Running History serialization performance tests...")
    
    test_history_serialization()
    test_mock_object_caching()
    
    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    main()
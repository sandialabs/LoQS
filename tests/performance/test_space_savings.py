"""
Simplified profiling script to demonstrate space savings from the new caching scheme.
"""

import tempfile
import os
import numpy as np
import h5py
from pathlib import Path

# Add project root to Python path
import sys
sys.path.append('/projects')

from tests.internal.test_serializable import MockSerializable
from loqs.internal.serializable import Serializable


def get_file_size(file_path):
    """Get file size in bytes."""
    return Path(file_path).stat().st_size


def main():
    """Run simplified profiling tests."""
    print("=== Serialization Space Savings Profiling ===")
    
    # Test 1: HDF5 dataset optimization vs groups storage
    print("\n1. HDF5 Iterable Optimization:")
    
    # Homogeneous data (should use dataset optimization)
    homogeneous_data = list(range(1000))
    
    with make_temp_path(suffix='.h5') as temp_file1:
        with h5py.File(temp_file1, 'w') as h5_file:
            root_group = h5_file.create_group('root')
            Serializable.encode(homogeneous_data, format="hdf5", h5_group=root_group)
        
        size_dataset = get_file_size(temp_file1)
        print(f"  Homogeneous list (dataset): {size_dataset:,} bytes")
        
    finally:
        if os.path.exists(temp_file1):
            os.unlink(temp_file1)
    
    # Heterogeneous data (should use groups storage)
    heterogeneous_data = [1, "string", 3.14] * 333
    
    with make_temp_path(suffix='.h5') as temp_file2:
        with h5py.File(temp_file2, 'w') as h5_file:
            root_group = h5_file.create_group('root')
            Serializable.encode(heterogeneous_data, format="hdf5", h5_group=root_group)
        
        size_groups = get_file_size(temp_file2)
        print(f"  Heterogeneous list (groups): {size_groups:,} bytes")
        print(f"  Space savings: {size_groups/size_dataset:.1f}x")
        
    finally:
        if os.path.exists(temp_file2):
            os.unlink(temp_file2)
    
    # Test 2: Serial ID caching
    print("\n2. Serial ID Caching:")
    
    # Create objects with identical content
    obj1 = MockSerializable(name="same", value=42, data={"key": "value"})
    obj2 = MockSerializable(name="same", value=42, data={"key": "value"})
    
    # Without caching
    with make_temp_path(suffix='.h5') as temp_file3:
        with h5py.File(temp_file3, 'w') as h5_file:
            root_group = h5_file.create_group('root')
            Serializable.encode(obj1, format="hdf5", h5_group=root_group, encode_cache=None, reset_encode_id=True)
            Serializable.encode(obj2, format="hdf5", h5_group=root_group, encode_cache=None)
        
        size_no_cache = get_file_size(temp_file3)
        print(f"  Without caching: {size_no_cache:,} bytes")
        
    finally:
        if os.path.exists(temp_file3):
            os.unlink(temp_file3)
    
    # With caching
    with make_temp_path(suffix='.h5') as temp_file4:
        with h5py.File(temp_file4, 'w') as h5_file:
            root_group = h5_file.create_group('root')
            cache = {}
            Serializable.encode(obj1, format="hdf5", h5_group=root_group, encode_cache=cache, reset_encode_id=True)
            Serializable.encode(obj2, format="hdf5", h5_group=root_group, encode_cache=cache)
        
        size_with_cache = get_file_size(temp_file4)
        print(f"  With caching: {size_with_cache:,} bytes")
        print(f"  Space savings: {(1 - size_with_cache/size_no_cache)*100:.1f}%")
        
    finally:
        if os.path.exists(temp_file4):
            os.unlink(temp_file4)
    
    # Test 3: Array compression
    print("\n3. Array Compression:")
    
    # Large array (should use compression)
    large_array = np.random.random((500, 500))
    
    with make_temp_path(suffix='.h5') as temp_file5:
        with h5py.File(temp_file5, 'w') as h5_file:
            root_group = h5_file.create_group('root')
            Serializable.encode(large_array, format="hdf5", h5_group=root_group)
        
        size_compressed = get_file_size(temp_file5)
        print(f"  Large array (compressed): {size_compressed:,} bytes")
        
    finally:
        if os.path.exists(temp_file5):
            os.unlink(temp_file5)
    
    # Small array (should not use compression)
    small_array = np.random.random((10, 10))
    
    with make_temp_path(suffix='.h5') as temp_file6:
        with h5py.File(temp_file6, 'w') as h5_file:
            root_group = h5_file.create_group('root')
            Serializable.encode(small_array, format="hdf5", h5_group=root_group)
        
        size_uncompressed = get_file_size(temp_file6)
        print(f"  Small array (uncompressed): {size_uncompressed:,} bytes")
        
    finally:
        if os.path.exists(temp_file6):
            os.unlink(temp_file6)
    
    print("\nProfiling completed successfully!")


if __name__ == "__main__":
    main()
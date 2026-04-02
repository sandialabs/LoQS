# Ralph Loop Context

## General rules

AVOID using Union and Optional in type annotations; instead, use Python 3.12+ syntax, i.e. Optional[Union[A, B]] -> A | B | None. For typing function arguments, use the Iterable instead of list/tuple/set and Mapping instead of dict. For typing return values, use explicit types rather than general ones, i.e. list instead of List. When importing general types, import them from collections.abc rather than typing if possible.

When searching the codebase (e.g. using grep, ls, find, etc.), ALWAYS explicitly exclude the files listed in .ignore

## Context added at 2026-03-29T01:21:42.147Z

You are currently improving how serialization is performed, specifically increasing the utility of caching and leveraging HDF5 dataset optimizations.
The majority of the serialization code is in loqs/internal/serializable.py and loqs/internal/encoder/*.py.
Most classes that can be serialized inherit Displayable, which itself inherits Serializable. The Displayable class should be treated as a pass-through inheritance, i.e. any changes to base serialization behavior should be implemented in Serializable and not be touched in Displayable.
ONLY change logic in "version" == 1 blocks of JSONEncoder/HDF5Encoder. Leave "version" == 0 blocks UNCHANGED.

### Caching Improvement Goals
Caching currently uses id(obj) as the key to check for cache hits/misses. However, this is too restrictive. There are cases where the objects are not exactly the same, but they have the same encoded object. In order to improve this, we will do the following:

1. Implement a Serializable.serial_id static function that recursively calls Serializable.serial_id on all of an object's SERIALIZE_ATTRS, in a similar way to how Serializable.encode and Serializable.decode work.
  1. In the base case, serial_id should return hash() of the object.
  1. If obj is a Serializable, serial_id should return on a hash of the tuple of the obj.SERIALIZE_ATTR serial IDs.
  1. If obj is a list, serial_id should return a hash of the tuple of serial ids of each element
  1. If obj is a dict, serial_id should be return a hash of the tuple of the serial ids of the list(obj.keys()) and list(obj.values()).
  1. If obj is a np.ndarray, serial_id should return a hash of the tuple of serial ids of the obj.shape and obj.flatten().tolist()
1. The encode_cache is currently id(obj) -> JSON/HDF5Encoder.ENCODE_ID. It should be modified to be Serializable.serial_id(obj) -> list of (id(obj), Encoder.ENCODE_ID) elements. An object is in cache if both its serial_id is a key and there is an entry with its id(obj) as the first entry of one of the list elements. The caching behaviour should be as follows:
  1. If serial_id(obj) is not in encode_cache, create [(id(obj), Encoder.ENCODE_ID)] as the cache value. This object will then have "cache_type" = "source" and "source_cache_id" = Encoder.ENCODE_ID (replacing "cache_id") added to its encoding data.
  1. If serial_id(obj) IS in encode_cache, but id(obj) is not the first element of any entries in the cache list, add (id(obj), Encoder.ENCODE_ID) to the list. This object will have "cache_type" = "copy", "reference_cache_id" = the ENCODE_ID of the first element in the cache for this serial id, and "source_cache_id" = the current Encoder.ENCODE_ID added to its encoding data.
  1. If serial_id(obj) is in the cache AND id(obj) is a first element of one of the entries in the cache list, do not add it to the cache and this object will have "cache_type" = "reference" and "reference_cache_id" = the ENCODE_ID (replacing "cache_id") corresponding to the elemnt of the cache list with matching id(obj) added to its encoding data.
1. The decode_cache is currently cache_id -> object. The structure of this will not change. However, the behavior when accessing it changes slightly:
  1. The decode_uncached_obj should remain the same.
  1. In the decode_cached_obj function, it should accept both encoded.get("cache_type", "") == "copy" and "reference". If "reference", the behavior is unchanged. If "copy", add another entry to the decode_cache with key encoded's "source_cache_id" and value is a copy of the element decode_cache corresponding to encoded's "reference_cache_id".
1. Any logic handling the difference between source, copy, and reference cache_types should be performed inside the Encoders rather than Serializable.
1. The goal of both changes is that serialized information is not duplicated in encoding, but we properly create the correct object instance structure when deserializing.
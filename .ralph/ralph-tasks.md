# Ralph Tasks

- [ ] Implement a better caching strategy as described in the .ralph/ralph-context.md
  - [ ] Implement a serial_id function that has a recursive structure through dicts, lists, arrays, and Serializable objects whose base case is a hash of immutably-casted objects
  - [ ] Modify the encode_cache to take serial_id(obj) keys and list of (id(obj), ENCODE_ID) values
  - [ ] Modify encode_decached_obj such that a new cache_type==source object adds the correct information into the cache
  - [ ] Modify encode_cached_obj such that it encodes either the new cache_type==copy encoding or the current cache_type==reference depending on whether id(obj) is in the cache list or not
  - [ ] Modify decode_cached_obj such that it handles the new cache_type==copy encoding
- [ ] Create tests to cover new code paths
  - [ ] Create tests for cases where objects have the same serializable information, i.e. equal serial_id(obj), but are different instances, i.e. nonequal id(obj).
  - [ ] Create tests where there are circular references
- [ ] Run all unit tests and fix any bugs
- [ ] Modify encode/decode_iterable such that it uses HDF5 datasets directly in cases where the iterable is a sequence of HDF5 native objects, e.g. ints, strs, bools, etc.
- [ ] Use the chunks kwarg and compression for large HDF5 datasets, especially in encode/decode_array
- [ ] Create test cases hitting both this new iterable encoding codepath and the original codepath (easily done with a heterogenous list)
- [ ] Run all unit tests and fix any bugs
- [ ] Create new profiling scripts that show what space savings this new caching scheme uses.
  - [ ] Test performance using Histories with Frames that store Instructions that are sometimes copies and sometimes the same object as the original Instruction
  - [ ] Simulate "old" style caching by always creating a copy of the Instruction
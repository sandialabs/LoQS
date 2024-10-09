"""TODO
"""

from __future__ import annotations

from abc import abstractmethod
from ast import literal_eval as make_tuple
from collections.abc import Mapping
import gzip
import importlib
import inspect
import json
import numpy as np
import re
import scipy.sparse as sps
import textwrap
from typing import Callable, ClassVar

from loqs.internal import SerializableViewer

class_location_changes = (
    {}
)  # (module, class) mapping from OLD to NEW locations


class Serializable:
    """
    The base class for all serializable objects.

    Derived classes should implement at least _to_serialization
    and _from_serialization (the abstract methods).
    They can just call super() versions, but they must at least be
    explicitly implemented in derived classes.

    Based heavily on pygsti.baseobjs.NicelySerializable.
    A "nicely serializable" object can be converted to and created from a
    native Python object (like a string or dict) that contains only other native
    Python objects.  In addition, there are constraints on the makeup of these
    objects so that they can be easily serialized to standard text-based formats,
    e.g. JSON.  For example, dictionary keys must be strings, and the list vs. tuple
    distinction cannot be assumed to be preserved during serialization.
    """

    # Class attributes
    CACHE_ON_SERIALIZE: ClassVar[bool] = False
    """Flag to indicate whether this class should be cached.

    Every Serializable object _can_ be cached, but caching does
    introduce some overhead. For cases where the serialized object
    is small or not frequently references, we can save time for very
    little filesize by not caching (the default behavior).

    Some large objects that are heavily referenced *should* use caching,
    however. Some examples: Instruction, InstructionStack, QECCode,
    QECCodePatch, any backend objects, etc.
    """

    ## ABSTRACT METHODS
    # Implement these in derived classes

    @abstractmethod
    def __hash__(self) -> int:
        """Hash for serializable object.

        This is required to enable caching while serializing.
        """
        pass

    @abstractmethod
    def _to_serialization(self, hash_to_serial_id_cache=None):
        state = {
            "module": self.__class__.__module__,
            "class": self.__class__.__name__,
            "version": 0,
        }
        return state

    @classmethod
    @abstractmethod
    def _from_serialization(cls, state, serial_id_to_obj_cache=None):
        c = cls._state_class(state)
        if not issubclass(c, cls):
            raise ValueError(
                "Class '%s' is trying to load a serialized '%s' object (not a subclass)!"
                % (
                    cls.__module__ + "." + cls.__name__,
                    state["module"] + "." + state["class"],
                )
            )
        implementing_cls = cls
        for candidate_cls in c.__mro__:
            if "_from_serialization" in candidate_cls.__dict__:
                implementing_cls = candidate_cls
                break

        if (
            implementing_cls == cls
        ):  # then there's no actual derived-class implementation to call!
            raise NotImplementedError(
                "Class '%s' doesn't implement _from_serialization!"
                % str(state["module"] + "." + state["class"])
            )
        else:
            return c._from_serialization(state, serial_id_to_obj_cache)

    ## PUBLIC CLASS METHODS
    # Primarily for deserialization

    @classmethod
    def from_serialization(cls, state, serial_id_to_obj_cache=None):
        """
        Create and initialize an object from a "nice" serialization.

        A "nice" serialization here means one created by a prior call to `to_serialization` using this
        class or a subclass of it.  Nice serializations adhere to additional rules (e.g. that dictionary
        keys must be strings) that make them amenable to common file formats (e.g. JSON).

        The `state` argument is typically a dictionary containing 'module' and 'state' keys specifying
        the type of object that should be created.  This type must be this class or a subclass of it.

        Parameters
        ----------
        state : object
            An object, usually a dictionary, representing the object to de-serialize.

        Returns
        -------
        object
        """
        if serial_id_to_obj_cache is None:
            serial_id_to_obj_cache = {}

        # Try to deserialize from cache first
        if state.get("type", "") == "cached_object_reference":
            if serial_id_to_obj_cache is None:
                raise RuntimeError(
                    "Object reference found but no object cache provided."
                )

            try:
                serial_id = state["cache_id"]
            except KeyError:
                raise RuntimeError(
                    "Object reference found but no id provided."
                )

            try:
                return serial_id_to_obj_cache[serial_id]
            except KeyError:
                raise RuntimeError(
                    f"Object reference found but source {serial_id} not available."
                )

        try:
            # Implementation note:
            # This method is similar to _from_serialization, but will defer to the method of a derived class
            # when once is specified in the state dictionary.  This method should thus be used when de-serializing
            # using a potential base class, i.e.  BaseClass._from_serialization_base(state).
            # (This method should rarely need to be overridden in derived (sub) classes.)
            if (
                isinstance(state, dict)
                and state["module"] == cls.__module__
                and state["class"] == cls.__name__
            ):
                # then the state is actually for this class and we should call its _from_serialization method:
                ret = cls._from_serialization(state, serial_id_to_obj_cache)
            else:
                # otherwise, this call functions as a base class call that defers to the correct derived class
                ret = Serializable._from_serialization(
                    state, serial_id_to_obj_cache
                )
        except RecursionError as e:
            raise NotImplementedError(
                "Hit recursion limit while deserializing, usually indicating "
                + "from_serialization was not implemented in a derived class."
            ) from e

        # Save new object in cache if it is a source
        if state.get("type", "") == "cached_object_source":
            try:
                serial_id = state["cache_id"]
            except KeyError:
                raise RuntimeError("Object source found but no id provided.")

            serial_id_to_obj_cache[serial_id] = ret

        return ret

    @classmethod
    def load(cls, f, format="json"):
        """
        Load an object of this type, or a subclass of this type, from an input stream.

        Parameters
        ----------
        f : file-like
            An open input stream to read from.

        format : {'json'}
            The format of the input stream data.

        Returns
        -------
        NicelySerializable
        """
        if format == "json":
            state = json.load(f)
        else:
            raise ValueError("Invalid `format` value: %s" % str(format))
        return cls.from_serialization(state)

    @classmethod
    def loads(cls, s, format="json"):
        """
        Load an object of this type, or a subclass of this type, from a string.

        Parameters
        ----------
        s : str
            The serialized object.

        format : {'json'}
            The format of the string data.

        Returns
        -------
        NicelySerializable
        """
        if format == "json":
            state = json.loads(s)
        else:
            raise ValueError("Invalid `format` value: %s" % str(format))
        return cls.from_serialization(state)

    @classmethod
    def read(cls, path, format=None):
        """
        Read an object of this type, or a subclass of this type, from a file.

        Parameters
        ----------
        path : str or Path or file-like
            The filename to open or an already open input stream.

        format : {'json', 'json.gz', None}
            The format of the file.  If `None` this is determined automatically
            by the filename extension of a given path.

        Returns
        -------
        NicelySerializable
        """
        if format is None:
            if str(path).endswith(".json"):
                format = "json"
            elif str(path).endswith(".json.gz"):
                format = "json.gz"
            else:
                raise ValueError(
                    "Cannot determine format from extension of filename: %s"
                    % str(path)
                )

        if format.endswith(".gz"):
            with gzip.open(str(path), "rt") as f:
                # Pass in format without .gz suffix
                return cls.load(f, format[:-3])

        # If no compression, open file normally
        with open(str(path), "r") as f:
            return cls.load(f, format)

    ## PUBLIC INSTANCE FUNCTIONS
    # Primarily for serializing

    def display(self):
        """Launch an interactive viewer for the object.

        This is a blocking operation until the viewer
        window is closed.
        """
        data = self.to_serialization()

        title = f"{self.__class__.__name__} "
        obj_name = getattr(self, "name", None)
        if obj_name is not None:
            title += f"({obj_name}) "
        title += "Viewer"

        app = SerializableViewer(data, title)
        app.mainloop()

    def dump(self, f, format="json", **format_kwargs):
        """
        Serializes and writes this object to a given output stream.

        Parameters
        ----------
        f : file-like
            A writable output stream.

        format : {'json', 'repr'}
            The format to write.

        format_kwargs : dict, optional
            Additional arguments specific to the format being used.
            For example, the JSON format accepts `indent` as an argument
            because `json.dump` does.

        Returns
        -------
        None
        """
        assert f is not None, "Must supply a valid `f` argument!"
        self._dump_or_dumps(f, format, **format_kwargs)

    def dumps(self, format="json", **format_kwargs):
        """
        Serializes this object and returns it as a string.

        Parameters
        ----------
        format : {'json', 'repr'}
            The format to write.

        format_kwargs : dict, optional
            Additional arguments specific to the format being used.
            For example, the JSON format accepts `indent` as an argument
            because `json.dump` does.

        Returns
        -------
        str
        """
        return self._dump_or_dumps(None, format, **format_kwargs)

    def to_serialization(self, hash_to_serial_id_cache=None):
        """
        Serialize this object in a way that adheres to "niceness" rules of common text file formats.

        Parameters
        ----------
        hash_to_serial_id_cache:
            A dictionary of already serialized id keys with their corresponding object values

        Returns
        -------
        object
            Usually a dictionary representing the serialized object, but may also be another native
            Python type, e.g. a string or list.
        """
        if hash_to_serial_id_cache is None:
            hash_to_serial_id_cache = {}

        try:
            cache_id = hash_to_serial_id_cache[hash(self)]
            return {
                "module": self.__class__.__module__,
                "class": self.__class__.__name__,
                "version": 0,
                "type": "cached_object_reference",
                "cache_id": cache_id,
            }
        except KeyError:
            # Cache miss
            pass

        state = self._to_serialization(hash_to_serial_id_cache)

        # Add this to the cache, if class marked as should be cached
        if self.CACHE_ON_SERIALIZE:
            cache_id = len(hash_to_serial_id_cache)
            hash_to_serial_id_cache[hash(self)] = cache_id
            state.update(
                {
                    "type": "cached_object_source",
                    "cache_id": cache_id,
                }
            )

        return state

    def write(self, path, format=None, **format_kwargs):
        """
        Writes this object to a file.

        Parameters
        ----------
        path : str or Path
            The name of the file that is written.

        format_kwargs : dict, optional
            Additional arguments specific to the format being used.
            For example, the JSON format accepts `indent` as an argument
            because `json.dump` does.

        Returns
        -------
        None
        """
        if format is None:
            if str(path).endswith(".json"):
                format = "json"
            elif str(path).endswith(".json.gz"):
                format = "json.gz"
            else:
                raise ValueError(
                    "Cannot determine format from extension of filename: %s"
                    % str(path)
                )

        if format.endswith(".gz"):
            with gzip.open(str(path), "wt") as f:
                # Pass in format without .gz suffix
                return self.dump(f, format[:-3], **format_kwargs)

        with open(str(path), "w") as f:
            self.dump(f, format, **format_kwargs)

    ## INTERNAL FUNCTIONS

    # With hash implemented, we get a (maybe not efficient) equality check
    def __eq__(self, other) -> bool:
        return hash(self) == hash(other)

    def _dump_or_dumps(self, f, format="json", **format_kwargs):
        """
        Serializes and writes this object to a given output stream.

        Parameters
        ----------
        f : file-like
            A writable output stream.  If `None`, then object is written
            as a string and returned.

        format : {'json', 'repr'}
            The format to write.

        format_kwargs : dict, optional
            Additional arguments specific to the format being used.
            For example, the JSON format accepts `indent` as an argument
            because `json.dump` does.

        Returns
        -------
        str or None
            If `f` is None, then the serialized object as a string is returned.  Otherwise,
            `None` is returned.
        """
        if format == "json":
            if "indent" not in format_kwargs:  # default indent=4 JSON argument
                format_kwargs = (
                    format_kwargs.copy()
                )  # don't update caller's dict!
                format_kwargs["indent"] = 4

            json_dict = self.to_serialization()
            if f is not None:
                json.dump(json_dict, f, **format_kwargs)
            else:
                return json.dumps(json_dict, **format_kwargs)
        else:
            raise ValueError("Invalid `format` argument: %s" % str(format))

    @classmethod
    def _state_class(cls, state, check_is_subclass=True):
        """Returns the class specified by the given state dictionary"""
        if (state["module"], state["class"]) in class_location_changes:
            state["module"], state["class"] = class_location_changes[
                state["module"], state["class"]
            ]
        try:
            m = importlib.import_module(state["module"])
            c = getattr(
                m, state["class"]
            )  # will raise AttributeError if class cannot be found
        except (ModuleNotFoundError, AttributeError) as e:
            raise ImportError(
                (
                    "Class or module not found when instantiating a Serializable"
                    f" {state['module']}.{state['class']} object!  If this class has"
                    " moved, consider adding (module, classname) mapping to"
                    " the loqs.internal.serializable.class_location_changes dict"
                )
            ) from e

        if check_is_subclass and not issubclass(c, cls):
            raise ValueError(
                "Expected a subclass or instance of '%s' but state dict has '%s'!"
                % (
                    cls.__module__ + "." + cls.__name__,
                    state["module"] + "." + state["class"],
                )
            )
        return c

    @classmethod
    def _check_compatible_state(cls, state):
        if state["module"] != cls.__module__ or state["class"] != cls.__name__:
            raise ValueError(
                "Serialization type mismatch: %s != %s"
                % (
                    state["module"] + "." + state["class"],
                    cls.__module__ + "." + cls.__name__,
                )
            )

    ## PUBLIC STATIC METHODS
    # Primarily helper functions kept in the Serializable namespace
    # This are all basically recursive functions that act on collections

    @staticmethod
    def deserialize(obj, serial_id_to_obj_cache=None):
        """Helper function to recursively unserialize objects."""
        if isinstance(obj, dict):
            if "module" in obj and "class" in obj:
                # This is a serialized class or object, try to deserialize
                if obj.get("as_type", False):
                    # If this is just the class, return it
                    return Serializable._deserialize_class(obj)

                if obj.get("type", None) == "matrix":
                    # This is a matrix, deserialize
                    return Serializable._deserialize_mx(obj["data"])

                # Otherwise, get the class and call its deserialization
                cls = Serializable._state_class(obj)
                return cls.from_serialization(obj, serial_id_to_obj_cache)

            # Otherwise, assume just a dict and recursively unserialize
            deserialized = {}
            for k, v in obj.items():
                if isinstance(k, list):
                    k = tuple(k)
                elif (
                    isinstance(k, str)
                    and k.startswith("(")
                    and k.endswith(")")
                ):
                    # This was a tuple
                    k = make_tuple(k)
                deserialized[k] = Serializable.deserialize(
                    v, serial_id_to_obj_cache
                )
            return deserialized
        elif isinstance(obj, (list, tuple)):
            return [
                Serializable.deserialize(e, serial_id_to_obj_cache)
                for e in obj
            ]
        elif isinstance(obj, str) and "def " in obj:
            # Assume this is a function definition
            return Serializable._deserialize_function(obj)

        # Otherwise, assume we are a built-in deserializable object
        return obj

    @staticmethod
    def hash(obj) -> int:
        """Helper function to recursively hash objects"""
        if isinstance(obj, dict):
            return hash(
                (
                    Serializable.hash(tuple(obj.keys())),
                    Serializable.hash(tuple(obj.values())),
                )
            )
        elif isinstance(obj, (tuple, list)):
            return hash(tuple(Serializable.hash(v) for v in obj))
        elif isinstance(obj, np.ndarray):
            return hash((tuple(obj.shape), tuple(obj.flatten().tolist())))
        return hash(obj)

    @staticmethod
    def serialize(obj, hash_to_serial_id_cache=None):
        """Helper function to recursively serialize objects."""
        if isinstance(obj, dict):
            serial_dict = {}
            for k, v in obj.items():
                if isinstance(k, tuple):
                    k = str(k)
                serial_dict[k] = Serializable.serialize(
                    v, hash_to_serial_id_cache
                )
            return serial_dict
        elif isinstance(obj, np.ndarray) or sps.issparse(obj):
            return {"type": "matrix", "data": Serializable._serialize_mx(obj)}
        elif isinstance(obj, (list, tuple, set)):
            return [
                Serializable.serialize(e, hash_to_serial_id_cache) for e in obj
            ]
        elif isinstance(obj, type):
            # This should be before function serialization,
            # so that we do not accidentally grab the class constructor
            # Constructor could still be initialized by explicitly serializing __init__
            return Serializable._serialize_class(obj)
        elif isinstance(obj, Callable):
            return Serializable._serialize_function(obj)
        elif isinstance(obj, Serializable):
            return obj.to_serialization(hash_to_serial_id_cache)

        # Otherwise, assume we are a built-in serializable object
        return obj

    ## PRIVATE STATIC FUNCTIONS
    # Helper functions for serializing/deserializing specific types or cache management

    @staticmethod
    def _deserialize_class(state: Mapping, **kwargs) -> type:
        if state.get("as_type", False):
            return Serializable._state_class(state, **kwargs)

        raise ValueError(
            "State does not correspond to a standalone class serialization,"
            + "i.e. does not have as_type: True in it. To get the class of "
            + "a serialized instance, use Serialized._state_class() instead."
        )

    # WARNING: This is inherently unsafe
    @staticmethod
    def _deserialize_function(src: str) -> Callable:
        # Execute the imports and function definition
        env = {}
        exec(src, env)

        # We need to find the function name
        # Search for last def, then first paren after it
        # Trim "def " and that should be the function name
        fn_defs = re.findall(r"^def .*\(", src, re.MULTILINE)
        last_fn_def = fn_defs[-1]
        key = last_fn_def[4:-1]

        # Pull the function out of the executed environment
        return env[key]

    @staticmethod
    def _deserialize_mx(mx):
        if mx is None:
            decoded = None
        elif isinstance(mx, dict):  # then a sparse mx
            assert mx["sparse_matrix_type"] == "csr"
            data = Serializable._deserialize_mx(mx["data"])
            indices = Serializable._deserialize_mx(mx["indices"])
            indptr = Serializable._deserialize_mx(mx["indptr"])
            decoded = sps.csr_matrix(
                (data, indices, indptr), shape=mx["shape"]
            )
        else:
            basemx = np.array(mx)
            if (
                basemx.dtype.kind == "U"
            ):  # character type array => complex numbers as strings
                decoded = np.array([complex(x) for x in basemx.flat])
                decoded = decoded.reshape(basemx.shape)
            else:
                decoded = basemx
        return decoded

    @staticmethod
    def _serialize_class(class_type) -> dict:
        state = {
            "module": class_type.__module__,
            "class": class_type.__name__,
            "version": 0,
            "as_type": True,
        }
        return state

    @staticmethod
    def _serialize_function(func) -> str:
        # Get source code
        src = textwrap.dedent(inspect.getsource(func))

        # Also try to get imports
        srcfile = inspect.getsourcefile(func)
        if srcfile is None:
            # We'll fail to get imports, just return source
            return src

        # Get all import lines
        with open(srcfile, "r") as f:
            import_lines = []
            multiline = ""
            for line in f.readlines():
                if len(multiline):
                    multiline += line
                    if ")" in line:
                        import_lines.append(textwrap.dedent(multiline))
                        multiline = ""
                elif "import " in line:
                    if "(" in line and ")" not in line:
                        multiline = line
                    else:
                        import_lines.append(textwrap.dedent(line))

        # Get all things that are imported
        needed_import_lines = []
        for line in import_lines:
            if " as " in line:
                entries = line.split(" as ")[1]
            else:
                entries = line.split("import ")[1]
            # Remove parentheses from multiline imports
            entries = [
                e.replace("(", "").replace(")", "") for e in entries.split(",")
            ]

            # Get rid of newline and whitespace for better searching
            entries = [e.strip() for e in entries if len(e.strip())]

            # If the imported thing is in our source code, we need this import
            if any([e in src for e in entries]):
                needed_import_lines.append(line)

        imports = "".join(needed_import_lines)
        return imports + src

    @staticmethod
    def _serialize_mx(mx):
        if mx is None:
            return None
        elif sps.issparse(mx):
            csr_mx = sps.csr_matrix(
                mx
            )  # convert to CSR and save in this format
            return {
                "sparse_matrix_type": "csr",
                "data": Serializable._serialize_mx(csr_mx.data),
                "indices": Serializable._serialize_mx(csr_mx.indices),
                "indptr": Serializable._serialize_mx(csr_mx.indptr),
                "shape": csr_mx.shape,
            }
        else:
            enc = (
                str
                if np.iscomplexobj(mx)
                else (
                    (lambda x: int(x))
                    if (mx.dtype == np.int64)
                    else (lambda x: x)
                )
            )
            encoded = np.array([enc(x) for x in mx.flat])
            return encoded.reshape(mx.shape).tolist()

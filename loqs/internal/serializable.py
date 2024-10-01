"""TODO
"""

from __future__ import annotations

from abc import abstractmethod
import importlib
import inspect
import json
import textwrap
from typing import Callable

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

    @classmethod
    def from_serialization(cls, state):
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
                ret = cls._from_serialization(state)
            else:
                # otherwise, this call functions as a base class call that defers to the correct derived class
                ret = Serializable._from_serialization(state)
        except RecursionError as e:
            raise NotImplementedError(
                "Hit recursion limit while deserializing, usually indicating "
                + "from_serialization was not implemented in a derived class."
            ) from e

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

        format : {'json', None}
            The format of the file.  If `None` this is determined automatically
            by the filename extension of a given path.

        Returns
        -------
        NicelySerializable
        """
        if format is None:
            if str(path).endswith(".json"):
                format = "json"
            else:
                raise ValueError(
                    "Cannot determine format from extension of filename: %s"
                    % str(path)
                )

        with open(str(path), "r") as f:
            return cls.load(f, format)

    @staticmethod
    def deserialize(obj):
        """Helper function to recursively unserialize objects."""
        if isinstance(obj, dict):
            if "module" in obj and "class" in obj:
                # This is a serialized object, try to deserialize
                cls = Serializable._state_class(obj)
                return cls.from_serialization(obj)

            # Otherwise, assume just a dict and recursively unserialize
            deserialized = {}
            for k, v in obj.items():
                if isinstance(k, list):
                    k = tuple(k)
                deserialized[k] = Serializable.deserialize(v)
            return deserialized
        elif isinstance(obj, (list, tuple)):
            return [Serializable.deserialize(e) for e in obj]
        elif isinstance(obj, str) and "def " in obj:
            # Assume this is a function definition
            return Serializable._deserialize_function(obj)

        # Otherwise, assume we are a built-in deserializable object
        return obj

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

    @staticmethod
    def serialize(obj):
        """Helper function to recursively serialize objects."""
        if isinstance(obj, dict):
            return {k: Serializable.serialize(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple, set)):
            return [Serializable.serialize(e) for e in obj]
        elif isinstance(obj, Callable):
            return Serializable._serialize_function(obj)
        elif isinstance(obj, Serializable):
            return obj.to_serialization()

        # Otherwise, assume we are a built-in serializable object
        return obj

    def to_serialization(self):
        """
        Serialize this object in a way that adheres to "niceness" rules of common text file formats.

        Returns
        -------
        object
            Usually a dictionary representing the serialized object, but may also be another native
            Python type, e.g. a string or list.
        """
        # This method is here to provide a space for us to insert some global serialization logic, if needed
        return self._to_serialization()

    def write(self, path, **format_kwargs):
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
        if str(path).endswith(".json"):
            format = "json"
        else:
            raise ValueError(
                "Cannot determine format from extension of filename: %s"
                % str(path)
            )

        with open(str(path), "w") as f:
            self.dump(f, format, **format_kwargs)

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

    @abstractmethod
    def _to_serialization(self):
        state = {
            "module": self.__class__.__module__,
            "class": self.__class__.__name__,
            "version": 0,
        }
        return state

    @classmethod
    @abstractmethod
    def _from_serialization(cls, state):
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
            return c._from_serialization(state)

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
            multiline = False
            for line in f.readlines():
                if multiline:
                    import_lines[-1] += line
                    if ")" in line:
                        multiline = False
                elif "import" in line:
                    import_lines.append(line)
                    if line.endswith("(\n"):
                        multiline = True

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
    def _deserialize_function(src: str) -> Callable:
        # Execute the imports and function definition
        env = {}
        exec(src, env)

        # We need to find the function name
        # Search for first def, then first paren after it
        # Trim "def " and that should be the function name
        start = src.find("def ")
        end = src.find("(", start)
        key = src[start + 4 : end]

        # Pull the function out of the executed environment
        return env[key]

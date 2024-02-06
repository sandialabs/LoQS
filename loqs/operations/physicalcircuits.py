"""Classes corresponding to physical circuit construction.

This file defines one interface class, :class:`ReturnsPygstiCircuit`,
which serves to localize our dependency on :class:`pygsti.circuits.Circuit`.

There are several user-facing classes designed to avoid the user
needing to build pyGSTi circuits directly:

- :class:`PhysicalCircuit`, a wrapper around :class:`pygsti.circuits.Circuit`
- :class:`CircuitPlaquetteFactory`, which takes template circuits and creates
    circuit "plaquettes" or fragments
- :class:`CircuitPlaquetteSpec`, which defines a specification for a
    :class:`CircuitPlaquetteFactory` to use
- :class:`PlaquetteCircuit`, which constructs a circuit from a
    :class:`CircuitPlaquetteFactory` and :class:`CircuitPlaquetteSpec`
"""

from abc import ABC, abstractmethod
from typing import Iterable, Mapping, Optional, TypeAlias, Union

from pygsti.circuits import Circuit as PygstiCircuit

from loqs.utils import IsCastable


class ReturnsPygstiCircuit(ABC):
    """An abstract base class for an object that returns a pyGSTi circuit.

    Currently, pygsti.circuits.Circuit are used as the intermediate
    representation for circuits acting on physical qubits. This class
    centralizes this dependency in case we decide to switch intermediate
    representations later.
    """

    @abstractmethod
    def get_circuit(
        self,
        qubit_labels: Optional[Iterable[str]] = None,
        omit_gates: Optional[Union[Iterable[str], str]] = None,
        compress: bool = False,
        delete_idle_layers: bool = False,
        return_editable: bool = False,
    ) -> PygstiCircuit:
        """Get a pygsti.circuits.Circuit acting on physical qubits.

        Parameters
        ----------
        qubit_labels: list of str, optional
            Qubit labels to use for the returned circuit. If not provided,
            the default qubit labels of the object are used.

        omit_gates: str or list of str, optional
            If provided, an operation (or list of operations) to replace with
            idles in the final circuit.

        compress: bool, optional
            If True, attempt to parallelize the circuit as much as possible.
            Defaults to False, maintaining the explicit layer structure of the
            underlying circuit.

        delete_idle_layers: bool, optional
            If True, drop any layers with no operations.
            Defaults to False, maintaining idle layers which may be used for
            scheduling later in circuit composition pipeline.

        return_editable: bool, optional
            If True, returns an editable :class:`pygsti.circuits.Circuit`.
            Defaults to False.

        Returns
        -------
        pygsti.circuits.Circuit
            A circuit acting on physical qubits
        """
        pass

    @abstractmethod
    def map_qubit_labels(self, qubit_mapping: Mapping[str, str]) -> None:
        """Substitute qubit labels.

        Parameters
        ----------
        qubit_mapping: dict
            Mapping from old qubit labels to new qubit labels.
            If a qubit label is not provided, it remains unchanged.
        """
        pass

    def _process_circuit(
        self,
        circuit: PygstiCircuit,
        omit_gates: Optional[Union[Iterable[str], str]] = None,
        compress: bool = False,
        delete_idle_layers: bool = False,
        return_editable: bool = False,
    ) -> PygstiCircuit:
        """Helper function to provide consistent circuit processing.

        Parameters
        ----------
        circuit: pygsti.circuits.Circuit
            Circuit to process

        Other Parameters:
            Refer to :meth:`ReturnsPygstiCircuit.get_circuit`

        Returns
        -------
        processed_circuit: pygsti.circuits.Circuit
            The processed Circuit
        """
        processed_circuit = circuit.copy(editable=True)

        if omit_gates is None:
            omit_gates = []
        elif isinstance(omit_gates, str):
            omit_gates = [omit_gates]
        for gate in omit_gates:
            processed_circuit.replace_gatename_with_idle_inplace(gate)

        if compress:
            processed_circuit = processed_circuit.parallelize()

        if delete_idle_layers:
            processed_circuit.delete_idle_layers_inplace()

        if not return_editable:
            processed_circuit.done_editing()

        return processed_circuit


class PhysicalCircuit(IsCastable, ReturnsPygstiCircuit):
    """Base class for circuits on physical qubits.

    This is currently mainly a wrapper class for pygsti.circuits.Circuit,
    on the off-chance that we want to choose a different intermediate
    representation later. Additionally, this enforces a consistent
    interface between basic circuits and those created by utility classes
    such as PlaquetteCircuit.

    TODO Examples
    """

    Castable: TypeAlias = Union[
        "PhysicalCircuit", PygstiCircuit, tuple, list, str
    ]
    """Type alias for allowed inputs to PhysicalCircuit()

    It needs to be either a :class:`PhysicalCircuit` already, a bare
    :class:`pygsti.circuits.Circuit`, or something that can be input in
    :meth:`pygsti.circuits.Circuit.cast`.
    Ideally we would take something like pygsti.circuits.Circuit.Castable here,
    but while that is not implemented in pyGSTi yet, we just take
    generic tuples and lists since a wide variety of things can be
    cast to Circuits and try to catch any errors during Circuit.cast().
    """

    def __init__(
        self,
        circuit: "PhysicalCircuit.Castable",
        qubit_labels: Optional[Iterable[str]] = None,
    ) -> None:
        """Initialize a PhysicalCircuit.

        Parameters
        ----------
        circuit: PhysicalCircuit.Castable
            Physical circuit to be wrapped

        qubit_labels: list of str, optional
            Default qubit labels to use for the circuit. If not provided,
            qubit labels are taken from the underlying circuit object.
        """
        if isinstance(circuit, PhysicalCircuit):
            self._circuit = circuit._circuit
        else:
            try:
                self._circuit = PygstiCircuit.cast(circuit)
            except Exception as e:
                raise ValueError(
                    f"Failed to cast {circuit} as a pyGSTi Circuit"
                ) from e

        if qubit_labels is None and isinstance(circuit, PhysicalCircuit):
            self.qubit_labels = circuit.qubit_labels
        elif qubit_labels is None:
            self.qubit_labels = self._circuit.line_labels
        else:
            assert set(self._circuit.line_labels).issubset(
                set(qubit_labels)
            ), "Circuit contains line labels that are not in qubit labels"
            self.qubit_labels = qubit_labels

    def get_circuit(
        self,
        qubit_labels: Optional[Iterable[str]] = None,
        omit_gates: Optional[Union[Iterable[str], str]] = None,
        compress: bool = False,
        delete_idle_layers: bool = False,
        return_editable: bool = False,
    ) -> PygstiCircuit:
        """Get a pygsti.circuits.Circuit acting on physical qubits.

        Parameters
        ----------
        Other Parameters
            Refer to :meth:`ReturnsPygstiCircuit.get_circuit`

        Returns
        -------
            Refer to :meth:`ReturnsPygstiCircuit.get_circuit`
        """
        circuit = self._circuit.copy()
        circuit.line_labels = (
            qubit_labels if qubit_labels is not None else self.qubit_labels
        )
        return self._process_circuit(
            circuit,
            omit_gates=omit_gates,
            compress=compress,
            delete_idle_layers=delete_idle_layers,
            return_editable=return_editable,
        )

    def map_qubit_labels(self, qubit_mapping: Mapping[str, str]) -> None:
        """Substitute qubit labels.

        Parameters
        ----------
        Other Parameters
            Refer to :meth:`ReturnsPygstiCircuit.map_qubit_labels`
        """
        # Add any missing qubits as passthrough for use with
        # the map_state_space_labels() function
        complete_mapping = {
            q: qubit_mapping.get(q, q) for q in self.qubit_labels
        }

        # Technically not necessary if using get_circuit(),
        # but setting this in case someone directly accesses _circuit
        self._circuit = self._circuit.map_state_space_labels(complete_mapping)

        self.qubit_labels = [complete_mapping[q] for q in self.qubit_labels]

    def __str__(self):
        return str(self.get_circuit())

    def __repr__(self):
        return repr(self.get_circuit())


class CircuitPlaquetteFactory(IsCastable, ReturnsPygstiCircuit):
    """Generate circuit plaquettes from templates.

    This object generates circuits plaquettes/fragments on a specified
    set of qubits from a set of circuit templates.
    For example, this could be used to generate stabilizer checks for
    a syndrome extraction circuit.

    Templates should take the form of a pyGSTi circuit with temporary
    line labels. These line labels will be replaced one-to-one by
    final qubit labels, so it is HIGHLY RECOMMENDED that the order is
    manually specified by the user.

    Examples
    --------

    Example template dictionary for weight-4 X and weight-2 Z checks
    for the surface code [1]_.

    >>> templates = {
    ...     # Fig 2a of [1]
    ...     "X": PhysicalCircuit([('Gh', 'aux'), ('Gcnot', 'aux', 'b'),
    ...         ('Gcnot', 'aux', 'a'), ('Gcnot', 'aux', 'd'),
    ...         ('Gcnot', 'aux', 'c'), ('Gh', 'aux'), ('Iz', 'aux')],
    ...         qubit_labels=['a', 'b', 'c', 'd', 'aux']),
    ...     # Fig 2b of [1] (including idle layers to match X-check H layers)
    ...     "Z": PhysicalCircuit([[], ('Gcnot', 'b', 'aux'),
    ...         ('Gcnot', 'a', 'aux'), ('Gcnot', 'd', 'aux'),
    ...         ('Gcnot', 'c', 'aux'), [], ('Iz','aux')],
    ...         qubit_labels=['a', 'b', 'c', 'd', 'aux']),
    ... }
    >>> factory = CircuitPlaquetteFactory(templates)

    Subsequent creation of a weight-4 X stabilizer extraction circuit.

    >>> c = factory.get_circuit("X", ["D0", "D1", "D2", "D3", "A4"])
    >>> print(repr(c))
    Circuit(Gh:A4Gcnot:A4:D1Gcnot:A4:D0Gcnot:A4:D3Gcnot:A4:D2Gh:A4Iz:A4\
@(D0,D1,D2,D3,A4))

    Creation of a "left triangle" weight-2 X stabilizer extraction circuit.

    >>> c = factory.get_circuit("X", [None, "D1", None, "D3", "A4"])
    >>> print(repr(c))
    Circuit(Gh:A4Gcnot:A4:D1[]Gcnot:A4:D3[]Gh:A4Iz:A4@(D1,D3,A4))

    Creation of a weight-4 stabilizer *preparation* circuit (no measurement).

    >>> c = factory.get_circuit("X", ["D0", "D1", "D2", "D3", "A4"],
    ... omit_gates="Iz")
    >>> print(repr(c))
    Circuit(Gh:A4Gcnot:A4:D1Gcnot:A4:D0Gcnot:A4:D3Gcnot:A4:D2Gh:A4[]\
@(D0,D1,D2,D3,A4))

    Additional templates can be added and used after creation.

    >>> factory.add_template("Zalt",
    ...     PhysicalCircuit([
    ...         [], ('Gcnot', 'b', 'aux'), ('Gcnot', 'd', 'aux'),
    ...         ('Gcnot', 'a', 'aux'), ('Gcnot', 'c', 'aux'), [], ('Iz','aux')
    ...     ], qubit_labels=['a', 'b', 'c', 'd', 'aux'])
    ... )
    >>> c = factory.get_circuit("Zalt", ["D0", "D1", "D2", "D3", "A4"])
    >>> print(repr(c))
    Circuit([]Gcnot:D1:A4Gcnot:D3:A4Gcnot:D0:A4Gcnot:D2:A4[]Iz:A4\
@(D0,D1,D2,D3,A4))

    .. [1] Y. Tomita and K.M. Svore, "Low-distance surface codes under
        realistic quantum simulations," Phys. Rev. A, vol. 90,
        pp. 062320, 2014.
    """

    Castable: TypeAlias = Union[
        "CircuitPlaquetteFactory", Mapping[str, PhysicalCircuit.Castable]
    ]
    """Type alias for allowed inputs to CircuitPlaquetteFactory
    """

    def __init__(
        self, circuit_templates: "CircuitPlaquetteFactory.Castable"
    ) -> None:
        """Initialize a CircuitPlaquetteFactory from circuit templates.

        Parameters
        ----------
        circuit_templates: CircuitPlaquetteFactory.Castable
            Another factory or a dict where keys will be string labels for each
            template type, and values will be :attr:`PhysicalCircuit.Castable`.
        """
        if isinstance(circuit_templates, CircuitPlaquetteFactory):
            self.circuit_templates = circuit_templates.circuit_templates
        else:
            self.circuit_templates = {
                k: PhysicalCircuit.cast(v)
                for k, v in circuit_templates.items()
            }

    def add_template(
        self, key: str, template: PhysicalCircuit.Castable
    ) -> None:
        """Add an additional template to the CircuitPlaquetteFactory.

        Parameters
        ----------
        key: str
            Key for new template type

        template: pygsti.circuits.Circuit
            New template circuit
        """
        assert key not in self.circuit_templates, "Template key already exists"
        self.circuit_templates[key] = PhysicalCircuit.cast(template)

    def get_circuit(
        self,
        template_key: str,
        qubit_labels: Iterable[Optional[str]],
        omit_gates: Optional[Union[Iterable[str], str]] = None,
        compress: bool = False,
        delete_idle_layers: bool = False,
        return_editable: bool = False,
    ) -> PygstiCircuit:
        """Create a stabilizer check circuit from one of the templates.

        If removing a line or operation results in an empty layer,
        that layer is retained to ensure no scheduling changes
        in the remainder of the circuit.

        Parameters
        ----------
        template_key: str
            Stabilizer template key for self.stabilizer_templates

        qubit_labels: list of str or None
            Qubit labels that match exactly with the template line_labels.
            If an entry is None, then that template line_label (and all
            gates that touch that line) are removed.
            For example, lower-weight checks with a commensurate
            schedule can be implemented this way.

        Other Parameters:
            Refer to :meth:`ReturnsPygstiCircuit.get_circuit`

        Returns
        -------
        circuit: pygsti.circuits.Circuit
            The qubit-replaced circuit fragment/plaquette
        """
        circuit = self.circuit_templates[template_key].get_circuit(
            return_editable=True
        )
        assert len(qubit_labels) == len(circuit.line_labels), (
            "Number of qubit labels should match template lines. If dropping "
            + "lines is desired, use None in correct place in qubit_labels."
        )

        # Either map the qubit or remove it if providing None
        # The latter feature is useful to get lower weight checks with the same
        # schedule as the full weight check
        mapping = {}
        for template_qubit, qubit in zip(circuit.line_labels, qubit_labels):
            if qubit is None:
                circuit.delete_lines(template_qubit, delete_straddlers=True)
            else:
                mapping[template_qubit] = qubit

        circuit.map_state_space_labels_inplace(mapping)

        return self._process_circuit(
            circuit,
            omit_gates=omit_gates,
            compress=compress,
            delete_idle_layers=delete_idle_layers,
            return_editable=return_editable,
        )

    def map_qubit_labels(self, qubit_mapping: Mapping[str, str]) -> None:
        """Substitute qubit labels.

        Parameters
        ----------
        Other Parameters
            Refer to :meth:`ReturnsPygstiCircuit.map_qubit_labels`
        """
        for key in self.circuit_templates:
            self.circuit_templates[key].map_qubit_labels(qubit_mapping)


class CircuitPlaquetteSpec(IsCastable):
    """Convenience type for specifying circuit plaquettes to stitch together.

    Plaquette specs are given as dictionaries where the keys are templates in
    a :class:`CircuitPlaquetteFactory` and values are lists of qubit label
    lists to be replaced by the factory. These dictionaries are type-aliased
    to :attr:`CircuitPlaquetteSpec.SpecStage` for convenience.

    All specified templates within a dictionary should be able to be done in
    parallel as the circuit layers will be concatenated. For specifications
    that involve qubit reuse or collisions, one can specify different "stages"
    by providing a list of :attr:`CircuitPlaquetteSpec.SpecStages`.

    For example, below we show the specification for Surface-17 and
    Surface-13 from [1]_.

    Examples
    --------
    >>> checks = {
    ...     "X": [
    ...         [None, None, "D1", "D2" , "A9"],
    ...         ["D0", "D1", "D3", "D4", "A11"],
    ...         ["D4", "D5", "D7", "D8", "A14"],
    ...         ["D6", "D7", None, None, "A16"],
    ...     ],
    ...     "Z": [
    ...         [None, "D0", None, "D3", "A10"],
    ...         ["D1", "D2", "D4", "D5", "A12"],
    ...         ["D3", "D4", "D6", "D7", "A13"],
    ...         ["D5", None, "D8", None, "A15"],
    ...     ],
    ... }
    >>> surface17_spec = CircuitPlaquetteSpec(checks)

    Note that the Surface-17 spec only has a single stage.

    >>> len(surface17_spec.stage_specs) == 1
    True

    We can alternatively specify a multi-stage spec such as Surface-13,
    which has auxiliary qubit reuse between stabilizer checks.

    >>> checks = [
    ...     {
    ...         # First we do all weight-4 checks
    ...         "X": [
    ...             ["D0", "D1", "D3", "D4", "A9"],
    ...             ["D4", "D5", "D7", "D8", "A12"],
    ...         ],
    ...         "Z": [
    ...             ["D1", "D2", "D4", "D5", "A10"],
    ...             ["D3", "D4", "D6", "D7", "A11"],
    ...         ],
    ...     },
    ...     {
    ...         # And then all weight-2 checks
    ...         "X": [["D1", "D2", None, None, "A10"],
    ...               [None, None, "D6", "D7", "A11"]],
    ...         "Z": [["D0", None, "D3", None, "A9"],
    ...               [None, "D5", None, "D8", "A12"]],
    ...     },
    ... ]
    >>> surface13_spec = CircuitPlaquetteSpec(checks)
    >>> len(surface13_spec.stage_specs) == 2
    True

    We can substitute different template keys into the specification.

    >>> surface17_spec.map_template_keys({"Z": "Zalt"})
    >>> list(surface17_spec.stage_specs[0].keys())
    ['X', 'Zalt']

    We can also substitute different qubit labels into the specification.
    Both substitution methods allow you to selectively making substitutions
    in a given set of spec stages also.
    Note that the qubit labels in the first stage change here...

    >>> surface13_spec.map_qubit_labels({"A10": "A13"}, stage_indices=[0])
    >>> surface13_spec.stage_specs[0]["Z"]
    [['D1', 'D2', 'D4', 'D5', 'A13'], ['D3', 'D4', 'D6', 'D7', 'A11']]

    but in the second stage, they remain unchanged.

    >>> surface13_spec.stage_specs[1]['X']
    [['D1', 'D2', None, None, 'A10'], [None, None, 'D6', 'D7', 'A11']]

    .. [1] Y. Tomita and K.M. Svore, "Low-distance surface codes under
        realistic quantum simulations," Phys. Rev. A, vol. 90,
        pp. 062320, 2014.
    """

    SpecStage: TypeAlias = Mapping[str, Iterable[Iterable[str]]]
    """Convenience type alias for a single stage specification dict.
    """

    Castable: TypeAlias = Union[
        "CircuitPlaquetteSpec", SpecStage, Iterable[SpecStage]
    ]
    """Types allowed to be input into :meth:`CircuitPlaquetteSpec`.
    """

    def __init__(self, spec: "CircuitPlaquetteSpec.Castable") -> None:
        """Initialize a CircuitPlaquetteSpec.

        Parameters
        ----------
        spec: CircuitPlaquetteSpec.Castable
            An existing Spec object, SpecStage dict, or list of SpecStage
            dicts.
        """
        if isinstance(spec, CircuitPlaquetteSpec):
            self.stage_specs = spec.stage_specs
        elif not isinstance(spec, (list, tuple)):
            self.stage_specs = [spec]
        else:
            self.stage_specs = spec

    def map_qubit_labels(
        self,
        qubit_mapping: Mapping[str, str],
        stage_indices: Optional[Iterable[int]] = None,
    ) -> None:
        """Substitute qubit labels in some spec stages.

        Parameters
        ----------
        qubit_mapping: dict
            Mapping from old qubit labels to new qubit labels.
            If a qubit label is not provided, it remains unchanged.

        stage_indices: list of int, optional
            Stage indices to apply the qubit mapping. If not provided,
            mapping is applied to all stages.
        """
        if stage_indices is None:
            stage_indices = list(range(len(self.stage_specs)))

        for i in stage_indices:
            spec = self.stage_specs[i]

            for key, qubit_label_list in spec.items():
                for j, qubit_labels in enumerate(qubit_label_list):
                    self.stage_specs[i][key][j] = [
                        qubit_mapping.get(q, q) for q in qubit_labels
                    ]

    def map_template_keys(
        self,
        key_mapping: Mapping[str, str],
        stage_indices: Optional[Iterable[int]] = None,
    ) -> None:
        """Substitute template keys in some spec stages.

        Parameters
        ----------
        key_mapping: dict
            Mapping from old template keys to new template keys.
            If a template key is not provided, it remains unchanged.

        stage_indices: list of int, optional
            Stage indices to apply the key mapping. If not provided,
            mapping is applied to all stages.
        """
        if stage_indices is None:
            stage_indices = list(range(len(self.stage_specs)))

        for i in stage_indices:
            spec = self.stage_specs[i]
            self.stage_specs[i] = {
                key_mapping.get(k, k): v for k, v in spec.items()
            }


class PlaquetteCircuit(ReturnsPygstiCircuit):
    """Generate a circuit from circuit plaquette templates & specifications.

    This takes the circuit template factory and plaquette specifications,
    generates the individual plaquettes, and stitches them together
    into a full circuit.

    Examples
    --------

    Example template dictionary for weight-4 X and weight-2 Z checks
    for Surface-17 [1]_.

    >>> templates = {
    ...     # Fig 2a of [1]
    ...     "X": PhysicalCircuit([('Gh', 'aux'), ('Gcnot', 'aux', 'b'),
    ...         ('Gcnot', 'aux', 'a'), ('Gcnot', 'aux', 'd'),
    ...         ('Gcnot', 'aux', 'c'), ('Gh', 'aux'), ('Iz', 'aux')],
    ...         qubit_labels=['a', 'b', 'c', 'd', 'aux']),
    ...     # Fig 2b of [1] (including idle layers to match X-check H layers)
    ...     "Z": PhysicalCircuit([[], ('Gcnot', 'b', 'aux'),
    ...         ('Gcnot', 'a', 'aux'), ('Gcnot', 'd', 'aux'),
    ...         ('Gcnot', 'c', 'aux'), [], ('Iz','aux')],
    ...         qubit_labels=['a', 'b', 'c', 'd', 'aux']),
    ... }
    >>> checks = {
    ...     "X": [
    ...         [None, None, "D1", "D2" , "A9"],
    ...         ["D0", "D1", "D3", "D4", "A11"],
    ...         ["D4", "D5", "D7", "D8", "A14"],
    ...         ["D6", "D7", None, None, "A16"],
    ...     ],
    ...     "Z": [
    ...         [None, "D0", None, "D3", "A10"],
    ...         ["D1", "D2", "D4", "D5", "A12"],
    ...         ["D3", "D4", "D6", "D7", "A13"],
    ...         ["D5", None, "D8", None, "A15"],
    ...     ],
    ... }
    >>> qubits = [f"D{i}" for i in range(9)] + [f"A{i}" for i in range(9, 17)]
    >>> surface17_syndrome = PlaquetteCircuit(templates, checks, qubits)
    >>> c = surface17_syndrome.get_circuit()
    >>> print(repr(c))
    Circuit([Gh:A9Gh:A11Gh:A14Gh:A16][Gcnot:A11:D1Gcnot:A14:D5Gcnot:A16:D7\
Gcnot:D0:A10Gcnot:D2:A12Gcnot:D4:A13][Gcnot:A11:D0Gcnot:A14:D4Gcnot:A16:D6\
Gcnot:D1:A12Gcnot:D3:A13Gcnot:D5:A15][Gcnot:A9:D2Gcnot:A11:D4Gcnot:A14:D8\
Gcnot:D3:A10Gcnot:D5:A12Gcnot:D7:A13][Gcnot:A9:D1Gcnot:A11:D3Gcnot:A14:D7\
Gcnot:D4:A12Gcnot:D6:A13Gcnot:D8:A15][Gh:A9Gh:A11Gh:A14Gh:A16]\
[Iz:A9Iz:A11Iz:A14Iz:A16Iz:A10Iz:A12Iz:A13Iz:A15]\
@(D0,D1,D2,D3,D4,D5,D6,D7,D8,A9,A10,A11,A12,A13,A14,A15,A16))

    For advanced functionality, the circuits can also be compressed
    and operations can be omitted. The below example provides a minimum depth
    syndrome preparation circuit (although not this necessarily does not obey
    the schedule provided in the stabilizer templates).

    >>> c = surface17_syndrome.get_circuit(omit_gates='Iz', compress=True)
    >>> print(repr(c))
    Circuit([Gh:A9Gh:A11Gh:A14Gh:A16Gcnot:D0:A10Gcnot:D2:A12Gcnot:D4:A13]\
[Gcnot:A11:D1Gcnot:A14:D5Gcnot:A16:D7Gcnot:D3:A13Gcnot:A9:D2][Gcnot:A11:D0\
Gcnot:A14:D4Gcnot:A16:D6Gcnot:D1:A12Gcnot:D5:A15Gcnot:D3:A10Gcnot:D7:A13]\
[Gcnot:A11:D4Gcnot:A14:D8Gcnot:D5:A12Gcnot:A9:D1Gcnot:D6:A13Gh:A16]\
[Gcnot:A11:D3Gcnot:A14:D7Gcnot:D4:A12Gcnot:D8:A15Gh:A9][Gh:A11Gh:A14]\
@(D0,D1,D2,D3,D4,D5,D6,D7,D8,A9,A10,A11,A12,A13,A14,A15,A16))

    .. [1] Y. Tomita and K.M. Svore, "Low-distance surface codes under
        realistic quantum simulations," Phys. Rev. A, vol. 90,
        pp. 062320, 2014.
    """

    def __init__(
        self,
        factory: CircuitPlaquetteFactory.Castable,
        spec: CircuitPlaquetteSpec.Castable,
        qubit_labels: Optional[Iterable[str]] = None,
    ) -> None:
        """Initialize a PlaquetteCircuit from templates and specifications.

        Parameters
        ----------
        factory: CircuitPlaquetteFactory
            The factory containing the circuit plaquette templates

        spec: CircuitPlaquetteSpec.Castable
            The specification defining the circuit plaquettes

        qubit_labels: list of str, optional
            The qubit labels for the full syndrome circuit. If provided,
            must include every qubit label included in `spec`. If not provided,
            the union of qubit labels in `spec` is used.
        """
        self.factory = CircuitPlaquetteFactory.cast(factory)
        self.spec = CircuitPlaquetteSpec.cast(spec)

        for stage in self.spec.stage_specs:
            assert all(
                [
                    c in self.factory.circuit_templates.keys()
                    for c in stage.keys()
                ]
            ), "Not all specified circuits can be made by the provided factory"

        if qubit_labels is None:
            qubit_labels = set()
            for qubit_label_lists in self.spec.values():
                for qubit_label_list in qubit_label_lists:
                    qubit_labels += set(qubit_label_list)
            qubit_labels = list(qubit_labels)

        self.qubit_labels = qubit_labels

    def get_circuit(
        self,
        qubit_labels: Optional[Iterable[str]] = None,
        omit_gates: Optional[Union[Iterable[str], str]] = None,
        compress: bool = False,
        delete_idle_layers: bool = False,
        return_editable: bool = False,
        compress_stages_only: bool = False,
    ) -> PygstiCircuit:
        """Construct the full circuit by concatenating generated plaquettes.

        Parameters
        ----------
        compress_stages_only: bool, optional
            If True, then `compress=True` will only compress within each
            stage. Defaults to False, where `compress=True` will compress
            the entire circuit. Has no effect if `compress=False`.

        Other Parameters:
            Refer to :meth:`ReturnsPygstiCircuit.get_circuit`

        Returns
        -------
        circuit: pygsti.circuits.Circuit
            The syndrome circuit
        """
        if qubit_labels is None:
            qubit_labels = self.qubit_labels

        circuit = PygstiCircuit([], line_labels=qubit_labels, editable=True)
        for stage in self.spec.stage_specs:
            stage_layers = []
            for i, (key, qubit_list) in enumerate(stage.items()):
                for j, qubits in enumerate(qubit_list):
                    plaq_circuit = self.factory.get_circuit(key, qubits)
                    if i == 0 and j == 0:
                        # Just take layers directly
                        stage_layers.extend(
                            [
                                list(plaq_circuit.layer(j))
                                for j in range(plaq_circuit.num_layers)
                            ]
                        )
                    else:
                        # Concatenate layers
                        for k in range(plaq_circuit.num_layers):
                            if k > len(stage_layers) - 1:
                                stage_layers.append([])

                            stage_layers[k] += list(plaq_circuit.layer(k))

            # TODO: I expect this to fail on a collision
            # Figure out what that error is and put a nice error message
            stage_circuit = PygstiCircuit(
                stage_layers, line_labels=qubit_labels
            )

            if compress_stages_only and compress:
                stage_circuit = stage_circuit.parallelize()

            circuit.append_circuit_inplace(stage_circuit)

        return self._process_circuit(
            circuit,
            omit_gates=omit_gates,
            compress=compress and not compress_stages_only,
            delete_idle_layers=delete_idle_layers,
            return_editable=return_editable,
        )

    def map_qubit_labels(
        self,
        qubit_mapping: Mapping[str, str],
        stage_indices: Optional[Iterable[int]] = None,
    ) -> None:
        """Substitute qubit labels in some spec stages.

        Parameters
        ----------
        Other Parameters
            Refer to :meth:`CircuitPlaquetteSpec.map_qubit_labels`
        """
        self.factory.map_qubit_labels(qubit_mapping)
        self.spec.map_qubit_labels(qubit_mapping, stage_indices)

    def map_template_keys(
        self,
        key_mapping: Mapping[str, str],
        stage_indices: Optional[Iterable[int]] = None,
    ) -> None:
        """Substitute template keys in some spec stages.

        Parameters
        ----------
        Other Parameters
            Refer to :meth:`CircuitPlaquetteSpec.map_template_keys`
        """
        self.spec.map_template_keys(key_mapping, stage_indices)

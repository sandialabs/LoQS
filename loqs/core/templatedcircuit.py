"""TODO
"""

from __future__ import annotations
from collections.abc import Iterator

from typing import Iterable, Mapping, Optional, TypeAlias, Union

from loqs.backends import (
    CircuitBackendCastable,
    PhysicalCircuit,
    PhysicalCircuitContainer,
)
from loqs.utils import IsCastable


class CircuitTemplateFactory(PhysicalCircuitContainer, IsCastable):
    """Object to create circuits from templates.

    This object generates circuit fragments on a specified set of qubits
    from a set of circuit templates. For example, this could be used to
    generate stabilizer checks for a syndrome extraction circuit.

    Templates should take the form of a :class:`PhysicalCircuit` with temporary
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
    >>> factory = CircuitTemplateFactory(templates)

    Subsequent creation of a weight-4 X stabilizer extraction circuit.

    >>> c = factory.get_circuit("X", ["D0", "D1", "D2", "D3", "A4"])
    >>> print(repr(c))
    Physical circuit with pyGSTi backend:\
Circuit(Gh:A4Gcnot:A4:D1Gcnot:A4:D0Gcnot:A4:D3Gcnot:A4:D2Gh:A4Iz:A4\
@(D0,D1,D2,D3,A4))

    Creation of a "left triangle" weight-2 X stabilizer extraction circuit.

    >>> c = factory.get_circuit("X", [None, "D1", None, "D3", "A4"])
    >>> print(repr(c))
    Physical circuit with pyGSTi backend:\
Circuit(Gh:A4Gcnot:A4:D1[]Gcnot:A4:D3[]Gh:A4Iz:A4@(D1,D3,A4))

    Creation of a weight-4 stabilizer *preparation* circuit (no measurement).

    >>> c = factory.get_circuit("X", ["D0", "D1", "D2", "D3", "A4"],
    ... omit_gates="Iz")
    >>> print(repr(c))
    Physical circuit with pyGSTi backend:\
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
    Physical circuit with pyGSTi backend:\
Circuit([]Gcnot:D1:A4Gcnot:D3:A4Gcnot:D0:A4Gcnot:D2:A4[]Iz:A4\
@(D0,D1,D2,D3,A4))

    .. [1] Y. Tomita and K.M. Svore, "Low-distance surface codes under
        realistic quantum simulations," Phys. Rev. A, vol. 90,
        pp. 062320, 2014.
    """

    circuit_templates: dict[str, PhysicalCircuit]
    """Dictionary holding circuit templates with string keys"""

    @property
    def Castable(self) -> TypeAlias:
        """Type alias for allowed inputs to CircuitTemplateFactory.

        Either take another factory or something that maps string keys
        to something that the CircuitBackend can cast as a circuit."""
        return Union[
            CircuitTemplateFactory,
            Mapping[str, PhysicalCircuit],
            Mapping[str, self.circuit_backend.CircuitCastable],
        ]

    def __init__(
        self,
        circuit_templates: CircuitTemplateFactory.Castable,
        backend: CircuitBackendCastable,
    ) -> None:
        """Initialize a CircuitTemplateFactory from circuit templates.

        Parameters
        ----------
        circuit_templates: CircuitTemplateFactory.Castable
            Another factory or a dict where keys will be string labels for each
            template type, and values will be :attr:`PhysicalCircuit.Castable`.
        """
        super().__init__(backend)

        if isinstance(circuit_templates, CircuitTemplateFactory):
            self.circuit_templates = circuit_templates.circuit_templates
        else:
            self.circuit_templates = {
                k: PhysicalCircuit(v, backend=backend)
                for k, v in circuit_templates.items()
            }

    def add_template(
        self, key: str, template: CircuitTemplateFactory.CircuitCastable
    ) -> None:
        """Add an additional template to the CircuitTemplateFactory.

        Parameters
        ----------
        key: str
            Key for new template type

        template: pygsti.circuits.Circuit
            New template circuit
        """
        assert key not in self.circuit_templates, "Template key already exists"
        self.circuit_templates[key] = self.circuit_backend.cast_circuit(
            template
        )

    def get_bare_circuit(self, key: str) -> CircuitTemplateFactory.CircuitType:
        return self.circuit_templates[key]

    def get_bare_circuit_iter(
        self,
    ) -> Iterator[CircuitTemplateFactory.CircuitType]:
        return super().get_bare_circuit_iter()

    def get_processed_circuit(
        self,
        template_key: str,
        qubit_labels: Iterable[Optional[CircuitTemplateFactory.QubitTypes]],
        **kwargs,
    ) -> PhysicalCircuit:
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
            Refer to relevant backend's :meth:`process_circuit`

        Returns
        -------
        circuit: pygsti.circuits.Circuit
            The qubit-replaced circuit fragment/plaquette
        """
        circuit: PhysicalCircuit = self.circuit_templates[template_key].copy()
        old_qubit_labels = circuit.get_qubit_labels(circuit)

        qubits_to_delete = []
        qubits_to_map = {}
        for old_q, new_q in zip(old_qubit_labels, qubit_labels):
            if new_q is None:
                qubits_to_delete.append(old_q)
            else:
                qubits_to_map[old_q] = new_q

        circuit.delete_qubits_inplace(qubits_to_delete)
        circuit.map_qubit_labels_inplace(qubits_to_map)

        return circuit.process_circuit(**kwargs)


class CircuitTemplateSpec(IsCastable):
    """Convenience type for specifying circuit templates to stitch together.

    Template specs are given as dictionaries where the keys are templates in
    a :class:`CircuitTemplateFactory` and values are lists of qubit label
    lists to be replaced by the factory. These dictionaries are type-aliased
    to :attr:`CircuitTemplateSpec.SpecStage` for convenience.

    All specified templates within a dictionary should be able to be done in
    parallel as the circuit layers will be concatenated. For specifications
    that involve qubit reuse or collisions, one can specify different "stages"
    by providing a list of :attr:`CircuitTemplateSpec.SpecStages`.

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
    >>> surface17_spec = CircuitTemplateSpec(checks)

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
    >>> surface13_spec = CircuitTemplateSpec(checks)
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

    @property
    def QubitTypes(self) -> TypeAlias:
        """The allowed values of qubit types.

        This should be a subset of the QubitTypes of the intended
        CircuitBackend in :class:`CircuitTemplateFactory`.
        """
        return self._qubit_types

    @property
    def SpecStage(self) -> TypeAlias:
        """Convenience type alias for a single stage specification dict."""
        return Mapping[str, Iterable[Iterable[self.QubitTypes]]]

    @property
    def Castable(self):
        """Types allowed to be input into :meth:`CircuitTemplateSpec`."""
        return Union[
            CircuitTemplateSpec, self.SpecStage, Iterable[self.SpecStage]
        ]

    def __init__(self, spec: CircuitTemplateSpec.Castable) -> None:
        """Initialize a CircuitTemplateSpec.

        Parameters
        ----------
        spec:
            An existing Spec object, SpecStage dict, or list of SpecStage
            dicts.

        backend:
            The circuit backend this template
        """
        if isinstance(spec, CircuitTemplateSpec):
            self.stage_specs = spec.stage_specs
        elif not isinstance(spec, (list, tuple)):
            self.stage_specs = [spec]
        else:
            self.stage_specs = spec

    def map_qubit_labels(
        self,
        qubit_mapping: Mapping[QubitTypes, QubitTypes],
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


class TemplatedCircuit(PhysicalCircuitContainer):
    """Generate a circuit from circuit templates & specifications.

    This takes the circuit template factory and template specifications,
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
    >>> surface17_syndrome = TemplatedCircuit(templates, checks, qubits)
    >>> c = surface17_syndrome.get_circuit()
    >>> print(repr(c))
    Physical circuit with pyGSTi backend:\
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
    Physical circuit with pyGSTi backend:\
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
        factory: CircuitTemplateFactory.Castable,
        spec: CircuitTemplateSpec.Castable,
        qubit_labels: Optional[Iterable[TemplatedCircuit.QubitTypes]] = None,
    ) -> None:
        """Initialize a TemplatedCircuit from templates and specifications.

        Parameters
        ----------
        factory: CircuitTemplateFactory
            The factory containing the circuit plaquette templates

        spec: CircuitTemplateSpec.Castable
            The specification defining the circuit plaquettes

        qubit_labels: list, optional
            The qubit labels for the full syndrome circuit. If provided,
            must include every qubit label included in `spec`. If not provided,
            the union of qubit labels in `spec` is used.
        """
        self.factory = CircuitTemplateFactory.cast(factory)
        self.spec = CircuitTemplateSpec.cast(spec)

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

    def get_processed_circuit(
        self,
        qubit_labels: Optional[Iterable[TemplatedCircuit.QubitTypes]] = None,
        **kwargs,
    ) -> PhysicalCircuit:
        """Construct the full circuit by concatenating generated templates.

        Parameters
        ----------
        qubit_labels: list
            Qubit labels to use for the full circuit. Useful if a specific
            qubit ordering is desired, or additional idle qubits should be
            included.

        Other Parameters:
            Refer to the relevant backend's :meth:`process_circuit`

        Returns
        -------
        circuit: pygsti.circuits.Circuit
            The concatenated templated
        """
        if qubit_labels is None:
            qubit_labels = self.qubit_labels

        circuit = PhysicalCircuit([], line_labels=qubit_labels)
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
            stage_circuit = PhysicalCircuit(
                stage_layers,
                qubit_labels=qubit_labels,
                backend=self.circuit_backend,
            )

            circuit.append_circuit_inplace(stage_circuit)

        final_circuit = self.circuit_backend.process_circuit(circuit, **kwargs)
        final_circuit.finalize_circuit_inplace()

        return final_circuit

    def map_qubit_labels(
        self,
        qubit_mapping: Mapping[
            TemplatedCircuit.QubitTypes, TemplatedCircuit.QubitTypes
        ],
        stage_indices: Optional[Iterable[int]] = None,
    ) -> None:
        """Substitute qubit labels in some spec stages.

        Parameters
        ----------
        Other Parameters
            Refer to :meth:`CircuitTemplateSpec.map_qubit_labels`
        """
        self.factory.map_all_qubit_labels_inplace(qubit_mapping)
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
            Refer to :meth:`CircuitTemplateSpec.map_template_keys`
        """
        self.spec.map_template_keys(key_mapping, stage_indices)

"""TODO
"""

from __future__ import annotations

from collections.abc import Sequence, Mapping
from typing import Generic, Literal, MutableSequence, TypeAlias, TypeVar

from loqs.backends.circuit import BasePhysicalCircuit


# Generic physical circuit type for these containers
PhysicalCircuit = TypeVar("PhysicalCircuit", bound=BasePhysicalCircuit)

# Type aliases for static type checking
TemplateCircuitCastableTypes: TypeAlias = Mapping
"""Type alias for allowed inputs to CircuitTemplateFactory.

Either take another factory or something that maps string keys
to something that the circuit backend can cast as a circuit."""

TemplateSpecStageType: TypeAlias = Mapping[str, Sequence[Sequence]]
"""Convenience type alias for a single stage specification dict."""

TemplateSpecCastableTypes: TypeAlias = (
    TemplateSpecStageType | Sequence[TemplateSpecStageType]
)
"""Types allowed to be input into :meth:`CircuitTemplateSpec`."""


class TemplatedCircuit(Generic[PhysicalCircuit]):
    """Generate a circuit from circuit templates & specifications.

    This takes the circuit template factory and template specifications,
    generates the individual plaquettes, and stitches them together
    into a full circuit.

    Note that this class takes the type of `PhysicalCircuit` as a generic type.
    The user should specify which circuit backend the factory creates at
    runtime, e.g. TemplatedCircuit[PyGSTiPhysicalCircuit](...).
    The value of this generic should probably match that used in the underlying
    :class:`CircuitTemplateFactory`.

    Examples
    --------

    Example template dictionary for weight-4 X and weight-2 Z checks
    for Surface-17 [1]_.

    >>> from loqs.backends import PyGSTiPhysicalCircuit
    >>> templates = {
    ...     # Fig 2a of [1]
    ...     "X": PyGSTiPhysicalCircuit([('Gh', 'aux'), ('Gcnot', 'aux', 'b'),
    ...         ('Gcnot', 'aux', 'a'), ('Gcnot', 'aux', 'd'),
    ...         ('Gcnot', 'aux', 'c'), ('Gh', 'aux'), ('Iz', 'aux')],
    ...         qubit_labels=['a', 'b', 'c', 'd', 'aux']),
    ...     # Fig 2b of [1] (including idle layers to match X-check H layers)
    ...     "Z": PyGSTiPhysicalCircuit([[], ('Gcnot', 'b', 'aux'),
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
    >>> c = surface17_syndrome.get_processed_circuit()
    >>> print(repr(c))
    Physical pyGSTi circuit:
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

    >>> c = surface17_syndrome.get_processed_circuit(omit_gates='Iz')
    >>> print(repr(c))
    Physical pyGSTi circuit:
    Circuit([Gh:A9Gh:A11Gh:A14Gh:A16][Gcnot:A11:D1Gcnot:A14:D5Gcnot:A16:D7\
Gcnot:D0:A10Gcnot:D2:A12Gcnot:D4:A13][Gcnot:A11:D0Gcnot:A14:D4Gcnot:A16:D6\
Gcnot:D1:A12Gcnot:D3:A13Gcnot:D5:A15][Gcnot:A9:D2Gcnot:A11:D4Gcnot:A14:D8\
Gcnot:D3:A10Gcnot:D5:A12Gcnot:D7:A13][Gcnot:A9:D1Gcnot:A11:D3Gcnot:A14:D7\
Gcnot:D4:A12Gcnot:D6:A13Gcnot:D8:A15][Gh:A9Gh:A11Gh:A14Gh:A16][]\
@(D0,D1,D2,D3,D4,D5,D6,D7,D8,A9,A10,A11,A12,A13,A14,A15,A16))

    .. [1] Y. Tomita and K.M. Svore, "Low-distance surface codes under
        realistic quantum simulations," Phys. Rev. A, vol. 90,
        pp. 062320, 2014.
    """

    circuit_templates: dict[str, BasePhysicalCircuit]
    """Dictionary holding circuit templates with string keys."""

    template_stage_specs: list[Mapping[str, MutableSequence[MutableSequence]]]
    """List of template stage specifications"""

    default_circuit_backend: type[PhysicalCircuit]
    """Default circuit backend to use when casting template circuits."""

    def __init__(
        self,
        circuit_templates: TemplateCircuitCastableTypes,
        template_stage_specs: TemplateSpecCastableTypes,
        qubit_labels: Sequence | None = None,
        default_circuit_backend: (
            type[PhysicalCircuit] | Literal["auto"]
        ) = "auto",
    ) -> None:
        """Initialize a TemplatedCircuit from templates and specifications.

        Parameters
        ----------
        factory:
            The factory containing the circuit plaquette templates

        spec:
            The specification defining the circuit plaquettes

        qubit_labels:
            The qubit labels for the full syndrome circuit. If provided,
            must include every qubit label included in `spec`. If not provided,
            the union of qubit labels in `spec` is used.

        default_circuit_type:
            The default circuit backend to use when casting new template circuits.
            Defaults to :class:`PyGSTiPhysicalCircuit`
        """
        if default_circuit_backend == "auto":
            # Try to infer it from the circuit templates
            circuit_types = [
                type(circ)
                for circ in circuit_templates.values()
                if issubclass(type(circ), BasePhysicalCircuit)
            ]

            assert all(
                [ctype == circuit_types[0] for ctype in circuit_types]
            ), (
                "All circuits must be the same backend if using `default_circuit_backend='auto'`. "
                + "Either only give one circuit type or manually specify which should be the default backend."
            )

            self.default_circuit_backend = circuit_types[0]
        else:
            self.default_circuit_backend = default_circuit_backend

        self.circuit_templates = {
            k: self.default_circuit_backend.cast(v)
            for k, v in circuit_templates.items()
        }

        if not isinstance(template_stage_specs, Sequence):
            template_stage_specs = [template_stage_specs]

        self.template_stage_specs = []
        for stage in template_stage_specs:
            self.template_stage_specs.append(
                {
                    k: [list(qls) for qls in qubit_label_seq]
                    for k, qubit_label_seq in stage.items()
                }
            )

        for stage in self.template_stage_specs:
            assert all(
                [c in self.circuit_templates for c in stage.keys()]
            ), "Not all specified circuits can be made by the provided factory"

        if qubit_labels is None:
            qubits = set()
            for stage in self.template_stage_specs:
                for qubit_label_lists in stage.values():
                    for qubit_label_list in qubit_label_lists:
                        qubits.update(set(qubit_label_list))
            qubit_labels = list(qubits)

        self.qubit_labels = qubit_labels

    def add_template(self, key: str, template_circuit: object) -> None:
        """Add an additional template to the :class:`TemplatedCircuit`.

        Parameters
        ----------
        key:
            Key for new template type

        template_circuit:
            New template circuit
        """
        assert key not in self.circuit_templates, "Template key already exists"
        self.circuit_templates[key] = self.default_circuit_backend.cast(
            template_circuit
        )

    def get_processed_circuit(
        self,
        qubit_labels: Sequence | None = None,
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

        circuit: PhysicalCircuit = self.default_circuit_backend(
            [], qubit_labels
        )
        for stage in self.template_stage_specs:
            stage_circuit = self.default_circuit_backend([], qubit_labels)
            for key, qubit_list in stage.items():
                for qubits in qubit_list:
                    plaq_circuit = self.get_processed_circuit_template(
                        key, qubits, **kwargs
                    )

                    try:
                        stage_circuit.merge_inplace(plaq_circuit, 0)
                    except Exception as e:
                        raise ValueError(
                            "Failed to merge template circuits. Ensure no collisions "
                            + "in templated circuits + stage specifications."
                        ) from e

            circuit.append_inplace(stage_circuit)

        return circuit.process_circuit(**kwargs)

    def get_processed_circuit_template(
        self,
        template_key: str,
        qubit_labels: Sequence,
        **kwargs,
    ) -> BasePhysicalCircuit:
        """Create a stabilizer check circuit from one of the templates.

        If removing a line or operation results in an empty layer,
        that layer is retained to ensure no scheduling changes
        in the remainder of the circuit.

        Parameters
        ----------
        template_key:
            Stabilizer template key for :attr:`circuit_templates`

        qubit_labels:
            Qubit labels that match exactly with the template line_labels.
            If an entry is None, then that template line_label (and all
            gates that touch that line) are removed.
            For example, lower-weight checks with a commensurate
            schedule can be implemented this way.

        Other Parameters:
            Refer to relevant backend's :meth:`process_circuit`

        Returns
        -------
        circuit:
            The qubit-replaced circuit fragment/plaquette
        """
        circuit = self.circuit_templates[template_key].copy()
        old_qubit_labels = circuit.qubit_labels

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

    def map_qubit_labels_inplace(
        self,
        qubit_mapping: Mapping,
        stage_indices: Sequence[int] | None = None,
    ) -> None:
        """Substitute qubit labels in some spec stages.

        Parameters
        ----------
        qubit_mapping:
            Mapping from old qubit labels to new qubit labels.
            If a qubit label is not provided, it remains unchanged.

        stage_indices:
            Stage indices to apply the key mapping. If not provided,
            mapping is applied to all stages.
        """
        # Map qubit labels in circuit templates
        for v in self.circuit_templates.values():
            v.map_qubit_labels_inplace(qubit_mapping)

        # Map qubit labels in template stage specifications
        if stage_indices is None:
            stage_indices = list(range(len(self.template_stage_specs)))

        for i in stage_indices:
            spec = self.template_stage_specs[i]

            for key, qubit_label_list in spec.items():
                for j, qubit_labels in enumerate(qubit_label_list):
                    self.template_stage_specs[i][key][j] = [
                        qubit_mapping.get(q, q) for q in qubit_labels
                    ]

    def map_template_keys_inplace(
        self,
        key_mapping: Mapping[str, str],
        stage_indices: Sequence[int] | None = None,
    ) -> None:
        """Substitute template keys in some spec stages.

        Parameters
        ----------
        key_mapping:
            Mapping from old template keys to new template keys.
            If a template key is not provided, it remains unchanged.

        stage_indices:
            Stage indices to apply the key mapping. If not provided,
            mapping is applied to all stages.
        """
        if stage_indices is None:
            stage_indices = list(range(len(self.template_stage_specs)))

        for i in stage_indices:
            spec = self.template_stage_specs[i]
            self.template_stage_specs[i] = {
                key_mapping.get(k, k): v for k, v in spec.items()
            }

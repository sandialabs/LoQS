"""Creation of syndrome circuits from stabilizer plaquette templates.
"""

from typing import Iterable, Mapping, Optional, TypeAlias, Union

from pygsti.circuits import Circuit


class StabilizerPlaquetteFactory:
    """Generate stabilizer plaquettes from template circuits.

    This object takes templates for stabilizer plaquettes and
    generates stabilizer check circuits.

    Templates should take the form of a pyGSTi circuit with temporary
    line labels. These line labels will be replaced one-to-one by
    final qubit labels, so it is HIGHLY RECOMMENDED that the order is
    manually specified by the user.

    Examples
    --------

    Example template dictionary for weight-4 X and weight-2 Z checks
    for the surface code [1]_.

    >>> templates = {
    ...     # Fig 2a of [1]_
    ...     "X": Circuit([('Gh', 'aux'), ('Gcnot', 'aux', 'b'), ('Gcnot', 'aux', 'a'),
    ...                 ('Gcnot', 'aux', 'd'), ('Gcnot', 'aux', 'c'), ('Gh', 'aux'),
    ...                 ('Iz', 'aux')],
    ...                 line_labels=['a', 'b', 'c', 'd', 'aux']),
    ...     # Fig 2b of [1]_ (including idle layers to match X check H layers)
    ...     "Z": Circuit([[], ('Gcnot', 'b', 'aux'), ('Gcnot', 'a', 'aux'),
    ...                 ('Gcnot', 'd', 'aux'), ('Gcnot', 'c', 'aux'), [], ('Iz','aux')],
    ...                 line_labels=['a', 'b', 'c', 'd', 'aux']),
    ... }
    >>> factory = StabilizerPlaquetteFactory(templates)

    Subsequent creation of a weight-4 X stabilizer extraction circuit.

    >>> c = factory.get_circuit("X", ["D0", "D1", "D2", "D3", "A4"])
    >>> print(repr(c))
    Circuit(Gh:A4Gcnot:A4:D1Gcnot:A4:D0Gcnot:A4:D3Gcnot:A4:D2Gh:A4Iz:A4@(D0,D1,D2,D3,A4))

    Creation of a "left triangle" weight-2 X stabilizer extraction circuit.

    >>> c = factory.get_circuit("X", [None, "D1", None, "D3", "A4"])
    >>> print(repr(c))
    Circuit(Gh:A4Gcnot:A4:D1[]Gcnot:A4:D3[]Gh:A4Iz:A4@(D1,D3,A4))

    Creation of a weight-4 stabilizer *preparation* circuit (no measurement).

    >>> c = factory.get_circuit("X", ["D0", "D1", "D2", "D3", "A4"], omit_gates="Iz")
    >>> print(repr(c))
    Circuit(Gh:A4Gcnot:A4:D1Gcnot:A4:D0Gcnot:A4:D3Gcnot:A4:D2Gh:A4[]@(D0,D1,D2,D3,A4))

    .. [1] Y. Tomita and K.M. Svore, "Low-distance surface codes under realistic
        quantum simulations," Phys. Rev. A, vol. 90, pp. 062320, 2014.
    """

    def __init__(self, stabilizer_templates: Mapping[str, Circuit]) -> None:
        """Initialize a StabilizerPlaquetteFactory from stabilizer templates.

        Parameters
        ----------
        stabilizer_templates: dict
            Keys will be string labels for each stabilizer type,
            and values will be pyGSTi Circuits.
        """
        self.stabilizer_templates = stabilizer_templates

    def get_circuit(
        self,
        template_key: str,
        qubit_labels: Iterable[Optional[str]],
        omit_gates: Optional[Union[Iterable[str], str]] = None,
    ) -> Circuit:
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

        omit_gates: str or list of str, optional
            Gatenames that should be removed from the final circuit,
            For example, syndrome preparation (rather than extraction)
            can be implemented by omiting the midcircuit measurements.
            Defaults to None, and all gates are included.

        Returns
        -------
        circuit: pygsti.circuits.Circuit
            The qubit-replaced stabilizer check circuit
        """
        circuit = self.stabilizer_templates[template_key].copy(editable=True)
        assert len(qubit_labels) == len(circuit.line_labels), (
            "Number of qubit labels should match template lines. "
            + "If dropping lines is desired, use None in correct place in qubit_labels."
        )

        # Either map the qubit or remove it if providing None
        # The latter feature is useful to get lower weight checks with the same schedule
        mapping = {}
        for template_qubit, qubit in zip(circuit.line_labels, qubit_labels):
            if qubit is None:
                circuit.delete_lines(template_qubit, delete_straddlers=True)
            else:
                mapping[template_qubit] = qubit

        circuit.map_state_space_labels_inplace(mapping)

        # Optionally omit some gates, e.g. a measurement if interested
        # in the syndrome preparation rather than extraction circuit
        if omit_gates is not None:
            if isinstance(omit_gates, str):
                omit_gates = [omit_gates]

            for gate in omit_gates:
                circuit.replace_gatename_with_idle_inplace(gate)

        circuit.done_editing()
        return circuit


StabilizerPlaquetteSpec: TypeAlias = Mapping[str, Iterable[Iterable[str]]]
"""Convenience type for stabilizer plaquette specifications.

Keys are stabilizer type, values are list of qubit label lists.
For example, below we show the specification for Surface-17 from [1]_.

Examples
--------
>>> surface17_checks: StabilizerPlaquetteSpec = {
...     "X": [
...         ["A9", None, None, "D1", "D2"],
...         ["A11", "D0", "D1", "D3", "D4"],
...         ["A14", "D4", "D5", "D7", "D8"],
...         ["A16", "D6", "D7", None, None],
...     ],
...     "Z": [
...         ["A10", None, "D0", None, "D3"],
...         ["A12", "D1", "D2", "D4", "D5"],
...         ["A13", "D3", "D4", "D6", "D7"],
...         ["A15", "D5", None, "D8", None],
...     ],
... }

.. [1] Y. Tomita and K.M. Svore, "Low-distance surface codes under realistic
    quantum simulations," Phys. Rev. A, vol. 90, pp. 062320, 2014.
"""


class SyndromeCircuit:
    """Generate a syndrome circuit from stabilizer templates & specifications.

    This takes the stabilizer template factory and a stabilizer specification,
    generates the individual stabilizer checks, and stitches them together
    into a full syndrome circuit.

    Examples
    --------

    Example template dictionary for weight-4 X and weight-2 Z checks
    for Surface-17 [1]_.

    >>> templates = {
    ...     # Fig 2a of [1]_
    ...     "X": Circuit([('Gh', 'aux'), ('Gcnot', 'aux', 'b'), ('Gcnot', 'aux', 'a'),
    ...                 ('Gcnot', 'aux', 'd'), ('Gcnot', 'aux', 'c'), ('Gh', 'aux'),
    ...                 ('Iz', 'aux')],
    ...                 line_labels=['a', 'b', 'c', 'd', 'aux']),
    ...     # Fig 2b of [1]_ (including idle layers to match X check H layers)
    ...     "Z": Circuit([[], ('Gcnot', 'b', 'aux'), ('Gcnot', 'a', 'aux'),
    ...                 ('Gcnot', 'd', 'aux'), ('Gcnot', 'c', 'aux'), [], ('Iz','aux')],
    ...                 line_labels=['a', 'b', 'c', 'd', 'aux']),
    ... }
    >>> factory = StabilizerPlaquetteFactory(templates)
    >>> checks = {
    ...     "X": [
    ...         ["A9", None, None, "D1", "D2"],
    ...         ["A11", "D0", "D1", "D3", "D4"],
    ...         ["A14", "D4", "D5", "D7", "D8"],
    ...         ["A16", "D6", "D7", None, None],
    ...     ],
    ...     "Z": [
    ...         ["A10", None, "D0", None, "D3"],
    ...         ["A12", "D1", "D2", "D4", "D5"],
    ...         ["A13", "D3", "D4", "D6", "D7"],
    ...         ["A15", "D5", None, "D8", None],
    ...     ],
    ... }
    >>> qubits = [f"D{i}" for i in range(9)] + [f"A{i}" for i in range(9, 17)]
    >>> surface17_syndrome = SyndromeCircuit(factory, checks, qubits)
    >>> c = surface17_syndrome.get_circuit()
    >>> print(repr(c))
    Circuit([Gh:D2Gh:D4Gh:D8][Gcnot:D4:D0Gcnot:D8:D4Gcnot:D1:D5Gcnot:D3:D7][Gcnot:D2:A9Gcnot:D4:A11\
Gcnot:D8:A14Gcnot:A10:D3Gcnot:A12:D5Gcnot:A13:D7][Gcnot:D2:D1Gcnot:D4:D3Gcnot:D8:D7Gcnot:D4:D5\
Gcnot:D6:D7][Gcnot:D4:D1Gcnot:D8:D5Gcnot:D0:D3Gcnot:D2:D5Gcnot:D4:D7][Gh:D2Gh:D4Gh:D8][Iz:D2Iz:D4Iz:D8\
Iz:D3Iz:D5Iz:D7]@(D0,D1,D2,D3,D4,D5,D6,D7,D8,A9,A10,A11,A12,A13,A14,A15,A16))

    You can also provide a list of StabilizerPlaquetteSpecs in the case
    that not all checks can be performed simultaneously, i.e. there is
    auxiliary qubit reuse.

    >>> checks = [
    ...     {
    ...         # First we do all weight-4 checks
    ...         "X": [
    ...             ["A9", "D0", "D1", "D3", "D4"],
    ...             ["A12", "D4", "D5", "D7", "D8"],
    ...         ],
    ...         "Z": [
    ...             ["A10", "D1", "D2", "D4", "D5"],
    ...             ["A11", "D3", "D4", "D6", "D7"],
    ...         ],
    ...     },
    ...     {
    ...         # And then all weight-2 checks
    ...         # (note that these plaquettes are flipped from Surface-17 weight-2 checks)
    ...         "X": [["A10", "D1", "D2", None, None], ["A11", None, None, "D6", "D7"]],
    ...         "Z": [["A9", "D0", None, "D3", None], ["A12", None, "D5", None, "D8"]],
    ...     },
    ... ]
    >>> qubits = [f"D{i}" for i in range(9)] + [f"A{i}" for i in range(9, 13)]
    >>> surface13_syndrome = SyndromeCircuit(factory, checks, qubits)
    >>> c = surface13_syndrome.get_circuit()
    >>> print(repr(c))
    Circuit([Gh:D4Gh:D8][Gcnot:D4:D0Gcnot:D8:D4Gcnot:D1:D5Gcnot:D3:D7][Gcnot:D4:A9Gcnot:D8:A12\
Gcnot:A10:D5Gcnot:A11:D7][Gcnot:D4:D3Gcnot:D8:D7Gcnot:D4:D5Gcnot:D6:D7][Gcnot:D4:D1Gcnot:D8:D5Gcnot:D2:D5\
Gcnot:D4:D7][Gh:D4Gh:D8][Iz:D4Iz:D8Iz:D5Iz:D7]Gh:D7[Gcnot:D7:A11Gcnot:A12:D8]Gcnot:D7:D6Gcnot:D5:D8Gh:D7\
[Iz:D7Iz:D8]@(D0,D1,D2,D3,D4,D5,D6,D7,D8,A9,A10,A11,A12))

    For advanced functionality, the circuits can also be compressed and operations can be omitted.
    The below example provides a minimum depth syndrome preparation circuit (although not this
    necessarily does not obey the schedule provided in the stabilizer templates).

    >>> c = surface13_syndrome.get_circuit(omit_gates='Iz', compress_within_stages=True,
    ...     compress_between_stages=True)
    >>> print(repr(c))
    Circuit([Gh:D4Gh:D8Gcnot:D1:D5Gcnot:D3:D7][Gcnot:D4:D0Gcnot:A10:D5Gcnot:A11:D7]Gcnot:D8:D4[Gcnot:D4:A9\
Gcnot:D8:A12][Gcnot:D4:D3Gcnot:D8:D7][Gcnot:D4:D5Gcnot:D6:D7][Gcnot:D4:D1Gcnot:D8:D5][Gcnot:D2:D5Gcnot:D4:D7Gh:D8]\
[Gh:D4Gh:D7Gcnot:A12:D8][Gcnot:D7:A11Gcnot:D5:D8]Gcnot:D7:D6Gh:D7@(D0,D1,D2,D3,D4,D5,D6,D7,D8,A9,A10,A11,A12))

    .. [1] Y. Tomita and K.M. Svore, "Low-distance surface codes under realistic
        quantum simulations," Phys. Rev. A, vol. 90, pp. 062320, 2014.
    """

    def __init__(
        self,
        factory: StabilizerPlaquetteFactory,
        stabilizers: Union[
            StabilizerPlaquetteSpec, Iterable[StabilizerPlaquetteSpec]
        ],
        qubit_labels: Iterable[str],
    ) -> None:
        """Initialize a SyndromeCircuit from stabilizer templates and specifications.

        Parameters
        ----------
        factory: StabilizerPlaquetteFactory
            The factory containing the stabilizer plaquette templates

        stabilizers: StabilizerPlaquetteSpec or list of StabilizerPlaquetteSpec
            The stabilizer plaquette specifications to generate.
            If a single spec given, it will be promoted to a list
            with one element.

        qubit_labels: list of str
            The qubit labels for the full syndrome circuit.
            Must include every qubit label included in `stabilizers`
        """
        self.factory = factory
        if isinstance(stabilizers, (list, tuple)):
            self.stabilizer_stages = stabilizers
        else:
            # We only got one stage, make it a list for consistency
            self.stabilizer_stages = [stabilizers]

        for stage in self.stabilizer_stages:
            assert all(
                [
                    c in factory.stabilizer_templates.keys()
                    for c in stage.keys()
                ]
            ), "Not all provided checks can be created by the provided factory"

        self.qubit_labels = qubit_labels

    def get_circuit(
        self,
        omit_gates: Optional[Union[Iterable[str], str]] = None,
        compress_within_stages: bool = False,
        compress_between_stages: bool = False,
        delete_idle_layers: bool = True,
    ) -> Circuit:
        """Construct the full syndrome circuit.

        Parameters
        ----------
        omit_gates: str or list of str, optional
            Gatenames that should be removed from the final circuit,
            For example, syndrome preparation (rather than extraction)
            can be implemented by omiting the midcircuit measurements.
            Defaults to None, and all gates are included.

        compress_within_stages: bool, optional
            Whether to parallelize/compress the circuit for each stage.
            Defaults to False, which maintains the strict schedule imposed
            by the templates.

        compress_between_stages: bool, optional
            Whether to parallelize/compress the circuit *between* stages.
            Defaults to False, which keeps stabilizer checks within their
            own stage.

        delete_idle_layers: bool, optional
            Whether to delete layers that have no operations, which could
            occur depending on idle layers in the templates, dropped gates
            for low weight checks, or omitted gates. Defaults to True.

        Returns
        -------
        circuit: pygsti.circuits.Circuit
            The syndrome circuit
        """
        circuit = Circuit([], line_labels=self.qubit_labels, editable=True)
        for stage in self.stabilizer_stages:
            stage_layers = []
            for i, (key, qubit_list) in enumerate(stage.items()):
                for j, qubits in enumerate(qubit_list):
                    check_circuit = self.factory.get_circuit(
                        key, qubits, omit_gates=omit_gates
                    )
                    if i == 0 and j == 0:
                        # Just take layers directly
                        stage_layers.extend(
                            [
                                list(check_circuit.layer(j))
                                for j in range(check_circuit.num_layers)
                            ]
                        )
                    else:
                        # Concatenate layers
                        for k in range(check_circuit.num_layers):
                            if k > len(stage_layers) - 1:
                                stage_layers.append([])

                            stage_layers[k] += list(check_circuit.layer(k))

            # TODO: I expect this to fail on a collision
            # Figure out what that error is and put a nice error message
            stage_circuit = Circuit(
                stage_layers, line_labels=self.qubit_labels
            )

            if compress_within_stages:
                stage_circuit = stage_circuit.parallelize()

            circuit.append_circuit_inplace(stage_circuit)

        if compress_between_stages:
            circuit = circuit.parallelize()
        if delete_idle_layers:
            if circuit._static:
                circuit = circuit.copy(editable=True)
            circuit.delete_idle_layers_inplace()
        circuit.done_editing()

        return circuit

    def map_stabilizer_keys(
        self, key_mapping: Mapping[str, str]
    ) -> "SyndromeCircuit":
        new_stabilizer_stages = []
        for stage in self.stabilizer_stages:
            new_stage = {key_mapping.get(k, k): v for k, v in stage.items()}
            new_stabilizer_stages.append(new_stage)

        return SyndromeCircuit(
            self.factory, new_stabilizer_stages, self.qubit_labels
        )

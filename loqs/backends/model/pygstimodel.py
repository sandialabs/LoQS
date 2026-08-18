#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.1.1                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################


from __future__ import annotations

from collections.abc import Mapping
import warnings
import numpy as np
from typing import ClassVar, Sequence, TypeAlias, TypeVar, TYPE_CHECKING, Any

from loqs.backends.circuit import BasePhysicalCircuit
from loqs.backends.model import BaseNoiseModel, TimeDependentBaseNoiseModel
from loqs.backends.reps import (
    GateRep,
    InstrumentRep,
    KrausGateRep,
    OperationRep,
    PTMGateRep,
    QSimSuperopGateRep,
    RepConstructionError,
    UnitaryGateRep,
    OutcomeOperationDictInstrumentRep,
    ZBasisProjectionInstrumentRep,
    convert as convert_rep,
)
from loqs.internal.serializable import Serializable

# Conditional imports for PyGSTi
if TYPE_CHECKING:
    # Type checking imports - these won't be executed at runtime
    from pygsti.baseobjs import TensorProdBasis
    from pygsti.baseobjs.label import (
        Label,
        LabelStr,
        LabelTupTupWithTime,
        LabelTupWithTime,
    )
    from pygsti.evotypes import Evotype
    from pygsti.modelmembers.modelmember import ModelMember
    from pygsti.modelmembers.operations import EmbeddedOp, DenseOperator
    from pygsti.models import Model, ExplicitOpModel, ImplicitOpModel
else:
    # Runtime imports - these will be attempted only when needed
    from pygsti.baseobjs import TensorProdBasis
    from pygsti.baseobjs.label import (
        Label,
        LabelStr,
        LabelTupTupWithTime,
        LabelTupWithTime,
    )
    from pygsti.evotypes import Evotype
    from pygsti.modelmembers.modelmember import ModelMember
    from pygsti.modelmembers.operations import EmbeddedOp, DenseOperator
    from pygsti.models import Model, ExplicitOpModel, ImplicitOpModel


T = TypeVar("T", bound="PyGSTiNoiseModel")


PyGSTiModelLike: TypeAlias = (
    ExplicitOpModel | ImplicitOpModel | BaseNoiseModel
)
"""Types of pyGSTi models this backend can handle"""


# ---------------------------------------------------------------------------
# Workaround/detection for a pyGSTi `EmbeddedOp` memory-blowup bug.
#
# See `issues/pygsti-543.md` and `issues/pr-543.md` in the LoQS workspace repo
# for full details, and https://github.com/sandialabs/pyGSTi/issues/543 for
# the upstream pyGSTi issue.
#
# In short: pyGSTi commit 5c5b06a6d ("First pass at updating default evotype
# behavior") made `Evotype.cast(...)` prefer *dense* representations whenever
# an operator's own state space has dimension <= 64 (i.e. up to ~3 qubits).
# `EmbeddedOp` (used to embed a small local operator into a larger multi-qubit
# model) naively reuses the *embedded* operator's `Evotype` object -- which was
# decided based on that operator's own small local state space -- to decide
# its own representation type, even though `EmbeddedOp` itself spans the much
# larger *parent* state space. This can cause a single `EmbeddedOp` to
# allocate a dense matrix scaling as O(dim(parent state space)^2) instead of a
# much smaller "embedded" representation, e.g. ~2.1 GB for a single 1-qubit
# idle gate embedded into a 7-qubit register. As of this writing the bug is
# still present on pyGSTi's `develop` branch.
#
# This mainly bites user-defined custom (often time-dependent) pyGSTi
# operators, such as those in the `timedepmodel.md` tutorial, which are
# commonly built by subclassing `DenseOperator`/`LinearOperator`, passing
# `evotype` as a bare string (e.g. `"densitymx"`) without an explicit
# `state_space`, and then wrapping the result in an `EmbeddedOp` targeting a
# larger multi-qubit model. PyGSTi's own model-construction helpers (e.g.
# `create_crosstalk_free_model`) are unaffected, since they build their
# operators/evotypes differently.
# ---------------------------------------------------------------------------

_DENSE_EMBEDDING_DIM_THRESHOLD = 64
"""Mirrors the dimension threshold pyGSTi itself uses (as of the bug's
introduction) to decide whether to prefer dense representations. Above this
threshold, an `EmbeddedOp` choosing a dense representation is almost always
a symptom of the bug described above rather than intentional."""


class PyGSTiEmbeddedOpMemoryWarning(UserWarning):
    """Warned when a pyGSTi `EmbeddedOp` appears to be using a dense
    representation sized to its full (large) parent state space, rather than
    a compact embedded representation.

    This is a symptom of a known pyGSTi bug; see the module-level comment
    above [](api:PyGSTiNoiseModel) for details, and
    [](api:safe_time_dependent_evotype) for a workaround.

    To turn this warning into a hard error (e.g. to fail fast in a script or
    test suite), use the standard [](api:warnings) filtering mechanism:

    >>> import warnings
    >>> from loqs.backends.model.pygstimodel import PyGSTiEmbeddedOpMemoryWarning
    >>> warnings.simplefilter("error", PyGSTiEmbeddedOpMemoryWarning)
    """


def safe_time_dependent_evotype(evotype: str = "densitymx") -> Evotype:
    """Construct a pyGSTi [](api:pygsti.evotypes.Evotype) that is safe to use
    for custom (often time-dependent) operators that will later be embedded
    into a larger multi-qubit model via
    [](api:pygsti.modelmembers.operations.EmbeddedOp).

    This works around a pyGSTi bug where `EmbeddedOp` inherits the wrapped
    operator's own dense-representation preference (typically decided based
    on the wrapped operator's small, local state space) instead of deciding
    based on its own, potentially much larger, parent state space -- which
    can cause disproportionate (and potentially enormous) memory use. See the
    module-level comment above [](api:PyGSTiNoiseModel) and
    `issues/pr-543.md` for full details.

    Explicitly forcing `default_prefer_dense_reps=False` here sidesteps the
    issue entirely (regardless of pyGSTi version) since `EmbeddedOp` will
    then correctly choose its compact "embedded" representation.

    Parameters
    ----------
    evotype:
        Name of the underlying pyGSTi evolution type, e.g. `"densitymx"`
        (the default).

    Returns
    -------
    Evotype
        A pyGSTi `Evotype` object with dense representations disabled by
        default. Pass this directly wherever you would otherwise pass a bare
        evotype string, e.g. as the `evotype` argument of a custom
        `DenseOperator` subclass's `__init__`.

    Examples
    --------
    >>> class MyTimeDependentIdle(DenseOperator):
    ...     def __init__(self):
    ...         super().__init__(
    ...             np.identity(4, 'd'),
    ...             pygsti.BuiltinBasis("pp", 4),
    ...             safe_time_dependent_evotype("densitymx"),
    ...         )
    """
    return Evotype.cast(evotype, default_prefer_dense_reps=False)


def _iter_embedded_ops(op: ModelMember):
    """Recursively yield all `EmbeddedOp` instances reachable from `op`
    (including `op` itself, if applicable), by walking `submembers()`.
    """
    if isinstance(op, EmbeddedOp):
        yield op

    submembers = getattr(op, "submembers", None)
    if callable(submembers):
        for sub in submembers():
            yield from _iter_embedded_ops(sub)


def _check_op_for_dense_embedding_blowup(
    op: ModelMember,
    label: object,
    dim_threshold: int = _DENSE_EMBEDDING_DIM_THRESHOLD,
) -> None:
    """Check `op` (and anything nested within it) for the pyGSTi
    `EmbeddedOp`/`Evotype` dense-representation memory-blowup bug described
    in the module-level comment above, and warn (via
    [](api:PyGSTiEmbeddedOpMemoryWarning)) if found.

    Parameters
    ----------
    op:
        The pyGSTi model member to check (e.g. a gate or instrument member
        pulled from `gate_dict`/`inst_dict`).

    label:
        A label identifying `op`, used only to make the warning message
        actionable (e.g. the gate/instrument key it was looked up with).

    dim_threshold:
        Parent state space dimension above which a "dense" `EmbeddedOp`
        representation is treated as a bug symptom rather than intentional.
    """
    for embedded in _iter_embedded_ops(op):
        rep_type = getattr(embedded, "_rep_type", None)
        if rep_type != "dense":
            continue

        parent_dim = embedded.state_space.dim
        child_dim = embedded.embedded_op.state_space.dim
        if parent_dim <= dim_threshold or child_dim >= parent_dim:
            # Either the dense representation is small regardless (cheap),
            # or this isn't actually an embedding into a larger space.
            continue

        approx_bytes = parent_dim * parent_dim * 8  # assumes float64 entries
        warnings.warn(
            f"pyGSTi gate/instrument {label!r} contains an EmbeddedOp using "
            f"a dense representation sized to its full parent state space "
            f"(dim={parent_dim}, ~{approx_bytes / 1e9:.2f} GB) instead of a "
            f"compact embedded representation (embedded operator only acts "
            f"on a dim={child_dim} subspace). This is a known pyGSTi bug "
            "(see issues/pygsti-543.md and issues/pr-543.md) where "
            "EmbeddedOp inherits its wrapped operator's dense-representation "
            "preference, which was decided based on the wrapped operator's "
            "own small state space, rather than re-deciding based on its "
            "own (larger) parent state space. This most commonly happens "
            "when constructing a custom pyGSTi operator (e.g. subclassing "
            "DenseOperator) with evotype passed as a bare string and no "
            "explicit state_space, then embedding it via EmbeddedOp. "
            "Workaround: construct the operator's evotype explicitly with "
            "dense representations disabled, e.g. using "
            "loqs.backends.model.pygstimodel.safe_time_dependent_evotype(...) "
            "in place of a bare evotype string.",
            PyGSTiEmbeddedOpMemoryWarning,
            stacklevel=3,
        )


class PyGSTiNoiseModel(TimeDependentBaseNoiseModel):
    """Model backend for handling [](api:pygsti.models.model.OpModel) objects.

    PyGSTi models are inherently time-dependent, so this inherits from
    [](api:TimeDependentBaseNoiseModel) rather than
    [](api:BaseNoiseModel).
    However, time-dependent features are opt-in and require the user
    to specify `use_time_dependence=True` during initialization.
    """

    name: ClassVar[str] = "pyGSTi"

    _SERIALIZE_ATTRS = ["model", "qubit_aliases"]

    model: ExplicitOpModel | ImplicitOpModel
    """Underlying [](api:pygsti.models.explicitmodel.ExplicitOpModel) or [](api:pygsti.models.implicitmodel.ImplicitOpModel)
    """

    def __init__(
        self,
        model: PyGSTiModelLike,
        qubit_aliases: Mapping | Sequence | None = None,
        zbasis_proj_resets: bool = True,
        use_time_dependence: bool = False,
        default_gate_durations: (
            Mapping[Label | str, int | float] | None
        ) = None,
        default_instrument_durations: (
            Mapping[Label | str, int | float] | None
        ) = None,
        instrument_outcome_qubits: (
            Mapping[str | tuple[str, tuple[str | int, ...]], str | int | Sequence[str | int]]
            | None
        ) = None,
    ) -> None:
        """Initialize a PyGSTiModelBackend.

        Parameters
        ----------
        model:
            A pyGSTi model to use when looking up operations

        instrument_outcome_qubits:
            Names the classical register a joint instrument's outcome
            channel belongs to -- needed when an instrument's outcome
            labels don't resolve to one bit per physical qubit (e.g. a
            2Q/4Q parity check). Keys are a bare instrument name, or an
            `(name, aliased_qubits)` pair to disambiguate the same name on
            different qubit sets; values are the classical label(s) to
            record under. Ignored for instruments whose outcome labels
            already resolve one bit per physical qubit.
        """
        from loqs.backends import is_backend_available

        if not is_backend_available("pygsti_model"):
            raise ImportError(
                "PyGSTi model backend is not available. "
                "Please install pygsti: pip install loqs[pygsti]"
            )
        from loqs.backends.model import DictNoiseModel

        # Dispatch on the input type first, so `self.model` is always set
        # (or a specific error already raised) before it's used below.
        self.use_embedded_op = False
        if isinstance(model, ExplicitOpModel):
            self.model = model
            self.gate_dict = self.model.operations
            self.inst_dict = self.model.instruments
        elif isinstance(model, ImplicitOpModel):
            self.model = model
            self.gate_dict = self.model.operation_blks.get("layers", {})
            self.inst_dict = self.model.instrument_blks.get("layers", {})
            self.use_embedded_op = True
        elif isinstance(model, PyGSTiNoiseModel):
            self.model = model.model
            self.gate_dict = model.gate_dict
            self.inst_dict = model.inst_dict
            self.use_embedded_op = model.use_embedded_op
        elif isinstance(model, DictNoiseModel):
            raise NotImplementedError("TODO: Build explicit op model")
        else:
            raise TypeError(f"Cannot cast {type(model)} to PyGSTiNoiseModel")

        # Currently there is a pyGSTi bug deserializing models that have
        # non-int or strs that do not start with "Q". We enforce this on
        # the underlying pyGSTi model's own qubit labels, since the bug
        # lives in pyGSTi's own (de)serialization and can't be worked
        # around here. `qubit_aliases` lets you present different labels
        # to LoQS, but does not bypass this restriction.
        model_qubits = self.model.state_space.qubit_labels  # type: ignore
        assert all(
            [
                isinstance(q, int)
                or (isinstance(q, str) and q.startswith("Q"))
                for q in model_qubits
            ]
        ), (
            "Model must use int or str starting with Q labels for qubits. ",
            "For qubit labels outside of these restrictions, use `qubit_aliases` "
            "to present different labels to LoQS (the underlying pyGSTi model's "
            "own labels must still conform).",
        )

        if qubit_aliases is None:
            if isinstance(model, PyGSTiNoiseModel):
                # Copy-constructor: inherit the source model's aliases.
                self.qubit_aliases = dict(model.qubit_aliases)
            else:
                self.qubit_aliases = {k: k for k in model_qubits}
        elif isinstance(qubit_aliases, Mapping):
            assert all([q in qubit_aliases for q in model_qubits])
            self.qubit_aliases = dict(qubit_aliases)
        elif isinstance(qubit_aliases, Sequence):
            assert len(qubit_aliases) == len(model_qubits)
            self.qubit_aliases = {
                k: v for k, v in zip(model_qubits, qubit_aliases)
            }
        else:
            raise TypeError("Invalid type for qubit aliases")
        self.model_qubit_aliases = {
            v: k for k, v in self.qubit_aliases.items()
        }

        self.zbasis_proj_resets = zbasis_proj_resets

        if instrument_outcome_qubits is None and isinstance(model, PyGSTiNoiseModel):
            # Copy-constructor: inherit the source model's declarations.
            self.instrument_outcome_qubits = dict(model.instrument_outcome_qubits)
        else:
            self.instrument_outcome_qubits = dict(instrument_outcome_qubits or {})

        self.use_time_dependence = use_time_dependence
        self.default_gate_durations = default_gate_durations
        self.default_instrument_durations = default_instrument_durations

        # TODO: Crosstalk specification?

        self._gate_rep_cache = {}
        self._inst_rep_cache = {}

        # Tracks which gate_dict/inst_dict keys have already been checked for
        # the pyGSTi EmbeddedOp dense-representation memory-blowup bug (see
        # `_check_op_for_dense_embedding_blowup` above), so we only pay the
        # (small) cost of checking -- and only warn -- once per op.
        self._dense_embedding_checked_gate_keys: set = set()
        self._dense_embedding_checked_inst_keys: set = set()
        # Separate bookkeeping used by `check_for_dense_embedding_issues`,
        # which is keyed on raw gate_dict/inst_dict (pyGSTi Label) keys
        # rather than the (name, *qubits) tuples used internally by
        # `_get_gate_rep`/`_get_instrument_rep`.
        self._dense_embedding_checked_raw_gate_keys: set = set()
        self._dense_embedding_checked_raw_inst_keys: set = set()

    @property
    def gate_keys(self) -> list:
        keys = []
        for key in self.gate_dict.keys():
            name = key.name

            if isinstance(key, LabelStr):
                # LabelStr does not have any qubits with it
                keys.append((name,))
                continue

            aliased_qubits = [self.qubit_aliases[q] for q in key.qubits]
            keys.append((name, aliased_qubits))
        return keys

    @property
    def instrument_keys(self) -> list:
        keys = []
        for key in self.inst_dict.keys():
            name = key.name
            aliased_qubits = [self.qubit_aliases[q] for q in key.qubits]
            keys.append((name, aliased_qubits))
        return keys

    _output_gate_reps = [
        UnitaryGateRep,
        KrausGateRep,
        PTMGateRep,
        QSimSuperopGateRep,
    ]

    @property
    def output_gate_reps(self) -> list[type[GateRep]]:
        return self._output_gate_reps

    _output_instrument_reps = [
        ZBasisProjectionInstrumentRep,
        OutcomeOperationDictInstrumentRep,
    ]

    @property
    def output_instrument_reps(self) -> list[type[InstrumentRep]]:
        """Get the list of instrument representation classes this model can output.

        Returns
        -------
        list[type[InstrumentRep]]
            List of instrument representation classes that this model can output.

        Note
        ----
        This is not quite right currently. It returns all *possible* types,
        but often models will only allow one of the two types.
        """
        return self._output_instrument_reps

    def get_gate_duration(self, gate_label) -> int | float:
        """Get the duration of a gate operation.

        This method retrieves the duration of a gate operation from either the gate label
        itself (if time-dependent) or from the default gate durations mapping.

        Parameters
        ----------
        gate_label : Label | str
            The gate label to get the duration for.

        Returns
        -------
        int | float
            Duration of the gate operation.

        Raises
        ------
        ValueError
            If time dependence is not enabled and no default gate durations are provided,
            or if a LayerTupTupWithTime is unexpectedly provided.

        KeyError
            If the gate label is not found in the default gate durations.
        """
        if not self.use_time_dependence:
            return 0

        if isinstance(gate_label, LabelTupWithTime):
            return gate_label.time  # type: ignore
        elif isinstance(gate_label, LabelTupTupWithTime):
            # This represents a layer. Technically this should be fine to extract the time,
            # but we almost certainly didn't mean to pass a whole layer here
            raise ValueError(
                "Unexpectedly got a LayerTupTupWithTime (i.e., layer) in get_gate_duration"
            )

        # If we are here, we got a normal (probably LabelTup/LabelTupWithArgs)
        # and need to be looking up in the defaults
        if self.default_gate_durations is None:
            raise ValueError(
                "Time not specified in pyGSTi circuit label and no default gate durations given!"
            )

        duration = self.default_gate_durations.get(gate_label, None)
        if duration is None:
            # Failed specific lookup, let's try by name only
            duration = self.default_gate_durations.get(gate_label.name, None)  # type: ignore

        if duration is None:
            raise KeyError(
                f"{gate_label} not available by label or name in default gate durations!"
            )

        return duration

    def get_instrument_duration(self, inst_label) -> int | float:
        """Get the duration of an instrument operation.

        This method retrieves the duration of an instrument operation from either the
        instrument label itself (if time-dependent) or from the default instrument
        durations mapping.

        Parameters
        ----------
        inst_label : Label | str
            The instrument label to get the duration for.

        Returns
        -------
        int | float
            Duration of the instrument operation.

        Raises
        ------
        ValueError
            If time dependence is not enabled and no default instrument durations are provided,
            or if a LayerTupTupWithTime is unexpectedly provided.

        KeyError
            If the instrument label is not found in the default instrument durations.
        """
        if not self.use_time_dependence:
            return 0

        if isinstance(inst_label, LabelTupWithTime):
            return inst_label.time  # type: ignore
        elif isinstance(inst_label, LabelTupTupWithTime):
            # This represents a layer. Technically this should be fine to extract the time,
            # but we almost certainly didn't mean to pass a whole layer here
            raise ValueError(
                "Unexpectedly got a LayerTupTupWithTime (i.e., layer) in get_instrument_duration"
            )

        # If we are here, we got a normal (probably LabelTup/LabelTupWithArgs)
        # and need to be looking up in the defaults
        if self.default_instrument_durations is None:
            raise ValueError(
                "Time not specified in pyGSTi circuit label and no default instrument durations given!"
            )

        duration = self.default_instrument_durations.get(inst_label, None)
        if duration is None:
            # Failed specific lookup, let's try by name only
            duration = self.default_instrument_durations.get(inst_label.name, None)  # type: ignore

        if duration is None:
            raise KeyError(
                f"{inst_label} not available by label or name in default instrument durations!"
            )

        return duration

    def check_for_dense_embedding_issues(self) -> None:
        """Proactively scan every gate/instrument in this model for the
        pyGSTi `EmbeddedOp` dense-representation memory-blowup bug (see the
        module-level comment near the top of this file, and
        `issues/pygsti-543.md` / `issues/pr-543.md`), warning (via
        [](api:PyGSTiEmbeddedOpMemoryWarning)) for each affected gate/
        instrument found.

        This performs the same check that [](api:get_reps) already performs
        lazily (once per gate/instrument, the first time it's used), but lets
        you audit a model up front -- e.g. right after injecting custom
        (often time-dependent) operators into `gate_dict`/`inst_dict` -- so
        problems surface before running a potentially expensive/large
        simulation, rather than partway through one.

        This is not called automatically during `__init__` because
        `gate_dict`/`inst_dict` are live references to the underlying pyGSTi
        model's dictionaries, and users often mutate them (e.g. to inject
        custom time-dependent gates) *after* constructing a
        [](api:PyGSTiNoiseModel). Note that this method uses its own
        bookkeeping (keyed on the raw pyGSTi `Label` objects used by
        `gate_dict`/`inst_dict`) that is independent from -- and so may
        result in a gate/instrument being checked more than once relative to
        -- the lazy, per-(name, qubits) checks performed by `get_reps`.
        """
        for op_key, op in self.gate_dict.items():
            if op_key not in self._dense_embedding_checked_raw_gate_keys:
                _check_op_for_dense_embedding_blowup(op, op_key)
                self._dense_embedding_checked_raw_gate_keys.add(op_key)

        for inst_key, op in self.inst_dict.items():
            if inst_key not in self._dense_embedding_checked_raw_inst_keys:
                _check_op_for_dense_embedding_blowup(op, inst_key)
                self._dense_embedding_checked_raw_inst_keys.add(inst_key)

    def get_reps(
        self,
        circuit: BasePhysicalCircuit,
        gatereps: Sequence[type[GateRep]],
        instreps: Sequence[type[InstrumentRep]],
    ) -> list[OperationRep]:
        # Get bare circuit (avoiding a redundant copy if it's already the
        # right type).
        from loqs.backends import PyGSTiPhysicalCircuit

        if not isinstance(circuit, PyGSTiPhysicalCircuit):
            circuit = PyGSTiPhysicalCircuit(circuit)
        pygsti_circuit = circuit.circuit

        # Iterate through circuit and pull out representations
        reps = []
        for layer in pygsti_circuit.layertup:  # type: ignore
            for comp in layer.components:  # type: ignore
                name = comp.name
                aliased_qubits = comp.qubits  # The circuit is already aliased
                qubits = [self.model_qubit_aliases[q] for q in aliased_qubits]
                if name.startswith("G"):
                    gate_rep = self._get_gate_rep(comp.name, qubits, gatereps)
                    # We need to save with original (aliased) qubits
                    rep = gate_rep.with_qubit_labels(aliased_qubits)
                    duration = self.get_gate_duration(comp)
                elif comp.name.startswith("I"):
                    instrument_rep = self._get_instrument_rep(
                        comp.name, qubits, instreps
                    )
                    rep = instrument_rep.with_qubit_labels(aliased_qubits)
                    duration = self.get_instrument_duration(comp)
                else:
                    raise NotImplementedError("Can only handle G/I prefixes")

                reps.append(rep)

                # If using time-dependence, update layer time
                if self.use_time_dependence:
                    self.add_gate_duration_to_layer(duration)
            # If using time-dependence, move simulation time forward
            if self.use_time_dependence:
                self.add_layer_duration_to_current_time()
        return reps

    def _get_gate_rep(self, name, qubits, gatereps):
        op_key = (name,) + tuple(qubits)
        # Check cache
        for gaterep in gatereps:
            if (op_key, gaterep) in self._gate_rep_cache:
                return self._gate_rep_cache[op_key, gaterep]

        # Look up using unaliased qubits
        op = self.gate_dict[op_key]

        if op_key not in self._dense_embedding_checked_gate_keys:
            _check_op_for_dense_embedding_blowup(op, op_key)
            self._dense_embedding_checked_gate_keys.add(op_key)

        # if using time-dependence, update operator rep and clear cache
        if self.use_time_dependence:
            op.set_time(self.current_time)

        if self.use_embedded_op and isinstance(op, EmbeddedOp):
            # Sanity-check the tensor-product basis shape; op.to_dense()
            # already returns a PTM in the per-gate basis regardless.
            assert isinstance(self.model.basis, TensorProdBasis)
            assert op.target_labels is not None
            target_indices = [
                op.state_space.qubit_labels.index(q) for q in op.target_labels
            ]
            assert all(
                i < len(self.model.basis.component_bases) for i in target_indices
            )

            op = op.embedded_op

        # This is already in the "pp" basis convert()'s machinery assumes;
        # let convert() produce whichever concrete GateRep is requested.
        ptm_rep = PTMGateRep(op.to_dense(on_space="HilbertSchmidt"), qubits)

        errors = []
        for gaterep in gatereps:
            try:
                gate_rep = convert_rep(ptm_rep, gaterep)
            except RepConstructionError as e:
                errors.append(e)
                continue

            if not self.use_time_dependence:
                self._gate_rep_cache[op_key, gaterep] = gate_rep
            return gate_rep

        raise RepConstructionError(
            f"Failed to create gate rep for any of {gatereps}, with errors:"
            + "\n".join([str(e) for e in errors])
        )

    def _get_instrument_rep(self, name, qubits, instreps):
        inst_key = (name,) + tuple(qubits)
        # Check cache
        for instrep in instreps:
            if (inst_key, instrep) in self._inst_rep_cache:
                return self._inst_rep_cache[inst_key, instrep]

        def _make_rep(instrep):
            if instrep is ZBasisProjectionInstrumentRep:
                reset = 0 if self.zbasis_proj_resets else None
                return ZBasisProjectionInstrumentRep(reset, True, qubits)
            elif instrep is OutcomeOperationDictInstrumentRep:
                # TODO: What to do with key error?
                # Look up using unaliased qubits
                op = self.inst_dict[inst_key]

                if inst_key not in self._dense_embedding_checked_inst_keys:
                    _check_op_for_dense_embedding_blowup(op, inst_key)
                    self._dense_embedding_checked_inst_keys.add(inst_key)

                # if using time-dependence, update operator rep
                # `Instrument` itself has no `set_time` -- each individual
                # member operation does.
                if self.use_time_dependence:
                    for member_op in op.values():
                        member_op.set_time(self.current_time)

                outcome_ops = {}
                for k, v in op.items():
                    if isinstance(k, str) and k != "" and all(c in "01" for c in k):
                        # pyGSTi's usual '0'/'1'-character-string convention
                        # for a decomposable multi-bit outcome.
                        label = tuple(int(c) for c in k)
                    else:
                        # Any other label (e.g. 'even'/'odd' for a joint
                        # parity-check instrument) is used as-is.
                        label = k

                    # Wrap as pyGSTi's native PTM; each consuming backend
                    # converts to whatever concrete GateRep it needs.
                    outcome_ops[label] = PTMGateRep(
                        v.to_dense(on_space="HilbertSchmidt"), qubits
                    )

                # A label that isn't itself a sequence of bits is its own
                # single (joint) channel, regardless of qubit count.
                def _n_channels(label):
                    if (
                        isinstance(label, Sequence)
                        and not isinstance(label, str)
                        and all(b in (0, 1) for b in label)
                    ):
                        return len(label)
                    return 1

                channel_counts = {_n_channels(lbl) for lbl in outcome_ops}
                if len(channel_counts) != 1:
                    raise RepConstructionError(
                        f"instrument {name!r} outcome labels have inconsistent "
                        f"channel counts {channel_counts!r}"
                    )
                n_channels = next(iter(channel_counts))

                if n_channels == len(qubits):
                    # One classical bit per physical qubit.
                    outcome_qubits = qubits
                elif n_channels == 1:
                    # A joint outcome channel isn't owned by any one qubit;
                    # the caller must say which classical register it's in.
                    aliased_qubits = tuple(self.qubit_aliases[q] for q in qubits)
                    outcome_qubits = self.instrument_outcome_qubits.get(
                        (name, aliased_qubits),
                        self.instrument_outcome_qubits.get(name),
                    )
                    if outcome_qubits is None:
                        raise RepConstructionError(
                            f"instrument {name!r} on {aliased_qubits!r} has a "
                            "single joint outcome channel; add an entry to "
                            "instrument_outcome_qubits"
                        )
                else:
                    raise RepConstructionError(
                        f"instrument {name!r} outcome labels have {n_channels} "
                        f"channels, matching neither 1 nor len(qubits)={len(qubits)}"
                    )

                return OutcomeOperationDictInstrumentRep(
                    outcome_ops, True, qubits, outcome_qubits
                )
            else:
                raise RepConstructionError(
                    f"Cannot create instrument rep for {instrep}"
                )

        errors = []
        for instrep in instreps:
            try:
                instrument_rep = _make_rep(instrep)
            except RepConstructionError as e:
                errors.append(e)
                continue

            if not self.use_time_dependence:
                self._inst_rep_cache[inst_key, instrep] = instrument_rep
            return instrument_rep

        raise RepConstructionError(
            f"Failed to create instrument rep for any of {instreps}, with errors:"
            + "\n".join([str(e) for e in errors])
        )

    def _get_encoding_attr(self, attr, ignore_no_serialize_flags=False):
        if attr == "model":
            return self.model.to_nice_serialization()
        return super()._get_encoding_attr(attr, ignore_no_serialize_flags)

    @classmethod
    def _from_decoded_attrs(cls: type[T], attr_dict: Mapping) -> T:
        model = Model.from_nice_serialization(attr_dict["model"])
        qubit_aliases = attr_dict["qubit_aliases"]
        return cls(model, qubit_aliases)

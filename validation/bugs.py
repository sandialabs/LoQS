"""Confirmed codepack and core bugs, keyed by id, for strict known failures."""

KNOWN_BUGS: dict[str, str] = {
    "steane-1": (
        "A Steane re-prep keeps the previous run's `logical_pauli_frame` "
        "and `latest_syndrome`: prep, X on D0, QEC, X on D1, QEC, "
        "`FT Zero Prep`, `FT Logical Z Measure` reads 1 where 0 is certain "
        "(`codepack_7_1_3_quantinuum2021.py:150,193`; the preps reset only "
        "the physical state). Class: tracking (`reuse`, `outcome` and O3). "
        "Severity: M. Fixed in stage 8, by generation."
    ),
    "steane-2": (
        "Steane's H and S frame updates, feed-forward and decoder assign "
        "the updated patch into the incoming `PatchLayout` "
        "(`patches[patch_label] = new_patch`, `:262,301,804,944`), so an "
        "earlier frame's `patches` shows the later tracking data: after "
        "prep, X on D0, QEC the history frame of the prep re-reads "
        "syndrome (0,0,0,0,1,0) where it held (0,0,0,0,0,0). Class: "
        "tracking (O4: `idle`, `gate`, `word`, `reuse`, `mp-track`, and "
        "`independence` block schedules). Severity: M. Fixed in stage 8, by "
        "generation."
    ),
    "steane-3": (
        "The Steane pyGSTi ideal model has no `evotype`, so it is "
        "`densitymx` (`codepack_7_1_3_quantinuum2021.py:1041-1053`), and "
        "pyGSTi builds its state prep and POVM dense over the whole "
        "register: a two-patch register (20 qubits) tries to allocate an "
        "8 TiB state. The missing `Imrz` instrument was a false alarm: "
        "`PyGSTiNoiseModel` serves `Imrz` as a Z projection. Class: static "
        "(`model-coverage`). Severity: M. Fixed in stage 3 by the shared "
        "`create_ideal_model`; shown in stage 8."
    ),
    "steane-4": (
        "Steane's `template_data_qubits` counts the auxiliary qubit `A2` "
        "as a data qubit (`qubits[2:]`, `codepack_7_1_3_quantinuum2021.py:"
        "94`), so it lists eight data qubits where the code has seven. "
        "Class: static (`completeness`, `data-qubits`). Severity: M. Fixed "
        "in stage 8, by generation."
    ),
    "steane-5": (
        "Under `include_idles` the repeat-until-success reset is not "
        "padded (`codepack_7_1_3_quantinuum2021.py:159-164`): it is `Imrz` "
        "on the data qubits alone, so the auxiliary qubits get no idle "
        "during each prep attempt's reset. Class: static (`idle-padding`). "
        "Severity: M. Fixed in stage 8, by generation."
    ),
    "five-1": (
        "The 5-qubit H's `raw_syndrome` remap is wrong: Z on D0, Unflagged "
        "QEC, H leaves stored (0,1,1,0) where the true syndrome is "
        "(1,0,0,0) (`codepack_5_1_3_quantinuum2022.py:322-326`). Class: "
        "tracking (`gate` H, O2; `word`). Severity: B. Fix: stage 5."
    ),
    "five-2": (
        "The 5-qubit K has no `raw_syndrome` remap: X on D0, QEC, K leaves "
        "stored (0,0,0,1) where the true syndrome is (1,0,1,0) "
        "(`:262-275`). Class: tracking (`gate` K, O2; `word`). Severity: "
        "B. Fix: stage 5."
    ),
    "five-3": (
        "`PauliFrame`'s K rule sends X to Y where K sends X to Z (stim "
        "`C_ZYX`): X on D0, QEC, K tracks `YIIII` "
        "(`loqs/core/recordables/pauliframe.py:239-240`). Class: tracking "
        "(`gate` K, O3; `word`). Severity: B. Fix: stage 5."
    ),
    "five-4": (
        "The 5-qubit H permutation builds a fresh patch and drops the "
        "per-qubit Pauli frame: Z on D2, QEC, H turns `IIZII` into `IIIII` "
        "(`:313-320`). Class: tracking (`gate` H, O3; `word`). Severity: "
        "B. Fix: stage 5."
    ),
    "five-5": (
        "Non-FT logical measures read the Pauli frame without the "
        "prime-basis rotation: Z on D0, QEC, Non-FT X measure gives 0 "
        "where 1 is certain; FT X measure gives 1 (`:448-497`). Class: "
        "tracking (`idle`, `gate`, `word`, `reuse` with a Non-FT measure; "
        "`agreement`). Severity: B. Fix: stage 5."
    ),
    "five-6": (
        "No check in this stage. The 5-qubit FT measure decoder ignores "
        "frame propagation through Unprep (`:843-852`): in research, 888 of "
        "2,520 runs with a tracked frame plus a single Part I fault gave a "
        "wrong logical outcome. Class: fault tolerance (`ft`, tracked "
        "seed). Severity: B. Declared as an `ft` sweep in stage 5, which "
        "also fixes it."
    ),
    "five-7": (
        "No check in this stage. The flagged 5-qubit decoders neither diff "
        "against nor update `raw_syndrome` (`:2023-2031`), so a tracked "
        "error can trigger a spurious hook correction. By reading only: no "
        "`idle` or `gate` case with `Flagged QEC` was attributed to it. "
        "Class: fault tolerance (`ft`). "
        "Severity: B. Declared as an `ft` sweep in stage 5, which also "
        "fixes it."
    ),
    "five-8": (
        "`create_ideal_model` asserts 7 qubits (`:629`) and its pyGSTi "
        "model lacks `Imrz` and an evotype (`:677`), so two-patch runs, "
        "the offset label set and the pyGSTi `model-coverage` fail. Class: "
        "static, independence. Severity: M. Fix: stage 5."
    ),
    "five-9": (
        "The Unflagged Decoder hardcodes qubit `A0` (`:1822`), so permuted "
        "or renamed labels raise `KeyError: 'A0'`. Class: static "
        "(`labels`). Severity: M. Fix: stage 5."
    ),
    "five-10": (
        "5-qubit preps reset only the physical state, not the tracked "
        "frame or `raw_syndrome` (`:153-158`, `:209-230`): FT Minus Prep, "
        "X on D0, QEC, X on D1, QEC, FT Minus Prep, FT X measure gives 0 "
        "in 20 of 20 shots where 1 is certain, and the frame still reads "
        "`XXIII` after the re-prep. Class: tracking (`reuse`, outcome and "
        "O3). Severity: M. Fix: stage 5."
    ),
    "five-11": (
        "A serialized-then-read 5-qubit code raises `NameError: "
        "unflagged_lookup_table`: the Unflagged Decoder closes over a "
        "local (`:1807`, `:1841`). Class: static (`serialization`). "
        "Severity: M. Fixed upstream by #167 (merged in 86d81a7)."
    ),
    "five-12": (
        "`FT Logical Z Measure` is a composite of the `H` and `FT Logical "
        "X Measure` instruction objects (`:1000-1006`) and gives random "
        "outcomes on a certain Z state: prep, H, FT Z measure gives 0/1 "
        "50/50 where 1 is certain. The same stack with the nested "
        "measure given as a label gives 1 in 100 of 100 shots, so the "
        "composite's pulled-up kwargs "
        "(`loqs/core/instructions/builders.py:210-236`) are the likely "
        "cause (not isolated further). Class: truth. Severity: B. Fix: "
        "stage 5."
    ),
    "five-13": (
        "Under `include_idles` the 5-qubit repeat-until-success reset is "
        "not padded (`codepack_5_1_3_quantinuum2022.py:160-166`), the "
        "same bug as `steane-5`: the auxiliary qubits get no idle during "
        "each prep attempt's reset. Class: static (`idle-padding`). "
        "Severity: M. Fix: stage 10."
    ),
    "surface-1": (
        "The surface code's logical H is the transversal H and a software "
        "rotation of the stored frame and history that leaves the patch's "
        "qubits unrelabelled (`codepack_surf17_tomita2014.py:439-491`), "
        "so a QEC round after H measures the old checks on a rotated "
        "state: Zero Prep, H, QEC, FT X Measure gives random outcomes "
        "where 0 is certain (the same word without QEC, or with the QEC "
        "after H only, is right). Case: `surface.surf10.truth.truth."
        "Zero_Prep-H-FT_Logical_X_Measure-QEC` (O1, O3, outcome). Class: "
        "truth, tracking (`gate`, `word`), independence. Severity: B. "
        "Fix: stage 6."
    ),
    "surface-2": (
        "The transversal CNOT and joint-parity circuits pair static "
        "qubit lists (`codepack_surf17_multipatch.py:196-202,393-408`) "
        "where a rotated patch needs its live data qubits. Latent: no "
        "check fails today, because the logical H leaves the patch "
        "qubits unrelabelled (`surface-1`) and H followed by the CNOT, "
        "with the H before the first QEC round, reads right in "
        "`mp-truth`. It surfaces once `surface-1` is fixed, as wrong "
        "parities after H then CNOT. Class: truth (`mp-truth` with H "
        "inputs). Severity: B once exposed. Fix: stage 6, with H."
    ),
    "surface-3": (
        "The deferred decoder assigns the updated patch into the incoming "
        "`PatchLayout` (`patches[patch_label] = new_patch`, "
        "`codepack_surf17_tomita2014.py:671`, and the H permutation, "
        "`:491`), so a history frame's `patches` re-reads later tracking "
        "data: after Zero Prep, X on D0 and QEC, the frame after the prep "
        "re-reads syndrome (1,0,0,0,1,0,0,0) where it held zeros "
        "(`surface.surf10.tracking.idle.*` O4). Class: tracking (O4 on "
        "every case with a QEC round; `truth` with QEC; `independence`). "
        "Severity: M. Fix: stage 6."
    ),
    "surface-4": (
        "A surface re-prep resets only the physical state: `Zero Prep` "
        "and `Plus Prep` are bare circuits (`:362`), so the patch's "
        "`syndrome_history_*` and frame survive and the next readout "
        "decodes the earlier run's rounds: two X errors with a QEC after "
        "each, `Zero Prep`, FT Z Measure gives 1 for some placements "
        "where 0 is certain (`surface.surf10.tracking.reuse.*`, outcome). "
        "The surgery CNOT's ancilla reuse is the same bug: a second "
        "`surgery-cnot` in one program, or a re-prep of its patches, "
        "gives a wrong outcome (`surface.surf10.truth.mp-truth."
        "surgery-cnot-Plus_Prep-Plus_Prep-QEC-x2`, outcome; `mp-reuse`, "
        "outcome and O3). Class: tracking (`reuse`, `mp-reuse`), truth "
        "(`mp-truth` x2). Severity: M. Fix: stage 6."
    ),
    "surface-5": (
        "A serialized-then-read surface code raises `NameError: name "
        "'Z_stabilizers' is not defined` in its FT measure: the global "
        "parity `apply_fn` closes over the local check lists "
        "(`codepack_surf17_tomita2014.py:548-549`, used at `:820`, "
        "`:937`), which a JSON or HDF5 round trip drops. Every config "
        "and format (`surface.*.static.serialization.*`). Class: static "
        "(`serialization`). Severity: M. Fixed upstream by #167 (merged in "
        "86d81a7)."
    ),
    "surface-6": (
        "The `clean_diff` and `guarded_diff` readouts take round 0 as "
        "their reference and skip it as a detector "
        "(`codepack_surf17_tomita2014.py:890`, `:1007`), so an error "
        "that precedes the first QEC round is never decoded: Zero Prep, "
        "X on D0, QEC, FT Z Measure gives 1 with either diff mode where "
        "0 is certain, and 0 in `raw` mode "
        "(`surface.surf10.truth.agreement.numpy-dict-errX-Zero_Prep-none-"
        "QEC-[12]`; the `Measure[12]` tracking cases). The modes are for "
        "readouts after a reference round (the surgery); on a plain "
        "patch they return a wrong answer silently. Class: tracking. "
        "Severity: M (opt-in kwargs). Fix: stage 6 (raise, or document)."
    ),
    "surface-7": (
        "The joint and surgery parity outcomes are corrected only by "
        "the patches' pending Pauli-frame bits "
        "(`codepack_surf17_multipatch.py` `decode_apply_fn`, "
        "`codepack_surf17_surgery.py:1473-1475`), but the deferred "
        "decoder keeps no frame until the final decode, so an error that "
        "a QEC round already detected still flips the parity: Zero Prep "
        "on both patches, X on D0, QEC, `joint-zz` or `surgery-zz-*`, "
        "reads parity 1 where 0 is certain (`surface.surf10.tracking."
        "mp-track.joint-zz-Zero_Prep-Zero_Prep-QEC-pre`, outcome, 12 of "
        "54 placements: X or Y on a Z-support qubit). Not predicted. "
        "Class: tracking (`mp-track`, `mp-reuse`). Severity: B for the "
        "surgery parity and the Bell prep and surgery CNOT built on it "
        "(a main path: noisy runs meet errors before the merge); M for "
        "the joint parity, documented non-FT. Fix: stage 6."
    ),
}

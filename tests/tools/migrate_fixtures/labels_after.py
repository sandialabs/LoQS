from loqs.core.instructions import InstructionLabel

label_kwargs_only = InstructionLabel("Increment", increment_by=2, patch_label="L0")
label_args_only = InstructionLabel("Increment", _legacy_inst_args=(5,))
tuple_four = {"instruction": "Increment", "increment_by": 3, "patch_label": "L0"}
tuple_three_global = {"instruction": "Init Counter", "_legacy_inst_args": (7,)}
already_modern = InstructionLabel("Increment", patch_label="L0", counter=5)
resolved_object_positional = InstructionLabel(some_instruction_obj, patch_label="L0")
nonliteral_kwargs = InstructionLabel("Increment", patch_label="L0", **extra_kwargs)
unresolvable_instruction = InstructionLabel("Nonexistent Instruction", patch_label="L0")
splat_call = InstructionLabel(*label_tuple)

from loqs.core.instructions import InstructionLabel

label_kwargs_only = InstructionLabel(
    "Increment", "L0", (), {"increment_by": 2}
)
label_args_only = InstructionLabel("Increment", None, (5,), {})
tuple_four = ("Increment", "L0", (), {"increment_by": 3})
tuple_three_global = ("Init Counter", None, (7,))
already_modern = InstructionLabel("Increment", patch_label="L0", counter=5)
resolved_object_positional = InstructionLabel(some_instruction_obj, "L0", (), {})
nonliteral_kwargs = InstructionLabel("Increment", "L0", (), extra_kwargs)
unresolvable_instruction = InstructionLabel("Nonexistent Instruction", "L0", (), {})
splat_call = InstructionLabel(*label_tuple)

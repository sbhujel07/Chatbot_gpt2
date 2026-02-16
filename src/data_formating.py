def format_input(entry_text):
  instruction_text = (
      f"Below is an instruction that describe the task."
      f"Write a response that appropriately completes the request."
      f"\n\n###Instruction:\n{entry_text["instruction"]}"
  )

  input_text = f"\n\n###Input:\n{entry_text["input"]}" if entry_text["input"] else ""

  return instruction_text + input_text



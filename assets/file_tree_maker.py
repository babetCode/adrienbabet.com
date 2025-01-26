str = '''regions
  axial
    skull
    spine
    pelvis
    thorax
  appendicular
    upper extremity
      upper arm
      forearm
      hand
    lower extremity
      thigh
      lower leg
      foot'''

def format_filetree(input_text):
    lines = input_text.splitlines()  # Split the input into lines
    stack = []  # Stack to track the levels of indentation
    result = ["{{< filetree/container >}}"]  # Start with the container
    base_indent = "    "  # Base indentation for all inner content

    for line in lines:
        stripped_line = line.lstrip()  # Word without indentation
        if not stripped_line:  # Skip empty lines
            continue

        indentation = len(line) - len(stripped_line)  # Calculate indentation level

        # Close open folders if the current indentation is less than the stack's last level
        while stack and stack[-1] >= indentation:
            result.append(base_indent + " " * stack.pop() + "{{< /filetree/folder >}}")

        # Add the folder to the result
        result.append(base_indent + " " * indentation + f"{{{{< filetree/folder name=\"{stripped_line}\" >}}}}")

        # Push the current indentation to the stack
        stack.append(indentation)

    # Close any remaining open folders
    while stack:
        result.append(base_indent + " " * stack.pop() + "{{< /filetree/folder >}}")

    result.append("{{< /filetree/container >}}")  # Close the container
    return "\n".join(result)

print(format_filetree(str))
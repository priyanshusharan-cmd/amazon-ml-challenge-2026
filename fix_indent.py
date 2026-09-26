import re

with open("src/blocking.py", "r") as f:
    text = f.read()

# First, remove the try: I just added so we can start fresh
text = text.replace("            try:\n                import sys", "            import sys")

# We know the block starts with "            import sys"
# and ends right before "            finally:"
# Everything in between needs to be indented by 4 spaces, and wrapped in a try:

start_idx = text.find("            import sys")
end_idx = text.find("            finally:")

if start_idx != -1 and end_idx != -1:
    block = text[start_idx:end_idx]
    # indent the block
    indented_block = "\n".join("    " + line if line.strip() else line for line in block.split("\n"))
    
    # add the try:
    final_block = "            try:\n" + indented_block
    
    new_text = text[:start_idx] + final_block + text[end_idx:]
    with open("src/blocking.py", "w") as f:
        f.write(new_text)
    print("Fixed indentation and try/finally block!")
else:
    print("Could not find start or end indices!")

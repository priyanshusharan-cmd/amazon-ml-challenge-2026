with open("tests/test_layer4_fork_equivalence.py", "r") as f:
    text = f.read()

import_injection = """# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
"""

text = text.replace('sys.path.insert(0, "/Users/priyanshusharan/Documents/Codex/2026-09-26/amazon-ml-2026")', import_injection)

with open("tests/test_layer4_fork_equivalence.py", "w") as f:
    f.write(text)
print("Fixed test paths")

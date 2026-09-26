with open("scripts/benchmark_india_val.py", "r") as f:
    text = f.read()

# Add project root to sys.path
import_injection = """import sys
from pathlib import Path

# Add project root to path so we can import src from anywhere
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
"""

if "PROJECT_ROOT" not in text:
    text = text.replace("import time", import_injection + "\nimport time")

# Replace hardcoded paths with config
old_paths = """    cleaned_india = Path("/Users/priyanshusharan/Documents/Codex/2026-09-26/amazon-ml-2026/artifacts/cleaned/India")
    val_india = Path("/Users/priyanshusharan/Documents/Codex/2026-09-26/amazon-ml-2026/artifacts/val_split/India")"""

new_paths = """    from src.config import config
    cleaned_india = config.CLEANED_DIR / "India"
    val_india = config.VAL_SPLIT_DIR / "India"
"""

text = text.replace(old_paths, new_paths)
text = text.replace("Path('/Users/priyanshusharan/Documents/Codex/2026-09-26/amazon-ml-2026/artifacts", "config.ARTIFACTS_DIR / Path('")

with open("scripts/benchmark_india_val.py", "w") as f:
    f.write(text)
print("Fixed benchmark_india_val.py")

import re

# Update CONTEXT.md
try:
    with open("CONTEXT.md", "r") as f:
        ctx = f.read()

    new_ctx = ctx + "\n\n## Windows Compatibility\nWindows is fully supported but operates sequentially (1 worker) during the memory-intensive Blocking phase to prevent memory serialization crashes. macOS and Linux use `fork` for full parallelization.\n\n"
    new_ctx = new_ctx + "### Setup & Benchmark on Windows\n```powershell\npython -m venv .venv\n.\\.venv\\Scripts\\activate\npip install -r requirements.txt\npython scripts\\benchmark_india_val.py\n```\n"

    with open("CONTEXT.md", "w") as f:
        f.write(new_ctx)
except FileNotFoundError:
    pass

# Update OPERATOR_GUIDE.md
try:
    with open("OPERATOR_GUIDE.md", "r") as f:
        guide = f.read()

    new_guide = guide + "\n\n## Execution Commands\n\n### macOS / Linux\n```bash\n.venv/bin/python scripts/benchmark_india_val.py\n```\n\n### Windows\n```powershell\n.\\.venv\\Scripts\\python scripts\\benchmark_india_val.py\n```\n\n**Note for Windows:** The retrieval pipeline will dynamically fall back to sequential execution (`W=1`) to prevent Out-of-Memory crashes associated with `spawn` serialization.\n"

    with open("OPERATOR_GUIDE.md", "w") as f:
        f.write(new_guide)
except FileNotFoundError:
    pass

print("Docs updated")

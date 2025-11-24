import os
import sys

# Make the project root importable so sibling packages (e.g. `kxor`) can be imported.

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Now you can do: from kxor import some_module
# (Better long-term: make the repo a package or `pip install -e .` and use relative imports.)
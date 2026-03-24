from __future__ import annotations

import runpy
from pathlib import Path


generator_path = Path(__file__).with_name("generate_notebooks.py")
runpy.run_path(str(generator_path), run_name="__main__")
print("Rebuilt notebooks from the current generator.")

"""Configure the private robosuite environment; job stdout captures its logs."""
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec('robosuite')
root = Path(next(iter(spec.submodule_search_locations)))
private = root / 'macros_private.py'
if not private.exists():
    private.write_text('from robosuite.macros import *\nFILE_LOGGING_LEVEL = None\n')
print('Robosuite runtime configuration:', private)

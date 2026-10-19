"""Compatibility import for the preserved numbered preprocessing source."""
import importlib.util
from pathlib import Path
_spec=importlib.util.spec_from_file_location('_numbered_preprocess',Path(__file__).with_name('01_preprocess.py'))
_module=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_module)
uppercase_ratio=_module.uppercase_ratio

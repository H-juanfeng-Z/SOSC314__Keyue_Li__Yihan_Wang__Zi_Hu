"""Compatibility import; original feature definitions remain unchanged."""
import importlib.util
from pathlib import Path
_spec=importlib.util.spec_from_file_location('_numbered_features',Path(__file__).with_name('02_feature_extract_singlequestion.py'))
_module=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_module)
extract_features=_module.extract_features

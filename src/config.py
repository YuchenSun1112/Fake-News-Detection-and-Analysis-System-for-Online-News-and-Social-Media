from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_ROOT_CONFIG = Path(__file__).resolve().parents[1] / "config.py"
_SPEC = spec_from_file_location("_project_root_config", _ROOT_CONFIG)

if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load project config from {_ROOT_CONFIG}")

_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

for _name in dir(_MODULE):
    if _name.isupper():
        globals()[_name] = getattr(_MODULE, _name)


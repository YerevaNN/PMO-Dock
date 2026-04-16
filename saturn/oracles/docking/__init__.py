# Re-export DockingVina from the repo-root oracles (shared implementation).
# When Saturn runs, "oracles" is saturn/oracles (first on path), so oracles.docking
# is this package; callers expect "from oracles.docking import DockingVina" to work.
import os
import sys
import importlib.util

_repo_root = os.environ.get("PROJECT_ROOT") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_root_docking_path = os.path.join(_repo_root, "docking_oracle", "docking.py")
_spec = importlib.util.spec_from_file_location("_root_oracles_docking", _root_docking_path)
_root_docking = importlib.util.module_from_spec(_spec)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
_spec.loader.exec_module(_root_docking)

DockingVina = _root_docking.DockingVina

__all__ = ["DockingVina"]
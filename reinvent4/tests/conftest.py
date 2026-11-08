"""Let the PMODock tests run without a full REINVENT4 install.

``comp_pmodock`` imports two things from REINVENT4's own half of the
``reinvent_plugins.components`` namespace package: ``add_tag`` (a 3-line class decorator)
and ``ComponentResults`` (a plain dataclass). When REINVENT4 is installed those resolve
normally and the tests exercise the real API. When it is not -- the usual case on a CI
box or a laptop, since REINVENT4 needs torch and a 700 MB checkout -- we stand in
equivalents taken from upstream v4.5.11 so the budget logic stays testable.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PLUGIN_DIR = os.path.join(PROJECT_ROOT, "reinvent4", "plugins")
for path in (PROJECT_ROOT, PLUGIN_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


def _have_real_reinvent() -> bool:
    try:
        return importlib.util.find_spec("reinvent") is not None
    except (ImportError, ValueError):
        return False


def _install_stubs() -> None:
    import reinvent_plugins.components as components

    if not hasattr(components, "__path__"):  # pragma: no cover - namespace pkg always has one
        return

    add_tag_mod = types.ModuleType("reinvent_plugins.components.add_tag")

    def add_tag(label: str, text: str = "True"):
        def wrapper(cls):
            setattr(cls, label, text)
            return cls

        return wrapper

    add_tag_mod.add_tag = add_tag

    results_mod = types.ModuleType("reinvent_plugins.components.component_results")

    @dataclass
    class ComponentResults:
        scores: List[np.ndarray]
        scores_properties: Optional[List[Dict]] = None
        uncertainty: Optional[List[np.ndarray]] = None
        uncertainty_type: Optional[str] = None
        uncertainty_properties: Optional[List[Dict]] = None
        failures_properties: Optional[List[Dict]] = None
        metadata: Optional[Dict] = None

    results_mod.ComponentResults = ComponentResults

    sys.modules.setdefault("reinvent_plugins.components.add_tag", add_tag_mod)
    sys.modules.setdefault("reinvent_plugins.components.component_results", results_mod)


if not _have_real_reinvent():
    _install_stubs()

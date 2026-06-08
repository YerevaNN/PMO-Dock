"""Shim: repo-root imports resolve to the installable benchmark client."""
from benchmark.docking_oracle.docking_vina_client import DockingVinaClient

__all__ = ["DockingVinaClient"]

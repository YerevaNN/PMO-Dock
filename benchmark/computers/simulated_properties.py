from __future__ import annotations

import logging
from functools import cache


# IMPORTANT:
# `docking_oracle.docking` imports OpenBabel (pybel). On some nodes this can fail due
# to missing system libs. We only need OpenBabel when we actually run local docking;
# importing benchmark computers should not hard-fail for remote-oracle runs.
#
# So we provide lazy wrappers that import `docking_oracle.docking` only when called.

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DockingOracle:
    def __new__(cls, *args, **kwargs):
        from benchmark.docking_oracle.docking import DockingOracle as _DockingOracle

        return _DockingOracle(*args, **kwargs)


@cache
def quickvina_predictor(target: str, exhaustiveness: int | None = None):
    from benchmark.docking_oracle.docking import quickvina_predictor as _quickvina_predictor

    return _quickvina_predictor(target, exhaustiveness=exhaustiveness)


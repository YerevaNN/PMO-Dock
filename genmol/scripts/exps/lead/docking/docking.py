# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This file re-exports the shared local DockingVina from the repo-root oracles package.
# Original MOOD-based implementation replaced by shared oracles.docking (Even-More-PMO).
# ---------------------------------------------------------------

import os
import sys

_lead_docking_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_lead_docking_dir)))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
from benchmark.docking_oracle.docking import DockingVina

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

"""
Adapted from GEAM: https://openreview.net/forum?id=sLGliHckR8
https://anonymous.4open.science/r/GEAM-45EF/utils_sac/docking.py
https://anonymous.4open.science/r/GEAM-45EF/utils_sac/utils.py
"""
from typing import Tuple
import os
import sys
import numpy as np
from oracles.oracle_component import OracleComponent
from oracles.dataclass import OracleComponentParameters
from rdkit import Chem
from rdkit.Chem import Mol



# Repo root for shared oracles.docking
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
from oracles.docking import DockingVina

def run_vina(
    smis: np.ndarray[str], 
    predictor: DockingVina
) -> Tuple[np.ndarray[float], np.ndarray[float]]:
    raw_docking_scores = np.array(predictor.predict(smis))
    rewards = np.clip(raw_docking_scores, 0, None)
    return raw_docking_scores, rewards


class QuickVina2(OracleComponent):
    """
    QuickVina2 docking. 
    """
    def __init__(self, parameters: OracleComponentParameters):
        super().__init__(parameters)
        self.vina_oracle = DockingVina(parameters.specific_parameters["target"])
        
    def __call__(self, mols: np.ndarray[Mol]) -> np.ndarray[float]:
        smiles = np.vectorize(Chem.MolToSmiles)(mols)
        return self._compute_property(smiles)
    
    def _compute_property(
        self,
        smiles: np.ndarray[Mol],
    ) -> np.ndarray[float]:
        """
        Returns the QuickVina2 docking scores.
        """
        raw_vina = np.array(self.vina_oracle.predict(smiles))
        return raw_vina

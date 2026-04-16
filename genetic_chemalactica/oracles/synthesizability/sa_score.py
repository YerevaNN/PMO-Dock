import numpy as np
from rdkit.Chem import Mol

from genetic_chemalactica.oracles.synthesizability.sascorer import calculateScore

class SAScore:
    """Synthetic Accessibility score (RDKit Mol -> float)."""

    def __call__(self, mols: np.ndarray[Mol]) -> np.ndarray[float]:
        return np.vectorize(self._compute_property)(mols)

    def _compute_property(self, mol: Mol) -> float:
        """
        Wrapper function in case of exceptions.
        """
        try:
            return calculateScore(mol)
        except Exception:
            return 0.0

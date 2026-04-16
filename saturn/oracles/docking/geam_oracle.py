"""
Adapted from GEAM: https://openreview.net/forum?id=sLGliHckR8
https://anonymous.4open.science/r/GEAM-45EF/utils_sac/docking.py
https://anonymous.4open.science/r/GEAM-45EF/utils_sac/utils.py

The implementation below is based on the above code-base for fair and exact comparison with GEAM.
"""
from typing import Tuple, List
import os
import sys
from shutil import rmtree
import subprocess
import threading
import numpy as np
from oracles.oracle_component import OracleComponent
from oracles.dataclass import OracleComponentParameters
from rdkit import Chem
from rdkit.Chem import Mol, QED
from openbabel import pybel
from tdc import Oracle
# SA Score
from oracles.synthesizability.sascorer import calculateScore
import requests
import logging
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from docking_oracle.docking import DockingVina
from docking_oracle.docking_vina_client import DockingVinaClient
from utils.tasks import task_name2constraints, select_sigma
from utils.rewards import hit_reward



def reward_vina(
    smis: np.ndarray[str],
    predictor,  # Can be DockingVina or DockingVinaClient (both have predict method)
    seed: int = 0,
) -> Tuple[np.ndarray[float], np.ndarray[float]]:
    # Antitarget multi-seed aggregation (3× max/mean) is implemented in ``docking_oracle.docking.DockingVina.predict``.
    raw_docking_scores = -np.array(predictor.predict(smis, seed=seed))
    rewards = np.clip(raw_docking_scores, 0, None)
    logging.debug("reward_vina: n=%d", len(smis))
    return raw_docking_scores, rewards

def reward_pmo(
    smis: np.ndarray[str],
    target: str
) -> np.ndarray[float]:
    return np.array([Oracle(target)(mol) for mol in smis])

def reward_qed(
    mols: np.ndarray[Mol]
) -> np.ndarray[float]:
    return np.array([QED.qed(m) for m in mols])

def reward_sim(lead_fp, mols):
    mol_fps = [Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048) for mol in mols]
    return Chem.AllChem.DataStructs.BulkTanimotoSimilarity(lead_fp, mol_fps)

def reward_sa(
    mols: np.ndarray[Mol]
) -> Tuple[np.ndarray[float], np.ndarray[float]]:
    raw_sa = np.array([calculateScore(m) for m in mols])
    sa_rewards = np.array([(10 - raw_score) / 9 for raw_score in raw_sa])
    return raw_sa, sa_rewards




class GEAMOracle(OracleComponent):
    """
    GEAM's Oracle that combines 3 individual Oracle Components:
        1. QuickVina 2 Docking
        2. QED
        3. SA Score
    """
    def __init__(self, parameters: OracleComponentParameters):
        super().__init__(parameters)
        # Extract target, handling both string and list cases
        if parameters.specific_parameters["target"] in ["6nzp_7uyt", "6nzp_5ut5", "6nzp_7uyw"]:
            target = "6nzp"
            self.antitarget = parameters.specific_parameters["target"].split("_")[-1]
            self.vina_exhaustiveness = 8
        else:
            self.antitarget = None
            target = parameters.specific_parameters["target"] 
            self.vina_exhaustiveness = 1
        self.target = target if isinstance(target, str) else target[0] if isinstance(target, list) and len(target) > 0 else str(target)
        logging.debug("GEAMOracle target=%s exhaustiveness=%s", self.target, self.vina_exhaustiveness)
        # self.vina_exhaustiveness = parameters.vina_exhaustiveness
        # self.vina_seed = parameters.vina_seed
        # Check if docking Vina service URL is configured (from specific_parameters or DOCKING_VINA_URL)
        oracle_service_url = parameters.specific_parameters.get("oracle_url")
        if oracle_service_url is None:
            oracle_service_url = os.environ.get("DOCKING_VINA_URL")
        # Ensure oracle_service_url is a string if it exists
        if oracle_service_url and isinstance(oracle_service_url, list):
            oracle_service_url = oracle_service_url[0] if len(oracle_service_url) > 0 else None
        if oracle_service_url:
            self.vina_oracle = DockingVinaClient(str(oracle_service_url), self.target)
            if self.antitarget:
                self.vina_oracle_antitarget = DockingVinaClient(str(oracle_service_url), self.antitarget)
            else:
                self.vina_oracle_antitarget = None
        else:
            self.vina_oracle = DockingVina(self.target, exhaustiveness=self.vina_exhaustiveness)
            if self.antitarget:
                self.vina_oracle_antitarget = DockingVina(self.antitarget, exhaustiveness=self.vina_exhaustiveness)
            else:
                self.vina_oracle_antitarget = None
    def __call__(self, mols: np.ndarray[Mol]) -> np.ndarray[float]:
        # Filter out None molecules and convert to SMILES
        valid_mols = []
        valid_smiles = []
        for mol in mols:
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)
                valid_mols.append(mol)
                valid_smiles.append(smiles)
            else:
                raise ValueError("Invalid molecule (None) passed to GEAMOracle")
        
        return self._compute_property(np.array(valid_smiles), np.array(valid_mols))
    
    def _compute_property(
        self,
        smiles: np.ndarray[str],
        mols: np.ndarray[Mol],
    ) -> Tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
        """Run GEAM's Oracle and return the aggregated reward."""
        t0 = perf_counter()
        n = len(smiles)

        if self.target in ["jnk3", "drd2", "gsk3b"]:
            raw_ds = reward_pmo(smiles, self.target)
            ds_rewards = raw_ds
        else:
            raw_ds, ds_rewards = reward_vina(smiles, self.vina_oracle)

        qed_rewards = reward_qed(mols)
        raw_sa, sa_rewards = reward_sa(mols)

        if self.antitarget in ["7uyt", "5ut5", "7uyw"]:
            raw_ds_antitarget, ds_rewards_antitarget = reward_vina(smiles, self.vina_oracle_antitarget)
            ds_rewards_antitarget = (np.clip(ds_rewards_antitarget, 0, 20) / 20)
            ds_target_values = (np.clip(ds_rewards, 0, 20) / 20)
            specificity_reward = np.clip(ds_target_values - ds_rewards_antitarget, 0, 1)
            aggregated_rewards = specificity_reward * ds_target_values * qed_rewards * sa_rewards
            raw_ds_antitarget[raw_ds_antitarget == -99.9] = 99.9
            logging.info("GEAMOracle dual-target n=%d %.1fs", n, perf_counter() - t0)
            return raw_ds, raw_ds_antitarget, qed_rewards, raw_sa, aggregated_rewards

        if self.parameters.lead_smiles:
            lead_fp = Chem.AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(self.lead_smiles), 2, 2048)
            raw_similarity = reward_sim(lead_fp, mols)
            aggregated_rewards = (np.clip(ds_rewards, 0, 20) / 20) * qed_rewards * sa_rewards * raw_similarity
            raw_ds[raw_ds == -99.9] = 99.9
            logging.info("GEAMOracle lead n=%d %.1fs", n, perf_counter() - t0)
            return raw_ds, qed_rewards, raw_sa, raw_similarity, aggregated_rewards

        aggregated_rewards = (np.clip(ds_rewards, 0, 20) / 20) * qed_rewards * sa_rewards
        raw_ds[raw_ds == -99.9] = 99.9
        logging.info("GEAMOracle single n=%d %.1fs", n, perf_counter() - t0)
        return raw_ds, qed_rewards, raw_sa, aggregated_rewards


class HITOracle(OracleComponent):
    """
    HIT's Oracle that combines 3 individual Oracle Components:
        1. QuickVina 2 Docking
        2. QED
        3. SA Score
    """
    def __init__(self, parameters: OracleComponentParameters):
        super().__init__(parameters)
        # Extract target, handling both string and list cases
        self.antitarget = None
        if parameters.specific_parameters["target"] in ["6nzp_7uyt", "6nzp_5ut5", "6nzp_7uyw"]:
            target = "6nzp"
            self.antitarget = parameters.specific_parameters["target"].split("_")[-1]
        else:
            target = parameters.specific_parameters["target"] 
        self.target = target if isinstance(target, str) else target[0] if isinstance(target, list) and len(target) > 0 else str(target)
        # Check if docking Vina service URL is configured (from specific_parameters or DOCKING_VINA_URL)
        oracle_service_url = parameters.specific_parameters.get("oracle_url")
        if oracle_service_url is None:
            oracle_service_url = os.environ.get("DOCKING_VINA_URL")
        # Ensure oracle_service_url is a string if it exists
        if oracle_service_url and isinstance(oracle_service_url, list):
            oracle_service_url = oracle_service_url[0] if len(oracle_service_url) > 0 else None
        if oracle_service_url:
            self.vina_oracle = DockingVinaClient(str(oracle_service_url), self.target)
            if self.antitarget is not None:
                self.vina_oracle_antitarget = DockingVinaClient(str(oracle_service_url), self.antitarget)
        else:
            self.vina_oracle = DockingVina(self.target)
            if self.antitarget is not None:
                self.vina_oracle_antitarget = DockingVina(self.antitarget)
        # Determine task name based on target
        if self.target in ["jnk3", "drd2", "gsk3b"]:
            task_name = "hit.pmo"
        elif self.target == "6nzp":
            task_name = f"spec.{target}"
        else:
            task_name = f"hit.{self.target}"
        
        # Get hit ranges for this task
        try:
            constraints = task_name2constraints(task_name)
            self.hit_ranges = {
                "docking_score": constraints.get("docking_score", [0, 20]),
                "antitarget_docking_score": constraints.get("antitarget_docking_score", [0, 20]),
                "qed_score": constraints.get("qed_score", [0.5, 1.0]),
                "sa_score": constraints.get("sa_score", [1.0, 5.0])
            }
        except KeyError:
            # Default ranges if task not found
            self.hit_ranges = {
                "docking_score": [0, 20],
                "antitarget_docking_score": [0, 20],
                "qed_score": [0.5, 1.0],
                "sa_score": [1.0, 5.0]
            }

    
    def __call__(self, mols: np.ndarray[Mol]) -> np.ndarray[float]:
        # Filter out None molecules and convert to SMILES
        valid_mols = []
        valid_smiles = []
        for mol in mols:
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)
                valid_mols.append(mol)
                valid_smiles.append(smiles)
            else:
                raise ValueError("Invalid molecule (None) passed to HITOracle")
        
        return self._compute_property(np.array(valid_smiles), np.array(valid_mols))
    
    def _compute_property(
        self,
        smiles: np.ndarray[str],
        mols: np.ndarray[Mol],
    ) -> Tuple[np.ndarray[float], ...]:
        """Run HIT's Oracle and return the aggregated reward using hit_reward."""
        t0 = perf_counter()
        n = len(smiles)
        is_pmo = self.target in ["jnk3", "drd2", "gsk3b"]

        if is_pmo:
            raw_ds = reward_pmo(smiles, self.target)
            ds_values = raw_ds.copy()
        else:
            raw_ds, _ = reward_vina(smiles, self.vina_oracle)
            ds_values = raw_ds.copy()
            ds_values[raw_ds == -99.9] = 99.9

        ds_values_antitarget = None
        if self.antitarget is not None:
            raw_ds_antitarget, _ = reward_vina(smiles, self.vina_oracle_antitarget)
            ds_values_antitarget = raw_ds_antitarget.copy()
            ds_values_antitarget[raw_ds_antitarget == -99.9] = 99.9

        qed_values = reward_qed(mols)
        raw_sa, sa_rewards = reward_sa(mols)
        sa_values = sa_rewards

        docking_sigma = select_sigma("DOCKING") or 0.1 * 20
        qed_sigma = select_sigma("QED") or 0.1 * 1
        sa_sigma = 0.1 * 1

        aggregated_rewards = []
        for i in range(len(mols)):
            ds_val = None if (not is_pmo and ds_values[i] == 99.9) else ds_values[i]
            if ds_values_antitarget is not None:
                ds_val_antitarget = None if (not is_pmo and ds_values_antitarget[i] == 99.9) else ds_values_antitarget[i]
                measured = [ds_val, ds_val_antitarget, qed_values[i], sa_values[i]]
                sigmas = [docking_sigma, docking_sigma, qed_sigma, sa_sigma]
                hit_ranges_list = [
                    self.hit_ranges["docking_score"],
                    self.hit_ranges["antitarget_docking_score"],
                    self.hit_ranges["qed_score"],
                    self.hit_ranges["sa_score"]
                ]
            else:
                measured = [ds_val, qed_values[i], sa_values[i]]
                sigmas = [docking_sigma, qed_sigma, sa_sigma]
                hit_ranges_list = [
                    self.hit_ranges["docking_score"],
                    self.hit_ranges["qed_score"],
                    self.hit_ranges["sa_score"]
                ]
            reward = hit_reward(measured, sigmas, hit_ranges_list, prod=True, avg=False)
            aggregated_rewards.append(reward)

        aggregated_rewards = np.array(aggregated_rewards)

        if self.parameters.lead_smiles:
            lead_fp = Chem.AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(self.parameters.lead_smiles), 2, 2048)
            raw_similarity = reward_sim(lead_fp, mols)
            similarity_sigma = select_sigma("SIMILAR") or 0.1 * 1
            similarity_range = [0.4, 1.0]  # Default similarity range
            
            # Combine with similarity
            final_rewards = []
            for i in range(len(mols)):
                ds_val = None if (not is_pmo and ds_values[i] == 99.9) else ds_values[i]
                measured = [ds_val, qed_values[i], sa_values[i], raw_similarity[i]]
                sigmas = [docking_sigma, qed_sigma, sa_sigma, similarity_sigma]
                hit_ranges_list = [
                    self.hit_ranges["docking_score"],
                    self.hit_ranges["qed_score"],
                    self.hit_ranges["sa_score"],
                    similarity_range
                ]
                reward = hit_reward(measured, sigmas, hit_ranges_list, prod=True, avg=False)
                final_rewards.append(reward)
            
            aggregated_rewards = np.array(final_rewards)
            if not is_pmo:
                raw_ds[raw_ds == 99.9] = 99.9
            logging.info("HITOracle lead n=%d %.1fs", n, perf_counter() - t0)
            return raw_ds, qed_values, raw_sa, raw_similarity, aggregated_rewards

        if self.antitarget in ["7uyt", "5ut5", "7uyw"]:
            # Rewards from raw (0-20 scale): clip then cap at 20 and normalize
            ds_rewards_antitarget = np.clip(np.clip(ds_values_antitarget, 0, None), 0, 20) / 20
            ds_target_values = np.clip(np.clip(ds_values, 0, None), 0, 20) / 20
            ds_target_values[ds_values == 99.9] = 0
            ds_rewards_antitarget[ds_values_antitarget == 99.9] = 0
            specificity_reward = np.clip(ds_target_values - ds_rewards_antitarget, 0, 1)
            aggregated_rewards = specificity_reward * ds_target_values * qed_values * sa_rewards
            raw_ds_antitarget = ds_values_antitarget.copy()
            logging.info("HITOracle dual n=%d %.1fs", n, perf_counter() - t0)
            return raw_ds, raw_ds_antitarget, qed_values, raw_sa, aggregated_rewards

        if not is_pmo:
            raw_ds[raw_ds == -99.9] = 99.9
        logging.info("HITOracle single n=%d %.1fs", n, perf_counter() - t0)
        return raw_ds, qed_values, raw_sa, aggregated_rewards
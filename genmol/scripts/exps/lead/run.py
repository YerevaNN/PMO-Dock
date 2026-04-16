# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import sys
from copy import deepcopy
# Add the GenMol root directory to Python path to import both genmol and scripts modules
genmol_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
sys.path.append(os.path.join(genmol_root, 'src'))  # for genmol module
sys.path.append(genmol_root)  # for scripts module
# Add project root to path for utils imports
project_root = os.path.dirname(genmol_root)
sys.path.insert(0, project_root)

from time import time, sleep
import random
import argparse
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs, AllChem, QED, RDConfig
from omegaconf import OmegaConf

from utils.tasks import task_name2constraints, select_sigma, guassian_modifier

from scripts.exps.lead.docking.docking import DockingVina
from genmol.sampler import Sampler
from genmol.utils.utils_chem import cut
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
import requests
from typing import List
import logging

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

def setup_logging(log_file=None):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Remove all existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        # Ensure log file directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Create file handler with immediate flushing
        try:
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setLevel(logging.INFO)
            # Set formatter for file handler
            formatter = logging.Formatter(log_format, datefmt=date_format)
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)
        except Exception as e:
            # If file handler creation fails, log to stderr and continue with stdout only
            print(f"Warning: Could not create log file handler for {log_file}: {e}", file=sys.stderr)
    
    # Configure root logger - use force=True if available (Python 3.8+)
    try:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            datefmt=date_format,
            handlers=handlers,
            force=True  # Force reconfiguration even if already configured
        )
    except TypeError:
        # Python < 3.8 doesn't support force parameter
        # Manually configure the root logger
        root_logger.setLevel(logging.INFO)
        formatter = logging.Formatter(log_format, datefmt=date_format)
        for handler in handlers:
            handler.setFormatter(formatter)
            root_logger.addHandler(handler)
    
    # Return root logger to ensure all logging calls work
    logger = logging.getLogger()
    logger.info(f"Logging initialized. Log file: {log_file if log_file else 'stdout only'}")
    # Force flush to ensure message is written immediately
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.flush()
    return logger

class DockingVinaClient:
    """HTTP client for DockingVina oracle service"""
    
    def __init__(self, service_url: str, target: str):
        # Ensure service_url has a scheme (http:// or https://)
        service_url = service_url.rstrip('/')
        if not service_url.startswith(('http://', 'https://')):
            service_url = f'http://{service_url}'
        self.service_url = service_url
        self.target = target
        self.predict_url = f"{self.service_url}/predict/{target}"
        
    def predict(self, smiles_list: List[str]) -> List[float]:
        """
        Predict docking scores for a list of SMILES strings via HTTP request.
        Implements retry logic: up to 5 retries with 5-minute timeout per request.
        Returns 0.0 only for molecules where docking() computation failed (server returns 99.9).
        Raises exception after all retries fail.
        
        Args:
            smiles_list: List of SMILES strings (can be numpy array)
            
        Returns:
            List of docking scores (affinities). Docking computation errors (99.9) are assigned 0.0.
            
        Raises:
            requests.exceptions.RequestException: After 5 failed retry attempts
        """
        # Convert numpy array to list if needed (numpy arrays are not JSON serializable)
        if isinstance(smiles_list, np.ndarray):
            smiles_list = smiles_list.tolist()
        elif not isinstance(smiles_list, list):
            smiles_list = list(smiles_list)
        
        max_retries = 5
        request_timeout = 300  # 5 minutes per request
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.predict_url,
                    json={'smiles': smiles_list},
                    timeout=request_timeout
                )
                response.raise_for_status()
                result = response.json()
                scores = result['scores']
                
                # Replace any error scores (99.9) with 0.0
                # This happens when docking() computation fails for a molecule
                scores = [0.0 if score == 99.9 else score for score in scores]
                
                return scores
                
            except requests.exceptions.Timeout as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logging.warning(f"Request timeout (attempt {attempt + 1}/{max_retries}) for {self.target} "
                                  f"at {self.predict_url}. Retrying in 2 seconds...")
                    sleep(2)  # Wait 2 seconds before retry
                    continue
                else:
                    error_msg = (f"Request timeout after {max_retries} attempts for {self.target} "
                               f"at {self.predict_url}. All retries exhausted.")
                    logging.error(error_msg)
                    raise requests.exceptions.Timeout(error_msg) from e
                    
            except requests.exceptions.ConnectionError as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logging.warning(f"Connection error (attempt {attempt + 1}/{max_retries}) for {self.target} "
                                  f"at {self.service_url}. Retrying in 2 seconds...")
                    sleep(2)  # Wait 2 seconds before retry
                    continue
                else:
                    error_msg = (f"Failed to connect to oracle service at {self.service_url} after {max_retries} attempts. "
                               f"Please ensure the service is running and accessible. "
                               f"Error: {e}")
                    logging.error(error_msg)
                    raise requests.exceptions.ConnectionError(error_msg) from e
                    
            except requests.exceptions.RequestException as e:
                last_exception = e
                # Check if response contains error information
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_data = e.response.json()
                        if 'error' in error_data:
                            logging.error(f"Oracle service error for {self.target} at {self.predict_url}: {error_data['error']}")
                    except:
                        pass
                
                if attempt < max_retries - 1:
                    logging.warning(f"Request error (attempt {attempt + 1}/{max_retries}) for {self.target} "
                                  f"at {self.predict_url}: {e}. Retrying in 2 seconds...")
                    sleep(2)  # Wait 2 seconds before retry
                    continue
                else:
                    error_msg = (f"Oracle service error after {max_retries} attempts for {self.target} "
                               f"at {self.predict_url}: {e}")
                    logging.error(error_msg)
                    raise requests.exceptions.RequestException(error_msg) from e
                    
            except Exception as e:
                last_exception = e
                # Unexpected error
                if attempt < max_retries - 1:
                    logging.warning(f"Unexpected error (attempt {attempt + 1}/{max_retries}) for {self.target}: {e}. Retrying in 2 seconds...")
                    sleep(2)
                    continue
                else:
                    logging.error(f"Unexpected error after {max_retries} attempts for {self.target}: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
        
        # Should never reach here, but just in case
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError(f"All {max_retries} retry attempts failed for {self.target} without exception.")

class GenMolOpt():
    def __init__(self, args):
        super().__init__()
        self.args = args
        # df = pd.read_csv('scripts/exps/lead/docking/actives.csv')
        # Get the directory where the current script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if self.args.tox:
            print('Running TOX optimization')
            self.start_smiles = "CC/C(=C(/CC)c1ccc(O)cc1)c1ccc(O)cc1"
            start_mol = Chem.MolFromSmiles(self.start_smiles)
            self.start_fp = AllChem.GetMorganFingerprintAsBitVect(start_mol, 2, 2048)
            self.start_prop = 1.0
        else:
            # Load starting molecule from actives (used for both seed and random pool)
            df = pd.read_csv(os.path.join(script_dir, 'docking', 'actives.csv'))
            df = df[df['target'] == self.args.oracle_name]
            self.start_smiles = df['smiles'].iloc[self.args.start_mol_idx - 1]
            start_mol = Chem.MolFromSmiles(self.start_smiles)
            self.start_fp = AllChem.GetMorganFingerprintAsBitVect(start_mol, 2, 2048)
            self.start_prop = df['DS'].iloc[self.args.start_mol_idx - 1]
            if self.args.pool == 'seed':
                print(f'Start SMILES:\t{self.start_smiles}')
                print(f'Start DS:\t{self.start_prop}')

        # Check if oracle service URL is configured (from env var or config)
        if self.args.get('oracle_url') is not None:
            oracle_service_url = self.args.get('oracle_url')
        else:
            oracle_service_url = os.environ.get("DOCKING_VINA_URL")
        if oracle_service_url:
            self.predictor = DockingVinaClient(oracle_service_url, self.args.oracle_name)
        else:
            self.predictor = DockingVina(self.args.oracle_name)
        self.population = [(self.start_prop, frag) for frag in cut(self.start_smiles)]
        print(f'Initial population: {len(self.population)} frags')
        self.sampler = Sampler(self.args.model_path)

        reslut_dir = self.args.log_dir
        os.makedirs(reslut_dir, exist_ok=True)       
        self.fname = f'{reslut_dir}/results.csv'
        print(f'\033[92m{self.fname}\033[0m')

    def reward_vina(self, smiles_list):
        reward = - np.array(self.predictor.predict(smiles_list))
        reward = np.clip(reward, 0, None)
        return list(reward)
    
    def reward_qed(self, mols):
        """Return QED; invalid mols get zero to keep list alignment."""
        qed_scores = []
        for mol in mols:
            if mol is None:
                qed_scores.append(0.0)
                continue
            try:
                qed_scores.append(QED.qed(mol))
            except Exception:
                qed_scores.append(0.0)
        return qed_scores
    
    def reward_sa(self, mols):
        """Return normalized SA score; invalid mols default to zero."""
        sa_scores = []
        for mol in mols:
            if mol is None:
                sa_scores.append(0.0)
                continue
            try:
                sa = sascorer.calculateScore(mol)
            except Exception:
                sa_scores.append(0.0)
                continue
            sa_scores.append((10 - sa) / 9)
        return sa_scores
    
    def reward_sim(self, mols):
        """Return Tanimoto similarity; invalid mols get zero similarity."""
        similarities = []
        for mol in mols:
            if mol is None:
                similarities.append(0.0)
                continue
            try:
                mol_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
                similarities.append(DataStructs.TanimotoSimilarity(self.start_fp, mol_fp))
            except Exception:
                similarities.append(0.0)
        return similarities

    def reward_tox(self, mols):
        mol_objs = [Chem.MolFromSmiles(s) for s in mols]
        rs = self.reward_sa(mol_objs)
        rsim = self.reward_sim(mol_objs)
        return rs, rsim

    def hit_reward(self, smiles_list):
        molecule_rewards = []
        for i in range(len(prop_values['docking_score'])):
            smiles_reward = []
            for prop, rnge in task_format.items():
                sigma = select_sigma(prop)
                m = prop_values[prop][i]
                if m is None:
                    smiles_reward.append(0)
                elif rnge[0] <= m <= rnge[1]:
                    smiles_reward.append(1)
                else:
                    dist = min(abs(m - rnge[0]), abs(m - rnge[1]))
                    reward = guassian_modifier(dist, mu=0, sigma=sigma)
                    smiles_reward.append(reward)
            # Combine rewards across properties for this molecule
            if prod:
                molecule_rewards.append(np.array(smiles_reward).prod().item())
            elif avg:
                molecule_rewards.append(np.array(smiles_reward).mean().item())
            else:
                molecule_rewards.append(np.array(smiles_reward))
        return molecule_rewards

    def geam_reward(self, prop_lists: list[list[float]]) -> list[float]:
        return list(np.prod(prop_lists, axis=0))
        
    def reward(self, smiles_list):
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        rv = self.reward_vina(smiles_list)
        rq = self.reward_qed(mols)
        rs = self.reward_sa(mols)
        rsim = self.reward_sim(mols)
        return rv, rq, rs, rsim
    
    def attach(self, frag1, frag2):
        rxn = AllChem.ReactionFromSmarts('[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]')
        mols = rxn.RunReactants((Chem.MolFromSmiles(frag1), Chem.MolFromSmiles(frag2)))
        idx = np.random.randint(len(mols))
        return mols[idx][0]
    
    def update_population(self, smiles_list, prop_list, reward=None):
        rv_list, rq_list, rs_list, rsim_list = prop_list
        for rv, rq, rs, rsim, smiles in zip(rv_list, rq_list, rs_list, rsim_list, smiles_list):
            if rv > self.start_prop and rq >= 0.6 and rs >= 6/9 and rsim >= self.args.sim_thr:
                frags = {frag for frag in cut(smiles)}
                self.population.extend([(rv, frag) for frag in frags])
        self.population.sort(reverse=True)

    def update_population_tox(self, smiles_list, prop_list):
        rs_list, rsim_list = prop_list
        for rs, rsim, smiles in zip(rs_list, rsim_list, smiles_list):
            if rs >= 6/9 and rsim >= 0.6:
                frags = {frag for frag in cut(smiles)}
                self.population.extend([(rsim, frag) for frag in frags])
        self.population.sort(reverse=True)

    def generate(self):
        for _ in range(1000):
            frag1, frag2 = random.sample([frag for prop, frag in self.population], 2)
            smiles = Chem.MolToSmiles(self.attach(frag1, frag2))
            if smiles is None: continue
            smiles = self.sampler.mask_modification(smiles, min_len=50, gamma=self.args.gamma)
            if smiles is not None:
                smiles = sorted(smiles.split('.'), key=len)[-1]     # get the largest
            return smiles
            
    def record(self, smiles_list, prop_list):
        """Record results to CSV file with proper column structure"""
        # Initialize CSV file with headers if it doesn't exist
        if not os.path.exists(self.fname):
            with open(self.fname, 'w') as f:
                f.write('molecule,docking_score,qed_score,sa_score,similarity\n')
        
        rv_list, rq_list, rs_list, rsim_list = prop_list
        for i in range(len(rs_list)):
            rs_list[i] = -9 * rs_list[i] + 10
        
        with open(self.fname, 'a') as f:
            for i in range(len(smiles_list)):
                # Write: molecule, docking_score, qed_score, sa_score, similarity
                f.write(f'{smiles_list[i]},{rv_list[i]},{rq_list[i]},{rs_list[i]},{rsim_list[i]}\n')

    def record_tox(self, smiles_list, prop_list):
        """Record results to CSV file with proper column structure"""
        # Initialize CSV file with headers if it doesn't exist
        if not os.path.exists(self.fname):
            with open(self.fname, 'w') as f:
                f.write('molecule,sa_score,similarity_score\n')

        rs_list, rsim_list = prop_list
        for i in range(len(rs_list)):
            rs_list[i] = -9 * rs_list[i] + 10
        with open(self.fname, 'a') as f:
            for i in range(len(smiles_list)):
                f.write(f'{smiles_list[i]},{rs_list[i]},{rsim_list[i]}\n')

    def run(self):
        # Setup logging
        log_file = os.path.join(self.args.log_dir, 'run.log')
        logger = setup_logging(log_file)
        
        t_start = time()
        logger.info(f"Start: oracle={self.args.oracle_name}, seed={self.args.seed}, "
                   f"max_calls={self.args.max_oracle_calls}, tox={getattr(self.args, 'tox', False)}")
        
        num_of_molecules = [0]
        actual_calls = self.args.max_oracle_calls
        iteration = 0
        
        while num_of_molecules[-1] < actual_calls:
            iteration += 1
            
            if self.args.tox:
                # Generate molecules
                smiles_list = [self.generate() for _ in range(self.args.num_gen)]
                
                # Calculate rewards
                prop_list = self.reward_tox(smiles_list)
                
                # Update population
                self.update_population_tox(smiles_list, prop_list)
                
                # Record results
                self.record_tox(smiles_list, prop_list)
                
                # Count unique molecules (for tox, we track from the CSV)
                if os.path.exists(self.fname):
                    df = pd.read_csv(self.fname).drop_duplicates(subset=['molecule'])
                    num_of_molecules.append(len(df))
                else:
                    num_of_molecules.append(0)
            else:
                # Generate molecules
                smiles_list = [self.generate() for _ in range(self.args.num_gen)]
                
                # Calculate rewards
                prop_values = self.reward(smiles_list)
                
                # Update population
                self.update_population(smiles_list, prop_values, reward=self.args.reward)
                
                # Record results
                self.record(smiles_list, prop_values)
                
                # Count unique molecules
                df = pd.read_csv(self.fname).drop_duplicates(subset=['molecule'])
                num_of_molecules.append(len(df))
            
            logger.info(f"Iter {iteration}: {num_of_molecules[-1]} unique molecules")
            
            if len(num_of_molecules) > 5 and num_of_molecules[-1] - num_of_molecules[-5] < 5:
                logger.warning('Stopping: no new molecules generated')
                break
        
        total_time = time() - t_start
        logger.info(f"Complete: {total_time/60:.1f}min, {num_of_molecules[-1]} molecules")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',            type=str)
    parser.add_argument('-o', '--oracle_name',      type=str,   required=False, choices=['parp1', 'fa7', '5ht1b', 'braf', 'jak2'])
    parser.add_argument('--oracle_url',             type=str,   default=None)
    parser.add_argument('-i', '--start_mol_idx',    type=int,   required=False, choices=[0, 1, 2])
    parser.add_argument('-d', '--sim_thr',          type=float, required=False, choices=[0.4, 0.6])
    parser.add_argument('-s', '--seed',             type=int,                   choices=[1, 2, 3])
    parser.add_argument('-m', '--model_path',       type=str,                   default='model.ckpt')
    parser.add_argument('--num_gen',                type=int,                   default=100)
    parser.add_argument('--max_oracle_calls',       type=int,                   default=1000)
    parser.add_argument('--gamma',                  type=float,                 default=0)
    parser.add_argument('--tox',                                                action='store_true')
    cmd_args = parser.parse_args()
    
    # Load config from file and merge with command line args
    args = OmegaConf.load(cmd_args.config_file)
    GenMolOpt(args).run()

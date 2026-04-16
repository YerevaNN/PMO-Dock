import os
import sys
from copy import deepcopy
from typing import Any
# Add the GenMol root directory to Python path to import both genmol and scripts modules
genmol_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
sys.path.append(os.path.join(genmol_root, 'src'))  # for genmol module 
sys.path.append(genmol_root)  # for scripts module
# Add project root to path for utils imports
project_root = os.path.dirname(genmol_root)
sys.path.insert(0, project_root)

from time import time, sleep, perf_counter
import random
import argparse
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs, AllChem, QED, RDConfig
from omegaconf import OmegaConf
import requests
from typing import List, Optional
import logging
import resource
import sys

from utils.tasks import task_name2constraints, select_sigma, guassian_modifier
from utils.docking_vina_client import DockingVinaClient
from genmol.sampler import Sampler
from genmol.utils.utils_chem import cut
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
from get_vocab import get_vocab_from_zinc250k
import sascorer

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
    
    logger = logging.getLogger()
    # Force flush to ensure message is written immediately
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.flush()
    return logger


class GenMolOpt():
    def __init__(self, args):
        super().__init__()
        self.args = args

        if self.args.oracle_name in ['6nzp_7uyt', '6nzp_5ut5', '6nzp_7uyw']:
            self.target = self.args.oracle_name.split('_')[0]
            self.antitarget = self.args.oracle_name.split('_')[1]
        else:
            self.target = self.args.oracle_name
            self.antitarget = None
        script_dir = os.path.dirname(os.path.abspath(__file__))

        try:
            df = pd.read_csv(os.path.join(script_dir, 'vocab', 'frags.csv'))
        except FileNotFoundError:
            df = get_vocab_from_zinc250k(size=self.args.population_size)
        self.population = random.sample(list(zip([0]*len(df), df['frag'])), self.args.population_size)
        if not os.path.exists(self.args.model_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {self.args.model_path}")
        self.sampler = Sampler(self.args.model_path)

        if self.args.get('pmo-task', False):
            from tdc import Oracle
            self.oracle = Oracle(self.args.oracle_name)
        else:
            # Check if oracle service URL is configured (from env var or config)
            oracle_service_url = os.environ.get("DOCKING_VINA_URL")
            if oracle_service_url is None:
                oracle_service_url = self.args.get('oracle_url')
            if oracle_service_url:
                self.target_oracle = DockingVinaClient(oracle_service_url, self.target)
                if self.antitarget:
                    self.antitarget_oracle = DockingVinaClient(oracle_service_url, self.antitarget)
            else:
                from scripts.exps.lead.docking.docking import DockingVina
                self.target_oracle = DockingVina(self.target)
                if self.antitarget:
                    self.antitarget_oracle = DockingVina(self.antitarget)

        reslut_dir = self.args.output_dir
        os.makedirs(reslut_dir, exist_ok=True)       
        self.fname = f'{reslut_dir}/results.csv'

    def reward_vina(self, smiles_list):
        """Calculate Vina reward (negative docking score, clipped to >= 0)"""
        t0 = perf_counter()
        scores = self.target_oracle.predict(smiles_list)
        elapsed = perf_counter() - t0
        n = len(smiles_list)
        per_mol = (elapsed / n) if n else 0
        logging.info(
            f"⏱️  reward_vina (target {self.target}): {n} molecules in {elapsed:.2f}s "
            f"({per_mol:.3f}s per molecule)"
        )
        reward = - np.array(scores)
        reward = np.clip(reward, 0, None)
        return list(reward)

    def antitarget_reward(self, smiles_list):
        t0 = perf_counter()
        scores = self.antitarget_oracle.predict(smiles_list)
        elapsed = perf_counter() - t0
        n = len(smiles_list)
        per_mol = (elapsed / n) if n else 0
        logging.info(
            f"⏱️  antitarget_reward ({self.antitarget}): {n} molecules in {elapsed:.2f}s "
            f"({per_mol:.3f}s per molecule)"
        )
        reward = - np.array(scores)
        reward = np.clip(reward, 0, None)
        return list(reward)

    def reward_pmo(self, smiles_list):
        return [self.oracle(mol) for mol in smiles_list]
    
    def reward_qed(self, mols):
        return [QED.qed(m) for m in mols]
    
    def reward_sa(self, mols):
        return [(10 - sascorer.calculateScore(m)) / 9 for m in mols]
    
    def property2computer(self, property_name, smiles_list):
        if property_name == 'qed_score':
            mols = [Chem.MolFromSmiles(s) for s in smiles_list]
            return self.reward_qed(mols)
        elif property_name == 'sa_score':
            mols = [Chem.MolFromSmiles(s) for s in smiles_list]
            return self.reward_sa(mols)
        elif property_name == 'docking_score':
            if self.args['pmo-task']:
                return self.reward_pmo(smiles_list)
            else:
                return self.reward_vina(smiles_list)
        elif property_name == 'antitarget_score':
            return self.antitarget_reward(smiles_list)
        else:
            raise ValueError(f'Invalid property name: {property_name}')
    
    def hit_reward(self, task_format, prop_values, prod=True, avg=False):
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

    def reward(self, smiles_list, prop_names=['docking_score', 'qed_score', 'sa_score']):
        if self.antitarget:
            prop_names = list(prop_names) + ['antitarget_score']
        prop_values = {}
        for prop in prop_names:
            prop_values[prop] = self.property2computer(prop, smiles_list)
        return prop_values
    
    def attach(self, frag1, frag2):
        rxn = AllChem.ReactionFromSmarts('[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]')
        mols = rxn.RunReactants((Chem.MolFromSmiles(frag1), Chem.MolFromSmiles(frag2)))
        idx = np.random.randint(len(mols))
        return mols[idx][0]
    
    def update_population(self, smiles_list, prop_values, reward):
        if reward == 'hit':
            task_format = task_name2constraints(self.args.task_name)
            rewards = self.hit_reward(task_format, prop_values)
            for reward, smiles in zip(rewards, smiles_list):
                if reward > 0:
                    frags = {frag for frag in cut(smiles)}
                    self.population.extend([(reward, frag) for frag in frags])
        elif reward == 'original':
            rv_list = prop_values['docking_score']
            rq_list = prop_values['qed_score']
            rs_list = prop_values['sa_score']
            if self.antitarget:
                ra_list = prop_values['antitarget_score']
                for rv, rq, rs, ra, smiles in zip(rv_list, rq_list, rs_list, ra_list, smiles_list):
                    gap = np.clip(rv/20 - ra/20, 0, 1)
                    if rq >= 0.4 and rs >= 0.66 and gap > 0.0:
                        frags = {frag for frag in cut(smiles)}
                        self.population.extend([(gap, frag) for frag in frags])
            else:
                for rv, rq, rs, smiles in zip(rv_list, rq_list, rs_list, smiles_list):
                    if rq >= 0.5 and rs >= 0.55:
                        frags = {frag for frag in cut(smiles)}
                        self.population.extend([(rv, frag) for frag in frags])
        elif reward == 'geam':
            rv_list = prop_values['docking_score']
            rv_list = [x / 20 for x in rv_list]
            rq_list = prop_values['qed_score']
            rs_list = prop_values['sa_score']
            rewards = self.geam_reward([rv_list, rq_list, rs_list])
            for reward, smiles in zip(rewards, smiles_list):
                if reward > 0:
                    frags = {frag for frag in cut(smiles)}
                    self.population.extend([(reward, frag) for frag in frags])

        self.population.sort(reverse=True)
        self.population = self.population[:self.args.population_size]
        #print(f'Population: {self.population[:5]} \n ... \n {self.population[-5:]} \n length: {len(self.population)} \n')

    def generate(self):
        for attempt in range(1000):
            # Fix: Avoid using parameterized generics in isinstance
            frag1, frag2 = random.sample([frag for prop, frag in self.population], 2)
            smiles = Chem.MolToSmiles(self.attach(frag1, frag2))
            if smiles is None:
                continue
            try:
                smiles = self.sampler.mask_modification(smiles, min_len=50, gamma=self.args.gamma)
            except Exception:
                pass  # fall back to original SMILES from attach()
            if smiles is not None:
                smiles = sorted(smiles.split('.'), key=len)[-1]     # get the largest
            return smiles
        return None  # Return None if all 1000 attempts failed
            
    def record(self, smiles_list, prop_values):
        """Record results to CSV file with proper column structure using pandas"""
        rv_list = prop_values['docking_score']
        rq_list = prop_values['qed_score']
        rs_list = prop_values['sa_score']
        df = pd.DataFrame({
            'molecule': smiles_list,
            'docking_score': rv_list,
            'qed_score': rq_list,
            'sa_score': rs_list
        })
        if self.antitarget:
            ra_list = prop_values['antitarget_score']
            df['antitarget_score'] = ra_list
            gap_list = np.array(rv_list) - np.array(ra_list)
            df['gap'] = np.clip(gap_list, 0, 1)
        # Write header only if file does not exist
        if not os.path.exists(self.fname):
            df.to_csv(self.fname, index=False, mode='w')
        else:
            df.to_csv(self.fname, index=False, mode='a', header=False)

    def run(self):
        log_file = os.path.join(self.args.output_dir, 'run.log')
        logger = setup_logging(log_file)

        budget = self.args.max_oracle_calls
        t_start = time()
        spent = 0
        num_of_molecules = [0]
        iteration = 0

        logger.info(f"oracle budget {budget} | oracle={self.args.oracle_name} seed={self.args.seed}")

        while num_of_molecules[-1] < budget:
            iteration += 1
            smiles_list = []
            for _ in range(self.args.num_gen):
                s = self.generate()
                if s is not None:
                    smiles_list.append(s)

            if not smiles_list:
                logger.error("no valid molecules, stop")
                break

            prop_values = self.reward(smiles_list)
            self.update_population(smiles_list, prop_values, reward=self.args.reward)
            self.record(smiles_list, prop_values)

            spent += len(smiles_list)
            df = pd.read_csv(self.fname).drop_duplicates(subset=['molecule'])
            num_of_molecules.append(len(df))
            pct = 100 * spent / budget if budget else 0
            elapsed = (time() - t_start) / 60
            logger.info(f"oracle {spent}/{budget} ({pct:.0f}%) | iter {iteration} | unique {num_of_molecules[-1]} | {elapsed:.1f}min")

            if len(num_of_molecules) > 5 and num_of_molecules[-1] - num_of_molecules[-5] < 5:
                logger.warning("no new molecules, stop")
                break

        logger.info(f"done | oracle {spent}/{budget} | {num_of_molecules[-1]} unique | {(time() - t_start) / 60:.1f}min")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',            type=str,   default='scripts/exps/hit/configs/genmol_hit.yaml')
    parser.add_argument('--oracle_name',            type=str,   choices=['parp1', 'fa7', '5ht1b', 'braf', 'jak2', '6nzp_7uyt', '6nzp_5ut5', '6nzp_7uyw'])
    parser.add_argument('--oracle_url',             type=str,   default=None)
    parser.add_argument('--seed',                   type=int,   choices=[1, 2, 3, 4, 5])
    parser.add_argument('--model_path',             type=str,   default='model.ckpt')
    parser.add_argument('--num_gen',                type=int,   default=100)
    parser.add_argument('--gamma',                  type=float, default=0)
    parser.add_argument('--population_size',        type=int,   default=100)
    parser.add_argument('--max_oracle_calls',       type=int,   default=3000)
    parser.add_argument('--output_dir',             type=str)
    parser.add_argument('--reward',                 type=str,   choices=['hit', 'original', 'geam'], default='original')
    parser.add_argument('--task_name',              type=str,   default='hit')
    parser.add_argument('--pmo_task',               action='store_true')
    cmd_args = parser.parse_args()
    
    # Load config from file and merge with command line args
    args = OmegaConf.load(cmd_args.config_file)
    GenMolOpt(args).run()

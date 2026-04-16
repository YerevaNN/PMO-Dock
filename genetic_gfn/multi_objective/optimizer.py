import os
import yaml
import random
import shutil
import logging
import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
# import tdc
# from tdc.generation import MolGen
import wandb
# from main.utils.chem import *
try:
    # Only used for logging hypervolume; make optional so the code can run without BoTorch installed.
    from botorch.utils.multi_objective.hypervolume import Hypervolume
except Exception:
    Hypervolume = None

from oracle.scorer.scorer import get_scores, TARGET_CONFIGS, DOCKING_WINDOWS
from utils.metrics import compute_success, compute_diversity

# Any objective name present here is treated as a docking objective (scored via the docking oracle service).
DOCKING_TARGETS = {str(k).lower() for k in (TARGET_CONFIGS or {}).keys()}
DOCKING_NORM_DENOM = 20.0  # Keep consistent with scorer.py docking normalization (affinity / -20.0).


class Oracle:
    def __init__(self, args=None, mol_buffer={}):
        self.name = None
        self.evaluator = None
        self.task_label = None
        if args is None:
            self.max_oracle_calls = 10000
            self.freq_log = 100
        else:
            self.args = args
            self.max_oracle_calls = args.max_oracle_calls
            self.freq_log = args.freq_log
            self.weights = np.array(args.alpha_vector) 
        self.mol_buffer = mol_buffer
        # self.sa_scorer = tdc.Oracle(name = 'SA')
        # self.diversity_evaluator = tdc.Evaluator(name = 'Diversity')
        self.last_log = 0
        self.current_div = 1.
        self.hypervolume = None
        if Hypervolume is not None:
            try:
                self.hypervolume = Hypervolume(ref_point=torch.zeros(len(args.objectives)))
            except Exception:
                self.hypervolume = None
        
        # Hit task tracking
        self.hit_count = 0
        self.is_hit_task = False  # Will be set based on objectives
        # Lead task flag (similarity objective present); used to control CSV logging behavior.
        self.is_lead_task = False
        # Constraint ranges for hit task
        self.qed_min, self.qed_max = 0.5, 1.0
        self.sa_min, self.sa_max = 1.0, 5.0
        # Docking constraints are target-specific for hit task.
        # Raw docking scores are negative (kcal/mol). More negative = stronger binding.
        self._dock_range_by_target = {
            "parp1": (-20.0, -10.0),
            "fa7": (-20.0, -8.5),
            "5ht1b": (-20.0, -8.7845),
            "braf": (-20.0, -10.30),
            "jak2": (-20.0, -9.10),
        }
        self.dock_min, self.dock_max = self._dock_range_by_target["parp1"]
        
        # Single per-run CSV for recording molecules and 
        # This is always written as: <output_dir>/molecules.csv
        self.molecules_csv_file = None
        # Best-effort stop reason for end-of-run summaries (set by optimizers).
        # Examples: "max_oracle_calls", "convergence", "stuck_no_new_molecules", "exception:<type>".
        self.stop_reason = None
        if args and hasattr(args, 'output_dir'):
            csv_dir = args.output_dir
            os.makedirs(csv_dir, exist_ok=True)
            # CSV file will be set when task_label is set

        # Per-molecule score logging (raw + normalized + reward).
        # Enable with: `export LOG_REWARD_SCORES=1`
        self.log_reward_scores = bool(int(os.environ.get("LOG_REWARD_SCORES", "0")))
        # If set, only log every N newly-scored molecules (default 1 = all).
        self.log_reward_scores_every = max(1, int(os.environ.get("LOG_REWARD_SCORES_EVERY", "1")))
        self._log_reward_scores_counter = 0

        self._logger = logging.getLogger(__name__)
        if self.log_reward_scores:
            # Ensure we see INFO logs even if the app didn't configure logging.
            logging.basicConfig(level=logging.INFO)
        
    @property
    def budget(self):
        return self.max_oracle_calls

    def assign_evaluator(self, evaluator):
        self.evaluator = evaluator

    def sort_buffer(self):
        self.mol_buffer = dict(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][0], reverse=True))
    
    def record_objectives(self, smiles: str, objectives, raw_scores):
        """
        Record a molecule to the per-run molecules CSV with one column per objective (raw scores).
        Format: smiles,<obj1>,<obj2>,...,<objN>,\n
        """
        if self.molecules_csv_file is None:
            self.molecules_csv_file = os.path.join(self.args.output_dir, "molecules.csv")
            os.makedirs(os.path.dirname(self.molecules_csv_file), exist_ok=True)
            header = "smiles," + ",".join([str(o) for o in objectives]) + ",\n"
            with open(self.molecules_csv_file, "w") as f:
                f.write(header)

        parts = []
        for x in raw_scores:
            try:
                parts.append(f"{float(x):.6f}")
            except Exception:
                parts.append(str(x))
        with open(self.molecules_csv_file, "a") as f:
            f.write(f"{smiles}," + ",".join(parts) + ",\n")

    def save_result(self, suffix=None):
        
        if suffix is None:
            output_file_path = os.path.join(self.args.output_dir, 'results.yaml')
        else:
            output_file_path = os.path.join(self.args.output_dir, 'results_' + suffix + '.yaml')

        self.sort_buffer()
        with open(output_file_path, 'w') as f:
            yaml.dump(self.mol_buffer, f, sort_keys=False)

    def log_intermediate(self, mols=None, scores=None, finish=False):

        if finish:
            temp_top100 = list(self.mol_buffer.items())[:100]
            smis = [item[0] for item in temp_top100]
            scores = [item[1][0] for item in temp_top100]
            scores_each = [item[1][2] for item in temp_top100]  # normalized scores
            raw_scores_each = [item[1][3] if len(item[1]) > 3 else item[1][2] for item in temp_top100]  # raw scores
            n_calls = self.max_oracle_calls
        else:
            if mols is None and scores is None:
                if len(self.mol_buffer) <= self.max_oracle_calls:
                    # If not spefcified, log current top-100 mols in buffer
                    temp_top100 = list(self.mol_buffer.items())[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    scores_each = [item[1][2] for item in temp_top100]  # normalized scores
                    raw_scores_each = [item[1][3] if len(item[1]) > 3 else item[1][2] for item in temp_top100]  # raw scores
                    n_calls = len(self.mol_buffer)
                else:
                    results = list(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][1], reverse=False))[:self.max_oracle_calls]
                    temp_top100 = sorted(results, key=lambda kv: kv[1][0], reverse=True)[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    scores_each = [item[1][2] for item in temp_top100]  # normalized scores
                    raw_scores_each = [item[1][3] if len(item[1]) > 3 else item[1][2] for item in temp_top100]  # raw scores
                    n_calls = self.max_oracle_calls
            else:
                # Otherwise, log the input moleucles
                smis = [Chem.MolToSmiles(m) for m in mols]
                n_calls = len(self.mol_buffer)
                # For this case, we don't have scores_each, so use empty lists
                scores_each = []
                raw_scores_each = []
        
        # Uncomment this line if want to log top-10 moelucles figures, so as the best_mol key values.
        # temp_top10 = list(self.mol_buffer.items())[:10]

        avg_top1 = np.max(scores)
        avg_top10 = np.mean(sorted(scores, reverse=True)[:10])
        avg_top100 = np.mean(scores)
        # avg_sa = np.mean(self.sa_scorer(smis))
        # diversity_top100 = self.diversity_evaluator(smis)
        mols = []
        for s in smis:
            try:
                mol = Chem.MolFromSmiles(s)
                if mol:
                    mols.append(mol)
            except:
                pass
        
        diversity_top100 = compute_diversity(mols)
        hv = None
        if self.hypervolume is not None:
            try:
                hv = self.hypervolume.compute(torch.tensor(scores_each))
            except Exception:
                hv = None
            
        self.current_div = diversity_top100
        
        # Calculate hit ratio for hit task
        hit_ratio = self.hit_count / n_calls if n_calls > 0 else 0.0
        
        # Extract and calculate average property values
        log_str = f'{n_calls}/{self.max_oracle_calls} | '
        
        if self.is_hit_task and len(raw_scores_each) > 0:
            # Extract QED, SA, and docking scores from raw_scores_each
            qed_idx = self.objectives_lower.index('qed') if 'qed' in self.objectives_lower else None
            sa_idx = self.objectives_lower.index('sa') if 'sa' in self.objectives_lower else None
            dock_idx = None
            docking_targets = ['parp1', 'fa7', '5ht1b', 'braf', 'jak2']
            for i, obj_lower in enumerate(self.objectives_lower):
                if obj_lower in docking_targets:
                    dock_idx = i
                    break
            
            # Calculate averages
            if qed_idx is not None:
                avg_qed = np.mean([raw_scores[qed_idx] if len(raw_scores) > qed_idx else 0.0 
                                  for raw_scores in raw_scores_each])
                log_str += f'avg_qed: {avg_qed:.3f} | '
            if sa_idx is not None:
                avg_sa = np.mean([raw_scores[sa_idx] if len(raw_scores) > sa_idx else 0.0 
                                 for raw_scores in raw_scores_each])
                log_str += f'avg_sa: {avg_sa:.3f} | '
            if dock_idx is not None:
                avg_dock = np.mean([raw_scores[dock_idx] if len(raw_scores) > dock_idx else 0.0 
                                    for raw_scores in raw_scores_each])
                log_str += f'avg_dock: {avg_dock:.3f} | '
            
            log_str += f'hits: {self.hit_count} | hit_ratio: {hit_ratio:.4f}'
        else:
            # For non-hit tasks, show standard metrics
            log_str += f'avg_top1: {avg_top1:.3f} | '
            log_str += f'avg_top10: {avg_top10:.3f} | '
            log_str += f'avg_top100: {avg_top100:.3f} | '
            if hv is not None:
                log_str += f'hv: {hv:.3f} | '
            log_str += f'div: {diversity_top100:.3f}'
        
        print(log_str)

        try:
            if self.is_hit_task and len(raw_scores_each) > 0:
                # Log property values for hit task
                log_dict = {
                    "n_oracle": n_calls,
                    "diversity_top100": diversity_top100,
                }
                
                # Add property averages
                qed_idx = self.objectives_lower.index('qed') if 'qed' in self.objectives_lower else None
                sa_idx = self.objectives_lower.index('sa') if 'sa' in self.objectives_lower else None
                dock_idx = None
                docking_targets = ['parp1', 'fa7', '5ht1b', 'braf', 'jak2']
                for i, obj_lower in enumerate(self.objectives_lower):
                    if obj_lower in docking_targets:
                        dock_idx = i
                        break
                
                if qed_idx is not None:
                    avg_qed = np.mean([raw_scores[qed_idx] if len(raw_scores) > qed_idx else 0.0 
                                      for raw_scores in raw_scores_each])
                    log_dict["avg_qed"] = avg_qed
                if sa_idx is not None:
                    avg_sa = np.mean([raw_scores[sa_idx] if len(raw_scores) > sa_idx else 0.0 
                                     for raw_scores in raw_scores_each])
                    log_dict["avg_sa"] = avg_sa
                if dock_idx is not None:
                    avg_dock = np.mean([raw_scores[dock_idx] if len(raw_scores) > dock_idx else 0.0 
                                       for raw_scores in raw_scores_each])
                    log_dict["avg_dock"] = avg_dock
                
                log_dict["hit_count"] = self.hit_count
                log_dict["hit_ratio"] = hit_ratio
            else:
                # Standard logging for non-hit tasks
                log_dict = {
                    "avg_top1": avg_top1, 
                    "avg_top10": avg_top10, 
                    "avg_top100": avg_top100, 
                    "diversity_top100": diversity_top100,
                    "n_oracle": n_calls,
                }
                if hv is not None:
                    log_dict["hv"] = hv
            
            wandb.log(log_dict)
        except:
            pass


    def __len__(self):
        return len(self.mol_buffer)
    
    
    def set_objectives(self, objectives, alpha_vector):
        self.objectives = objectives
        self.weights = np.array(alpha_vector)
        # Store lowercase version for consistent comparison
        self.objectives_lower = [obj.lower() for obj in objectives]
        self.is_lead_task = ('similarity' in self.objectives_lower)
        
        # Check if this is a hit task (qed, sa, and a docking target)
        # NOTE: If similarity is present, treat this as a lead task and do NOT enable hit-task CSV logging.
        if (not self.is_lead_task) and len(objectives) == 3:
            if 'qed' in self.objectives_lower and 'sa' in self.objectives_lower:
                # Check if third objective is a docking target
                docking_targets = ['parp1', 'fa7', '5ht1b', 'braf', 'jak2']
                if any(target in self.objectives_lower for target in docking_targets):
                    self.is_hit_task = True
                    # Apply target-specific docking window when available (defaults to parp1 window)
                    target = None
                    for t in docking_targets:
                        if t in self.objectives_lower:
                            target = t
                            break
                    if target is not None:
                        self.dock_min, self.dock_max = self._dock_range_by_target.get(
                            target, self._dock_range_by_target["parp1"]
                        )
    
    def _check_hit(self, qed_score, sa_score, dock_score_raw):
        """Check if molecule satisfies all hit task constraints."""
        qed_ok = self.qed_min <= qed_score <= self.qed_max
        sa_ok = self.sa_min <= sa_score <= self.sa_max
        dock_ok = self.dock_min <= dock_score_raw <= self.dock_max
        return qed_ok and sa_ok and dock_ok
    
    def moo_evaluator_batch(self, mols):
        """
        Evaluate a batch of molecules for multi-objective optimization.
        Returns lists of (rewards, scores_array, raw_scores_array) for each molecule.
        """
        from rdkit import Chem
        
        num_mols = len(mols)
        num_objs = len(self.objectives)
        
        print(f"[DEBUG moo_evaluator_batch] Starting with {num_mols} molecules, {num_objs} objectives: {self.objectives}")
        
        # Initialize 2D arrays: (num_molecules x num_objectives)
        all_scores = np.zeros((num_mols, num_objs))
        all_raw_scores = np.zeros((num_mols, num_objs))
        
        # Find indices for each objective type
        qed_idx = self.objectives_lower.index('qed') if 'qed' in self.objectives_lower else None
        sa_idx = self.objectives_lower.index('sa') if 'sa' in self.objectives_lower else None
        
        # Get scores for all molecules for each objective (batched call per objective)
        for i, obj in enumerate(self.objectives):
            obj_lower = str(obj).lower()
            print(f"[DEBUG moo_evaluator_batch] Processing objective {i}/{num_objs}: {obj} ({obj_lower})")
            try:
                if obj_lower in DOCKING_TARGETS:
                    # Docking: get both normalized and raw scores for all molecules
                    print(f"[DEBUG moo_evaluator_batch] Calling get_scores for docking target {obj_lower}")
                    norm_scores_list, raw_scores_list = get_scores(obj, mols, return_normalized=True, return_raw_scores=True)
                    print(f"[DEBUG moo_evaluator_batch] Got {len(norm_scores_list)} norm scores, {len(raw_scores_list)} raw scores")
                    all_scores[:, i] = norm_scores_list
                    all_raw_scores[:, i] = raw_scores_list
                elif i == qed_idx:
                    # QED: raw score is already in [0, 1]
                    print(f"[DEBUG moo_evaluator_batch] Calling get_scores for QED")
                    raw_scores_list = get_scores(obj, mols, return_normalized=False)
                    print(f"[DEBUG moo_evaluator_batch] Got {len(raw_scores_list)} QED scores")
                    all_raw_scores[:, i] = raw_scores_list
                    all_scores[:, i] = raw_scores_list
                elif i == sa_idx:
                    # SA: get raw score and normalize
                    print(f"[DEBUG moo_evaluator_batch] Calling get_scores for SA")
                    from oracle.scorer.scorer import normalize_sa_score
                    raw_scores_list = get_scores(obj, mols, return_normalized=False)
                    print(f"[DEBUG moo_evaluator_batch] Got {len(raw_scores_list)} SA scores")
                    all_raw_scores[:, i] = raw_scores_list
                    all_scores[:, i] = [normalize_sa_score(s) for s in raw_scores_list]
                else:
                    # Fallback
                    print(f"[DEBUG moo_evaluator_batch] Calling get_scores for other objective")
                    raw_scores_list = get_scores(obj, mols, return_normalized=False)
                    print(f"[DEBUG moo_evaluator_batch] Got {len(raw_scores_list)} scores")
                    all_raw_scores[:, i] = raw_scores_list
                    all_scores[:, i] = raw_scores_list
            except Exception as e:
                print(f"[DEBUG moo_evaluator_batch] ERROR getting scores for objective {obj}: {e}")
                import traceback
                traceback.print_exc()
                raise
        
        # Now compute rewards, record CSV, and track hits for each molecule
        print(f"[DEBUG moo_evaluator_batch] Computing rewards for {num_mols} molecules")
        rewards = []
        
        for mol_idx, mol in enumerate(mols):
            scores = all_scores[mol_idx]
            raw_scores = all_raw_scores[mol_idx]
            
            # Standard weighted sum reward
            reward = np.matmul(scores, self.weights.reshape(-1, 1))
            
            # Selectivity reward override (when anti-target is present with 6nzp)
            try:
                obj_lowers = list(self.objectives_lower)
                if "6nzp" in obj_lowers:
                    i_main = obj_lowers.index("6nzp")
                    anti_candidates = [i for i, o in enumerate(obj_lowers) if (o in DOCKING_TARGETS and o != "6nzp")]
                    if anti_candidates:
                        i_anti = anti_candidates[0]
                        
                        def _dock_norm_from_affinity(aff) -> float:
                            try:
                                aff_f = float(aff)
                            except Exception:
                                return 0.0
                            if aff_f == -1.0:
                                return 0.0
                            try:
                                x = aff_f / (-DOCKING_NORM_DENOM)
                            except Exception:
                                x = 0.0
                            return float(max(0.0, min(1.0, x)))
                        
                        ds_target_norm = float(max(0.0, min(1.0, scores[i_main])))
                        ds_anti_norm = _dock_norm_from_affinity(raw_scores[i_anti])
                        gap = float(max(0.0, min(1.0, ds_target_norm - ds_anti_norm)))
                        
                        qed_val = float(max(0.0, min(1.0, scores[qed_idx]))) if qed_idx is not None else 0.0
                        sa_val = float(max(0.0, min(1.0, scores[sa_idx]))) if sa_idx is not None else 0.0
                        
                        selectivity_vec = np.array([ds_target_norm, gap, qed_val, sa_val])
                        reward = np.matmul(selectivity_vec, self.weights.reshape(-1, 1))
            except Exception:
                pass
            
            rewards.append(float(reward))
            
            # CSV recording
            try:
                mol_smiles = Chem.MolToSmiles(mol)
                if mol_smiles:
                    self.record_objectives(mol_smiles, self.objectives, raw_scores)
            except Exception:
                pass
            
            # Hit tracking
            if self.is_hit_task:
                try:
                    mol_smiles = Chem.MolToSmiles(mol)
                    if mol_smiles:
                        qed_idx2 = self.objectives_lower.index('qed') if 'qed' in self.objectives_lower else None
                        sa_idx2 = self.objectives_lower.index('sa') if 'sa' in self.objectives_lower else None
                        dock_idx2 = None
                        hit_docking_targets = ['parp1', 'fa7', '5ht1b', 'braf', 'jak2']
                        for j, obj_lower in enumerate(self.objectives_lower):
                            if obj_lower in hit_docking_targets:
                                dock_idx2 = j
                                break
                        if qed_idx2 is not None and sa_idx2 is not None and dock_idx2 is not None:
                            is_hit = self._check_hit(raw_scores[qed_idx2], raw_scores[sa_idx2], raw_scores[dock_idx2])
                            if is_hit:
                                self.hit_count += 1
                except Exception:
                    pass
        
        print(f"[DEBUG moo_evaluator_batch] Returning {len(rewards)} rewards")
        return rewards, all_scores, all_raw_scores
    
    def moo_evaluator(self, mol):
        """
        Evaluate molecule for multi-objective optimization.
        For hit task: R(x) = DSd(x) × QED(x) × SAc(x) where all are normalized to [0, 1]
        Otherwise: weighted sum
        """
        scores = np.zeros(len(self.objectives))
        raw_scores = np.zeros(len(self.objectives))  # Store raw scores for hit checking
        

        # Find indices for each objective type
        qed_idx = self.objectives_lower.index('qed') if 'qed' in self.objectives_lower else None
        sa_idx = self.objectives_lower.index('sa') if 'sa' in self.objectives_lower else None
        # Get scores for each objective
        for i, obj in enumerate(self.objectives):
            obj_lower = str(obj).lower()
            if obj_lower in DOCKING_TARGETS:
                # Docking objective (including selectivity anti-targets like 7uyt/7uyw/5ut5).
                norm_scores, raw_scores_list = get_scores(obj, [mol], return_normalized=True, return_raw_scores=True)
                scores[i] = norm_scores[0]
                raw_scores[i] = raw_scores_list[0]
            elif i == qed_idx:
                # QED: get raw score (already in [0, 1])
                raw_score = get_scores(obj, [mol], return_normalized=False)[0]
                raw_scores[i] = raw_score
                scores[i] = raw_score
            elif i == sa_idx:
                # SA: get raw score and normalize
                raw_score = get_scores(obj, [mol], return_normalized=False)[0]
                raw_scores[i] = raw_score
                from oracle.scorer.scorer import normalize_sa_score
                scores[i] = normalize_sa_score(raw_score)
            else:
                # Fallback (shouldn't happen in hit task)
                raw_score = get_scores(obj, [mol], return_normalized=False)[0]
                raw_scores[i] = raw_score
                scores[i] = raw_score
        

            # Hit task: use the original reward form (weighted sum via alpha_vector),
            # but keep hit checking + CSV recording based on *raw* scores.
            #
            # Here `scores` for hit task are direction-correct, normalized values:
            # - docking: DSd in [0,1]
            # - QED: already in [0,1]
            # - SA: SAc in [0,1] (lower SA -> higher SAc)
        reward = np.matmul(scores, self.weights.reshape(-1, 1))

        # Selectivity reward override (when anti-target is present with 6nzp):
        # Final reward:
        #   R = ds_target_norm(x) * (1 - gap) * qed(x) * sa(x)
        # where:
        #   gap = clip(ds_target_norm - ds_anti_norm, 0..1)
        #
        # ds_target_norm is the normalized docking score for 6nzp in [0,1] (already in `scores`).
        # ds_anti_norm is derived from the raw docking affinity (UNFLIPPED) for the anti-target objective.
        try:
            obj_lowers = list(self.objectives_lower)
            if "6nzp" in obj_lowers:
                i_main = obj_lowers.index("6nzp")
                anti_candidates = [i for i, o in enumerate(obj_lowers) if (o in DOCKING_TARGETS and o != "6nzp")]
                if anti_candidates:
                    i_anti = anti_candidates[0]

                    def _dock_norm_from_affinity(aff) -> float:
                        # Mirror scorer behavior: docking failure becomes raw=-1.0 and norm=0.0.
                        try:
                            aff_f = float(aff)
                        except Exception:
                            return 0.0
                        if aff_f == -1.0:
                            return 0.0
                        try:
                            x = aff_f / (-DOCKING_NORM_DENOM)
                        except Exception:
                            x = 0.0
                        return float(max(0.0, min(1.0, x)))

                    ds_target_norm = float(scores[i_main])
                    ds_target_norm = float(max(0.0, min(1.0, ds_target_norm)))
                    ds_anti_norm = _dock_norm_from_affinity(raw_scores[i_anti])

                    gap = float(max(0.0, min(1.0, ds_target_norm - ds_anti_norm)))

                    qed_val = float(scores[qed_idx]) if qed_idx is not None else 0.0
                    sa_val = float(scores[sa_idx]) if sa_idx is not None else 0.0
                    qed_val = float(max(0.0, min(1.0, qed_val)))
                    sa_val = float(max(0.0, min(1.0, sa_val)))
                    # IMPORTANT: do NOT overwrite `scores` here.
                    # `scores` must stay aligned with `self.objectives` for correct logging/CSV/debugging.
                    selectivity_vec = np.array([ds_target_norm, gap, qed_val, sa_val])
                    reward = np.matmul(selectivity_vec, self.weights.reshape(-1, 1))
        except Exception:
            pass
        
        # Always write molecules.csv for any successfully-scored molecule.
        from rdkit import Chem
        try:
            mol_smiles = Chem.MolToSmiles(mol)
        except Exception:
            mol_smiles = None

        if mol_smiles:
            try:
                self.record_objectives(mol_smiles, self.objectives, raw_scores)
            except Exception:
                pass

            # Optional hit-count tracking (only for true hit tasks).
            if self.is_hit_task:
                try:
                    qed_idx2 = self.objectives_lower.index('qed') if 'qed' in self.objectives_lower else None
                    sa_idx2 = self.objectives_lower.index('sa') if 'sa' in self.objectives_lower else None
                    dock_idx2 = None
                    hit_docking_targets = ['parp1', 'fa7', '5ht1b', 'braf', 'jak2']
                    for j, obj_lower in enumerate(self.objectives_lower):
                        if obj_lower in hit_docking_targets:
                            dock_idx2 = j
                            break
                    if qed_idx2 is not None and sa_idx2 is not None and dock_idx2 is not None:
                        is_hit = self._check_hit(raw_scores[qed_idx2], raw_scores[sa_idx2], raw_scores[dock_idx2])
                        if is_hit:
                            self.hit_count += 1
                except Exception:
                    pass
        

        return reward, scores, raw_scores

    def _log_reward_scores(self, smiles: str, reward, scores: np.ndarray, raw_scores: np.ndarray):
        """
        Log the exact per-objective values used for reward calculation:
        - raw_scores: oracle raw outputs (used for hit checking + CSV)
        - scores: normalized/direction-correct values used in reward computation
        - reward: scalar used by optimizer
        """
        self._log_reward_scores_counter += 1
        if (self._log_reward_scores_counter % self.log_reward_scores_every) != 0:
            return

        try:
            reward_f = float(reward)
        except Exception:
            try:
                reward_f = float(np.array(reward).reshape(-1)[0])
            except Exception:
                reward_f = 0.0

        # Best-effort hit flag (based on raw scores) for debugging.
        is_hit = None
        if self.is_hit_task:
            try:
                qed_idx = self.objectives_lower.index('qed') if 'qed' in self.objectives_lower else None
                sa_idx = self.objectives_lower.index('sa') if 'sa' in self.objectives_lower else None
                dock_idx = None
                docking_targets = ['parp1', 'fa7', '5ht1b', 'braf', 'jak2']
                for i, obj_lower in enumerate(self.objectives_lower):
                    if obj_lower in docking_targets:
                        dock_idx = i
                        break
                if qed_idx is not None and sa_idx is not None and dock_idx is not None:
                    is_hit = bool(self._check_hit(raw_scores[qed_idx], raw_scores[sa_idx], raw_scores[dock_idx]))
            except Exception:
                is_hit = None

        parts = []
        for i, obj in enumerate(self.objectives):
            try:
                parts.append(
                    f"{obj}:raw={float(raw_scores[i]):.6f}|norm={float(scores[i]):.6f}"
                )
            except Exception:
                parts.append(f"{obj}:raw=?|norm=?")

        # Single-line log for easy grepping.
        self._logger.info(
            "[reward_scores] n=%d smiles=%s reward=%.6f is_hit=%s %s",
            len(self.mol_buffer) + 1,
            smiles,
            reward_f,
            str(is_hit),
            " ".join(parts),
        )

    def score_smi(self, smi):
        """
        Function to score one molecule

        Argguments:
            smi: One SMILES string represnets a moelcule.

        Return:
            score: a float represents the property of the molecule.
        """
        if len(self.mol_buffer) > self.max_oracle_calls:
            return 0
        if smi is None:
            return 0
        mol = Chem.MolFromSmiles(smi)
        if mol is None or len(smi) == 0:
            return 0
        else:
            smi = Chem.MolToSmiles(mol)
            if smi in self.mol_buffer:
                pass
            else:
                reward, scores, raw_scores = self.moo_evaluator(mol)
                if self.log_reward_scores:
                    try:
                        self._log_reward_scores(smi, reward, scores, raw_scores)
                    except Exception:
                        pass
                # Store: [reward, oracle_call_number, normalized_scores, raw_scores]
                self.mol_buffer[smi] = [float(reward), len(self.mol_buffer)+1, scores, raw_scores]
            return self.mol_buffer[smi][0]
    
    def score_smi_batch(self, smis):
        """
        Function to score a batch of molecules efficiently.

        Arguments:
            smis: List of SMILES strings representing molecules.

        Return:
            scores: List of floats representing the properties of the molecules.
        """
        print(f"[DEBUG score_smi_batch] Starting with {len(smis)} SMILES, buffer_size={len(self.mol_buffer)}")
        
        # Step 1: Collect valid, uncached molecules
        mols_to_score = []
        smi_to_mol_map = {}  # Maps canonical SMILES to RDKit mol object
        
        for smi in smis:
            if len(self.mol_buffer) >= self.max_oracle_calls:
                break
            if smi is None:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None or len(smi) == 0:
                continue
            
            smi_canon = Chem.MolToSmiles(mol)
            if smi_canon not in self.mol_buffer:
                # Only score molecules not in buffer
                if smi_canon not in smi_to_mol_map:
                    mols_to_score.append(mol)
                    smi_to_mol_map[smi_canon] = mol
        
        print(f"[DEBUG score_smi_batch] Found {len(mols_to_score)} new molecules to score (rest are cached)")
        
        # Step 2: Batch score all new molecules
        if mols_to_score:
            print(f"[DEBUG score_smi_batch] Calling moo_evaluator_batch with {len(mols_to_score)} molecules")
            try:
                rewards, all_scores, all_raw_scores = self.moo_evaluator_batch(mols_to_score)
                print(f"[DEBUG score_smi_batch] moo_evaluator_batch returned {len(rewards)} rewards")
            except Exception as e:
                print(f"[DEBUG score_smi_batch] ERROR in moo_evaluator_batch: {e}")
                import traceback
                traceback.print_exc()
                raise
            
            # Step 3: Store results in buffer
            print(f"[DEBUG score_smi_batch] Storing {len(mols_to_score)} results in buffer")
            for idx, mol in enumerate(mols_to_score):
                smi_canon = Chem.MolToSmiles(mol)
                reward = rewards[idx]
                scores = all_scores[idx]
                raw_scores = all_raw_scores[idx]
                
                # Optional reward logging
                if self.log_reward_scores:
                    try:
                        self._log_reward_scores(smi_canon, reward, scores, raw_scores)
                    except Exception:
                        pass
                
                # Store: [reward, oracle_call_number, normalized_scores, raw_scores]
                self.mol_buffer[smi_canon] = [float(reward), len(self.mol_buffer)+1, scores, raw_scores]
            print(f"[DEBUG score_smi_batch] Buffer now has {len(self.mol_buffer)} molecules")
        
        # Step 4: Return scores for all input SMILES (including cached and invalid)
        print(f"[DEBUG score_smi_batch] Building score_list for {len(smis)} input SMILES")
        score_list = []
        for smi in smis:
            if smi is None:
                score_list.append(0)
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None or len(smi) == 0:
                score_list.append(0)
                continue
            
            smi_canon = Chem.MolToSmiles(mol)
            score_list.append(self.mol_buffer.get(smi_canon, [0])[0])
        
        print(f"[DEBUG score_smi_batch] Returning {len(score_list)} scores")
        return score_list

    def __call__(self, smiles_lst):
        """
        Score molecules (single SMILES string or list of SMILES).
        """
        print(f"[DEBUG Oracle.__call__] Received {type(smiles_lst).__name__}, len={len(smiles_lst) if isinstance(smiles_lst, list) else 'N/A'}")
        
        if type(smiles_lst) == list:
            # Batch scoring for list of SMILES
            print(f"[DEBUG Oracle.__call__] Calling score_smi_batch with {len(smiles_lst)} SMILES")
            score_list = self.score_smi_batch(smiles_lst)
            print(f"[DEBUG Oracle.__call__] Got {len(score_list) if score_list else 0} scores back, type={type(score_list).__name__}")
            if len(self.mol_buffer) % self.freq_log == 0 and len(self.mol_buffer) > self.last_log:
                self.sort_buffer()
                self.log_intermediate()
                self.last_log = len(self.mol_buffer)
                self.save_result(self.task_label)
        else:
            # Single SMILES string
            print(f"[DEBUG Oracle.__call__] Calling score_smi with single SMILES")
            score_list = self.score_smi(smiles_lst)
            if len(self.mol_buffer) % self.freq_log == 0 and len(self.mol_buffer) > self.last_log:
                self.sort_buffer()
                self.log_intermediate()
                self.last_log = len(self.mol_buffer)
                self.save_result(self.task_label)
        print(f"[DEBUG Oracle.__call__] Returning score_list, type={type(score_list).__name__}")
        return score_list

    @property
    def finish(self):
        return len(self.mol_buffer) >= self.max_oracle_calls
    
class BaseOptimizer:

    def __init__(self, args=None):
        self.model_name = "Default"
        self.args = args
        self.n_jobs = args.n_jobs
        # self.pool = joblib.Parallel(n_jobs=self.n_jobs)
        self.smi_file = args.smi_file
        self.oracle = Oracle(args=self.args)
        # Populated after each optimize() call (before reset()) for external callers like run.py.
        self.last_run_summary = None
        # if self.smi_file is not None:
        #     self.all_smiles = self.load_smiles_from_file(self.smi_file)
        # else:
        #     data = MolGen(name = 'ZINC')
        #     self.all_smiles = data.get_data()['smiles'].tolist()
            
        # self.sa_scorer = tdc.Oracle(name = 'SA')
        # self.diversity_evaluator = tdc.Evaluator(name = 'Diversity')
        # self.filter = tdc.chem_utils.oracle.filter.MolFilter(filters = ['PAINS', 'SureChEMBL', 'Glaxo'], property_filters_flag = False)

    # def load_smiles_from_file(self, file_name):
    #     with open(file_name) as f:
    #         return self.pool(delayed(canonicalize)(s.strip()) for s in f)
            
    def sanitize(self, mol_list):
        new_mol_list = []
        smiles_set = set()
        for mol in mol_list:
            if mol is not None:
                try:
                    smiles = Chem.MolToSmiles(mol)
                    if smiles is not None and smiles not in smiles_set:
                        smiles_set.add(smiles)
                        new_mol_list.append(mol)
                except ValueError:
                    print('bad smiles')
        return new_mol_list
        
    def sort_buffer(self):
        self.oracle.sort_buffer()
    
    def log_intermediate(self, mols=None, scores=None, finish=False):
        self.oracle.log_intermediate(mols=mols, scores=scores, finish=finish)
    
    def log_result(self):

        print(f"Logging final results...")

        # import ipdb; ipdb.set_trace()

        log_num_oracles = [100, 500, 1000, 3000, 5000, 10000]
        assert len(self.mol_buffer) > 0 

        results = list(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][1], reverse=False))
        if len(results) > 10000:
            results = results[:10000]
        
        results_all_level = []
        for n_o in log_num_oracles:
            results_all_level.append(sorted(results[:n_o], key=lambda kv: kv[1][0], reverse=True))
        
        # Currently logging the top-100 moelcules, will update to PDD selection later
        # data = [[i+1, results_all_level[-1][i][1][0], results_all_level[-1][i][1][1], \
        #         wandb.Image(Draw.MolToImage(Chem.MolFromSmiles(results_all_level[-1][i][0]))), results_all_level[-1][i][0]] for i in range(100)]
        # columns = ["Rank", "Score", "#Oracle", "Image", "SMILES"]
        # wandb.log({"Top 100 Molecules": wandb.Table(data=data, columns=columns)})
        
        # # Log batch metrics at various oracle calls
        # data = [[log_num_oracles[i]] + self._analyze_results(r) for i, r in enumerate(results_all_level)]
        # columns = ["#Oracle", "avg_top100", "avg_top10", "avg_top1", "Diversity", "avg_SA", "%Pass", "Top-1 Pass"]
        # wandb.log({"Batch metrics at various level": wandb.Table(data=data, columns=columns)})
        
    def save_result(self, suffix=None):

        print(f"Saving molecules...")
        
        if suffix is None:
            output_file_path = os.path.join(self.args.output_dir, 'results.yaml')
        else:
            output_file_path = os.path.join(self.args.output_dir, 'results_' + suffix + '.yaml')

        self.sort_buffer()
        with open(output_file_path, 'w') as f:
            yaml.dump(self.mol_buffer, f, sort_keys=False)
    
    def _analyze_results(self, results):
        results = results[:100]
        scores_dict = {item[0]: item[1][0] for item in results}
        smis = [item[0] for item in results]
        scores = [item[1][0] for item in results]
        smis_pass = self.filter(smis)
        if len(smis_pass) == 0:
            top1_pass = -1
        else:
            top1_pass = np.max([scores_dict[s] for s in smis_pass])
        return [np.mean(scores), 
                np.mean(scores[:10]), 
                np.max(scores), 
                self.diversity_evaluator(smis), 
                np.mean(self.sa_scorer(smis)), 
                float(len(smis_pass) / 100), 
                top1_pass]

    def reset(self):
        del self.oracle
        self.oracle = Oracle(args=self.args, mol_buffer={})

    @property
    def mol_buffer(self):
        return self.oracle.mol_buffer

    @property
    def finish(self):
        return self.oracle.finish
        
    def _optimize(self, oracle, config):
        raise NotImplementedError
            
    def optimize(self, oracle, config, seed=0, project="test"):
        # run = wandb.init(project=project, config=config, reinit=True, entity="mol_opt")
        if self.args.wandb != 'disabled':
            project = 'pmo' if self.args.method.startswith('genetic_gfn') else 'pmo_baselines'
            run = wandb.init(project=project, group=oracle.name, config=config, reinit=True)
            wandb.config.oracle = oracle.name
            wandb.config.method = self.args.method
            wandb.run.name = oracle.name + "_" + self.args.method + "_" + self.args.run_name + "_" + str(seed) + "_" + wandb.run.id
        # wandb.run.name = self.model_name + "_" + oracle.name + "_" + wandb.run.id
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        self.seed = seed
        # Set task_label for CSV file naming (use run_name if available, otherwise generate from seed)
        if hasattr(self.args, 'run_name') and self.args.run_name != "default":
            self.oracle.task_label = self.args.run_name + "_seed" + str(seed)
        else:
            self.oracle.task_label = self.model_name + "_seed" + str(seed)
        # --- Per-run docking temp directory cleanup ---
        # Docking creates temporary ligand/pdbqt files. To avoid collisions in parallel runs and to
        # keep the workspace clean, we route docking temp files into a per-run directory and delete
        # it after the run finishes (best-effort).
        output_dir_abs = os.path.abspath(self.args.output_dir)
        # Use task_label (includes run_name + seed) so parallel runs with the same seed don't collide.
        task_label = getattr(self.oracle, "task_label", f"seed_{seed}")
        desired_tmp = os.path.join(output_dir_abs, "docking_tmp", str(task_label))
        existing_tmp = os.environ.get("DOCKING_TMP_DIR")
        if existing_tmp:
            docking_tmp_dir = existing_tmp
            # Only auto-delete if it's inside this run's output dir (safety).
            cleanup_tmp = os.path.abspath(docking_tmp_dir).startswith(output_dir_abs + os.sep) or os.path.abspath(docking_tmp_dir) == output_dir_abs
            set_env = False
        else:
            docking_tmp_dir = desired_tmp
            cleanup_tmp = True
            set_env = True

        os.makedirs(docking_tmp_dir, exist_ok=True)
        if set_env:
            os.environ["DOCKING_TMP_DIR"] = docking_tmp_dir

        try:
            self._optimize(oracle, config)
            # If the inner loop ended without explicitly setting a stop reason, assume budget reached
            # when mol_buffer hit the cap; otherwise mark as "completed" (best-effort).
            if getattr(self.oracle, "stop_reason", None) is None:
                try:
                    if self.finish:
                        self.oracle.stop_reason = "max_oracle_calls"
                    else:
                        self.oracle.stop_reason = "completed"
                except Exception:
                    self.oracle.stop_reason = "completed"
        except Exception as e:
            # Surface a structured reason for run.py summaries, then re-raise.
            try:
                self.oracle.stop_reason = f"exception:{type(e).__name__}"
            except Exception:
                pass
            raise
        finally:
            if cleanup_tmp:
                shutil.rmtree(docking_tmp_dir, ignore_errors=True)
            if set_env:
                os.environ.pop("DOCKING_TMP_DIR", None)
        if self.args.log_results:
            self.log_result()
        # Use task_label to avoid overwriting when multiple runs share the same output_dir.
        self.save_result(self.oracle.task_label)

        # Save a minimal run summary for callers (before reset() clears the oracle buffer).
        try:
            yaml_path = os.path.join(self.args.output_dir, f"results_{self.oracle.task_label}.yaml")
            mol_csv_path = getattr(self.oracle, "molecules_csv_file", None)
            self.last_run_summary = {
                "task_label": str(getattr(self.oracle, "task_label", "")),
                "output_dir": str(getattr(self.args, "output_dir", "")),
                "stop_reason": str(getattr(self.oracle, "stop_reason", "")),
                "n_unique_molecules": int(len(getattr(self.oracle, "mol_buffer", {}) or {})),
                "results_yaml": str(yaml_path),
                "molecules_csv": str(mol_csv_path) if mol_csv_path else "",
            }
        except Exception:
            self.last_run_summary = None
        # self.reset()
        if self.args.wandb != 'disabled':
            run.finish()
        self.reset()

    def production(self, oracle, config, num_runs=5, project="production"):
        # Production seed pool (hard-coded).
        # NOTE: Intentionally fixed to seeds 0-9 so `--n_runs 10` runs exactly these.
        seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        if num_runs is None:
            num_runs = len(seeds)

        if num_runs > len(seeds):
            raise ValueError(f"Requested num_runs={num_runs} but only {len(seeds)} seeds are available/provided.")

        seeds = seeds[:num_runs]
        for seed in seeds:
            self.optimize(oracle, config, seed, project)
            self.reset()


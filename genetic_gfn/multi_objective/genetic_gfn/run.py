import os
import sys
import numpy as np
import time
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
# Ensure multi_objective/ is importable regardless of current working directory.
# This enables imports like `from optimizer import BaseOptimizer` and `from oracle...`.
multi_objective_dir = os.path.abspath(os.path.join(path_here, ".."))
if multi_objective_dir not in sys.path:
    sys.path.insert(0, multi_objective_dir)

# Keep repo root on sys.path (historical behavior).
sys.path.append('/'.join(path_here.rstrip('/').split('/')[:-2]))
from optimizer import BaseOptimizer
from genetic_gfn.utils import Variable, seq_to_smiles, unique
from genetic_gfn.model import RNN
from genetic_gfn.data_structs import Vocabulary, Experience
import torch
from rdkit import Chem

from joblib import Parallel
from genetic_gfn.graph_ga_expert import GeneticOperatorHandler


def sanitize(smiles):
    canonicalized = []
    for s in smiles:
        try:
            canonicalized.append(Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True))
        except:
            pass
    return canonicalized


class Genetic_GFN_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "genetic_gfn"

    def _optimize(self, oracle, config):

        # self.oracle.assign_evaluator(oracle)
        self.oracle.set_objectives(*(oracle))

        path_here = os.path.dirname(os.path.realpath(__file__))
        restore_prior_from=os.path.join(path_here, 'data/Prior.ckpt')
        restore_agent_from=restore_prior_from 
        voc = Vocabulary(init_from_file=os.path.join(path_here, "data/Voc"))

        Prior = RNN(voc)
        Agent = RNN(voc)

        # By default restore Agent to same model as Prior, but can restore from already trained Agent too.
        # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
        # to the CPU.
        if torch.cuda.is_available():
            Prior.rnn.load_state_dict(torch.load(os.path.join(path_here,'data/Prior.ckpt')))
            Agent.rnn.load_state_dict(torch.load(restore_agent_from))
        else:
            Prior.rnn.load_state_dict(torch.load(os.path.join(path_here, 'data/Prior.ckpt'), map_location=lambda storage, loc: storage))
            Agent.rnn.load_state_dict(torch.load(restore_agent_from, map_location=lambda storage, loc: storage))

        # We dont need gradients with respect to Prior
        for param in Prior.rnn.parameters():
            param.requires_grad = False

        # optimizer = torch.optim.Adam(Agent.rnn.parameters(), lr=config['learning_rate'])
        log_z = torch.nn.Parameter(torch.tensor([5.]).cuda())
        optimizer = torch.optim.Adam([{'params': Agent.rnn.parameters(), 
                                        'lr': config['learning_rate']},
                                    {'params': log_z, 
                                        'lr': config['lr_z']}])

        # For policy based RL, we normally train on-policy and correct for the fact that more likely actions
        # occur more often (which means the agent can get biased towards them). Using experience replay is
        # therefor not as theoretically sound as it is for value based RL, but it seems to work well.
        experience = Experience(voc, max_size=config['num_keep'])

        ga_handler = GeneticOperatorHandler(mutation_rate=config['mutation_rate'], 
                                            population_size=config['population_size'])
        pool = Parallel(n_jobs=config['num_jobs'])

        # Optional warm-start: if a lead/seed molecule is provided, score it once and
        # add it to both the oracle buffer and experience replay memory.
        # This makes it eligible for GA mating pool selection and early replay sampling.
        seed_mol = getattr(self.args, "seed_mol", "") if hasattr(self, "args") else ""
        seed_smi_canon = None
        seed_score = None
        force_seed_in_first_mating_pool = False
        if isinstance(seed_mol, str) and seed_mol.strip():
            try:
                m = Chem.MolFromSmiles(seed_mol.strip())
                if m is not None:
                    seed_smi_canon = Chem.MolToSmiles(m)
                    seed_score = float(self.oracle([seed_smi_canon])[0])
                    experience.add_experience([(seed_smi_canon, seed_score)])
                    # Force-include the seed in the *first* GA mating pool even if it is not in top-k yet.
                    # After the first GA step, it will only remain in the mating pool naturally (if it stays in top-k).
                    force_seed_in_first_mating_pool = True
                    print(f"[warm_start] added seed_mol to pool: smiles={seed_smi_canon} score={seed_score:.6f}")
                else:
                    print(f"[warm_start] seed_mol invalid (RDKit MolFromSmiles failed): {seed_mol!r}")
            except Exception as e:
                print(f"[warm_start] failed to add seed_mol={seed_mol!r}: {e}")

        print("Model initialized, starting training...")
        t_start = time.time()

        # Emit a lightweight progress heartbeat to stdout so users can tail logs.
        # We log approximately every `args.freq_log` *new oracle calls*.
        freq_log = int(getattr(self.args, "freq_log", 100) or 100)
        last_progress_calls = 0

        step = 0
        patience = 0
        prev_n_oracles = 0
        stuck_cnt = 0

        # Timing metrics for bottleneck analysis
        total_oracle_time = 0.0
        total_gpu_sampling_time = 0.0
        total_gpu_training_time = 0.0
        total_ga_time = 0.0

        while True:

            if len(self.oracle) > 100:
                self.sort_buffer()
                old_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
            else:
                old_scores = 0
            
            # Sample from Agent
            t_gpu_sample_start = time.time()
            seqs, agent_likelihood, entropy = Agent.sample(config['batch_size'])
            t_gpu_sample_end = time.time()
            total_gpu_sampling_time += (t_gpu_sample_end - t_gpu_sample_start)

            # Get prior likelihood and score
            # Convert all sequences to SMILES (keep all batch_size molecules, no deduplication)
            smiles = seq_to_smiles(seqs, voc)
            if config['valid_only']:
                smiles = sanitize(smiles)
            
            # Send full batch to oracle (optimizer handles caching for duplicates in buffer)
            print(f"[DEBUG run.py] Step {step}: About to call oracle with {len(smiles)} SMILES")
            print(f"[DEBUG run.py] First 3 SMILES: {smiles[:3] if len(smiles) >= 3 else smiles}")
            t_oracle_start = time.time()
            oracle_result = self.oracle(smiles)
            print(f"[DEBUG run.py] Oracle returned type={type(oracle_result).__name__}, len={len(oracle_result) if hasattr(oracle_result, '__len__') else 'N/A'}")
            score = np.array(oracle_result)
            print(f"[DEBUG run.py] Converted to numpy array, shape={score.shape}, dtype={score.dtype}")
            
            # Verify length matching
            if len(smiles) != len(score):
                print(f"[ERROR run.py] LENGTH MISMATCH! len(smiles)={len(smiles)}, len(score)={len(score)}")
                print(f"[ERROR run.py] This will cause issues in zip(smiles, score)")
            else:
                print(f"[DEBUG run.py] Length check OK: both have {len(smiles)} elements")
            
            t_oracle_end = time.time()
            oracle_time_batch = t_oracle_end - t_oracle_start
            total_oracle_time += oracle_time_batch
            print(f"[timing] step={step} agent_oracle: {len(smiles)} mols (batch_size={config['batch_size']}) in {oracle_time_batch:.3f}s ({oracle_time_batch/max(len(smiles),1):.4f}s/mol)")
            
            # Now deduplicate for experience replay (to avoid adding duplicate experiences)
            # Use SMILES-based deduplication since seq-based might not match sanitized smiles
            seen = set()
            unique_smiles = []
            unique_scores = []
            print(f"[DEBUG run.py] Starting deduplication from {len(smiles)} molecules")
            for smi, sc in zip(smiles, score):
                if smi not in seen:
                    seen.add(smi)
                    unique_smiles.append(smi)
                    unique_scores.append(sc)
            
            print(f"[DEBUG run.py] After deduplication: {len(unique_smiles)} unique molecules (removed {len(smiles) - len(unique_smiles)} duplicates)")
            smiles_unique = unique_smiles
            score_unique = np.array(unique_scores)
            print(f"[DEBUG run.py] Adding {len(smiles_unique)} experiences to replay buffer")

            if self.finish:
                try:
                    self.oracle.stop_reason = "max_oracle_calls"
                except Exception:
                    pass
                print('max oracle hit')
                break 

            # Progress heartbeat (stdout).
            try:
                n_calls = int(len(self.oracle))
            except Exception:
                n_calls = 0
            if freq_log > 0 and n_calls >= (last_progress_calls + freq_log):
                # Snap to the last completed logging boundary for stable output.
                last_progress_calls = (n_calls // freq_log) * freq_log
                best_smi = None
                best_reward = None
                try:
                    # mol_buffer entries: {smiles: (reward, ...)}
                    self.oracle.sort_buffer()
                    best_smi, best_elem = next(iter(self.oracle.mol_buffer.items()))
                    if isinstance(best_elem, (list, tuple)) and len(best_elem) > 0:
                        best_reward = float(best_elem[0])
                    else:
                        best_reward = float(best_elem)
                except Exception:
                    pass
                dt_s = time.time() - t_start
                if best_reward is not None and best_smi is not None:
                    print(
                        f"[progress] oracle_calls={n_calls} step={step} best_reward={best_reward:.6f} "
                        f"best_smiles={best_smi} elapsed_s={dt_s:.1f}"
                    )
                else:
                    print(f"[progress] oracle_calls={n_calls} step={step} elapsed_s={dt_s:.1f}")

            # early stopping
            if len(self.oracle) > 1000:
                self.sort_buffer()
                new_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
                if new_scores == old_scores:
                    patience += 1
                    if patience >= self.args.patience:
                        self.log_intermediate(finish=True)
                        try:
                            self.oracle.stop_reason = "convergence"
                        except Exception:
                            pass
                        print('convergence criteria met, abort ...... ')
                        break
                else:
                    patience = 0

            # early stopping
            if prev_n_oracles < len(self.oracle):
                stuck_cnt = 0
            else:
                stuck_cnt += 1
                if stuck_cnt >= 10:
                    self.log_intermediate(finish=True)
                    try:
                        self.oracle.stop_reason = "stuck_no_new_molecules"
                    except Exception:
                        pass
                    print('cannot find new molecules, abort ...... ')
                    break
            
            prev_n_oracles = len(self.oracle)

            # Calculate augmented likelihood
            # augmented_likelihood = prior_likelihood.float() + 500 * Variable(score).float()
            # reinvent_loss = torch.pow((augmented_likelihood - agent_likelihood), 2)
            # print('REINVENT:', reinvent_loss.mean().item())

            # Then add new experience (use unique molecules only to avoid duplicate experiences)
            new_experience = zip(smiles_unique, score_unique)
            experience.add_experience(new_experience)

            if config['population_size'] and len(self.oracle) > config['population_size']:
                self.oracle.sort_buffer()
                pop_smis, pop_scores = tuple(map(list, zip(*[(smi, elem[0]) for (smi, elem) in self.oracle.mol_buffer.items()])))

                # GA mating pool is the current top `num_keep` molecules by reward.
                # If a `seed_mol` was provided, force it into the *first* GA mating pool to guide early search,
                # but do not keep it forever (afterwards it must remain in the top list naturally).
                mating_smis = pop_smis[:config['num_keep']]
                mating_scores = pop_scores[:config['num_keep']]
                if force_seed_in_first_mating_pool and seed_smi_canon:
                    if seed_smi_canon not in mating_smis:
                        # Insert at front and trim back to keep pool size constant.
                        mating_smis = [seed_smi_canon] + mating_smis
                        mating_scores = [float(seed_score) if seed_score is not None else 0.0] + mating_scores
                        mating_smis = mating_smis[:config['num_keep']]
                        mating_scores = mating_scores[:config['num_keep']]
                    force_seed_in_first_mating_pool = False

                populations = (mating_smis, mating_scores)
                # populations = select_pop(pop_smis, pop_scores, config['population_size'], rank_coefficient=config['rank_coefficient'])

                for g in range(config['ga_generations']):
                    t_ga_start = time.time()
                    child_smis, child_n_atoms, pop_smis, pop_scores = ga_handler.query(
                            query_size=config['offspring_size'], mating_pool=populations, pool=pool, 
                            rank_coefficient=config['rank_coefficient'], 
                        )
                    t_ga_end = time.time()
                    ga_time_batch = t_ga_end - t_ga_start
                    total_ga_time += ga_time_batch

                    t_oracle_start = time.time()
                    child_score = np.array(self.oracle(child_smis))
                    t_oracle_end = time.time()
                    oracle_time_batch = t_oracle_end - t_oracle_start
                    total_oracle_time += oracle_time_batch
                    print(f"[timing] step={step} ga_gen={g} oracle: {len(child_smis)} mols in {oracle_time_batch:.3f}s ({oracle_time_batch/max(len(child_smis),1):.4f}s/mol)")
                
                    new_experience = zip(child_smis, child_score)
                    experience.add_experience(new_experience)

                    # import pdb; pdb.set_trace()
                    populations = (pop_smis+child_smis, pop_scores+child_score.tolist())

                    if self.finish:
                        try:
                            self.oracle.stop_reason = "max_oracle_calls"
                        except Exception:
                            pass
                        print('max oracle hit')
                        break
                
            # Experience Replay
            # First sample
            print(f"[DEBUG run.py] Starting training phase, experience buffer size={len(experience)}")
            avg_loss = 0.
            t_training_start = time.time()
            if config['experience_replay'] and len(experience) > config['experience_replay']:
                print(f"[DEBUG run.py] Running {config['experience_loop']} training loops")
                for _ in range(config['experience_loop']):
                    if config['rank_coefficient'] > 0:
                        exp_seqs, exp_score = experience.rank_based_sample(config['experience_replay'], config['rank_coefficient'])
                    else:
                        exp_seqs, exp_score = experience.sample(config['experience_replay'])

                    exp_agent_likelihood, _ = Agent.likelihood(exp_seqs.long())
                    prior_agent_likelihood, _ = Prior.likelihood(exp_seqs.long())

                    reward = torch.tensor(exp_score).cuda()
                    exp_forward_flow = exp_agent_likelihood + log_z
                    exp_backward_flow = reward * config['beta']
                    loss = torch.pow(exp_forward_flow - exp_backward_flow, 2).mean()

                    # KL penalty
                    if config['penalty'] == 'prior_kl':
                        loss_p = (exp_agent_likelihood - prior_agent_likelihood).mean()
                        loss += config['kl_coefficient']*loss_p

                    # print(loss.item())
                    avg_loss += loss.item()/config['experience_loop']

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            t_training_end = time.time()
            training_time_step = t_training_end - t_training_start
            total_gpu_training_time += training_time_step
            print(f"[DEBUG run.py] Training phase complete, took {training_time_step:.3f}s")

            step += 1
            print(f"[DEBUG run.py] ===== Completed step {step-1}, moving to step {step} =====")
            
            # Log timing breakdown every 10 steps
            if step % 10 == 0:
                elapsed = time.time() - t_start
                oracle_pct = (total_oracle_time / elapsed) * 100 if elapsed > 0 else 0
                gpu_sample_pct = (total_gpu_sampling_time / elapsed) * 100 if elapsed > 0 else 0
                gpu_train_pct = (total_gpu_training_time / elapsed) * 100 if elapsed > 0 else 0
                ga_pct = (total_ga_time / elapsed) * 100 if elapsed > 0 else 0
                other_pct = 100 - oracle_pct - gpu_sample_pct - gpu_train_pct - ga_pct
                print(f"[timing_breakdown] step={step} elapsed={elapsed:.1f}s | "
                      f"oracle={total_oracle_time:.1f}s ({oracle_pct:.1f}%) | "
                      f"gpu_sample={total_gpu_sampling_time:.1f}s ({gpu_sample_pct:.1f}%) | "
                      f"gpu_train={total_gpu_training_time:.1f}s ({gpu_train_pct:.1f}%) | "
                      f"ga={total_ga_time:.1f}s ({ga_pct:.1f}%) | "
                      f"other={other_pct:.1f}%")


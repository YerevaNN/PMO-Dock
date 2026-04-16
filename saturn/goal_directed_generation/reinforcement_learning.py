"""
Adapted from https://github.com/MolecularAI/Reinvent with code additions for:
    1. Augmented Memory: https://pubs.acs.org/doi/10.1021/jacsau.4c00066
    2. Beam Enumeration: https://openreview.net/forum?id=7UhxsmbdaQ
    3. Hallucinated Memory (GraphGA based: https://pubs.rsc.org/en/content/articlelanding/2019/sc/c8sc05372c)
"""
from typing import Tuple
import os
import logging
import time
import torch
import numpy as np

import utils.chemistry_utils as chemistry_utils
from utils.chemistry_utils import is_encodable
from goal_directed_generation.utils import sample_unique_sequences
from utils.utils import to_tensor, setup_logging, get_gpu_memory_stats, format_gpu_memory

from oracles.oracle import Oracle
from goal_directed_generation.dataclass import GoalDirectedGenerationConfiguration

from models.generator import Generator
from experience_replay.replay_buffer import ReplayBuffer
from diversity_filter.diversity_filter import DiversityFilter
from hallucinated_memory.utils import initialize_hallucinator
from beam_enumeration.beam_enumeration import BeamEnumeration

# Syntheseus oracle for custom results write-out
from oracles.synthesizability.syntheseus import Syntheseus



class ReinforcementLearningAgent:
    """
    RL agent for goal-directed generation.
    """
    def __init__(
        self,
        logging_frequency: int,
        logging_path: str,
        model_checkpoints_dir: str,
        oracle: Oracle,
        configuration: GoalDirectedGenerationConfiguration,
        device: str
    ):
        self.prior = Generator.load_from_file(configuration.reinforcement_learning.prior, device)
        # Prior model is not updated so disable gradients
        self._disable_prior_gradients()
        self.agent = Generator.load_from_file(configuration.reinforcement_learning.agent, device)
        self.device = self.agent.device
        # In case the Agent is to be trained on CPU, move also the Prior to CPU to avoid tensors on different devices
        self.prior.network.to(self.device)

        # Seed for documentation
        self.seed = configuration.seed

        # Oracle
        self.oracle = oracle
        # RL parameters
        self.batch_size = configuration.reinforcement_learning.batch_size
        self.learning_rate = configuration.reinforcement_learning.learning_rate
        self.sigma = configuration.reinforcement_learning.sigma
        self.augmented_memory = configuration.reinforcement_learning.augmented_memory
        self.augmentation_rounds = configuration.reinforcement_learning.augmentation_rounds
        self.selective_memory_purge = configuration.reinforcement_learning.selective_memory_purge

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(parameters=configuration.experience_replay)
        
        if self.replay_buffer.lead_smiles_in_buffer:
            self.oracle = self.replay_buffer.prepopulate_buffer_with_lead_smiles(self.oracle)
        else:
            self.oracle = self.replay_buffer.prepopulate_buffer(self.oracle)

        # Diversity Filter
        self.diversity_filter = DiversityFilter(configuration.diversity_filter)

        # Hallucinated Memory
        self.execute_hallucinated_memory = configuration.hallucinated_memory.execute_hallucinated_memory
        self.hallucinator = initialize_hallucinator(
            prior=self.prior,
            parameters=configuration.hallucinated_memory
        )

        # Beam Enumeration
        self.execute_beam_enumeration = configuration.beam_enumeration.execute_beam_enumeration
        self.beam_enumeration = BeamEnumeration(
            k=configuration.beam_enumeration.beam_k,
            beam_steps=configuration.beam_enumeration.beam_steps,
            substructure_type=configuration.beam_enumeration.substructure_type.lower(),
            substructure_min_size=configuration.beam_enumeration.structure_min_size,
            pool_size=configuration.beam_enumeration.pool_size,
            pool_saving_frequency=configuration.beam_enumeration.pool_saving_frequency,
            patience=configuration.beam_enumeration.patience,
            token_sampling_method=configuration.beam_enumeration.token_sampling_method,
            filter_patience_limit=configuration.beam_enumeration.filter_patience_limit
        )

        # Only the Agent is updated so the Prior does not need an optimizer
        self.optimizer = torch.optim.Adam(self.agent.get_network_parameters(), lr=self.learning_rate)

        # Model checkpointing save directory
        self.model_checkpoints_dir = model_checkpoints_dir
        os.makedirs(self.model_checkpoints_dir, exist_ok=True)
        self.logging_path = logging_path
        self.logging_frequency = logging_frequency
        self.logging_multiple = 1

        # Best Agent checkpointing
        self.best_agent_reward = float("-inf")
        self.patience = 0

        # Set up logging
        setup_logging(logging_path)

        # Log GPU memory after loading both models (prior + agent)
        mem_str = format_gpu_memory(self.device)
        if mem_str:
            logging.getLogger(__name__).info("GPU memory after loading models: %s", mem_str)
  
    def run(self):
        log = logging.getLogger(__name__)
        mol_counts_trend = [0]
        start_time = time.perf_counter()
        batch_number = 0
        log.info("RL start budget=%s", self.oracle.budget)
        while not self.oracle.budget_exceeded():
            batch_number += 1
            batch_start = time.perf_counter()

            seqs, smiles, _ = sample_unique_sequences(self.agent, self.batch_size)
            reset, validity = self._validity_drift_guard(smiles)
            if reset:
                continue

            smiles = chemistry_utils.remove_molecules_with_radicals(smiles)
            if len(smiles) == 0:
                log.debug("Batch %s: no valid SMILES after radicals filter, skipping", batch_number)
                continue

            if (self.execute_beam_enumeration) and (len(self.beam_enumeration.pool) != 0):
                seqs, smiles = self.beam_enumeration.filter_batch(seqs, smiles)
            if len(smiles) == 0:
                self.beam_enumeration.filtered_epoch_updates()
                if self.beam_enumeration.patience_limit_reached():
                    log.info("Beam enumeration patience limit reached; stopping")
                    break
                continue

            smiles, penalized_rewards = self.oracle(smiles, self.diversity_filter)
            if self.execute_beam_enumeration:
                self.beam_enumeration.epoch_updates(
                    agent=self.agent,
                    num_valid_smiles=len(smiles),
                    mean_reward=penalized_rewards.mean(),
                    oracle_calls=self.oracle.calls,
                )

            if (self.execute_hallucinated_memory) and (len(self.replay_buffer.memory) == self.replay_buffer.memory_size):
                hallucinated_smiles = self.hallucinator.hallucinate(self.replay_buffer.memory)
                hallucinated_smiles, hallucinated_penalized_rewards = self.oracle(
                    hallucinated_smiles, self.diversity_filter, is_hallucinated_batch=True
                )
                self.hallucinator.epoch_updates(
                    oracle_calls=self.oracle.calls,
                    buffer_rewards=self.replay_buffer.memory["reward"],
                    hallucinations=hallucinated_smiles,
                    hallucination_rewards=hallucinated_penalized_rewards,
                )
            else:
                hallucinated_smiles, hallucinated_penalized_rewards = np.array([]), np.array([])

            smiles = np.concatenate((smiles, hallucinated_smiles), 0)
            penalized_rewards = np.concatenate((penalized_rewards, hallucinated_penalized_rewards), 0)

            loss = self.compute_loss(smiles, penalized_rewards)
            self.replay_buffer.add(smiles=smiles, rewards=penalized_rewards)

            mol_counts_trend.append(self.oracle.calls)
            if len(mol_counts_trend) > 10 and mol_counts_trend[-1] == mol_counts_trend[-11]:
                log.warning("No new molecules generated for 10 batches; stopping")
                break

            er_smiles, er_rewards = self.replay_buffer.sample_memory()
            er_loss = self.compute_loss(er_smiles, er_rewards)
            loss = torch.cat((loss, er_loss), 0)
            self.backpropagate(loss)

            if self.augmented_memory and len(self.replay_buffer.memory) > 0:
                if self.selective_memory_purge:
                    self.replay_buffer.selective_memory_purge(smiles, penalized_rewards)
                for _ in range(self.augmentation_rounds):
                    randomized_smiles = chemistry_utils.randomize_smiles_batch(smiles, self.prior)
                    loss = self.compute_loss(randomized_smiles, penalized_rewards)
                    randomized_buffer_smiles, randomized_buffer_rewards = self.replay_buffer.augmented_memory_replay(self.prior)
                    augmented_memory_loss = self.compute_loss(randomized_buffer_smiles, randomized_buffer_rewards)
                    loss = torch.cat((loss, augmented_memory_loss), 0)
                    self.backpropagate(loss)

            self._write_out_results()
            if self.oracle.calls > self.logging_frequency * self.logging_multiple:
                self.agent.save(os.path.join(self.model_checkpoints_dir, f"{self.agent.model_architecture}_{self.oracle.calls}_agent.ckpt"))
                self.logging_multiple += 1

            if (np.mean(penalized_rewards) > self.best_agent_reward) and (validity > 0.5):
                self.best_agent_reward = np.mean(penalized_rewards)
                self.agent.save(os.path.join(self.model_checkpoints_dir, "best_agent.ckpt"))

            batch_elapsed = time.perf_counter() - batch_start
            mem_str = format_gpu_memory(self.device)
            log.info(
                "batch=%s calls=%s/%s validity=%.0f%% reward_mean=%.4f time=%.1fs %s",
                batch_number,
                self.oracle.calls,
                self.oracle.budget,
                validity * 100,
                float(np.mean(penalized_rewards)),
                batch_elapsed,
                mem_str if mem_str else "",
            )

        self._write_out_results()
        total_time = time.perf_counter() - start_time
        log.info(
            "RL done batches=%s calls=%s/%s wall_time=%.1fs",
            batch_number,
            self.oracle.calls,
            self.oracle.budget,
            total_time,
        )
        self.agent.save(os.path.join(self.model_checkpoints_dir, f"final_{self.agent.model_architecture}_agent.ckpt"))

    def compute_loss(
        self, 
        smiles: np.ndarray[str],
        rewards: np.ndarray[float]
    ) -> torch.Tensor:
        """
        Compute the loss for the RL agent.
        Based on REINVENT's original loss function: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0235-x
        """
        if len(smiles) != 0:
            # Filter out SMILES that cannot be encoded by the vocabulary
            # This can happen when SMILES contain tokens not in the vocabulary (e.g., from replay buffer)
            encodable_mask = np.array([is_encodable(smile, self.prior) for smile in smiles])
            
            if not encodable_mask.any():
                # No encodable SMILES, return empty tensor
                return torch.tensor([], dtype=torch.float64, device=self.device)
            
            # Filter SMILES and rewards to only include encodable ones
            encodable_smiles = smiles[encodable_mask]
            encodable_rewards = rewards[encodable_mask]
            
            # NOTE: likelihood_smiles returns the NLL so negation recovers the log-likelihood
            prior_log_likelihoods = -self.prior.likelihood_smiles(encodable_smiles)
            agent_log_likelihoods = -self.agent.likelihood_smiles(encodable_smiles)
            augmented_log_likelihoods = prior_log_likelihoods + self.sigma * to_tensor(encodable_rewards, self.device)
            loss = torch.pow((augmented_log_likelihoods - agent_log_likelihoods), 2)
            return loss
        else:
            return torch.tensor([], dtype=torch.float64, device=self.device)

    def backpropagate(self, loss: torch.Tensor) -> None:
        """
        Agent update via backpropagation.
        Directly returns if the loss is empty.
        """
        if len(loss) > 0:
            loss = loss.mean()
            self.optimizer.zero_grad() 
            loss.backward()
            self.optimizer.step()
        else:
            return

    def _disable_prior_gradients(self):
        """Disable gradients for the Prior as it is not updated."""
        for param in self.prior.get_network_parameters():
            param.requires_grad = False

    def _validity_drift_guard(self, smiles: np.ndarray[str]) -> Tuple[bool, float]:
        """Guard against agent drift (invalid SMILES). Returns (reset_done, validity)."""
        validity = chemistry_utils.batch_validity(smiles)
        if validity == 0.0:
            if self.patience == 10:
                logging.getLogger(__name__).warning("Validity 0 for 10 batches; resetting agent to best checkpoint")
                self._reset_agent()
                self.patience = 0
                return True, validity
            self.patience += 1
            return False, validity
        self.patience = 0
        return False, validity

    def _reset_agent(self):
        """Reset the Agent to the best checkpoint."""
        self.agent = Generator.load_from_file(os.path.join(
            self.model_checkpoints_dir, 
            "best_agent.ckpt"
        ), self.device)

    def _write_out_results(self):
        """
        Writes out the following results:
            1. Oracle History
            2. Number of Oracle Repeats
            3. Beam Enumeration History
            4. Hallucination History
            5. Syntheseus Synthesis Graphs
        """
        base_save_path = os.path.dirname(self.logging_path)
        self.oracle.write_out_oracle_history(base_save_path)
        self.oracle.write_out_repeat_history(base_save_path)

        if self.execute_beam_enumeration:
            self.beam_enumeration.end_actions(self.oracle.calls)

        if self.execute_hallucinated_memory:
            self.hallucinator.write_out_history(base_save_path)

        for oracle in self.oracle.oracle:
            if isinstance(oracle, Syntheseus):
                oracle._write_out_smiles_rxn_tracker()        
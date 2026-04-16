"""
Parent script that executes Sample Efficient Generative Molecular Design using Memory Manipulation (Saturn).
Takes as input a JSON configuration file that specifies all parameters for the generatve experiment.
Adapted from https://github.com/MolecularAI/Reinvent/input.py.
"""
import argparse
import json
import logging
import os
import sys
import time
import traceback
import yaml

# Path: Saturn first (oracles, goal_directed_generation, ...), then repo root (utils.docking_vina_client, oracles.docking, etc.)
saturn_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(saturn_dir)
sys.path.insert(0, saturn_dir)
sys.path.insert(1, repo_root)
os.environ.setdefault("EVEN_MORE_PMO_ROOT", repo_root)

from utils.utils import set_seed_everywhere

# Goal-Directed Generation
from goal_directed_generation.reinforcement_learning import ReinforcementLearningAgent
from goal_directed_generation.dataclass import ReinforcementLearningParameters, GoalDirectedGenerationConfiguration
from experience_replay.dataclass import ExperienceReplayParameters
from hallucinated_memory.dataclass import HallucinatedMemoryParameters
from beam_enumeration.dataclass import BeamEnumerationParameters
from diversity_filter.dataclass import DiversityFilterParameters

# Oracle (for Goal-Directed Generation)
from oracles.oracle import Oracle
from oracles.dataclass import OracleConfiguration

# Scoring
from scoring.scorer import Scorer
from scoring.dataclass import ScoringConfiguration

# Reaction-based Enumeration
from enumeration.enumeration import rxn_based_enumeration


parser = argparse.ArgumentParser(description="Run Saturn.")
parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Path to the JSON/YAML configuration file.",
)

LOG_FMT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def read_config_file(path: str):
    """Read configuration file, supporting both JSON and YAML formats."""
    with open(path, "r") as f:
        file_content = f.read()
    try:
        return yaml.safe_load(file_content)
    except yaml.YAMLError:
        pass
    try:
        return json.loads(file_content.replace("\r", "").replace("\n", ""))
    except (ValueError, KeyError, TypeError):
        pass
    return None


def main():
    main_start_time = time.perf_counter()
    args = parser.parse_args()
    config_path = os.path.abspath(args.config)

    # Single logging setup: config path in first line so failed hparam runs are traceable
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FMT,
        datefmt=LOG_DATEFMT,
        force=True,
    )
    log = logging.getLogger(__name__)
    log.info("config=%s", config_path)

    config = read_config_file(args.config)
    if config is None:
        log.error("Invalid or unsupported config file: %s", config_path)
        raise SystemExit(1)

    # Allow configs to use ${PROJECT_ROOT}, ${OUT_DIR}, etc.
    from benchmark.paths import expand_env_vars

    config = expand_env_vars(config)

    running_mode = config["running_mode"].lower()
    device = config["device"]
    seed = config["seed"]
    model_architecture = config["model_architecture"]
    set_seed_everywhere(seed, device)

    log.info("mode=%s device=%s seed=%s model=%s", running_mode, device, seed, model_architecture)

    if running_mode == "distribution_learning":
        from distribution_learning.distribution_learning import DistributionLearningTrainer
        from distribution_learning.dataclass import DistributionLearningConfiguration
        distribution_learning_trainer = DistributionLearningTrainer(
            logging_path=config["logging"]["logging_path"],
            model_checkpoints_dir=config["logging"]["model_checkpoints_dir"],
            configuration=DistributionLearningConfiguration(
                seed,
                model_architecture,
                **config["distribution_learning"]["parameters"],
            ),
        )
        distribution_learning_trainer.run()

    elif running_mode == "goal_directed_generation":
        oracle = Oracle(OracleConfiguration(**config["oracle"]))
        is_component_syntheseus = [c.get("name") == "syntheseus" for c in config["oracle"]["components"]]
        if any(is_component_syntheseus):
            idx = is_component_syntheseus.index(True)
            syntheseus_params = config["oracle"]["components"][idx]["specific_parameters"]
            syntheseus_oracle = next(orac for orac in oracle.oracle if orac.name == "syntheseus")
            if syntheseus_params.get("enforced_reactions", {}).get("seed_reactions"):
                log.info("Seeding replay buffer via rxn-based enumeration")
                seeding_smiles = rxn_based_enumeration(
                    prior_path=config["goal_directed_generation"]["reinforcement_learning"]["prior"],
                    device=device,
                    syntheseus_params=syntheseus_params,
                    syntheseus_oracle=syntheseus_oracle,
                    n_seeds=config["goal_directed_generation"]["experience_replay"]["memory_size"],
                )
                config["goal_directed_generation"]["experience_replay"]["smiles"] = seeding_smiles

        reinforcement_learning_agent = ReinforcementLearningAgent(
            logging_frequency=config["logging"]["logging_frequency"],
            logging_path=config["logging"]["logging_path"],
            model_checkpoints_dir=config["logging"]["model_checkpoints_dir"],
            oracle=oracle,
            configuration=GoalDirectedGenerationConfiguration(
                seed,
                model_architecture,
                ReinforcementLearningParameters(**config["goal_directed_generation"]["reinforcement_learning"]),
                ExperienceReplayParameters(**config["goal_directed_generation"]["experience_replay"]),
                DiversityFilterParameters(**config["goal_directed_generation"]["diversity_filter"]),
                HallucinatedMemoryParameters(**config["goal_directed_generation"]["hallucinated_memory"]),
                BeamEnumerationParameters(**config["goal_directed_generation"]["beam_enumeration"]),
            ),
            device=device,
        )
        log.info("Starting goal-directed generation (RL)")
        rl_start = time.perf_counter()
        reinforcement_learning_agent.run()
        log.info("RL finished in %.1fs", time.perf_counter() - rl_start)

    elif running_mode in ["scoring", "scorer"]:
        oracle = Oracle(OracleConfiguration(**config["oracle"]))
        scorer = Scorer(
            config["logging"]["logging_path"],
            oracle=oracle,
            diversity_filter_configuration=DiversityFilterParameters(**config["goal_directed_generation"]["diversity_filter"]),
            configuration=ScoringConfiguration(**config["scoring"]),
        )
        scorer.run()

    else:
        log.error("Unknown running_mode=%s", running_mode)
        raise ValueError(f"Running mode '{running_mode}' is not implemented.")

    elapsed = time.perf_counter() - main_start_time
    log.info("Saturn completed successfully in %.1fs (config=%s)", elapsed, config_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt=LOG_DATEFMT, force=True)
        log = logging.getLogger(__name__)
        log.error("Saturn failed: %s", e)
        log.error("argv: %s", sys.argv)
        log.error(traceback.format_exc())
        raise SystemExit(1)

import argparse
import os
import sys
from typing import List
import gc
import time
import logging

# Add parent directory to path for imports.
# IMPORTANT: insert at the front so `genetic_chemalactica/utils/*` overrides top-level `utils/*`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark.benchmark_timer import BenchmarkTimer

from genetic_chemalactica.oracles.oracle import select_oracle
from benchmark.guacamol_assets import lead_seed_smiles

parp1_0 = lead_seed_smiles("parp1", 0)
parp1_1 = lead_seed_smiles("parp1", 1)
parp1_2 = lead_seed_smiles("parp1", 2)
fa7_0 = lead_seed_smiles("fa7", 0)
fa7_1 = lead_seed_smiles("fa7", 1)
fa7_2 = lead_seed_smiles("fa7", 2)
_5ht1b_0 = lead_seed_smiles("5ht1b", 0)
_5ht1b_1 = lead_seed_smiles("5ht1b", 1)
_5ht1b_2 = lead_seed_smiles("5ht1b", 2)
braf_0 = lead_seed_smiles("braf", 0)
braf_1 = lead_seed_smiles("braf", 1)
braf_2 = lead_seed_smiles("braf", 2)
jak2_0 = lead_seed_smiles("jak2", 0)
jak2_1 = lead_seed_smiles("jak2", 1)
jak2_2 = lead_seed_smiles("jak2", 2)

from genetic.genetic_utils import Entry, Pool, canonicalize
#from utils import generate_random_number

from logging_ import logger

from utils.mol import find_valid_mols


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


class TempScheduler:

    def __init__(self, start_temp, end_temp, num_steps):
        assert start_temp >= end_temp
        self._start_temp = start_temp
        self._end_temp = end_temp
        self._temp_step = (start_temp - end_temp) / num_steps

    def step(self, i):
        self.temp = self._start_temp - self._temp_step * i
        self.temp = max(self.temp, self._end_temp + self._temp_step)


def props_dict_from_task(full_task_name: str):
    prop_dict = {
        # "pmo.jnk3": "",
        "pmo.median1": (
            ("SIMILAR", lambda: f"CC12CCC(CC1=O)C2(C)C 0.55"),
            ("SIMILAR", lambda: f"CC1CCC(C(C)C)C(O)C1 0.55")
        ),
        "pmo.median2": (
            ("SIMILAR", lambda: f"CN1CC(=O)N2C(Cc3c([nH]c4ccccc34)C2c2ccc3c(c2)OCO3)C1=O 0.55"),
            ("SIMILAR", lambda: f"CCCc1nn(C)c2c(=O)[nH]c(-c3cc(S(=O)(=O)N4CCN(C)CC4)ccc3OCC)nc12 0.55")
        ),
        "pmo.scaffold_hop": (
            "[SIMILAR]CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C 0.80[/SIMILAR]"
        ),
        "pmo.sitagliptin_mpo": (
            ("SIMILAR", lambda: f"C=CC(=O)Nc1cc(Nc2nccc(-c3cn(C)c4ccccc34)n2)c(OC)cc1N(C)CCN(C)C 0.01"),
            ("CLOGP", lambda: "2.02"),
            ("TPSA", lambda: "77.04"),
            ("FORMULA", lambda: "C16H15F6N5O2")
        ),
        # "pmo.fexofenadine_mpo": (
        #     ("SIMILAR", lambda: f"CC(C)(C(=O)O)c1ccc(C(O)CCCN2CCC(C(O)(c3ccccc3)c3ccccc3)CC2)cc1 {generate_random_number(0.8, 1.0):.2f}"),
        #     ("TPSA", lambda: f"{generate_random_number(90, 140):.2f}"),
        #     ("CLOGP", lambda: f"{generate_random_number(-1, 4):.2f}")
        # ),
        # "pmo.ranolazine_mpo": (
        #     ("SIMILAR", lambda: f"COc1ccccc1OCC(O)CN1CCN(CC(=O)Nc2c(C)cccc2C)CC1 {generate_random_number(0.7, 1.0):.2f}"),
        #     ("CLOGP", lambda: f"{generate_random_number(7, 13):.2f}"),
        #     ("TPSA", lambda: f"{generate_random_number(95, 140):.2f}")
        # ),
        # "pmo.perindopril_mpo": (
        #     ("SIMILAR", lambda: f"CCCC(NC(C)C(=O)N1C(C(=O)O)CC2CCCCC21)C(=O)OCC {generate_random_number(0.9, 1.0):.2f}"),
        #     ("NUMAROMATICRINGS", lambda: "2")
        # ),
        # "pmo.zaleplon_mpo": (
        #     ("SIMILAR", lambda: f"CCN(C(C)=O)c1cccc(-c2ccnc3c(C#N)cnn23)c1 {generate_random_number(0.9, 1.0):.2f}"),
        #     ("FORMULA", lambda: "C19H17N3O2")
        # ),
        # "pmo.amlodipine_mpo": (
        #     ("SIMILAR", lambda: f"CCOC(=O)C1=C(COCCN)NC(C)=C(C(=O)OC)C1c1ccccc1Cl {generate_random_number(0.9, 1.0):.2f}"),
        #     ("NUMRINGS", lambda: "3")
        # ),
        "geam.parp1": (
            # ("QED", lambda: f"{generate_random_number(0.5, 0.94):.2f}"),
            # ("SAS", lambda: f"{generate_random_number(1, 5):.2f}")
        ),
        "geam.fa7": (
            # ("QED", lambda: f"{generate_random_number(0.5, 0.94):.2f}"),
            # ("SAS", lambda: f"{generate_random_number(1, 5):.2f}")
        ),
        "geam.5ht1b": (
            # ("QED", lambda: f"{generate_random_number(0.5, 0.94):.2f}"),
            # ("SAS", lambda: f"{generate_random_number(1, 5):.2f}")
        ),
        "geam.braf": (
            # ("QED", lambda: f"{generate_random_number(0.5, 0.94):.2f}"),
            # ("SAS", lambda: f"{generate_random_number(1, 5):.2f}")
        ),
        "geam.jak2": (
            # ("QED", lambda: f"{generate_random_number(0.5, 0.94):.2f}"),
            # ("SAS", lambda: f"{generate_random_number(1, 5):.2f}")
        ),
        "dock.parp1": (
            ("QED", lambda: "[0.5, 0.94]"),
            ("SAS", lambda: "[1, 5]")
        ),
        "dock.fa7": (
            ("QED", lambda: "[0.5, 0.94]"),
            ("SAS", lambda: "[1, 5]")
        ),
        "dock.5ht1b": (
            ("QED", lambda: "[0.5, 0.94]"),
            ("SAS", lambda: "[1, 5]")
        ),
        "dock.braf": (
            ("QED", lambda: "[0.5, 0.94]"),
            ("SAS", lambda: "[1, 5]")
        ),
        "dock.jak2": (
            ("QED", lambda: "[0.5, 0.94]"),
            ("SAS", lambda: "[1, 5]")
        ),
        "lead.parp1_04_0": (
            ("SIMILAR", lambda: f"{parp1_0} [0.40,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.parp1_04_1": (
            ("SIMILAR", lambda: f"{parp1_1} [0.40,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.parp1_04_2": (
            ("SIMILAR", lambda: f"{parp1_2} [0.40,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.parp1_06_0": (
            ("SIMILAR", lambda: f"{parp1_0} [0.60,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.parp1_06_1": (
            ("SIMILAR", lambda: f"{parp1_1} [0.60,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.parp1_06_2": (
            ("SIMILAR", lambda: f"{parp1_2} [0.60,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.fa7_04_0": (
            ("SIMILAR", lambda: f"{fa7_0} [0.40,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.fa7_04_1": (
            ("SIMILAR", lambda: f"{fa7_1} [0.40,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.fa7_04_2": (
            ("SIMILAR", lambda: f"{fa7_2} [0.40,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.fa7_06_0": (
            ("SIMILAR", lambda: f"{fa7_0} [0.60,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.fa7_06_1": (
            ("SIMILAR", lambda: f"{fa7_1} [0.60,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.fa7_06_2": (
            ("SIMILAR", lambda: f"{fa7_2} [0.60,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.5ht1b_04_0": (
            ("SIMILAR", lambda: f"{_5ht1b_0} [0.40,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.5ht1b_04_1": (
            ("SIMILAR", lambda: f"{_5ht1b_1} [0.40,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.5ht1b_04_2": (
            ("SIMILAR", lambda: f"{_5ht1b_2} [0.40,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.5ht1b_06_0": (
            ("SIMILAR", lambda: f"{_5ht1b_0} [0.60,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.5ht1b_06_1": (
            ("SIMILAR", lambda: f"{_5ht1b_1} [0.60,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.5ht1b_06_2": (
            ("SIMILAR", lambda: f"{_5ht1b_2} [0.60,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.braf_04_0": (
            ("SIMILAR", lambda: f"{braf_0} [0.40,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.braf_04_1": (
            ("SIMILAR", lambda: f"{braf_1} [0.40,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.braf_04_2": (
            ("SIMILAR", lambda: f"{braf_2} [0.40,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.braf_06_0": (
            ("SIMILAR", lambda: f"{braf_0} [0.60,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.braf_06_1": (
            ("SIMILAR", lambda: f"{braf_1} [0.60,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.braf_06_2": (
            ("SIMILAR", lambda: f"{braf_2} [0.60,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.jak2_04_0": (
            ("SIMILAR", lambda: f"{jak2_0} [0.40,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.jak2_04_1": (
            ("SIMILAR", lambda: f"{jak2_1} [0.40,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.jak2_04_2": (
            ("SIMILAR", lambda: f"{jak2_2} [0.40,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.jak2_06_0": (
            ("SIMILAR", lambda: f"{jak2_0} [0.60,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.jak2_06_1": (
            ("SIMILAR", lambda: f"{jak2_1} [0.60,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.jak2_06_2": (
            ("SIMILAR", lambda: f"{jak2_2} [0.60,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),  
        "lead.jnk3_04_0": (
            ("SIMILAR", lambda: f"{parp1_0} [0.40,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.drd2_04_0": (
            ("SIMILAR", lambda: f"{parp1_0} [0.40,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.gsk3b_04_0": (
            ("SIMILAR", lambda: f"{parp1_0} [0.40,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "hit.jnk3_04_0": (
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]"),
            ("DOCKING SCORE", lambda: "[0.60,1.00]")
        ),
        "hit.drd2_04_0": (
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]"),
            ("DOCKING SCORE", lambda: "[0.60,1.00]")
        ),
        "hit.gsk3b_04_0": (
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]"),
            ("DOCKING SCORE", lambda: "[0.60,1.00]")
        ),
        "spec.6nzp_7uyt": (
            ("QED", lambda: "[0.40,1.00]"),
            ("SAS", lambda: "[1.00,4.00]"),
            ("DOCKING SCORE", lambda: "[10.67,20.00]")
        ),
        "spec.6nzp_5ut5": (
            ("QED", lambda: "[0.40,1.00]"),
            ("SAS", lambda: "[1.00,4.00]"),
            ("DOCKING SCORE", lambda: "[10.67,20.00]")
        ),
        "spec.6nzp_7uyw": (
            ("QED", lambda: "[0.40,1.00]"),
            ("SAS", lambda: "[1.00,4.00]"),
            ("DOCKING SCORE", lambda: "[10.67,20.00]")
        ),
    }
    prop_list = prop_dict[full_task_name]

    task_prompt = ""
    for prop in prop_list:
        start_tag = f"[{prop[0]}]"
        end_tag = f"[/{prop[0]}]"
        value = prop[1]()
        task_prompt += f"{start_tag}{value}{end_tag}"

    return task_prompt


def create_prompts(
    pool: Pool,
    model_args,
    genetic_args,
    include_start_token: bool,
    task_name: str
):
    prompts = []
    for n in range(genetic_args.num_prompts * 2):
        similar_entries = pool.random_subset(genetic_args.num_similars)
        prompt = ""

        # add similar molecules in the prompt
        for entry in similar_entries:
            sim_value = np.random.uniform(genetic_args.sim_vals[0], genetic_args.sim_vals[1])
            prompt += f"{model_args.sim_tags[0]}{entry.mol} {sim_value:.2f}{model_args.sim_tags[1]}"

        prompt += props_dict_from_task(task_name)

        # add start of molecule tag
        if include_start_token:
            prompt += model_args.mol_tags[0]
        prompts.append(prompt)
    return prompts


def search(args):
    log_dir = args.oracle.log_dir
    # Setup logging
    run_log_path = os.path.join(log_dir, "run.log")
    setup_logging(run_log_path)
    
    logger.info(f"Running Genetic")
    timer = BenchmarkTimer(log_dir=log_dir)

    checkpoint_path = os.environ.get("HF_LOCAL_MODEL_DIR", "").strip() or str(args.model.checkpoint_path)
    tokenizer_path = os.environ.get("HF_LOCAL_TOKENIZER_DIR", "").strip() or str(args.model.tokenizer_path)

    # Load model with memory-efficient settings
    # Use device_map dict format for explicit single-device placement
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": args.device},  # Explicitly map all layers to the specified device
        attn_implementation="sdpa",  # Use efficient SDPA attention
        local_files_only=True,
    )
    model.eval()  # Set to eval mode to disable gradients and reduce memory
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        padding_side="left",
        local_files_only=True,
    )
    tokenizer.pad_token_id = tokenizer.get_vocab()[args.model.pad_token]

    # Optional: HTTP DockingVina service URL. If absent, property_computers uses local quickvina.
    vina_url = OmegaConf.select(args, "vina_url", default=None)
    # Optional: disable the internal ORACLES_APP HTTP wrapper and compute props in-process.
    use_oracles_app = OmegaConf.select(args, "oracle.use_oracles_app", default=False)
    oracle = select_oracle(
        reward_type=args.oracle.reward_type,
        task_name=args.oracle.name,
        log_dir=args.oracle.log_dir,
        max_oracle_calls=args.oracle.max_calls,
        vina_url=vina_url,
        use_oracles_app=use_oracles_app,
        bench_timer=timer,
    )
    pool = Pool(args.genetic.pool_size)
    
    # Track oracle call statistics
    total_oracle_calls = 0
    total_wasted_calls = 0
    # temp_scheduler = TempScheduler(
    #     start_temp=args.genetic.temp_schedule[0],
    #     end_temp=args.genetic.temp_schedule[1],
    #     num_steps=args.oracle.max_calls // args.genetic.num_prompts
    # )

    entries: List[Entry] = []
    # random_smiles = pd.read_csv("/auto/home/tigranfahradyan/vector_dbs/random_pubchem_smiles.csv")["smiles"].values
    # scores = oracle(random_smiles)
    # for mol, score in zip(random_smiles, scores, strict=True):
    #     entries.append(
    #         Entry(
    #             smiles=mol,
    #             score=score,
    #             representation=args.model.representation
    #         )
    #     )
    # pool.add(entries)
    # entries.clear()

    num_iter = 0
    oracle_calls_trend = []
    while not oracle.finish():
        oracle_calls_trend.append(len(oracle.mol_buffer))
        if len(oracle_calls_trend) > 10:
            if oracle_calls_trend[-1] - oracle_calls_trend[-5] <= 5:
                logger.info("Stopping: no new molecules generated")
                break
        num_iter = len(oracle.mol_buffer) // args.genetic.num_prompts
        # temp_scheduler.step(num_iter)
        logger.info(f"Iter: {num_iter}")

        # Generation timing: prompts/setup vs LM forward vs post-process (add gen_train phase here if you fine-tune).
        with timer.phase("gen_prompts"):
            prompts = create_prompts(
                pool,
                args.model,
                args.genetic,
                include_start_token=True,
                task_name=args.oracle.name
            )
            if "chemlactica" in args.model.checkpoint_path.lower():
                prompts = [tokenizer.eos_token + p for p in prompts]

            gen_config = OmegaConf.to_container(args.generation)
            gen_batch_size = getattr(args.model, 'gen_batch_size', 32)
            if gen_batch_size is None or gen_batch_size > len(prompts):
                gen_batch_size = len(prompts)

        with timer.phase("gen_model"):
            all_generated_tokens = []
            for start_idx in range(0, len(prompts), gen_batch_size):
                end_idx = min(start_idx + gen_batch_size, len(prompts))
                batch_prompts = prompts[start_idx:end_idx]

                data = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(args.device)

                generated_texts = model.generate(
                    **data,
                    **gen_config
                )
                batch_generated_tokens = tokenizer.batch_decode(generated_texts)
                all_generated_tokens.extend(batch_generated_tokens)

                del data
                del generated_texts
                torch.cuda.empty_cache()

        with timer.phase("gen_post"):
            generated_tokens = all_generated_tokens
            valid_mols = find_valid_mols(generated_tokens, args.model.mol_tags[0], args.model.mol_tags[1])
            valid_mols = [canonicalize(mol) for mol in valid_mols]

        # collect all molecules generated from all prompts
        mols = []
        for mol in valid_mols:
            mols.append(mol)
        
        # keep only unique molecules
        unique_mols, unique_inds = np.unique(mols, return_index=True)
        unique_prompts = np.array(prompts)[unique_inds]
        assert len(unique_prompts) == len(unique_mols)

        # keep only the unique molecules and prompts
        remaining_num_mols = args.genetic.num_prompts - len(entries)
        unique_mols = unique_mols[:remaining_num_mols]
        unique_prompts = unique_prompts[:remaining_num_mols]

        # Count wasted calls (molecules already in buffer)
        wasted_calls = sum(1 for mol in unique_mols if mol in oracle.mol_buffer)
        new_mols_count = len(unique_mols) - wasted_calls
        
        # Oracle timing is split inside the oracle: oracle_props (new molecules only) vs oracle_scoring
        # (reward + log for the whole batch, including duplicates / “wasted” calls).
        oracle_call_start = time.time()
        scores = oracle(unique_mols, unique_prompts)
        oracle_call_time = time.time() - oracle_call_start
        
        # Update statistics
        oracle_calls_after = len(oracle.mol_buffer)
        total_oracle_calls += len(unique_mols)
        total_wasted_calls += wasted_calls
        
        for mol, score in zip(unique_mols, scores, strict=True):
            entries.append(
                Entry(
                    smiles=mol,
                    score=score
                )
            )

        # Counts / oracle stats go to the run log only; timing_profile.csv stays time-only (phase seconds).
        logger.info(
            f"Iter {num_iter}: oracle_time={oracle_call_time:.2f}s, unique_mols={len(unique_mols)}, "
            f"new_mols={new_mols_count}, wasted={wasted_calls}, total_wasted={total_wasted_calls}, "
            f"cumulative_eval_time={oracle.time_spent_on_evaluation:.2f}s, buffer={oracle_calls_after}/{args.oracle.max_calls}"
        )
        timer.log_iteration(num_iter)

        # add the molecules to the pool
        if len(entries) == args.genetic.num_prompts:
            num_iter += 1
            pool.add(entries)
            entries.clear()
    
    # Final summary
    final_oracle_calls = len(oracle.mol_buffer)
    wasted_percentage = (total_wasted_calls/total_oracle_calls*100) if total_oracle_calls > 0 else 0
    
    logger.info(f"Final: total_wasted={total_wasted_calls} ({wasted_percentage:.1f}%), total_oracle_time={oracle.time_spent_on_evaluation:.2f}s, calls={final_oracle_calls}/{args.oracle.max_calls}")
    timer.log_summary()
    
    logger.info(f"Run log saved to: {run_log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
       
    parser.add_argument("--config_file", required=False, type=str)
    parser.add_argument("--seeds", nargs="+", required=False, type=int)
    parser.add_argument("--oracle.name", required=False, type=str)
    #parser.add_argument("--oracle_calls", required=False, type=int)
    parser.add_argument("--n_gpus", required=False, default=1, type=int)
    parser.add_argument("--hparam_config", type=str, required=False, default=None)

    cmd_args = parser.parse_args()

    args = OmegaConf.load(cmd_args.config_file)
    # Set DOCKING_VINA_URL from config (vina_url written by genetic_runner) so docking code sees it
    vina_url = OmegaConf.select(args, "vina_url", default=None)
    if vina_url:
        os.environ["DOCKING_VINA_URL"] = str(vina_url)
    search(args)
    
    
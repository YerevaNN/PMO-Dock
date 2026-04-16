import os
import logging
from argparse import ArgumentParser
from collections import defaultdict, namedtuple

from mcts.run import search as mcts_search
from genetic.run import search as genetic_search
from grpo.hf.run import search as grpo_search
from logging_ import init_logger, logger
from utils import set_seed

import tomli_w
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


class ExperimentArgs:
    pass


def args_to_two_level_dict(args) -> defaultdict:
    args_dict = defaultdict(defaultdict)
    for k, v in vars(args).items():
        if "." in k:
            first_level_key, second_level_key = k.split(".", 1)
            args_dict[first_level_key][second_level_key] = v
    return args_dict


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--parent_log_dirs", nargs="+", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--oracle.names", nargs="+", type=str, required=True)
    parser.add_argument("--oracle.max_calls", type=int, required=True)
    parser.add_argument("--url", type=str, required=False, default=None)
    parser.add_argument("--experiment_id", type=str, required=False, default="none")
    cmd_args = parser.parse_args()

    init_logger(logging.INFO)
    set_seed(cmd_args.seed)

    # create the log dir files
    log_dirs = []
    for parent_log_dir in cmd_args.parent_log_dirs:
        log_dir = os.path.join(parent_log_dir, f"seed-{cmd_args.seed}")
        os.makedirs(log_dir, exist_ok=True)
        log_dirs.append(log_dir)

    args_dict = defaultdict(defaultdict)
    # add logging configs
    args_dict['logging']['dirs'] = log_dirs
    # args_dict['logging']['file_path'] = log_file_path
    args_dict['model']['url'] = cmd_args.url
    args_dict['experiment']['id'] = cmd_args.experiment_id

    # add the toml configs
    toml_args_dict = tomllib.load(open(os.path.join(cmd_args.parent_log_dirs[0], "config.toml"), "rb"))
    for section, section_args in toml_args_dict.items():
        for k, v in section_args.items():
            args_dict[section][k] = v

    # add command line configs
    cmd_args_dict = args_to_two_level_dict(cmd_args)
    for section, section_args in cmd_args_dict.items():
        for k, v in section_args.items():
            args_dict[section][k] = v

    args = ExperimentArgs()

    # add the arguments to the args
    for k, v in args_dict.items():
        setattr(args, k, namedtuple(k.title(), v.keys())(**v))

    logger.info(f"Experiment id: {cmd_args.experiment_id}")

    if cmd_args.method == "mcts":
        mcts_search(args)
    elif cmd_args.method == "genetic":
        genetic_search(args)
    elif cmd_args.method == "grpo":
        grpo_search(args)
    else:
        raise ValueError(f"Invalid method {cmd_args.method}")

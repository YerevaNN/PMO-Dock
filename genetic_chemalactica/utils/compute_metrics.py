from typing import List
import argparse
from pathlib import Path
import yaml
import os
import sys
import numpy as np
import pandas as pd
import time

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.file import merge_csvs, load_csv
from benchmark.guacamol_assets import lead_seed_smiles, perindopril_smiles

parp1_0 = lead_seed_smiles("parp1", 0)
parp1_1 = lead_seed_smiles("parp1", 1)
parp1_2 = lead_seed_smiles("parp1", 2)
fa7_0 = lead_seed_smiles("fa7", 0)
fa7_1 = lead_seed_smiles("fa7", 1)
fa7_2 = lead_seed_smiles("fa7", 2)
braf_0 = lead_seed_smiles("braf", 0)
braf_1 = lead_seed_smiles("braf", 1)
braf_2 = lead_seed_smiles("braf", 2)
jak2_0 = lead_seed_smiles("jak2", 0)
jak2_1 = lead_seed_smiles("jak2", 1)
jak2_2 = lead_seed_smiles("jak2", 2)
_5ht1b_0 = lead_seed_smiles("5ht1b", 0)
_5ht1b_1 = lead_seed_smiles("5ht1b", 1)
_5ht1b_2 = lead_seed_smiles("5ht1b", 2)


def compute_top_auc(buffer, top_n, finish, freq_log, max_oracle_calls):
    sum = 0
    prev = 0
    called = 0
    ordered_results = list(sorted(buffer.items(), key=lambda kv: kv[1][1], reverse=False))
    for idx in range(freq_log, min(len(buffer), max_oracle_calls), freq_log):
        temp_result = ordered_results[:idx]
        temp_result = list(sorted(temp_result, key=lambda kv: kv[1][0], reverse=True))[:top_n]
        top_n_now = np.mean([item[1][0] for item in temp_result])
        sum += freq_log * (top_n_now + prev) / 2
        prev = top_n_now
        called = idx
    temp_result = list(sorted(ordered_results, key=lambda kv: kv[1][0], reverse=True))[:top_n]
    top_n_now = np.mean([item[1][0] for item in temp_result])
    sum += (len(buffer) - called) * (top_n_now + prev) / 2
    if finish and len(buffer) < max_oracle_calls:
        sum += (max_oracle_calls - len(buffer)) * top_n_now
    return sum / max_oracle_calls


MEDIAN_DOCKING_SCORES = {
    "6nzp": 10.67,
    "parp1": 10.0,
    "fa7": 8.5, 
    "5ht1b": 8.7845,
    "braf": 10.3,
    "jak2": 9.1,
}


def task_name2hit_thresholds(task_name):
    return {
        "spec.6nzp_7uyt": {
            "DOCKING.6nzp": lambda v: v >= 10.67,
            "DOCKING.7uyt": lambda v: v >= 0.0,
            "SAS": lambda v: v <= 4.0,
            "QED": lambda v: v >= 0.4
        },
        "spec.6nzp_5ut5": {
            "DOCKING.6nzp": lambda v: v >= 10.67,
            "DOCKING.5ut5": lambda v: v >= 0.0,
            "SAS": lambda v: v <= 4.0,
            "QED": lambda v: v >= 0.4
        },
        "spec.6nzp_7uyw": {
            "DOCKING.6nzp": lambda v: v >= 10.67,
            "DOCKING.7uyw": lambda v: v >= 0.0,
            "SAS": lambda v: v <= 4.0,
            "QED": lambda v: v >= 0.4
        },
        "dock.parp1": {
            "DOCKING.parp1": lambda v: v >= 10.0,
            "SAS": lambda v: v <= 5,
            "QED": lambda v: v >= 0.5
        },
        "dock.fa7": {
            "DOCKING.fa7": lambda v: v >= 8.5,
            "SAS": lambda v: v <= 5,
            "QED": lambda v: v >= 0.5
        },
        "dock.5ht1b": {
            "DOCKING.5ht1b": lambda v: v >= 8.7845,
            "SAS": lambda v: v <= 5,
            "QED": lambda v: v >= 0.5
        },
        "dock.braf": {
            "DOCKING.braf": lambda v: v >= 10.3,
            "SAS": lambda v: v <= 5,
            "QED": lambda v: v >= 0.5
        },
        "dock.jak2": {
            "DOCKING.jak2": lambda v: v >= 9.1,
            "SAS": lambda v: v <= 5,
            "QED": lambda v: v >= 0.5
        },
        "pmo.perindopril_mpo": {
            f"SIMILAR.{perindopril_smiles}": lambda v: v >= 0.5,
            "NUMAROMATICRINGS": lambda v: v == 2
        },
        "pmo.perindopril_mpo_prop": {
            f"SIMILAR.{perindopril_smiles}": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 5,
            "QED": lambda v: v >= 0.5
        },
        "lead.parp1_04_0": {
            f"SIMILAR.{parp1_0}": lambda v: v >= 0.4,
            "QED": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 4
        },
        "lead.parp1_04_1": {
            f"SIMILAR.{parp1_1}": lambda v: v >= 0.4,
            "QED": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 4
        },
        "lead.parp1_04_2": {
            f"SIMILAR.{parp1_2}": lambda v: v >= 0.4,
            "QED": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 4
        },
        "lead.parp1_06_0": {
            f"SIMILAR.{parp1_0}": lambda v: v >= 0.6,
            "QED": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 4
        },
        "lead.parp1_06_1": {
            f"SIMILAR.{parp1_1}": lambda v: v >= 0.6,
            "QED": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 4
        },
        "lead.parp1_06_2": {
            f"SIMILAR.{parp1_2}": lambda v: v >= 0.6,
            "QED": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 4
        },
        "lead.fa7_04_0": {
            f"SIMILAR.{fa7_0}": lambda v: v >= 0.4,
            "QED": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 4
        },
        "lead.fa7_04_1": {
            f"SIMILAR.{fa7_1}": lambda v: v >= 0.4,
            "QED": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 4
        },
        "lead.fa7_04_2": {
            f"SIMILAR.{fa7_2}": lambda v: v >= 0.4,
            "QED": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 4
        },
        "lead.fa7_06_0": {
            f"SIMILAR.{fa7_0}": lambda v: v >= 0.6,
            "QED": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 4
        },
        "lead.fa7_06_1": {
            f"SIMILAR.{fa7_1}": lambda v: v >= 0.6,
            "QED": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 4
        },
        "lead.fa7_06_2": {
            f"SIMILAR.{fa7_2}": lambda v: v >= 0.6,
            "QED": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 4
        },
        "lead.5ht1b_04_0": {
            f"SIMILAR.{_5ht1b_0}": lambda v: v >= 0.4,
            "QED": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 4
        },
        "lead.5ht1b_04_1": {
            f"SIMILAR.{_5ht1b_1}": lambda v: v >= 0.4,
            "QED": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 4
        },
        "lead.5ht1b_04_2": {
            f"SIMILAR.{_5ht1b_2}": lambda v: v >= 0.4,
            "QED": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 4
        },
        "lead.5ht1b_06_0": {
            f"SIMILAR.{_5ht1b_0}": lambda v: v >= 0.6,
            "QED": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 4
        },
        "lead.5ht1b_06_1": {
            f"SIMILAR.{_5ht1b_1}": lambda v: v >= 0.6,
            "QED": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 4
        },
        "lead.5ht1b_06_2": {
            f"SIMILAR.{_5ht1b_2}": lambda v: v >= 0.6,
            "QED": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 4
        },
        "lead.braf_04_0": {
            f"SIMILAR.{braf_0}": lambda v: v >= 0.4,
            "QED": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 4
        },
        "lead.braf_04_1": {
            f"SIMILAR.{braf_1}": lambda v: v >= 0.4,
            "QED": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 4
        },
        "lead.braf_04_2": {
            f"SIMILAR.{braf_2}": lambda v: v >= 0.4,
            "QED": lambda v: v >= 0.6,  
            "SAS": lambda v: v <= 4
        },
        "lead.braf_06_0": {
            f"SIMILAR.{braf_0}": lambda v: v >= 0.6,
            "QED": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 4
        },
        "lead.braf_06_1": {
            f"SIMILAR.{braf_1}": lambda v: v >= 0.6,
            "QED": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 4
        },
        "lead.braf_06_2": {
            f"SIMILAR.{braf_2}": lambda v: v >= 0.6,
            "QED": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 4
        },
        "lead.jak2_04_0": {
            f"SIMILAR.{jak2_0}": lambda v: v >= 0.4,
            "QED": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 4
        },
        "lead.jak2_04_1": {
            f"SIMILAR.{jak2_1}": lambda v: v >= 0.4,
            "QED": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 4
        },
        "lead.jak2_04_2": {
            f"SIMILAR.{jak2_2}": lambda v: v >= 0.4,
            "QED": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 4
        },
        "lead.jak2_06_0": {
            f"SIMILAR.{jak2_0}": lambda v: v >= 0.6,
            "QED": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 4
        },
        "lead.jak2_06_1": {
            f"SIMILAR.{jak2_1}": lambda v: v >= 0.6,
            "QED": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 4
        },
        "lead.jak2_06_2": {
            f"SIMILAR.{jak2_2}": lambda v: v >= 0.6,
            "QED": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 4
        },
        "lead.jnk3_04_0": {
            f"SIMILAR.{parp1_0}": lambda v: v >= 0.4,
            "QED": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 4
        },
        "lead.jnk3_06_0": {
            f"SIMILAR.{parp1_0}": lambda v: v >= 0.6,
            "QED": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 4
        },
        "lead.drd2_04_0": {
            f"SIMILAR.{parp1_0}": lambda v: v >= 0.4,
            "QED": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 4
        },
        "lead.drd2_06_0": {
            f"SIMILAR.{parp1_0}": lambda v: v >= 0.6,
            "QED": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 4
        },
        "lead.gsk3b_04_0": {
            f"SIMILAR.{parp1_0}": lambda v: v >= 0.4,
            "QED": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 4
        },
        "lead.gsk3b_06_0": {
            f"SIMILAR.{parp1_0}": lambda v: v >= 0.6,
            "QED": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 4
        },
        "lead_no_sim.jnk3_04_0": {
            "QED": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 4
        },
        "lead_no_sim.drd2_04_0": {
            "QED": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 4
        },
        "lead_no_sim.gsk3b_04_0": {
            "QED": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 4
        },
        "hit.jnk3_04_0": {
            "JNK3": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 5,
            "QED": lambda v: v >= 0.5
        },
        "hit.drd2_04_0": {
            "DRD2": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 5,
            "QED": lambda v: v >= 0.5
        },
        "hit.gsk3b_04_0": {
            "GSK3B": lambda v: v >= 0.6,
            "SAS": lambda v: v <= 5,
            "QED": lambda v: v >= 0.5
        }
    }[task_name]


def compute_hit_ratio(
    mol_dfs,
    max_oracle_calls: int,
    task_name: str
):
    hit_thresholds = task_name2hit_thresholds(task_name)

    hit_ratios = []
    for mol_df in mol_dfs:
        # print(len(mol_df), max_oracle_calls)
        assert len(mol_df) <= max_oracle_calls
        print("Unique molecules:", len(mol_df['molecule'].unique()), "Total molecules:", len(mol_df))

        hr = 0
        for index, row in mol_df.iterrows():
            if np.all([
                func(row[prop_name])
                for prop_name, func in hit_thresholds.items()
            ]):
                hr += 1
                    
        hit_ratios.append(hr / max_oracle_calls)

    return np.mean(hit_ratios) * 100, np.std(hit_ratios) * 100


def compute_hit_top_k_docking_score(
    mol_dfs,
    max_oracle_calls: int,
    task_name: str,
    top_k: int,
):
    hit_thresholds = task_name2hit_thresholds(task_name)
    target = task_name.split(".")[-1]
    target = target.split("_")[0]  # in case of something like lead.parp1_0.4_1
    top_k_avg_scores = []
    dfs_containing_hits = 0
    for mol_df in mol_dfs:
        assert len(mol_df) == max_oracle_calls

        top_k_scores = []
        for index, row in mol_df.iterrows():
            if np.all([
                func(row[prop_name])
                for prop_name, func in hit_thresholds.items()
            ]):
                if f"DOCKING.{target}" in row:
                    top_k_scores.append(row[f"DOCKING.{target}"])
                else:
                    top_k_scores.append(row[f"{target.upper()}"])

        top_k_scores = sorted(top_k_scores, reverse=True)[:top_k]
        print(top_k_scores)
        if len(top_k_scores) > 0:
            dfs_containing_hits += 1
            top_k_avg = np.mean(top_k_scores) 
            top_k_avg_scores.append(top_k_avg)

    # print(top_k_avg_scores)
    return np.mean(top_k_avg_scores), np.std(top_k_avg_scores), dfs_containing_hits

def compute_hit_k_difference(
    mol_dfs,
    max_oracle_calls: int,
    task_name: str,
    top_k: int,
):
    pass

def compute_auc(buffer, top_n, finish, freq_log, max_oracle_calls):
    sum = 0
    prev = 0
    called = 0
    ordered_results = list(sorted(buffer.items(), key=lambda kv: kv[1][-1], reverse=False))
    for idx in range(freq_log, min(len(buffer), max_oracle_calls), freq_log):
        temp_result = ordered_results[:idx]
        temp_result = list(sorted(temp_result, key=lambda kv: kv[1][0], reverse=True))[:top_n]
        top_n_now = np.mean([item[1][0] for item in temp_result])
        sum += freq_log * (top_n_now + prev) / 2
        prev = top_n_now
        called = idx
    temp_result = list(sorted(ordered_results, key=lambda kv: kv[1][0], reverse=True))[:top_n]
    top_n_now = np.mean([item[1][0] for item in temp_result])
    sum += (len(buffer) - called) * (top_n_now + prev) / 2
    if finish and len(buffer) < max_oracle_calls:
        sum += (max_oracle_calls - len(buffer)) * top_n_now
    return sum / max_oracle_calls


def compute_top_10_auc(mol_buffers, max_oracle_calls):
    top_10_aucs = []
    for mol_buffer in mol_buffers:
        top_10_auc = compute_auc(mol_buffer, top_n=10, finish=True, freq_log=100, max_oracle_calls=max_oracle_calls)
        top_10_aucs.append(top_10_auc)
    return np.mean(top_10_aucs), np.std(top_10_aucs)


def compute_avg(mol_df, top_n, max_oracle_calls):
    assert len(mol_df) == max_oracle_calls
    mol_df.sort_values(by="score", inplace=True, ascending=False)
    return sum(mol_df["score"].values[:top_n]) / top_n
    # mol_list = [(v[0], v[-1]) for v in buffer.values()]
    # sorted_mol_list = sorted(mol_list, key=lambda x: x[-1])[:max_oracle_calls]
    # scores = [e[0] for e in sorted_mol_list]
    # return sum(sorted(scores, reverse=True)[:top_n]) / top_n


def compute_top_k_avg(mol_dfs, k, max_oracle_calls):
    top_k_avgs = []
    for mol_df in mol_dfs:
        top_k_avg = compute_avg(mol_df, top_n=k, max_oracle_calls=max_oracle_calls)
        top_k_avgs.append(top_k_avg)
    return np.mean(top_k_avgs), np.std(top_k_avgs)


def compute_metric(
    mol_dfs: List[pd.DataFrame],
    metric_name: str,
    max_oracle_calls: int,
    k: int=None,
    task_name: str=None
):
    if "hit_ratio" in metric_name:
        mean, std = compute_hit_ratio(mol_dfs, max_oracle_calls, task_name)
        num_dfs = len(mol_dfs)
    elif "hit_top_k_docking_score" in metric_name:
        mean, std, num_dfs = compute_hit_top_k_docking_score(mol_dfs, max_oracle_calls, task_name, top_k=k)
    elif metric_name == "top_k_avg":
        mean, std = compute_top_k_avg(mol_dfs, k, max_oracle_calls)
        num_dfs = len(mol_dfs)
    elif metric_name == "hit_k_difference_avg":
        mean, std = compute_hit_k_difference(mol_dfs, max_oracle_calls, task_name, top_k=k)
        num_dfs = len(mol_dfs)
    else:
        raise ValueError(f"{metric_name} metric is not supported")
    
    return mean, std, num_dfs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--metric_names", type=str, nargs="+", required=True)
    parser.add_argument("--max_oracle_calls", type=int, required=True)
    parser.add_argument("--task_name", type=str, required=False, default=None)
    parser.add_argument("--k", type=int, required=False, default=150)
    parser.add_argument('--hparams', type=str, nargs="+", required=False, default=None)       
    parser.add_argument('--pad', type=str, required=False, default=None)
    parser.add_argument('--sep', type=str, required=False, default=";")
    args = parser.parse_args()

    #params_to_track = ["grpo.learning_rate", "grpo.beta"]
    params_to_track = []
    if args.hparams:
        report_dfs = {}
        for metric_name in args.metric_names:
            report_dfs[metric_name] = pd.DataFrame()

        ld = Path(args.log_dir)
        for d in ld.iterdir():
            if d.is_dir() and d.name.startswith("exp-"):
                config_dict = yaml.safe_load(open(str(d / "config-1.yaml"), "r"))
                hparams_combination = [config_dict["grpo"][param] for param in args.hparams]
                hparams_combination = str(tuple(hparams_combination))

                task_dirs = [d_ for d_ in d.iterdir() if d_.is_dir()]
                for task_dir in task_dirs:
                    for seed_dir in Path(task_dir).iterdir():
                        merge_csvs(seed_dir, num_gens=16, sep=args.sep)

                    mol_df = load_csv(task_dir, args.max_oracle_calls, unique=True, sep=args.sep, pad=args.pad)
                    for metric_name in args.metric_names:
                        mean, std, num_dfs = compute_metric(
                            mol_df,
                            metric_name=metric_name,
                            k=args.k,
                            max_oracle_calls=args.max_oracle_calls,
                            task_name=task_dir.name
                        )

                        report_dfs[metric_name].loc[hparams_combination, task_dir.name] = f"{mean:.4f} ± {std:.4f} ({num_dfs})"
            
        for metric, df in report_dfs.items():
            #preparing df to write into csv
            df = df.T
            tuples = [eval(c) for c in df.columns]
            sorted_tuples = sorted(tuples, key=lambda x: (x[1], x[0]))
            sorted_cols = [str(t) for t in sorted_tuples]
            df = df[sorted_cols]
            hparam_str = "_".join([param for param in args.hparams])
            df.to_csv(f"{args.log_dir}/report_{metric}_{hparam_str}.csv", mode="w", header=True)

    else:
        task_dir = str(Path(args.log_dir) / args.task_name)
        for seed_dir in Path(task_dir).glob("seed-*"):
            # if not os.path.exists(os.path.join(seed_dir, "mols.csv")):
            merge_csvs(seed_dir, num_gens=16, sep=args.sep)

        metric_dict = {}
        mol_df = load_csv(task_dir, args.max_oracle_calls, unique=True, sep=args.sep, pad=args.pad)
        for metric_name in args.metric_names:
            mean, std, seed_count = compute_metric(
                mol_df,
                metric_name=metric_name,
                k=args.k,
                max_oracle_calls=args.max_oracle_calls,
                task_name=args.task_name
            )
            metric_dict[metric_name] = {
                "mean": mean,
                "std": std,
                "seed_count": seed_count
            }

        config_dict = yaml.safe_load(open(str(Path(task_dir) / "config-1.yaml"), "r"))
        for param in params_to_track:
            if '.' in param:
                a, b = param.split('.')
                param_value = config_dict[a][b]
            else:
                param_value = config_dict[param]
            print(f"{param}: {config_dict[a][b]}", end=" ")
        for metric_name in args.metric_names:
            print(f"{metric_name}: {metric_dict[metric_name]['mean']:.4f} ± {metric_dict[metric_name]['std']: .4f} ({metric_dict[metric_name]['seed_count']})", end=", ")
        print()

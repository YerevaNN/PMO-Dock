from typing import List, Dict
import os
import torch
import yaml
from pathlib import Path
from functools import cache

import torch
import pandas as pd


@cache
def cached_load_yaml(seed_dir: str):
    return yaml.safe_load(open(os.path.join(seed_dir, "mols.yaml"), "r"))


@cache
def cached_load_csv(seed_file_path: str, sep: str):
    return pd.read_csv(os.path.join(seed_file_path), sep=sep)


def merge_csvs(csv_dir, num_gens: int, sep: str):
    dfs = []
    for d in Path(csv_dir).iterdir():
        if d.name == "mols.csv":
            continue
        try:
            dfs.append(pd.read_csv(str(d), sep=sep))
        except Exception as e:
            print(f"Could not read {d}: {e}")

    if len(dfs) == 0:
        return pd.DataFrame()
    min_length = min(len(df) for df in dfs)
    merged_results = []
    for i in range(0, min_length, num_gens):
        for df in dfs:
            merged_results.append(df.iloc[i:i + num_gens])

    # print(merged_results)
    pd.concat(merged_results, ignore_index=True).to_csv(
        os.path.join(csv_dir, "mols.csv"),
        index=False,
        sep=sep
    )


def load_yamls(
    log_dir: List[str],
    max_oracle_calls: int,
    pad: bool=True
) -> Dict:
    mol_buffers = []
    seed_dirs = [d for d in Path(log_dir).iterdir() if d.name.startswith("seed-")]
    for dir in seed_dirs:
        # print(f"Reading {dir}...")
        mol_buffer = cached_load_yaml(dir)
        if len(mol_buffer) < max_oracle_calls:
            if pad:
                print(f"{dir} has less than {max_oracle_calls} mols, padding with max value")
                difference = max_oracle_calls - len(mol_buffer)
                max_entry = max(mol_buffer.values(), key=lambda v: v[0])
                padded_buffer = {str(i * 100): max_entry for i in range(difference)}
                assert len(padded_buffer) == difference
                mol_buffer.update(padded_buffer)
                assert len(mol_buffer) == max_oracle_calls
            else:
                print(f"{dir} has less than {max_oracle_calls} mols, skipping")
                continue
        mol_buffers.append(mol_buffer)
    return mol_buffers


def load_csv(
    log_dir: str,
    max_oracle_calls: int,
    sep: str,
    unique: bool=False,
    pad: str=None,
):
    mol_dfs = []
    seed_dirs = [d for d in Path(log_dir).iterdir() if d.name.startswith("seed-")]
    for seed_dir in seed_dirs:
        seed_file_path = os.path.join(seed_dir, "mols.csv")
        if not os.path.exists(seed_file_path):
            print(f"{seed_file_path} does not exist skipping...")
            continue
        mol_df = cached_load_csv(seed_file_path, sep=sep)
        # print(mol_df)
        total_count = len(mol_df)
        # print(f", ", end="")
        if unique:
            mol_df = mol_df.drop_duplicates(subset="mol")
        print(f"{seed_dir} - total: {total_count}, unique: {len(mol_df)}")
        if len(mol_df) < max_oracle_calls:
            if pad == "min":
                print(f"{seed_dir} has less than {max_oracle_calls} mols, padding with {pad} score")
                difference = max_oracle_calls - len(mol_df)
                min_entry = mol_df.loc[mol_df['score'].idxmin()]
                pad_df = pd.DataFrame([min_entry] * difference, columns=mol_df.columns)
                mol_df = pd.concat([mol_df, pad_df], ignore_index=True)
                assert len(mol_df) == max_oracle_calls
            elif pad == "max":
                print(f"{seed_dir} has less than {max_oracle_calls} mols, padding with {pad} score")
                difference = max_oracle_calls - len(mol_df)
                max_entry = mol_df.loc[mol_df['score'].idxmax()]
                pad_df = pd.DataFrame([max_entry] * difference, columns=mol_df.columns)
                mol_df = pd.concat([mol_df, pad_df], ignore_index=True)
                assert len(mol_df) == max_oracle_calls
            else:
                print(f"{seed_dir} has less than {max_oracle_calls} mols, skipping")
                continue
        mol_df = mol_df.head(max_oracle_calls)
        mol_dfs.append(mol_df)

    return mol_dfs


def select_dtype(dtype: str):
    return {
        "bf16": torch.bfloat16,
        "fp32": torch.float32
    }[dtype]
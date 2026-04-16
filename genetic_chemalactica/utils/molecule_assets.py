from __future__ import annotations

import re

import numpy as np

from benchmark.actives_loader import actives_smiles_by_target
from utils.rewards import guassian_modifier


def parse_molecular_formula(formula: str):
    matches = re.findall(r"([A-Z][a-z]*)(\d*)", formula)
    results = []
    for elem, cnt in matches:
        results.append((elem, 1 if not cnt else int(cnt)))
    return results


def isomer_scoring(formula1: str, formula2: str):
    atom2cnt_lst1 = parse_molecular_formula(formula1)
    atom2cnt_lst2 = parse_molecular_formula(formula2)
    res = []
    for (target_atom, target_cnt) in atom2cnt_lst2:
        for (atom, cnt) in atom2cnt_lst1:
            if target_atom == atom:
                res.append(guassian_modifier(abs(target_cnt - cnt), mu=0, sigma=1.0))
                break
    total_num_atoms1 = sum([cnt for (_, cnt) in atom2cnt_lst1])
    total_num_atoms2 = sum([cnt for (_, cnt) in atom2cnt_lst2])
    res.append(guassian_modifier(abs(total_num_atoms1 - total_num_atoms2), mu=0, sigma=2.0))
    return np.array(res).prod().tolist()


osimertinib_smiles = "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc2nccc(n2)c3cn(C)c4ccccc34"
fexofenadine_smiles = "CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4"
ranolazine_smiles = "COc1ccccc1OCC(O)CN2CCN(CC(=O)Nc3c(C)cccc3C)CC2"
perindopril_smiles = "O=C(OCC)C(NC(C(=O)N1C(C(=O)O)CC2CCCCC12)C)CCC"
amlodipine_smiles = "Clc1ccccc1C2C(=C(/N/C(=C2/C(=O)OCC)COCCN)C)\\C(=O)OC"
sitagliptin_smiles = "Fc1cc(c(F)cc1F)CC(N)CC(=O)N3Cc2nnc(n2CC3)C(F)(F)F"
zaleplon_smiles = "O=C(C)N(CC)C1=CC=CC(C2=CC=NC3=C(C=NN23)C#N)=C1"


def lead_seed_smiles(target: str, idx: int) -> str:
    by_target = actives_smiles_by_target()
    return by_target[target][int(idx)]


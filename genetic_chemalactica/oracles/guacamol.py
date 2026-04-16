from rdkit import Chem
import numpy as np

from benchmark.computers.property_computers import (
    compute_similarity,
    compute_clogp,
    compute_tpsa,
    compute_formula,
    compute_num_aromatic_rings
)
from benchmark.actives_loader import actives_smiles_by_target


def guassian_modifier(x, mu, sigma):
    return np.exp(-0.5 * np.power((x - mu) / sigma, 2.0))


def parse_molecular_formula(formula):
    """
    Parse a molecular formulat to get the element types and counts.

    Args:
        formula: molecular formula, f.i. "C8H3F3Br"

    Returns:
        A list of tuples containing element types and number of occurrences.
    """
    import re

    matches = re.findall(r"([A-Z][a-z]*)(\d*)", formula)

    # Convert matches to the required format
    results = []
    for match in matches:
        # convert count to an integer, and set it to 1 if the count is not visible in the molecular formula
        count = 1 if not match[1] else int(match[1])
        results.append((match[0], count))

    return results


def isomer_scoring(formula1, formula2):
    atom2cnt_lst1 = parse_molecular_formula(formula1)
    atom2cnt_lst2 = parse_molecular_formula(formula2)

    res = []
    for (target_atom, target_cnt) in atom2cnt_lst2:
        for (atom, cnt) in atom2cnt_lst1:
            if target_atom == atom:
                res.append(guassian_modifier(abs(target_cnt - cnt), mu=0, sigma=1.0))
                continue

    total_num_atoms1 = sum([cnt for (_, cnt) in atom2cnt_lst1])
    total_num_atoms2 = sum([cnt for (_, cnt) in atom2cnt_lst2])
    res.append(guassian_modifier(abs(total_num_atoms1 - total_num_atoms2), mu=0, sigma=2.0))
    
    return np.array(res).prod().tolist()


osimertinib_smiles = "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc2nccc(n2)c3cn(C)c4ccccc34"
fexofenadine_smiles = "CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4"
ranolazine_smiles = "COc1ccccc1OCC(O)CN2CCN(CC(=O)Nc3c(C)cccc3C)CC2"
perindopril_smiles = "O=C(OCC)C(NC(C(=O)N1C(C(=O)O)CC2CCCCC12)C)CCC"
perindopril_rdkit_mol = Chem.MolFromSmiles(perindopril_smiles)
amlodipine_smiles = "Clc1ccccc1C2C(=C(/N/C(=C2/C(=O)OCC)COCCN)C)\C(=O)OC"
sitagliptin_smiles = "Fc1cc(c(F)cc1F)CC(N)CC(=O)N3Cc2nnc(n2CC3)C(F)(F)F"
sitagliptin_rdkit_mol = Chem.MolFromSmiles(sitagliptin_smiles)
zaleplon_smiles = "O=C(C)N(CC)C1=CC=CC(C2=CC=NC3=C(C=NN23)C#N)=C1"
zaleplon_rdkit_mol = Chem.MolFromSmiles(zaleplon_smiles)

# Benchmark-owned seed molecules (actives) for lead tasks.
_ACT = actives_smiles_by_target()
parp1_0, parp1_1, parp1_2 = _ACT["parp1"][:3]
fa7_0, fa7_1, fa7_2 = _ACT["fa7"][:3]
_5ht1b_0, _5ht1b_1, _5ht1b_2 = _ACT["5ht1b"][:3]
braf_0, braf_1, braf_2 = _ACT["braf"][:3]
jak2_0, jak2_1, jak2_2 = _ACT["jak2"][:3]

def sitagliptin_mpo(rdkit_mols):
    similarity = compute_similarity(rdkit_mols, sitagliptin_rdkit_mol)
    clogp = compute_clogp(rdkit_mols)
    tpsa = compute_tpsa(rdkit_mols)
    formula = compute_formula(rdkit_mols)
    return np.array([similarity, clogp, tpsa, formula])


def perindopril_mpo(rdkit_mols):
    similarity = compute_similarity(rdkit_mols, perindopril_rdkit_mol)
    num_ar_rings = compute_num_aromatic_rings(rdkit_mols)
    return np.array([similarity, num_ar_rings])


def zaleplon_mpo(rdkit_mols):
    similairty = compute_similarity(rdkit_mols, zaleplon_rdkit_mol)
    formula = compute_formula(rdkit_mols)
    return np.array([similairty, formula])
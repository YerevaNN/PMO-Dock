from typing import List
import re

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

RDLogger.DisableLog('rdApp.*')


def compute_fingerprint(rdkit_mol, fing_type):
    if fing_type == "morgan":
        return AllChem.GetMorganFingerprintAsBitVect(rdkit_mol, 2, nBits=2048)
    else:
        raise ValueError(f"{fing_type} not supported.")


def tanimoto_similarity(fing1, fing2):
    return DataStructs.TanimotoSimilarity(fing1, fing2)


def is_valid_smiles(mol_str: str):
    try:
        return (
            ',' not in mol_str
            and
            Chem.MolFromSmiles(mol_str) is not None
        )
    except Exception as e:
        # print(e)
        return False


def find_valid_mols(completions, mol_start_token, mol_end_token, canonicalize=True):
    valid_mols = []
    for completion in completions:
        # find all occurances of the mol_start_token
        start_token_inds: List[int] = [i for i in range(len(completion)) if completion.startswith(mol_start_token, i)]
        end_token_ind: int = completion.find(mol_end_token)
        
        # make sure the start and end tags are present
        if len(start_token_inds) == 0:
            # logger.error(f"Generated molecule {mol} does not have {self.mol_start_token}.")
            valid_mols.append("")
            continue
        if end_token_ind == -1:
            # logger.error(f"Generated molecule {mol} does not have {self.mol_end_token}.")
            valid_mols.append("")
            continue

        start_token_ind = start_token_inds.pop(0)
        # make sure that all start tags after the first tags are right to the first end tag
        if len(start_token_inds) > 0 and start_token_inds[0] < end_token_ind:
            valid_mols.append("")
            continue

        # extract the molecule representation and check the validity
        mol_repr: str = completion[start_token_ind + len(mol_start_token):end_token_ind]
        if is_valid_smiles(mol_repr):
            valid_mols.append(mol_repr)
        else:
            valid_mols.append("")

    return valid_mols


def find_valid_mols_from_cot_regex(generations, mol_start_token, mol_end_token):
    valid_mols = []
    for gen in generations:
        pattern = r".*\[SMILES](.*?)\[/SMILES]"
        # last_smiles = re.search(pattern, gen).group(1)
        last_smiles = re.findall(pattern, gen)
        if len(last_smiles) == 0:
            last_smiles = ""
        else:
            last_smiles = last_smiles[0]
        
        if is_valid_smiles(last_smiles):
            valid_mols.append(last_smiles)
        else:
            valid_mols.append("")
    
    return valid_mols


def find_valid_mols_from_cot(completions):
    valid_mols = []
    invalid_mols = []
    for completion in completions:
        mol = completion.split("[SMILES]")[-1].split("[/SMILES]")[0]   
        if is_valid_smiles(mol):
            invalid_mols.append("")
            valid_mols.append(mol)
        else:
            valid_mols.append("")
            invalid_mols.append(mol)
            
    return valid_mols, invalid_mols
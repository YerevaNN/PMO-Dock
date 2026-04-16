import math

import numpy as np

from benchmark.guacamol_assets import (
    amlodipine_smiles,
    fexofenadine_smiles,
    lead_seed_smiles,
    osimertinib_smiles,
    perindopril_smiles,
    ranolazine_smiles,
    sitagliptin_smiles,
    zaleplon_smiles,
)

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

def select_sigma(prop_name: str):
    coef = 0.1
    prop_name2sigma = {
        "QED": coef * 1,
        "CLOGP": coef * 18,
        "SAS": coef * 7,
        "TPSA": coef * 100,
        "WEIGHT": coef * 1000,
        "RINGCOUNT": coef * 5,
        "NUMAROMATICRINGS": coef * 5,

        # Toxometris properties
        "SOLUBILITY": coef * 5,
        "SOLUBILITY_REL": coef * 0.5,
        "TOXICITY": coef * 1,
        "TOXICITY_REL": coef * 0.5,

        # Binding predictors
        "JNK3": coef * 1,
        "DRD2": coef * 1,
        "GSK3B": coef * 1
    }
    
    if prop_name in prop_name2sigma:
        return prop_name2sigma[prop_name]
    elif "DOCKING" in prop_name:
        return coef * 20
    elif "SIMILAR" in prop_name:
        return coef * 1
    else:
        return None
        # raise ValueError(f"Cannot select sigma for {prop_name}")


def randint(min_value: int, max_value: int):
    return np.random.randint(min_value, max_value + 1)


def randfloat(min_value: float, max_value: float):
    return np.random.uniform(   min_value, max_value)


def task_name2grpo_prompt(task_name: str, start_token: str, ranged: bool=True):
    task_name2randomized_prompt = {
        "dock.parp1": (
            f"[PROPERTY]parp1 {randfloat(10.0, 20.0):.2f}[/PROPERTY]"
            f"[QED]{randfloat(0.5, 1.0):.2f}[/QED]"
            f"[SAS]{randfloat(1.0, 5.0):.2f}[/SAS]"
        ),
        "dock.fa7": (
            f"[PROPERTY]fa7 {randfloat(10.0, 20.0):.2f}[/PROPERTY]"
            f"[QED]{randfloat(0.5, 1.0):.2f}[/QED]"
            f"[SAS]{randfloat(1.0, 5.0):.2f}[/SAS]"
        ),
        "dock.5ht1b": (
            f"[PROPERTY]5ht1b {randfloat(10.0, 20.0):.2f}[/PROPERTY]"
            f"[QED]{randfloat(0.5, 1.0):.2f}[/QED]"
            f"[SAS]{randfloat(1.0, 5.0):.2f}[/SAS]"
        ),
        "dock.braf": (
            f"[PROPERTY]braf {randfloat(10.0, 20.0):.2f}[/PROPERTY]"
            f"[QED]{randfloat(0.5, 1.0):.2f}[/QED]"
            f"[SAS]{randfloat(1.0, 5.0):.2f}[/SAS]"
        ),
        "dock.jak2": (
            f"[PROPERTY]jak2 {randfloat(10.0, 20.0):.2f}[/PROPERTY]"
            f"[QED]{randfloat(0.5, 1.0):.2f}[/QED]"
            f"[SAS]{randfloat(1.0, 5.0):.2f}[/SAS]"
        ),
        "pmo.perindopril_mpo": (
            f"[SIMILAR]{perindopril_smiles} {randfloat(0.95, 1.0):.2f}[/SIMILAR]"
            f"[NUMAROMATICRINGS]{randint(2, 2)}[/NUMAROMATICRINGS]"
        ),
        "pmo.perindopril_mpo_prop": (
            f"[PROPERTY]SIMILAR {randfloat(0.2, 1.0):.2f}[/PROPERTY]"
            f"[QED]{randfloat(0.5, 1.0):.2f}[/QED]"
            f"[SAS]{randfloat(1.0, 5.0):.2f}[/SAS]"
        ),
        "tox.DES": (
            f"[SIMILAR]{perindopril_smiles} {randfloat(0.4, 1.0):.2f}[/SIMILAR]"
            f"[SAS]{randfloat(1.0, 5.0):.2f}[/SAS]"
            f"[PROPERTY]solubility {randfloat(-4, 0.5):.2f}[/PROPERTY]"
            f"[PROPERTY]solubility_rel {randfloat(0.6, 1.0):.2f}[/PROPERTY]"
            f"[PROPERTY]toxicity {randfloat(0.0, 0.0):.2f}[/PROPERTY]"
            f"[PROPERTY]toxicity_rel {randfloat(0.6, 1.0):.2f}[/PROPERTY]"
        ),
        "tox.solubility": (
            f"[SIMILAR]{perindopril_smiles} {randfloat(0.4, 1.0):.2f}[/SIMILAR]"
            f"[PROPERTY]solubility {randfloat(-4, 0.5):.2f}[/PROPERTY]"
            f"[PROPERTY]solubility_rel {randfloat(0.6, 1.0):.2f}[/PROPERTY]"
        ),
        "tox.toxicity": (
            f"[SIMILAR]{perindopril_smiles} {randfloat(0.4, 1.0):.2f}[/SIMILAR]"
            f"[PROPERTY]toxicity {randfloat(0.0, 0.0):.2f}[/PROPERTY]"
            f"[PROPERTY]toxicity_rel {randfloat(0.6, 1.0):.2f}[/PROPERTY]"
        )
    }

    task_name2prompt = {
        "spec.6nzp_7uyt": (
            f"[DOCKING_SCORE][10.67,20.00][/DOCKING_SCORE]"
            f"[QED][0.40,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
        ),
        "spec.6nzp_5ut5": (
            f"[DOCKING_SCORE][10.67,20.00][/DOCKING_SCORE]"
            f"[QED][0.40,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
        ),
        "spec.6nzp_7uyw": (
            f"[DOCKING_SCORE][10.67,20.00][/DOCKING_SCORE]"
            f"[QED][0.40,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
        ),
        "dock.parp1": (
            f"[DOCKING_SCORE][10.00,20.00][/DOCKING_SCORE]"
            f"[QED][0.50,1.00][/QED]"
            f"[SAS][1.00,5.00][/SAS]"
        ),
        "dock.fa7": (
            f"[DOCKING_SCORE][8.50,20.00][/DOCKING_SCORE]"
            f"[QED][0.50,1.00][/QED]"
            f"[SAS][1.00,5.00][/SAS]"
        ),
        "dock.5ht1b": (
            f"[DOCKING_SCORE][8.79,20.00][/DOCKING_SCORE]"
            f"[QED][0.50,1.00][/QED]"
            f"[SAS][1.00,5.00][/SAS]"
        ),
        "dock.braf": (
            f"[DOCKING_SCORE][10.30,20.00][/DOCKING_SCORE]"
            f"[QED][0.50,1.00][/QED]"
            f"[SAS][1.00,5.00][/SAS]"
        ),
        "dock.jak2": (
            f"[DOCKING_SCORE][9.10,20.00][/DOCKING_SCORE]"
            f"[QED][0.50,1.00][/QED]"
            f"[SAS][1.00,5.00][/SAS]"
        ),
        "pmo.perindopril_mpo": (
            f"[SIMILAR]{perindopril_smiles} [0.95,1.00][/SIMILAR]"
            f"[NUMAROMATICRINGS][2,2][/NUMAROMATICRINGS]"
        ),
        "pmo.perindopril_mpo_prop": (
            f"[DOCKING_SCORE][10.00,20.00][/DOCKING_SCORE]"
            f"[QED][0.50,1.00][/QED]"
            f"[SAS][1.00,5.00][/SAS]"
        ),
        "tox.DES": (
            f"[SIMILAR]{perindopril_smiles} [0.40,1.00][/SIMILAR]"
            f"[SAS][1.00,5.00][/SAS]"
            f"[SOLUBILITY][-4.00,0.50][/SOLUBILITY]"
            f"[SOLUBILITY_REL][0.60,1.00][/SOLUBILITY_REL]"
            f"[TOXICITY][0.00,0.00][/TOXICITY]"
            f"[TOXICITY_REL][0.60,1.00][/TOXICITY_REL]"
        ),
        "tox.solubility": (
            f"[SIMILAR]{perindopril_smiles} [0.40,1.00][/SIMILAR]"
            f"[SOLUBILITY][-4.00,0.50][/SOLUBILITY]"
            f"[SOLUBILITY_REL][0.60,1.00][/SOLUBILITY_REL]"
        ),
        "tox.toxicity": (
            f"[SIMILAR]{perindopril_smiles} [0.40,1.00][/SIMILAR]"
            f"[TOXICITY][0.00,0.00][/TOXICITY]"
            f"[TOXICITY_REL][0.60,1.00][/TOXICITY_REL]"
        ),
        "lead.parp1_04_0": (
            f"[SIMILAR]{parp1_0} [0.40,1.00][/SIMILAR]"
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE]20.00[/DOCKING_SCORE]"
        ),
        "lead.parp1_04_1": (
            f"[SIMILAR]{parp1_1} [0.40,1.00][/SIMILAR]"
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE]20.00[/DOCKING_SCORE]"
        ),
        "lead.parp1_04_2": (
            f"[SIMILAR]{parp1_2} [0.40,1.00][/SIMILAR]"
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE]20.00[/DOCKING_SCORE]"
        ),
        "lead.parp1_06_0": (
            f"[SIMILAR]{parp1_0} [0.60,1.00][/SIMILAR]"
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE]20.00[/DOCKING_SCORE]"
        ),
        "lead.parp1_06_1": (
            f"[SIMILAR]{parp1_1} [0.60,1.00][/SIMILAR]"
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE]20.00[/DOCKING_SCORE]"
        ),
        "lead.parp1_06_2": (
            f"[SIMILAR]{parp1_2} [0.60,1.00][/SIMILAR]"
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE]20.00[/DOCKING_SCORE]"
        ),
        "lead.fa7_04_0": (
            f"[SIMILAR]{fa7_0} [0.40,1.00][/SIMILAR]"
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE]20.00[/DOCKING_SCORE]"
        ),
        "lead.fa7_04_1": (
            f"[SIMILAR]{fa7_1} [0.40,1.00][/SIMILAR]"
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE]20.00[/DOCKING_SCORE]"
        ),
        "lead.fa7_04_2": (
            f"[SIMILAR]{fa7_2} [0.40,1.00][/SIMILAR]"
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE]20.00[/DOCKING_SCORE]"
        ),
        "lead.fa7_06_0": (
            f"[SIMILAR]{fa7_0} [0.60,1.00][/SIMILAR]"
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE]20.00[/DOCKING_SCORE]"
        ),
        "lead.fa7_06_1": (
            f"[SIMILAR]{fa7_1} [0.60,1.00][/SIMILAR]"
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE]20.00[/DOCKING_SCORE]"
        ),
        "lead.fa7_06_2": (
            f"[SIMILAR]{fa7_2} [0.60,1.00][/SIMILAR]"
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE]20.00[/DOCKING_SCORE]"
        ),
        "lead.5ht1b_04_0": (
            f"[SIMILAR]{_5ht1b_0} [0.40,1.00][/SIMILAR]"
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE]20.00[/DOCKING_SCORE]"
        ),
        "lead.5ht1b_04_1": (
            f"[SIMILAR]{_5ht1b_1} [0.40,1.00][/SIMILAR]"
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE]20.00[/DOCKING_SCORE]"
        ),
        "lead.5ht1b_04_2": (
            f"[SIMILAR]{_5ht1b_2} [0.40,1.00][/SIMILAR]"
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE]20.00[/DOCKING_SCORE]"
        ),
        "lead.5ht1b_06_0": (
            f"[SIMILAR]{_5ht1b_0} [0.60,1.00][/SIMILAR]"
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE]20.00[/DOCKING_SCORE]"
        ),
        "lead.5ht1b_06_1": (
            f"[SIMILAR]{_5ht1b_1} [0.60,1.00][/SIMILAR]"
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE]20.00[/DOCKING_SCORE]"
        ),
        "lead.5ht1b_06_2": (
            f"[SIMILAR]{_5ht1b_2} [0.60,1.00][/SIMILAR]"
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE]20.00[/DOCKING_SCORE]"
        ),
        "lead.braf_04_0": (
            f"[SIMILAR]{braf_0} [0.40,1.00][/SIMILAR]"
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE]20.00[/DOCKING_SCORE]"
        ),
        "lead.braf_04_1": (
            f"[SIMILAR]{braf_1} [0.40,1.00][/SIMILAR]"
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE]20.00[/DOCKING_SCORE]"
        ),
        "lead.braf_04_2": (
            f"[SIMILAR]{braf_2} [0.40,1.00][/SIMILAR]"
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE]20.00[/DOCKING_SCORE]"
        ),
        "lead.braf_06_0": (
            f"[SIMILAR]{braf_0} [0.60,1.00][/SIMILAR]"
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE]20.00[/DOCKING_SCORE]"
        ),
        "lead.braf_06_1": (
            f"[SIMILAR]{braf_1} [0.60,1.00][/SIMILAR]"
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE]20.00[/DOCKING_SCORE]"
        ),
        "lead.braf_06_2": (
            f"[SIMILAR]{braf_2} [0.60,1.00][/SIMILAR]"
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE]20.00[/DOCKING_SCORE]"
        ),
        "lead.jak2_04_0": (
            f"[SIMILAR]{jak2_0} [0.40,1.00][/SIMILAR]"
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE]20.00[/DOCKING_SCORE]"
        ),
        "lead.jak2_04_1": (
            f"[SIMILAR]{jak2_1} [0.40,1.00][/SIMILAR]"
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE]20.00[/DOCKING_SCORE]"
        ),
        "lead.jak2_04_2": (
            f"[SIMILAR]{jak2_2} [0.40,1.00][/SIMILAR]"
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE]20.00[/DOCKING_SCORE]"
        ),
        "lead.jak2_06_0": (
            f"[SIMILAR]{jak2_0} [0.60,1.00][/SIMILAR]"
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE]20.00[/DOCKING_SCORE]"
        ),
        "lead.jak2_06_1": (
            f"[SIMILAR]{jak2_1} [0.60,1.00][/SIMILAR]"
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE]20.00[/DOCKING_SCORE]"
        ),
        "lead.jak2_06_2": (
            f"[SIMILAR]{jak2_2} [0.60,1.00][/SIMILAR]"
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE]20.00[/DOCKING_SCORE]"
        ),
        "lead.jnk3_04_0": (
            f"[SIMILAR]{parp1_0} [0.40,1.00][/SIMILAR]"
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE]20.00[/DOCKING_SCORE]"
        ),
        "lead.jnk3_06_0": (
            f"[SIMILAR]{parp1_0} [0.60,1.00][/SIMILAR]"
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE]20.00[/DOCKING_SCORE]"
        ),
        "lead.drd2_04_0": (
            f"[SIMILAR]{parp1_0} [0.40,1.00][/SIMILAR]"
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE]20.00[/DOCKING_SCORE]"
        ),
        "lead.drd2_06_0": (
            f"[SIMILAR]{parp1_0} [0.60,1.00][/SIMILAR]"
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE]20.00[/DOCKING_SCORE]"
        ),
        "lead.gsk3b_04_0": (
            f"[SIMILAR]{parp1_0} [0.40,1.00][/SIMILAR]"
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE]20.00[/DOCKING_SCORE]"
        ),
        "lead.gsk3b_06_0": (
            f"[SIMILAR]{parp1_0} [0.60,1.00][/SIMILAR]"
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE]20.00[/DOCKING_SCORE]"
        ),
        "lead_no_sim.jnk3_04_0": (
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE][0.60,1.00][/DOCKING_SCORE]"
        ),
        "lead_no_sim.drd2_04_0": (
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE][0.60,1.00][/DOCKING_SCORE]"
        ),
        "lead_no_sim.gsk3b_04_0": (
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE][0.60,1.00][/DOCKING_SCORE]"
        ),
        "hit.jnk3_04_0": (
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE]20.00[/DOCKING_SCORE]"
        ),
        "hit.drd2_04_0": (
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE]20.00[/DOCKING_SCORE]"
        ),
        "hit.gsk3b_04_0": (
            f"[QED][0.60,1.00][/QED]"
            f"[SAS][1.00,4.00][/SAS]"
            f"[DOCKING_SCORE]20.00[/DOCKING_SCORE]"
        )
    }
    if ranged is False and task_name in task_name2randomized_prompt:
        return task_name2randomized_prompt[task_name] + start_token
    if ranged is True and task_name in task_name2prompt:
        return task_name2prompt[task_name] + start_token
    raise ValueError(f"{task_name} does not exist")


def task_name2computer_names(task_name: str):
    task_name2computer_names_dict = {
        # pmo oracles
        "pmo.osimertinib_mpo": [f"SIMILAR.{osimertinib_smiles}", "TPSA", "CLOGP"],
        "pmo.fexofenadine_mpo": [f"SIMILAR.{fexofenadine_smiles}", "TPSA", "CLOGP"],
        "pmo.ranolazine_mpo": [f"SIMILAR.{ranolazine_smiles}", "CLOGP", "TPSA"],
        "pmo.perindopril_mpo": [f"SIMILAR.{perindopril_smiles}", "NUMAROMATICRINGS"],
        "pmo.perindopril_mpo_prop": [f"SIMILAR.{perindopril_smiles}", "QED", "SAS"],
        "pmo.amlodipine_mpo": [f"SIMILAR.{amlodipine_smiles}", "NUMRINGS"],
        "pmo.sitagliptin_mpo": [f"SIMILAR.{sitagliptin_smiles}", "CLOGP", "TPSA", "FORMULA"],
        "pmo.zaleplon_mpo": [f"SIMILAR.{zaleplon_smiles}", "FORMULA"],

        # spec oracles
        "spec.6nzp_7uyt": ["DOCKING.6nzp", "DOCKING.7uyt", "QED", "SAS"],
        "spec.6nzp_5ut5": ["DOCKING.6nzp", "DOCKING.5ut5", "QED", "SAS"],
        "spec.6nzp_7uyw": ["DOCKING.6nzp", "DOCKING.7uyw", "QED", "SAS"],

        # docking oracles
        "dock.parp1": ["DOCKING.parp1", "QED", "SAS"],
        "dock.jak2": ["DOCKING.jak2", "QED", "SAS"],
        "dock.braf": ["DOCKING.braf", "QED", "SAS"],
        "dock.fa7": ["DOCKING.fa7", "QED", "SAS"],
        "dock.5ht1b": ["DOCKING.5ht1b", "QED", "SAS"],

        # toxometris oracles
        "tox.DES": ["SIMILAR.CC/C(=C(/CC)c1ccc(O)cc1)c1ccc(O)cc1", "SAS", "SOLUBILITY", "SOLUBILITY_REL", "TOXICITY", "TOXICITY_REL"],
        "tox.solubility": ["SOLUBILITY", "SOLUBILITY_REL", "SIMILAR.CC/C(=C(/CC)c1ccc(O)cc1)c1ccc(O)cc1"],
        "tox.toxicity": ["TOXICITY", "TOXICITY_REL", "SIMILAR.CC/C(=C(/CC)c1ccc(O)cc1)c1ccc(O)cc1"],
        
        # lead oracles
        "lead.parp1_04_0": ["DOCKING.parp1", f"SIMILAR.{parp1_0}", "QED", "SAS"],
        "lead.parp1_04_1": ["DOCKING.parp1", f"SIMILAR.{parp1_1}", "QED", "SAS"],
        "lead.parp1_04_2": ["DOCKING.parp1", f"SIMILAR.{parp1_2}", "QED", "SAS"],
        "lead.parp1_06_0": ["DOCKING.parp1", f"SIMILAR.{parp1_0}", "QED", "SAS"],
        "lead.parp1_06_1": ["DOCKING.parp1", f"SIMILAR.{parp1_1}", "QED", "SAS"],
        "lead.parp1_06_2": ["DOCKING.parp1", f"SIMILAR.{parp1_2}", "QED", "SAS"],
        "lead.fa7_04_0": ["DOCKING.fa7", f"SIMILAR.{fa7_0}", "QED", "SAS"],
        "lead.fa7_04_1": ["DOCKING.fa7", f"SIMILAR.{fa7_1}", "QED", "SAS"],
        "lead.fa7_04_2": ["DOCKING.fa7", f"SIMILAR.{fa7_2}", "QED", "SAS"],
        "lead.fa7_06_0": ["DOCKING.fa7", f"SIMILAR.{fa7_0}", "QED", "SAS"],
        "lead.fa7_06_1": ["DOCKING.fa7", f"SIMILAR.{fa7_1}", "QED", "SAS"],
        "lead.fa7_06_2": ["DOCKING.fa7", f"SIMILAR.{fa7_2}", "QED", "SAS"],
        "lead.5ht1b_04_0": ["DOCKING.5ht1b", f"SIMILAR.{_5ht1b_0}", "QED", "SAS"],
        "lead.5ht1b_04_1": ["DOCKING.5ht1b", f"SIMILAR.{_5ht1b_1}", "QED", "SAS"],
        "lead.5ht1b_04_2": ["DOCKING.5ht1b", f"SIMILAR.{_5ht1b_2}", "QED", "SAS"],
        "lead.5ht1b_06_0": ["DOCKING.5ht1b", f"SIMILAR.{_5ht1b_0}", "QED", "SAS"],
        "lead.5ht1b_06_1": ["DOCKING.5ht1b", f"SIMILAR.{_5ht1b_1}", "QED", "SAS"],
        "lead.5ht1b_06_2": ["DOCKING.5ht1b", f"SIMILAR.{_5ht1b_2}", "QED", "SAS"],
        "lead.braf_04_0": ["DOCKING.braf", f"SIMILAR.{braf_0}", "QED", "SAS"],
        "lead.braf_04_1": ["DOCKING.braf", f"SIMILAR.{braf_1}", "QED", "SAS"],
        "lead.braf_04_2": ["DOCKING.braf", f"SIMILAR.{braf_2}", "QED", "SAS"],
        "lead.braf_06_0": ["DOCKING.braf", f"SIMILAR.{braf_0}", "QED", "SAS"],
        "lead.braf_06_1": ["DOCKING.braf", f"SIMILAR.{braf_1}", "QED", "SAS"],
        "lead.braf_06_2": ["DOCKING.braf", f"SIMILAR.{braf_2}", "QED", "SAS"],
        "lead.jak2_04_0": ["DOCKING.jak2", f"SIMILAR.{jak2_0}", "QED", "SAS"],
        "lead.jak2_04_1": ["DOCKING.jak2", f"SIMILAR.{jak2_1}", "QED", "SAS"],
        "lead.jak2_04_2": ["DOCKING.jak2", f"SIMILAR.{jak2_2}", "QED", "SAS"],
        "lead.jak2_06_0": ["DOCKING.jak2", f"SIMILAR.{jak2_0}", "QED", "SAS"],
        "lead.jak2_06_1": ["DOCKING.jak2", f"SIMILAR.{jak2_1}", "QED", "SAS"],
        "lead.jak2_06_2": ["DOCKING.jak2", f"SIMILAR.{jak2_2}", "QED", "SAS"],
        "lead.jnk3_04_0": ["JNK3", f"SIMILAR.{parp1_0}", "QED", "SAS"],
        "lead.drd2_04_0": ["DRD2", f"SIMILAR.{parp1_0}", "QED", "SAS"],
        "lead.gsk3b_04_0": ["GSK3B", f"SIMILAR.{parp1_0}", "QED", "SAS"],
        
        "lead_no_sim.jnk3_04_0": ["JNK3", "QED", "SAS"],
        "lead_no_sim.drd2_04_0": ["DRD2", "QED", "SAS"],
        "lead_no_sim.gsk3b_04_0": ["GSK3B", "QED", "SAS"],

        "hit.jnk3_04_0": ["JNK3", "QED", "SAS"],
        "hit.drd2_04_0": ["DRD2", "QED", "SAS"],
        "hit.gsk3b_04_0": ["GSK3B", "QED", "SAS"],
    }
    if task_name in task_name2computer_names_dict.keys():
        return task_name2computer_names_dict[task_name]
    
    raise ValueError(f"No task name {task_name}")


def task_name2hit_ranges(task_name: str):
    return {
        # pmo oracles
        "pmo.perindopril_mpo_prop": [[0.6, 1.0], [0.5, 1.0], [1.0, 5.0]],
        
        # spec oracles
        "spec.6nzp_7uyt": [[10.67, math.inf], [0.0, math.inf], [0.4, 1.0], [1.0, 4.0]],
        "spec.6nzp_5ut5": [[10.67, math.inf], [0.0, math.inf], [0.4, 1.0], [1.0, 4.0]],
        "spec.6nzp_7uyw": [[10.67, math.inf], [0.0, math.inf], [0.4, 1.0], [1.0, 4.0]],

        # docking oracles
        "dock.parp1": [[10.0, math.inf], [0.5, 1.0], [1.0, 5.0]],
        "dock.fa7": [[8.5, math.inf], [0.5, 1.0], [1.0, 5.0]],
        "dock.5ht1b": [[8.7845, math.inf], [0.5, 1.0], [1.0, 5.0]],
        "dock.braf": [[10.3, math.inf], [0.5, 1.0], [1.0, 5.0]],
        "dock.jak2": [[9.1, math.inf], [0.5, 1.0], [1.0, 5.0]],


        # lead oracles
        "lead.parp1_04_0": [None, [0.4, 1.0], [0.6, 1.0], [1.0, 4.0]],
        "lead.parp1_04_1": [None, [0.4, 1.0], [0.6, 1.0], [1.0, 4.0]],
        "lead.parp1_04_2": [None, [0.4, 1.0], [0.6, 1.0], [1.0, 4.0]],
        "lead.parp1_06_0": [None, [0.6, 1.0], [0.6, 1.0], [1.0, 4.0]],
        "lead.parp1_06_1": [None, [0.6, 1.0], [0.6, 1.0], [1.0, 4.0]],
        "lead.parp1_06_2": [None, [0.6, 1.0], [0.6, 1.0], [1.0, 4.0]],
        "lead.fa7_04_0": [None, [0.4, 1.0], [0.6, 1.0], [1.0, 4.0]],
        "lead.fa7_04_1": [None, [0.4, 1.0], [0.6, 1.0], [1.0, 4.0]],
        "lead.fa7_04_2": [None, [0.4, 1.0], [0.6, 1.0], [1.0, 4.0]],
        "lead.fa7_06_0": [None, [0.6, 1.0], [0.6, 1.0], [1.0, 4.0]],
        "lead.fa7_06_1": [None, [0.6, 1.0], [0.6, 1.0], [1.0, 4.0]],
        "lead.fa7_06_2": [None, [0.6, 1.0], [0.6, 1.0], [1.0, 4.0]],
        "lead.5ht1b_04_0": [None, [0.4, 1.0], [0.6, 1.0], [1.0, 4.0]],
        "lead.5ht1b_04_1": [None, [0.4, 1.0], [0.6, 1.0], [1.0, 4.0]],
        "lead.5ht1b_04_2": [None, [0.4, 1.0], [0.6, 1.0], [1.0, 4.0]],
        "lead.5ht1b_06_0": [None, [0.6, 1.0], [0.6, 1.0], [1.0, 4.0]],
        "lead.5ht1b_06_1": [None, [0.6, 1.0], [0.6, 1.0], [1.0, 4.0]],
        "lead.5ht1b_06_2": [None, [0.6, 1.0], [0.6, 1.0], [1.0, 4.0]],
        "lead.braf_04_0": [None, [0.4, 1.0], [0.6, 1.0], [1.0, 4.0]],
        "lead.braf_04_1": [None, [0.4, 1.0], [0.6, 1.0], [1.0, 4.0]],
        "lead.braf_04_2": [None, [0.4, 1.0], [0.6, 1.0], [1.0, 4.0]],
        "lead.braf_06_0": [None, [0.6, 1.0], [0.6, 1.0], [1.0, 4.0]],
        "lead.braf_06_1": [None, [0.6, 1.0], [0.6, 1.0], [1.0, 4.0]],
        "lead.braf_06_2": [None, [0.6, 1.0], [0.6, 1.0], [1.0, 4.0]],
        "lead.jak2_04_0": [None, [0.4, 1.0], [0.6, 1.0], [1.0, 4.0]],
        "lead.jak2_04_1": [None, [0.4, 1.0], [0.6, 1.0], [1.0, 4.0]],
        "lead.jak2_04_2": [None, [0.4, 1.0], [0.6, 1.0], [1.0, 4.0]],
        "lead.jak2_06_0": [None, [0.6, 1.0], [0.6, 1.0], [1.0, 4.0]],
        "lead.jak2_06_1": [None, [0.6, 1.0], [0.6, 1.0], [1.0, 4.0]],
        "lead.jak2_06_2": [None, [0.6, 1.0], [0.6, 1.0], [1.0, 4.0]],
        "lead.jnk3_04_0": [None, [0.4, 1.0], [0.6, 1.0], [1.0, 4.0]],
        "lead.drd2_04_0": [None, [0.4, 1.0], [0.6, 1.0], [1.0, 4.0]],
        "lead.gsk3b_04_0": [None, [0.4, 1.0], [0.6, 1.0], [1.0, 4.0]],
        "lead.jnk3_06_0": [None, [0.6, 1.0], [0.6, 1.0], [1.0, 4.0]],
        "lead.drd2_06_0": [None, [0.6, 1.0], [0.6, 1.0], [1.0, 4.0]],
        "lead.gsk3b_06_0": [None, [0.6, 1.0], [0.6, 1.0], [1.0, 4.0]],

        "lead_no_sim.jnk3_04_0": [None, [0.6, 1.0], [1.0, 4.0]],
        "lead_no_sim.drd2_04_0": [None, [0.6, 1.0], [1.0, 4.0]],
        "lead_no_sim.gsk3b_04_0": [None, [0.6, 1.0], [1.0, 4.0]],

        "hit.jnk3_04_0": [[0.6, 1.0], [1.0, 5.0], [0.5, 1.0]],
        "hit.drd2_04_0": [[0.6, 1.0], [1.0, 5.0], [0.5, 1.0]],
        "hit.gsk3b_04_0": [[0.6, 1.0], [1.0, 5.0], [0.5, 1.0]]
    }[task_name]
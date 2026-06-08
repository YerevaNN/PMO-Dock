import math

import numpy as np

from benchmark.actives_loader import lead_seed_smiles
from benchmark.docking_oracle.docking import ANTITARGET_RECEPTORS
from benchmark.spec_tasks import spec_task_name

_SPEC_PROMPT = (
    f"[DOCKING_SCORE][10.67,20.00][/DOCKING_SCORE]"
    f"[QED][0.40,1.00][/QED]"
    f"[SAS][1.00,4.00][/SAS]"
)
_SPEC_HIT_RANGES = [[10.67, math.inf], [0.0, math.inf], [0.4, 1.0], [1.0, 4.0]]


def _spec_prompt_dict():
    return {spec_task_name(at): _SPEC_PROMPT for at in ANTITARGET_RECEPTORS}


def _spec_computer_names():
    return {
        spec_task_name(at): ["DOCKING.6nzp", f"DOCKING.{at}", "QED", "SAS"]
        for at in ANTITARGET_RECEPTORS
    }


def _spec_hit_ranges_dict():
    return {spec_task_name(at): _SPEC_HIT_RANGES for at in ANTITARGET_RECEPTORS}


_ALLOWED_TASK_PREFIXES = ("hit.", "lead.", "spec.")
_ALLOWED_HIT_REWARD_TYPES = frozenset({"hit", "max", "geam"})
_HIT_TARGETS = frozenset({"parp1", "fa7", "5ht1b", "braf", "jak2"})


def validate_task_name(task_name: str) -> str:
    if not task_name or "." not in task_name:
        raise ValueError(
            f"Task name '{task_name}' is invalid. "
            "Use hit.<target>, lead.<target>_<sim>_<seed>, or spec.6nzp_<antitarget>."
        )
    if task_name.startswith("dock."):
        raise ValueError(
            f"Task name '{task_name}' is not supported. "
            "Use hit.<target> instead of dock.<target> (e.g. hit.parp1)."
        )
    if task_name.lower().startswith("geam") or task_name.lower().split(".", 1)[0] == "geam":
        raise ValueError(
            f"Task name '{task_name}' is not supported. "
            "GEAM is a reward/oracle mode in Saturn and GenMol, not a genetic_chemalactica task prefix."
        )
    if task_name.startswith(("pmo.", "lead_no_sim.")):
        raise ValueError(
            f"Task name '{task_name}' is not supported in genetic_chemalactica. "
            "Use hit.<target>, lead.<target>_<sim>_<seed>, or spec.6nzp_<antitarget>."
        )
    if not any(task_name.startswith(prefix) for prefix in _ALLOWED_TASK_PREFIXES):
        raise ValueError(
            f"Task name '{task_name}' must start with one of: {', '.join(_ALLOWED_TASK_PREFIXES)}"
        )
    return task_name


def is_hit_docking_task(task_name: str) -> bool:
    """True for hit.<target> docking tasks (not lead or spec)."""
    if not task_name.startswith("hit."):
        return False
    target = task_name.split(".", 1)[1]
    return target in _HIT_TARGETS


def validate_reward_type(task_name: str, reward_type: str) -> str:
    """Validate --reward_type for hit.<target> docking tasks (hit, max, or geam)."""
    reward_type = (reward_type or "hit").strip().lower()
    if is_hit_docking_task(task_name) and reward_type not in _ALLOWED_HIT_REWARD_TYPES:
        raise ValueError(
            f"reward_type '{reward_type}' is invalid for {task_name}. "
            f"Allowed values: {', '.join(sorted(_ALLOWED_HIT_REWARD_TYPES))}."
        )
    return reward_type


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
    validate_task_name(task_name)
    task_name2randomized_prompt = {
        "hit.parp1": (
            f"[PROPERTY]parp1 {randfloat(10.0, 20.0):.2f}[/PROPERTY]"
            f"[QED]{randfloat(0.5, 1.0):.2f}[/QED]"
            f"[SAS]{randfloat(1.0, 5.0):.2f}[/SAS]"
        ),
        "hit.fa7": (
            f"[PROPERTY]fa7 {randfloat(10.0, 20.0):.2f}[/PROPERTY]"
            f"[QED]{randfloat(0.5, 1.0):.2f}[/QED]"
            f"[SAS]{randfloat(1.0, 5.0):.2f}[/SAS]"
        ),
        "hit.5ht1b": (
            f"[PROPERTY]5ht1b {randfloat(10.0, 20.0):.2f}[/PROPERTY]"
            f"[QED]{randfloat(0.5, 1.0):.2f}[/QED]"
            f"[SAS]{randfloat(1.0, 5.0):.2f}[/SAS]"
        ),
        "hit.braf": (
            f"[PROPERTY]braf {randfloat(10.0, 20.0):.2f}[/PROPERTY]"
            f"[QED]{randfloat(0.5, 1.0):.2f}[/QED]"
            f"[SAS]{randfloat(1.0, 5.0):.2f}[/SAS]"
        ),
        "hit.jak2": (
            f"[PROPERTY]jak2 {randfloat(10.0, 20.0):.2f}[/PROPERTY]"
            f"[QED]{randfloat(0.5, 1.0):.2f}[/QED]"
            f"[SAS]{randfloat(1.0, 5.0):.2f}[/SAS]"
        ),
    }

    task_name2prompt = {
        **_spec_prompt_dict(),
        "hit.parp1": (
            f"[DOCKING_SCORE][10.00,20.00][/DOCKING_SCORE]"
            f"[QED][0.50,1.00][/QED]"
            f"[SAS][1.00,5.00][/SAS]"
        ),
        "hit.fa7": (
            f"[DOCKING_SCORE][8.50,20.00][/DOCKING_SCORE]"
            f"[QED][0.50,1.00][/QED]"
            f"[SAS][1.00,5.00][/SAS]"
        ),
        "hit.5ht1b": (
            f"[DOCKING_SCORE][8.79,20.00][/DOCKING_SCORE]"
            f"[QED][0.50,1.00][/QED]"
            f"[SAS][1.00,5.00][/SAS]"
        ),
        "hit.braf": (
            f"[DOCKING_SCORE][10.30,20.00][/DOCKING_SCORE]"
            f"[QED][0.50,1.00][/QED]"
            f"[SAS][1.00,5.00][/SAS]"
        ),
        "hit.jak2": (
            f"[DOCKING_SCORE][9.10,20.00][/DOCKING_SCORE]"
            f"[QED][0.50,1.00][/QED]"
            f"[SAS][1.00,5.00][/SAS]"
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
    }
    if ranged is False and task_name in task_name2randomized_prompt:
        return task_name2randomized_prompt[task_name] + start_token
    if ranged is True and task_name in task_name2prompt:
        return task_name2prompt[task_name] + start_token
    raise ValueError(f"{task_name} does not exist")


def task_name2computer_names(task_name: str):
    task_name = validate_task_name(task_name)
    task_name2computer_names_dict = {
        # spec oracles
        **_spec_computer_names(),

        # docking oracles
        "hit.parp1": ["DOCKING.parp1", "QED", "SAS"],
        "hit.jak2": ["DOCKING.jak2", "QED", "SAS"],
        "hit.braf": ["DOCKING.braf", "QED", "SAS"],
        "hit.fa7": ["DOCKING.fa7", "QED", "SAS"],
        "hit.5ht1b": ["DOCKING.5ht1b", "QED", "SAS"],

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
    }
    if task_name in task_name2computer_names_dict.keys():
        return task_name2computer_names_dict[task_name]
    
    raise ValueError(f"No task name {task_name}")


def task_name2hit_ranges(task_name: str):
    task_name = validate_task_name(task_name)
    return {
        # spec oracles
        **_spec_hit_ranges_dict(),

        # docking oracles
        "hit.parp1": [[10.0, math.inf], [0.5, 1.0], [1.0, 5.0]],
        "hit.fa7": [[8.5, math.inf], [0.5, 1.0], [1.0, 5.0]],
        "hit.5ht1b": [[8.7845, math.inf], [0.5, 1.0], [1.0, 5.0]],
        "hit.braf": [[10.3, math.inf], [0.5, 1.0], [1.0, 5.0]],
        "hit.jak2": [[9.1, math.inf], [0.5, 1.0], [1.0, 5.0]],


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
    }[task_name]
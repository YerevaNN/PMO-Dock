import math
import re

import numpy as np

from benchmark.rewards import gaussian_modifier


def _parse_molecular_formula(formula: str):
    matches = re.findall(r"([A-Z][a-z]*)(\d*)", formula)
    return [(elem, 1 if not cnt else int(cnt)) for elem, cnt in matches]


def isomer_scoring(formula1: str, formula2: str):
    atom2cnt_lst1 = _parse_molecular_formula(formula1)
    atom2cnt_lst2 = _parse_molecular_formula(formula2)
    res = []
    for target_atom, target_cnt in atom2cnt_lst2:
        for atom, cnt in atom2cnt_lst1:
            if target_atom == atom:
                res.append(gaussian_modifier(abs(target_cnt - cnt), mu=0, sigma=1.0))
                break
    total_num_atoms1 = sum(cnt for _, cnt in atom2cnt_lst1)
    total_num_atoms2 = sum(cnt for _, cnt in atom2cnt_lst2)
    res.append(gaussian_modifier(abs(total_num_atoms1 - total_num_atoms2), mu=0, sigma=2.0))
    return np.array(res).prod().tolist()


def compute_gaussian_err(measured, real, sigmas, agg=True):
    errors = []
    for m, r, sigma in zip(measured, real, sigmas, strict=True):
        if type(m) == str:
            score = isomer_scoring(m, r)
        else:
            score = gaussian_modifier(m - r, mu=0, sigma=sigma)
        errors.append(score)

    if agg:
        return np.array(errors).prod().tolist()
    return errors

def compute_geam_reward(measured):
    docking_scores = measured[0]
    sa_scores = measured[1]
    qed_scores = measured[2]
    
    # Formula used in GEAM paper
    trans_sa_scores = (10 - sa_scores) / 9
    aggregated_scores = (np.clip(docking_scores, 0, 20) / 20) * qed_scores * trans_sa_scores
    return aggregated_scores


def hit_reward(measured, sigmas, hit_ranges, prod=True, avg=False):
    rewards = []
    for m, rnge, sigma in zip(measured, hit_ranges, sigmas, strict=True):
        if m is None:
            rewards.append(0)
        elif rnge[0] <= m <= rnge[1]:
            rewards.append(1)
        else:
            dist = min(abs(m - rnge[0]), abs(m - rnge[1]))
            reward = gaussian_modifier(dist, mu=0, sigma=sigma)
            rewards.append(reward)

    if prod:
        return np.array(rewards).prod().item()
    elif avg:
        return np.array(rewards).mean().item()
    return np.array(rewards)


def hit_docking_score_reward(measured, sigmas, hit_ranges):
    docking_score = measured[0]
    computed_hit_reward = hit_reward(measured[1:], sigmas[1:], hit_ranges[1:])
    return docking_score / 20 * computed_hit_reward


def hit_spec_reward(measured, sigmas, hit_ranges):
    assert len(measured) == 4
    target_docking_score = measured[0]
    antitarget_docking_score = measured[1]
    gap = np.clip(target_docking_score / 20 - antitarget_docking_score / 20, 0, 1)
    computed_hit_reward = hit_reward(measured, sigmas, hit_ranges)
    return gap * computed_hit_reward

def hit_similarity_reward(measured, sigmas, hit_ranges):
    assert len(measured) == 3
    similarity_score = measured[0]
    computed_hit_reward = hit_reward(measured[1:], sigmas[1:], hit_ranges[1:])
    return similarity_score * computed_hit_reward



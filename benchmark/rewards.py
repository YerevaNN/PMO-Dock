"""
Shared reward helpers for hit/lead tasks.
"""
import numpy as np

def gaussian_modifier(x, mu, sigma):
    return np.exp(-0.5 * np.power((x - mu) / sigma, 2.0))

def select_sigma(prop_name: str):
    """Shared sigma for reward scaling; supports both root and genetic_chemalactica/saturn key names."""
    coef = 0.1
    prop_name2sigma = {
        "qed_score": coef * 1,
        "sa_score": coef * 7,
        "docking_score": coef * 20,
        "CLOGP": coef * 18,
        "TPSA": coef * 100,
        # genetic_chemalactica / saturn naming
        "QED": coef * 1,
        "SAS": coef * 7,
        "WEIGHT": coef * 1000,
        "RINGCOUNT": coef * 5,
        "NUMAROMATICRINGS": coef * 5,
        "SOLUBILITY": coef * 5,
        "SOLUBILITY_REL": coef * 0.5,
        "TOXICITY": coef * 1,
        "TOXICITY_REL": coef * 0.5,
        "JNK3": coef * 1,
        "DRD2": coef * 1,
        "GSK3B": coef * 1,
    }
    if prop_name in prop_name2sigma:
        return prop_name2sigma[prop_name]
    if "docking_score" in prop_name or "DOCKING" in prop_name:
        return coef * 20
    if "SIMILAR" in prop_name:
        return coef * 1
    return None

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

def compute_geam_reward(measured):
    """GEAM paper formula: (docking/20) * qed * (10-sa)/9."""
    docking_scores = measured[0]
    sa_scores = measured[1]
    qed_scores = measured[2]
    trans_sa_scores = (10 - sa_scores) / 9
    aggregated_scores = (np.clip(docking_scores, 0, 20) / 20) * qed_scores * trans_sa_scores
    return aggregated_scores

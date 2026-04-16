def task_name2constraints(task_name):
    if task_name in ["spec.6nzp_7uyt", "spec.6nzp_5ut5", "spec.6nzp_7uyw"]:
        return {
            'qed_score': [0.40, 1.00],
            'sa_score': [1.00, 4.00],
            'docking_score': [10.67, 20.00],
            'antitarget_docking_score': [0.00, 20.00]
        }
    return {
        'lead.sim_04': {
            'qed_score': [0.60, 1.00],
            'sa_score': [1.00, 4.00],
            'similarity_score': [0.40, 1.00]
        },
        'lead.sim_06': {
            'qed_score': [0.60, 1.00],
            'sa_score': [1.00, 4.00],
            'similarity_score': [0.60, 1.00]
        },
        'hit.parp1': {
            'qed_score': [0.50, 1.00],
            'sa_score': [1.00, 5.00],
            'docking_score': [10.00, 20.00]
        },
        'hit.fa7': {
            'qed_score': [0.50, 1.00],
            'sa_score': [1.00, 5.00],
            'docking_score': [8.50, 20.00]
        },
        'hit.5ht1b': {
            'qed_score': [0.50, 1.00],
            'sa_score': [1.00, 5.00],
            'docking_score': [8.7845, 20.00]
        },
        'hit.braf': {
            'qed_score': [0.50, 1.00],
            'sa_score': [1.00, 5.00],
            'docking_score': [10.30, 20.00]
        },
        'hit.jak2': {
            'qed_score': [0.50, 1.00],
            'sa_score': [1.00, 5.00],
            'docking_score': [9.10, 20.00]
        },
        "hit.pmo": {
            'qed_score': [0.50, 1.00],
            'sa_score': [1.00, 5.00],
            'docking_score': [0.60, 1.00]
        }
    }[task_name]
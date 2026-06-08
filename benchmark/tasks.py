from benchmark.spec_tasks import SPEC_TASK_CONSTRAINTS, is_spec_task_name


def task_name2constraints(task_name):
    if is_spec_task_name(task_name):
        return dict(SPEC_TASK_CONSTRAINTS)
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
        }
    }[task_name]


def lead_qed_sa_hit_constraints(sim: str):
    """Lead docking curves: ``sim`` is ``'04'`` or ``'06'`` (similarity tier)."""
    return task_name2constraints(f"lead.sim_{sim}")
# Category-best hyperparameters

This table is the source of truth for the frozen validation-sweep configuration of each algorithm × task. Runner YAML/JSON files under each method are convenience copies of these values. If a config file disagrees with this table, **use the table**.


## Frozen values (test leaderboard)

| Algorithm | Task | exp | Tuned hyperparameters |
|-----------|------|-----|------------------------|
| ChemLlama | Hit | 6 | `pool_size=10`, `num_prompts=50`, `num_similars=3` |
| ChemLlama | Lead | **7** | `pool_size=10`, `num_prompts=50`, `num_similars=4` |
| ChemLlama | Spec | 27 | `pool_size=20`, `num_prompts=100`, `num_similars=4` |
| Genetic GFN | Hit | 2 | `rank_coefficient=0.01`, `kl_coefficient=0.0005` |
| Genetic GFN | Lead | 3 | `rank_coefficient=0.03`, `kl_coefficient=0.001` |
| Genetic GFN | Spec | 2 | `rank_coefficient=0.01`, `kl_coefficient=0.0005` |
| GenMol | Hit | 117 | `softmax_temp=1.5`, `randomness=1.2`, `gamma=0.5`, `pool=random`, `reward=original` |
| GenMol | Lead | 19 | `softmax_temp=1.5`, `randomness=3`, `gamma=0`, `pool=seed`, `reward=original` |
| GenMol | Spec | 20 | `softmax_temp=0.5`, `randomness=0.5`, `gamma=0.1`, `pool=random`, `reward=original` |
| Saturn | Hit | 10 | `sigma=256`, `execute_hallucinated_memory=false`, reward `geam` |
| Saturn | Lead | 15 | `sigma=512`, `execute_hallucinated_memory=true`, reward `geam` |
| Saturn | Spec | 11 | `sigma=512`, `execute_hallucinated_memory=false`, reward `geam` |


Untuned knobs stay at each method’s default config.

## Grid → exp index

Decode any `exp-N` with the product order below. First listed axis varies slowest.

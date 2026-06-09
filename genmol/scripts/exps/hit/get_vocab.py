import os
import sys
from collections import defaultdict
from tqdm import trange
import pandas as pd

from genmol.utils.utils_chem import cut
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

_genmol_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def _zinc250k_csv() -> str:
    return os.environ.get("GENMOL_ZINC250K_CSV", os.path.join(_genmol_root, "data", "zinc250k.csv"))


def _hit_vocab_dir() -> str:
    return os.environ.get(
        "GENMOL_HIT_VOCAB_DIR",
        os.path.join(_genmol_root, "scripts", "exps", "hit", "vocab"),
    )


def get_vocab_from_zinc250k(mol_df=None, size=None):
    if mol_df is not None:
        df = mol_df
    else:
        df = pd.read_csv(_zinc250k_csv())
    # construct vocabulary
    frags = []
    for i in trange(len(df)):
        frags.extend(cut(df['smiles'].iloc[i]))
    # Drop duplications
    frags = list(set(frags))

    foldername = _hit_vocab_dir()
    if not os.path.exists(foldername):
        os.mkdir(foldername)
    
    df = pd.DataFrame(frags, columns=['frag'])
    df['size'] = df['frag'].apply(lambda frag: Chem.MolFromSmiles(frag).GetNumAtoms())
    if size is not None:
        df = df.sample(n=size, random_state=42)
    df.to_csv(os.path.join(foldername, 'frags.csv'), index=False)

    return df

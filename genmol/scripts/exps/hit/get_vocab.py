import os
import sys
from collections import defaultdict
from tqdm import trange
import pandas as pd

from genmol.utils.utils_chem import cut
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

def get_vocab_from_zinc250k(mol_df=None, size=None):
    if mol_df is not None:
        df = mol_df
    else:
        df = pd.read_csv(f"{os.environ['PROJECT_ROOT']}/GenMol/data/zinc250k.csv")
    # construct vocabulary
    frags = []
    for i in trange(len(df)):
        frags.extend(cut(df['smiles'].iloc[i]))
    # Drop duplications
    frags = list(set(frags))

    foldername = f'{os.environ["PROJECT_ROOT"]}/GenMol/scripts/exps/hit/vocab'
    if not os.path.exists(foldername):
        os.mkdir(foldername)
    
    df = pd.DataFrame(frags, columns=['frag'])
    df['size'] = df['frag'].apply(lambda frag: Chem.MolFromSmiles(frag).GetNumAtoms())
    if size is not None:
        df = df.sample(n=size, random_state=42)
    df.to_csv(os.path.join(foldername, 'frags.csv'), index=False)

    return df

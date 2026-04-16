"""
DEPRECATED: algorithms should import benchmark computers from `benchmark.computers`.

This module remains as a thin compatibility shim.
"""

from benchmark.computers.property_computers import *  # noqa: F403


def select_prop_computer(computer_name, vina_url=None):
    name_to_computer = {
        "QED": compute_qed,
        "TPSA": compute_tpsa,
        "SAS": compute_sas,
        "CLOGP": compute_clogp,
        "WEIGHT": compute_weight,
        "FORMULA": compute_formula,
        "NUMAROMATICRINGS": compute_num_aromatic_rings,
        "RINGCOUNT": compute_num_rings,

        # Toxometris computers
        "SOLUBILITY": partial(compute_toxometris_score, assay="solubility"),
        "SOLUBILITY_REL": partial(compute_toxometris_score, assay="solubility", reliability=True),
        "TOXICITY": partial(compute_toxometris_score, assay="ames"),
        "TOXICITY_REL": partial(compute_toxometris_score, assay="ames", reliability=True),

        # Binding predictors
        "JNK3": partial(predict_binding_score, protein="jnk3"),
        "DRD2": partial(predict_binding_score, protein="drd2"),
        "GSK3B": partial(predict_binding_score, protein="gsk3b"),
    }
    if computer_name in name_to_computer:
        return name_to_computer[computer_name]
    
    arg = ".".join(computer_name.split(".")[1:])
    computer_name = computer_name.split(".")[0]
    if computer_name == "SIMILAR":
        mol = Chem.MolFromSmiles(arg)
        return partial(compute_similarity, rdkit_mol1=mol)
    elif computer_name == "DOCKING":
        target = arg
        return partial(compute_quickvina_docking_score, target=target, vina_url=vina_url)
    
    raise ValueError(f"Oracle with name {computer_name} does not exist.")


def rel_err(measured, real):
    return np.abs(measured - real) / real


def compute_qed_sas_docking(rdkit_mols, target: str, vina_url=None):
    computer_names = ["QED", "SAS", f"DOCKING.{target}"]
    return dynamic_computer(rdkit_mols, computer_names, vina_url=vina_url)


def compute_toxometris_score(rdkit_mols, assay, reliability=False):
    smiles_list = [Chem.MolToSmiles(rdkit_mol) for rdkit_mol in rdkit_mols]
    url = "https://stage.toxometris.ai/v1/gentox/predict_assay_api"
    headers = {
        "Authorization": f"Bearer {os.environ['TOXOMETRIS_API_KEY']}",
        "Content-Type": "application/json"
    }
    payload = {
        "smiles": smiles_list,
        "assay": assay
    }
    response = requests.post(url, headers=headers, json=payload).json()
    if response["status"] != "success":
        raise ValueError(response["message"])
    
    if reliability:
        rel_scores = []
        for score in response["data"]:
            value = score["reliability"] if score["validity"] else 0.5
            rel_scores.append(value)
        return np.array(rel_scores)
    else:
        assay_scores = []
        for score in response["data"]:
            invalid_value = {
                "solubility": -10,
                "ames": 0.0,
            }[assay]
            value = score["value"] if score["validity"] else invalid_value
            assay_scores.append(value)
        return np.array(assay_scores)


def geam_docking_oracle(rdkit_mols, target: str, verb: bool=True, vina_url=None):
    docking_scores = compute_quickvina_docking_score(rdkit_mols, target, verb=verb, vina_url=vina_url)
    qed_scores = compute_qed(rdkit_mols, verb=verb)
    sa_scores = compute_sas(rdkit_mols)

    # Formula used in GEAM paper
    trans_sa_scores = (10 - sa_scores) / 9
    aggregated_scores = (np.clip(docking_scores, 0, 20) / 20) * qed_scores * trans_sa_scores
    return aggregated_scores, docking_scores, qed_scores, sa_scores

# def compute_quickvina_docking_score(rdkit_mols, target: str):
#     predictor = quickvina_predictor(target)
#     smiles_list = [Chem.MolToSmiles(rdkit_mol) for rdkit_mol in rdkit_mols]
#     scores = -np.array(predictor.predict(smiles_list))
#     return np.clip(scores, 0, None)
def compute_quickvina_docking_score(rdkit_mols, target: str, verb=True, vina_url=None):
    """
    Compute docking scores using either DockingVina service or local quickvina predictor.
    Skips invalid molecules (None) to save oracle budget.
    
    Args:
        rdkit_mols: List of RDKit molecule objects (may contain None for invalid molecules)
        target: Target protein name (e.g., 'parp1', 'fa7', '5ht1b', 'braf', 'jak2')
        verb: Verbose flag (default: True)
        vina_url: Optional vina service URL (takes precedence over env var)
    
    Returns:
        Array of docking scores (clipped to >= 0). Invalid molecules get score 0.0.
    """
    # Filter out None values and track valid indices
    valid_indices = []
    valid_mols = []
    for i, rdkit_mol in enumerate(rdkit_mols):
        if rdkit_mol is not None:
            valid_indices.append(i)
            valid_mols.append(rdkit_mol)
    
    # Initialize result array with default score (0.0) for all molecules
    scores = np.zeros(len(rdkit_mols), dtype=np.float32)
    
    # If no valid molecules, return zeros
    if len(valid_mols) == 0:
        return scores
    
    # Convert valid molecules to SMILES
    smiles_list = [Chem.MolToSmiles(rdkit_mol) for rdkit_mol in valid_mols]
    
    # Check if vina/oracle service URL is configured (parameter > env). Prefer ORACLE_SERVICE_URL; VINA_SERVICE_URL is fallback.
    oracle_service_url = vina_url or os.environ.get("VINA_SERVICE_URL") or os.environ.get("ORACLE_SERVICE_URL")
    logging.getLogger(__name__).info("oracle_service_url: %s (from %s)", oracle_service_url, "request param" if vina_url else "env var")
    if oracle_service_url:
        # Use DockingVina service - only for valid molecules
        client = DockingVinaClient(oracle_service_url, target)
        valid_scores = client.predict(smiles_list)
        # Convert to numpy array and process like local predictor
        valid_scores = -np.array(valid_scores)  # Negative affinity to positive score
        valid_scores = np.clip(valid_scores, 0, None)
        # Map valid scores back to their original positions
        scores[valid_indices] = valid_scores
    else:
        # Use local quickvina predictor (original behavior)
        predictor = quickvina_predictor(target)
        valid_scores = -np.array(predictor.predict(smiles_list))
        valid_scores = np.clip(valid_scores, 0, None)
        # Map valid scores back to their original positions
        scores[valid_indices] = valid_scores
    
    return scores


def dynamic_computer(rdkit_mols, computer_names, verb=True, vina_url=None):
    scores_dict = {}
    for computer_n in computer_names:
        prop_computer = select_prop_computer(computer_n, vina_url=vina_url)
        p_scores = prop_computer(rdkit_mols, verb=verb)
        scores_dict[computer_n] = p_scores
    return scores_dict


def compute_qed(rdkit_mols, verb=True):
    qed_scores = []
    for rdkit_mol in rdkit_mols:
        try:
            qed_scores.append(qed(rdkit_mol))
        except Exception as e:
            if verb:
                print("Could not compute QED for", rdkit_mol, e)
            qed_scores.append(None)
    return np.array(qed_scores)


def compute_clogp(rdkit_mols, verb=True):
    logp_scores = []
    for rdkit_mol in rdkit_mols:
        try:
            logp_scores.append(Crippen.MolLogP(rdkit_mol))
        except Exception as e:
            if verb:
                print("Could not compute CLOGP for", rdkit_mol, e)
            logp_scores.append(None)
    return np.array(logp_scores)


def compute_tpsa(rdkit_mols, verb=True):
    tpsa = []
    for rdkit_mol in rdkit_mols:
        try:
            tpsa.append(rdMolDescriptors.CalcTPSA(rdkit_mol))
        except Exception as e:
            if verb:
                print("Could not compute TPSA for", rdkit_mol, e)
            tpsa.append(None)
    return np.array(tpsa)
        
        
def compute_weight(rdkit_mols, verb=True):
    weights = []
    for rdkit_mol in rdkit_mols:
        try:
            weights.append(rdMolDescriptors.CalcExactMolWt(rdkit_mol))
        except Exception as e:
            if verb:
                print("Could not compute weight for", rdkit_mol, e)
            weights.append(None)
    return np.array(weights)


def compute_num_aromatic_rings(rdkit_mols, verb=True):
    num_arom_rings = []
    for rdkit_mol in rdkit_mols:
        try:
            num_arom_rings.append(rdMolDescriptors.CalcNumAromaticRings(rdkit_mol))
        except Exception as e:
            if verb:
                print("Could not compute weight for", rdkit_mol, e)
            num_arom_rings.append(None)
    return np.array(num_arom_rings)
    # return np.vectorize(rdMolDescriptors.CalcNumAromaticRings)(rdkit_mols)


def compute_num_rings(rdkit_mols, verb=True):
    num_rings = []
    for rdkit_mol in rdkit_mols:
        try:
            num_rings.append(rdMolDescriptors.CalcNumRings(rdkit_mol))
        except Exception as e:
            if verb:
                print("Could not compute weight for", rdkit_mol, e)
            num_rings.append(None)
    return np.array(num_rings)
    # return np.vectorize(rdMolDescriptors.CalcNumRings)(rdkit_mols)


def compute_formula(rdkit_mols, verb=True):
    formulas = []
    for rdkit_mol in rdkit_mols:
        try:
            formulas.append(rdMolDescriptors.CalcMolFormula(rdkit_mol))
        except Exception as e:
            if verb:
                print("Could not compute formula for", rdkit_mol, e)
            formulas.append(None)
    return np.array(formulas)


def compute_sas(rdkit_mols, verb=True):
    sa_scores = []
    for rdkit_mol in rdkit_mols:
        try:
            sa_scores.append(sascorer.calculateScore(rdkit_mol))
        except Exception as e:
            if verb:
                print("Could not compute SA score for", rdkit_mol, e)
            sa_scores.append(None)
    return np.array(sa_scores)


def compute_fing(rdkit_mols, verb=True):
    fings = []
    for rdkit_mol in rdkit_mols:
        try:
            fings.append(AllChem.GetMorganFingerprintAsBitVect(rdkit_mol, 2, nBits=2048))
        except Exception as e:
            if verb:
                print("Could not compute fingerprint for", rdkit_mol, e)
            fings.append(None)
    return fings


def compute_similarity(rdkit_mols, rdkit_mol1, verb=True):
    print(f"Computing similarity between {len(rdkit_mols)} generated molecules and {rdkit_mol1} seed molecule.")
    fings = compute_fing(rdkit_mols)
    fing1 = compute_fing([rdkit_mol1])[0]
    return np.array([DataStructs.TanimotoSimilarity(f, fing1) for f in fings])


def compute_similarity_fing(fings, fing1):
    return np.array(BulkTanimotoSimilarity(fing1, fings))


def predict_binding_score(rdkit_mols, protein, verb=True):
    """Compute binding scores using subprocess to call binding_predictors.py in bind_oracles env."""

    smiles_list = [Chem.MolToSmiles(rdkit_mol) for rdkit_mol in rdkit_mols]
    try:
        # Create the command to run binding_predictors.py in the bind_oracles environment
        predictor_script = resolve_from_project_root(
            "genetic_chemalactica", "oracles", "binding_predictors.py"
        )
        cmd = [
            "conda", "run", "-n", "bind_oracles", "python", 
            str(predictor_script),
            "--protein", protein
        ] + ["--smiles_lst"] + smiles_list
        
        # Run the command and capture output
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=str(resolve_from_project_root()),
            check=True
        )
        
        # Parse the JSON output
        scores = json.loads(result.stdout.strip())
        return np.array(scores)
        
    except subprocess.CalledProcessError as e:
        print(f"Error running binding predictors for {protein}: {e}")
        print(f"stderr: {e.stderr}")
        return np.array([None] * len(smiles_list))
    except json.JSONDecodeError as e:
        print(f"Error parsing binding predictor output for {protein}: {e}")
        return np.array([None] * len(smiles_list))
    except Exception as e:
        print(f"Unexpected error in binding score computation for {protein}: {e}")
        return np.array([None] * len(smiles_list))
from flask import Flask, request, jsonify
from rdkit import Chem
import argparse
import os
import socket
import time

import numpy as np
from benchmark.computers.property_computers import (
    dynamic_computer,
    compute_qed_sas_docking,
    geam_docking_oracle,
)


app = Flask(__name__)


@app.route('/geam', methods=['POST'])
def send_geam():
    data = request.json  # Assuming you are sending JSON data in the request body
    if not isinstance(data, dict):
        return jsonify({'error': 'Expected a dict'}), 400
    
    rdkit_mols = [Chem.MolFromSmiles(s) for s in data["mols"]]
    target = data["target"]
    vina_url = data.get('vina_url')  # Extract vina_url from request if provided
    scores, docking_scores, qed_scores, sa_scores = geam_docking_oracle(rdkit_mols, target, vina_url=vina_url)
    return jsonify({
        "scores": scores.tolist(),
        "docking_scores": docking_scores.tolist(),
        "sa_scores": sa_scores.tolist(),
        "qed_scores": qed_scores.tolist()
    }), 200


@app.route(f"/sas_qed_docking", methods=['POST'])
def send_sas_qed_docking():
    data = request.json  # Assuming you are sending JSON data in the request body
    if not isinstance(data, dict):
        return jsonify({'error': 'Expected a dict'}), 400 
    
    target = data["target"]
    smiles = data["mols"]
    vina_url = data.get('vina_url')  # Extract vina_url from request if provided

    rdkit_mols = np.vectorize(Chem.MolFromSmiles)(smiles)
    scores_dict = compute_qed_sas_docking(rdkit_mols, target, vina_url=vina_url)

    scores_dict = {k: v.tolist() for k, v in scores_dict.items()}
    return jsonify(scores_dict), 200


@app.route(f"/dynamic", methods=['POST'])
def send_dynamic():
    data = request.json  # Assuming you are sending JSON data in the request body
    if not isinstance(data, dict):
        return jsonify({'error': 'Expected a dict'}), 400 
    
    computer_names = data['computer_names']
    smiles = data['mols']
    vina_url = data.get('vina_url')  # Extract vina_url from request if provided
    rdkit_mols = np.vectorize(Chem.MolFromSmiles)(smiles)
    scores_dict = dynamic_computer(rdkit_mols, computer_names, vina_url=vina_url)

    scores_dict = {k: v.tolist() for k, v in scores_dict.items()}
    return jsonify(scores_dict), 200


@app.route(f"/dynamic_max", methods=['POST'])
def send_dynamic_max():
    data = request.json  # Assuming you are sending JSON data in the request body
    if not isinstance(data, dict):
        return jsonify({'error': 'Expected a dict'}), 400 
    
    computer_names = data['computer_names']
    smiles = data['mols']
    num_eval = data["num_eval"]
    vina_url = data.get('vina_url')  # Extract vina_url from request if provided
    rdkit_mols = np.vectorize(Chem.MolFromSmiles)(smiles)
    scores_dict_list = []
    for _ in range(num_eval):
        scores_dict_list.append(dynamic_computer(rdkit_mols, computer_names, vina_url=vina_url))

    scores_dict = {}
    for key in computer_names:
        values = np.array([scores_dict_list[i][key] for i in range(num_eval)])
        scores_dict[key] = values.max(axis=0)

    scores_dict = {k: v.tolist() for k, v in scores_dict.items()}
    return jsonify(scores_dict), 200


@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'hello'}), 200


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=False, default=5454)
    parser.add_argument("--num_processes", type=int, required=False, default=128)
    parser.add_argument(
        "--port_max_tries",
        type=int,
        required=False,
        default=5,
        help="If port is in use, retry with port+1 up to this many times.",
    )
    parser.add_argument(
        "--port_file",
        type=str,
        required=False,
        default=None,
        help="If set, write the bound port to this file.",
    )
    args = parser.parse_args()
    
    base_port = int(args.port)
    max_tries = max(1, int(args.port_max_tries))

    chosen_port = None
    last_err = None
    for i in range(max_tries):
        port = base_port + i
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", port))
            chosen_port = port
            break
        except OSError as e:
            last_err = e
            time.sleep(0.2)
        finally:
            try:
                s.close()
            except Exception:
                pass

    if chosen_port is None:
        raise OSError(
            f"Could not bind any port in range [{base_port}, {base_port + max_tries - 1}]"
        ) from last_err

    if args.port_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.port_file)), exist_ok=True)
        with open(args.port_file, "w") as f:
            f.write(str(chosen_port))

    print(f"ORACLES_APP_PORT: {chosen_port}", flush=True)
    app.run(
        debug=False,
        port=chosen_port,
        host="0.0.0.0",
        processes=args.num_processes,
        threaded=False,
    )
        
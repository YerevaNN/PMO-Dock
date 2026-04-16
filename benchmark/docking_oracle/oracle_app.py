#!/usr/bin/env python3

"""
Flask service for DockingOracle predictions.

Moved under `benchmark/docking_oracle/` so the benchmark owns the oracle.
"""

import traceback
import resource
from flask import Flask, request, jsonify

from benchmark.docking_oracle.docking import DockingOracle


try:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    print(f"Current file descriptor limits: soft={soft}, hard={hard}")
    target_limit = min(65536, hard) if hard > 0 else 65536
    resource.setrlimit(resource.RLIMIT_NOFILE, (target_limit, hard))
    new_soft, new_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    print(f"File descriptor limit set to: soft={new_soft}, hard={new_hard}")
except (ValueError, OSError) as e:
    print(f"Warning: Could not increase file descriptor limit: {e}")
except Exception as e:
    print(f"Warning: Unexpected error setting file descriptor limit: {e}")


app = Flask(__name__)
oracles = {}
TARGETS = ["parp1", "jak2", "braf", "fa7", "5ht1b", "6nzp", "7uyt", "5ut5", "7uyw"]


def initialize_oracles(exhaustiveness=None):
    global oracles
    if exhaustiveness is not None:
        print(f"Initializing oracles with exhaustiveness={exhaustiveness}")
    for target in TARGETS:
        try:
            oracles[target] = DockingOracle(target, exhaustiveness)
            print(f"Initialized {target} oracle")
        except Exception as e:
            print(f"Error initializing {target}: {e}")
            traceback.print_exc()
    print(f"Oracle service ready: {len(oracles)} targets initialized")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "initialized_targets": list(oracles.keys())})


@app.route("/predict/<target>", methods=["POST"])
def predict(target):
    if target not in oracles:
        return (
            jsonify(
                {"error": f"Target {target} not available. Available targets: {list(oracles.keys())}"}
            ),
            400,
        )

    try:
        data = request.get_json()
        if not data or "smiles" not in data:
            return jsonify({'error': 'Missing "smiles" key in request body'}), 400

        smiles_list = data["smiles"]
        if not isinstance(smiles_list, list):
            return jsonify({"error": '"smiles" must be a list'}), 400

        seed = data.get("seed", 42)
        oracle = oracles[target]
        print(f"[oracle_app] Received batch for target {target}: {len(smiles_list)} molecules, seed={seed}")
        scores = oracle.predict(smiles_list, seed)
        print(f"[oracle_app] Completed batch for target {target}: {len(smiles_list)} molecules scored")
        return jsonify({"target": target, "scores": scores, "count": len(scores)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500


@app.route("/targets", methods=["GET"])
def list_targets():
    return jsonify({"targets": list(oracles.keys())})


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="DockingOracle Flask Service")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--exhaustiveness", type=int, default=None)
    args = parser.parse_args()
    initialize_oracles(exhaustiveness=args.exhaustiveness)
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()


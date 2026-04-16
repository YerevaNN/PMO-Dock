from __future__ import annotations

import json
from argparse import ArgumentParser

from tdc import Oracle


def predict_binding_score(smiles_lst: list[str], protein: str) -> list[float]:
    oracle = Oracle(protein)
    return list(map(oracle, smiles_lst))


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--smiles_lst", type=str, nargs="+", required=True)
    parser.add_argument("--protein", type=str, required=True)

    args = parser.parse_args()

    if args.protein not in ["jnk3", "drd2", "gsk3b"]:
        raise ValueError(f"Invalid protein {args.protein}")
    print(json.dumps(predict_binding_score(args.smiles_lst, args.protein)))


if __name__ == "__main__":
    main()
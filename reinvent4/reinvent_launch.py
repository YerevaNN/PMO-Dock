#!/usr/bin/env python
"""Launch REINVENT4 with a seed that actually takes effect.

REINVENT4 v4.5.11 cannot be seeded through its own CLI. ``reinvent/Reinvent.py`` does::

    seed = input_config.get("seed", None)
    if args.seed is not None:
        set_seed(seed)

which reads the seed from the *config* but gates it on the *command line* flag. Both routes
are dead ends:

* ``ReinventConfig`` sets ``extra="forbid"`` and declares no ``seed`` field, so a config
  carrying ``seed`` is rejected at validation, before that line is ever reached;
* with no config seed, ``input_config.get("seed")`` is ``None``, and ``set_seed(None)``
  returns immediately -- so ``--seed N`` is accepted, advertised as "Sets the random seeds
  for reproducibility", and silently does nothing.

Upstream's own fix would be ``set_seed(args.seed)``. We do not patch upstream, so this
launcher applies the seed itself and then hands over to REINVENT4's normal entry point.
Without it every REINVENT4 run in the benchmark would be unseeded, and the seed-to-seed
spread the paper reports for the other four methods would be, for this method, noise.

Usage (what ``runner_common.run_one`` invokes)::

    python -m reinvent4.reinvent_launch --seed 0 /path/to/config.json
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("config", help="REINVENT4 config file")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    # Imported here, not at module scope, so --help works without a REINVENT4 install.
    from reinvent.Reinvent import main_script
    from reinvent.utils.helpers import set_seed

    set_seed(args.seed)

    # REINVENT4 parses its own command line; give it one it understands. --seed is left
    # out deliberately: upstream would only re-run the no-op above.
    sys.argv = ["reinvent", args.config]
    main_script()


if __name__ == "__main__":
    main()

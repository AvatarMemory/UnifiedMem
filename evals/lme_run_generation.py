"""Compatibility wrapper around evals.run_generation.

This file is kept so legacy commands keep working after consolidating
LongMemEval generation into `evals/run_generation.py`.
"""

import os
import sys


if __package__ is None and __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from evals.run_generation import check_args, main, parse_args


if __name__ == "__main__":
    args = parse_args()
    check_args(args)
    main(args)

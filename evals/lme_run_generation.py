"""Compatibility wrapper around evals.run_generation.

This file is kept so legacy commands keep working after consolidating
LongMemEval generation into `evals/run_generation.py`.
"""

from evals.run_generation import check_args, main, parse_args


if __name__ == "__main__":
    args = parse_args()
    check_args(args)
    main(args)

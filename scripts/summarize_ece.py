import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lnl_foundation.results import build_ece_reports


def main():
    parser = argparse.ArgumentParser(description="Build standalone Stage-2 ECE reports.")
    parser.add_argument("--outputs_root", default="outputs")
    args = parser.parse_args()
    result = build_ece_reports(args.outputs_root)
    print(f"ECE runs: {result['runs']}")
    print(f"ECE summary: {result['summary']}")
    print(f"ECE table: {result['table']}")
    print(f"Rows={result['n_runs']}, groups={result['n_groups']}")


if __name__ == "__main__":
    main()

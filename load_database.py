from pickle import load
from symreg.dataset import load
from lzma import open as lzma_open
import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a dataset of symbolic regression formulas.",
        allow_abbrev=False,
    )
    parser.add_argument("input", type=str, help="input file")
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        default=None,
        help="maximum number of formulas to load",
    )
    parser.add_argument(
        "-o",
        "--offset",
        type=int,
        default=0,
        help="offset to start loading formulas from",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    start = time.time()
    formulas = load(args.input, args.count, args.offset)
    end = time.time()

    print(f"Loaded {len(formulas)} formulas in {end - start:.2f} seconds.")


if __name__ == "__main__":
    main()

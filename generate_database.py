from symreg.random import *
from concurrent.futures import ProcessPoolExecutor
import tqdm
import lzma
import pickle
import time
import argparse

np.seterr(all="ignore")

options = RandomOptions()


def generate_bucket(k: int):
    """Generate a bucket (of size k) of random formulas."""
    formulas = []
    for _ in range(k):
        formulas.append(random_formula(options))
    return formulas


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def main():
    parser = argparse.ArgumentParser(
        description="Generate a database of random formulas."
    )
    parser.add_argument(
        "-o", "--output", type=str, default="formulas.pkl.xz", help="output file"
    )

    generation_group = parser.add_argument_group("Generation options")
    generation_group.add_argument(
        "-c", "--count", type=int, default=10000, help="Count of formulas to generate."
    )
    generation_group.add_argument(
        "-j", "--jobs", type=int, default=None, help="Maximum number of jobs/workers."
    )
    generation_group.add_argument(
        "-b", "--bucket", type=int, default=1000, help="Size of bucket."
    )

    formula_group = parser.add_argument_group("Formula options")
    formula_group.add_argument(
        "--seed", type=int, default=None, help="random seed to use."
    )
    formula_group.add_argument(
        "--max-depth", type=int, default=5, help="Maximum depth of the formula."
    )
    formula_group.add_argument(
        "--max-arity", type=int, default=2, help="Maximum arity of the formula."
    )

    args = parser.parse_args()

    bucket_count = args.count // args.bucket

    if args.seed is not None:
        np.random.seed(args.seed)

    # Generate formulas
    print("Generating formulas...")
    progress_bar = tqdm.tqdm(total=args.count, leave=False)
    start = time.time()
    executor = ProcessPoolExecutor(max_workers=args.jobs)
    formulas = set()
    # Generate formulas in parallel (each bucket is generated in a separate process)
    for batch in executor.map(generate_bucket, [args.bucket] * bucket_count):
        formulas.update(batch)
        progress_bar.update(len(batch))
    progress_bar.close()
    end = time.time()
    print(f"Generated {len(formulas)} formulas. Time: {end - start:.2f} seconds.")

    # Save formulas to disk using lzma compression and pickle serialization
    print("Saving formulas...")
    progress_bar = tqdm.tqdm(total=args.count, leave=False)
    start = time.time()
    with lzma.open(args.output, "wb") as f:
        # Dump general metadata for the dataset
        pickle.dump({
            "bucket_size": args.bucket,
            "count": args.count,
        }, f, pickle.HIGHEST_PROTOCOL)

        # Dumps formulas in chunks (so we don't have to load the whole dataset at once)
        for chunk in chunks(list(formulas), args.bucket):
            pickle.dump(chunk, f, pickle.HIGHEST_PROTOCOL)
            progress_bar.update(len(chunk))
    progress_bar.close()
    end = time.time()
    print(f"Formulas saved to {args.output}. Time: {end - start:.2f} seconds.")


if __name__ == "__main__":
    main()

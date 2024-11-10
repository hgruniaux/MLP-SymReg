from symreg.random import *
from concurrent.futures import ProcessPoolExecutor
import tqdm
import pickle
import time
import argparse

np.seterr(all="ignore")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a dataset of symbolic regression formulas.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "-o", "--output", type=str, default="formulas.pkl.gz", help="output file"
    )

    generation_group = parser.add_argument_group("Generation options")
    generation_group.add_argument(
        "-c", "--count", type=int, default=10000, help="count of formulas to generate"
    )
    generation_group.add_argument(
        "-j", "--jobs", type=int, default=None, help="maximum number of jobs/workers"
    )
    generation_group.add_argument(
        "-b", "--bucket", type=int, default=1000, help="size of bucket"
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

    return parser.parse_args()


options = RandomOptions()
# Some general options
options.simplify = True
options.must_have_variable = True
options.definition_set = None
options.add_affine_functions = True


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


def generate_formulas(jobs: int, count: int, bucket_size: int):
    print("Generating formulas...")
    progress_bar = tqdm.tqdm(total=count, leave=False)
    start = time.time()
    executor = ProcessPoolExecutor(max_workers=jobs)
    formulas = set()

    # Generate formulas in parallel (each bucket is generated in a separate process)
    bucket_count = count // bucket_size
    for batch in executor.map(generate_bucket, [bucket_size] * bucket_count):
        formulas.update(batch)
        progress_bar.update(len(batch))
    progress_bar.close()
    end = time.time()
    print(f"Generated {len(formulas)} formulas. Time: {end - start:.2f} seconds.")

    return list(formulas)


def save_formulas(output_filename: str, formulas: list, bucket_size: int):
    print("Saving formulas...")
    progress_bar = tqdm.tqdm(total=len(formulas), leave=False)
    start = time.time()

    if output_filename.endswith(".xz"):
        from lzma import open as lzma_open

        open_func = lzma_open
    elif output_filename.endswith(".gz"):
        from gzip import open as gzip_open

        open_func = gzip_open
    elif output_filename.endswith(".bz2"):
        from bz2 import open as bz2_open

        open_func = bz2_open
    else:
        open_func = open

    with open_func(output_filename, "wb") as f:
        # Dumps formulas in chunks (so we don't have to load the whole dataset at once)
        for chunk in chunks(formulas, bucket_size):
            pickle.dump(chunk, f, pickle.HIGHEST_PROTOCOL)
            progress_bar.update(len(chunk))
    progress_bar.close()
    end = time.time()
    print(f"Formulas saved to {output_filename}. Time: {end - start:.2f} seconds.")


def main():
    args = parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    options.max_depth = args.max_depth
    options.allowed_variables = [k for k in range(args.max_arity)]

    # Generate formulas
    formulas = generate_formulas(args.jobs, args.count, args.bucket)

    # Save formulas to disk using lzma compression and pickle serialization
    save_formulas(args.output, formulas, args.bucket)


if __name__ == "__main__":
    main()

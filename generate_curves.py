from symreg.formula import *
from symreg.random import *
from concurrent.futures import ProcessPoolExecutor
from gzip import open as gzip_open
import numpy as np
import tqdm.auto as tqdm
import pickle

np.seterr(all="ignore")

MAX_DEPTH = 5
MAX_RANGE = 10
MIN_SAMPLES = 10
MAX_SAMPLES = 100
NOISE_MEAN = 0
NOISE_STD = 0.5

CHUNK_COUNT = 1000
CHUNK_SIZE = 1000

def generate_one_formula():
    while True:
        x_range  = np.random.random(2) * MAX_RANGE - MAX_RANGE / 2
        samples = np.random.randint(MIN_SAMPLES, MAX_SAMPLES)
        x = np.linspace(x_range[0], x_range[1], samples)

        random_options = RandomOptions()
        random_options.definition_set = x
        random_options.max_depth = MAX_DEPTH

        formula = random_formula(random_options)

        y = formula(x)
        if not np.isfinite(y).all():
            continue

        y += np.random.normal(NOISE_MEAN, NOISE_STD, np.shape(y))

        return x, y, formula

def generate_one_chunk(chunk_size: int):
    formulas = []
    for _ in range(chunk_size):
        formulas.append(generate_one_formula())
    return formulas

def generate(output):
    executor = ProcessPoolExecutor()
    progress_bar = tqdm.tqdm(total=CHUNK_COUNT * CHUNK_SIZE, unit=" formulas", leave=False)
    for i, chunk in enumerate(executor.map(generate_one_chunk, [CHUNK_SIZE] * CHUNK_COUNT)):
        pickle.dump(chunk, output, pickle.HIGHEST_PROTOCOL)
        progress_bar.update(CHUNK_SIZE)
    progress_bar.close()

if __name__ == "__main__":
    print("Generating formulas...")
    with gzip_open("data.pkl.gz", "wb") as output:
        generate(output)
    print(f"Done. Generated {CHUNK_COUNT * CHUNK_SIZE} formulas.")

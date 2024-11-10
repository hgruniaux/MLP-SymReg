from pickle import load
from symreg.formula import *
from lzma import open as lzma_open
import time

MAX_LOAD_COUNT = None

running_count = 0
start = time.time()
formulas = []
with lzma_open("formulas.pkl.xz", "rb") as f:
    metadata = load(f)
    bucket_size = metadata["bucket_size"]
    count = metadata["count"]

    while True:
        try:
            formulas_bucket = load(f)
            assert(len(formulas_bucket) == bucket_size) # ensure file is correctly formatted
            formulas += formulas_bucket

            running_count += len(formulas_bucket)
            if MAX_LOAD_COUNT is not None and running_count > MAX_LOAD_COUNT:
                formulas = formulas[:MAX_LOAD_COUNT]
                running_count = MAX_LOAD_COUNT
                break
        except EOFError:
            break
end = time.time()
print(f"Loaded {running_count} formulas in {end - start:.2f} seconds.")

print(len(formulas))

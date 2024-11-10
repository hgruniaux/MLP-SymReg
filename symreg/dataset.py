import pickle
from symreg.formula import *
from typing import List


def load(filename: str, max_count: int | None = None) -> List[Formula]:
    """
    Load formulas from a file that was generated by `generate_database.py`.
    Such file must have the extension `.pkl.xz` or `.pkl`.

    Parameters
    ----------
    filename : str
        The filename to load from.
    max_count : int, optional
        The maximum number of formulas to load. If None, all formulas are loaded.

    Returns
    -------
    List[Formula]
        The formulas loaded from the file.
    """

    formulas = []

    if filename.endswith(".xz"):
        from lzma import open as lzma_open
        open_func = lzma_open
    elif filename.endswith(".gz"):
        from gzip import open as gzip_open
        open_func = gzip_open
    elif filename.endswith(".bz2"):
        from bz2 import open as bz2_open
        open_func = bz2_open
    else:
        open_func = open

    with open_func(filename, "rb") as f:
        metadata = pickle.load(f)
        bucket_size = metadata["bucket_size"]
        count = metadata["count"]

        while True:
            try:
                formulas_bucket = pickle.load(f)
                if len(formulas_bucket) != bucket_size:
                    raise ValueError("File is incorrectly formatted")
                formulas += formulas_bucket

                if max_count is not None and len(formulas) >= max_count:
                    formulas = formulas[:max_count]
                    break
            except EOFError:
                break

        if len(formulas) > count:
            raise ValueError("File is incorrectly formatted")

    return formulas

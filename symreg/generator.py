from symreg.formula import *

class Generator:
  """
  Abstract base class for all generators.
  """

  def generate(self) -> Formula:
    """
    Generates a new formula.
    """
    raise NotImplementedError

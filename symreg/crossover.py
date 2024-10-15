from typing import List
from symreg.formula import *

class Crossover:
  """
  Abstract base class for all crossover algorithms.
  """

  def crossover(self, candidates: List[Formula]) -> Formula:
    """
    Generates a new formula by recombining parts from candidates.
    """
    raise NotImplementedError

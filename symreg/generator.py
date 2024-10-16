from symreg.formula import *
from symreg.random import *

class Generator:
    """
    Abstract base class for all generators.
    """

    def generate(self) -> Formula:
        """
        Generates a new formula.
        """
        raise NotImplementedError

class RandomGenerator(Generator):
    """
    Generates random formulas using the symreg.random module.
    """

    def __init__(self, options: RandomOptions = RandomOptions()):
        super().__init__()
        self.options = options

    def generate(self):
        return random_formula(self.options)

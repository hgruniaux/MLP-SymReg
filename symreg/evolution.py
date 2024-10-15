from dataclasses import dataclass
from symreg.formula import *
from symreg.crossover import *
from symreg.mutator import *
from symreg.generator import *
import random


@dataclass
class Options:
    """
    Options for the evolutionary algorithm.
    """

    min_depth: int = 1
    """The minimum depth for generating mathematical expressions."""

    max_depth: int = 3
    """The maximum depth for generating mathematical expressions."""

    population_size: int = 500
    """Size of a population at each generation.
    Too large population may slow down considerably the system."""

    k_best: int = 5
    """The count of best candidates to be considered to generate the new generation."""

    iterations: int = 10
    """Count of iterations (generations) to simulate."""

    show_every: int = 1
    """Show information and the best canditate at every requested generation.
    Set to None to hide intermediate information."""

    generator: Generator = None

    crossover_rate: float = 0.05
    """Probability to generate a formula in the new generation using cross-over
    (picking random parts from two formula of the previous generation)."""

    crossover: Crossover = None

    mutation_rate: float = 0.95
    """Probability to generate a formula in the new generation using mutation
    from a formula of the previous generation."""

    mutator: Mutator = None


def fitness(formula: Formula, x, y) -> float:
    """Computes the mean squared error (MSE)."""
    predicted_y = formula(x)
    return np.mean((y - predicted_y) ** 2)


def run(x, y, options: Options):
    # Generate initial population
    generation = [options.generator.generate() for _ in range(options.population_size)]

    crossover_count = int(options.crossover_rate * options.population_size)
    mutation_count = int(options.mutation_rate * options.population_size)
    random_count = options.population_size - crossover_count - mutation_count

    for i in range(options.iterations):
        fitness_values = [fitness(f, x, y) for f in generation]
        best_candidates_indices = np.argpartition(fitness_values, options.k_best)
        best_candidates: List[Formula] = generation[best_candidates_indices]

        generation = []

        # Generate formulas by recombining random parts of the previous best candidates (cross-over)
        for _ in range(crossover_count):
            formula = options.crossover.crossover(best_candidates)
            generation.append(formula)

        # Generate formulas by applying random mutations to the previous best candidates (mutations)
        for _ in range(mutation_count):
            formula = random.choice(best_candidates)
            options.mutator.mutate(formula)
            generation.append(formula)

        # Generate completely new random formulas
        for _ in range(random_count):
            formula = options.generator.generate()
            generation.append(formula)

        if i % options.show_every == 0:
            print(f"Iteration {i}:")
            for j, candidate in enumerate(best_candidates):
                print(f"    Best candidate {j}:")
                print(f"        Formula: {candidate}")
                print(f"        Fitness: {fitness(candidate, x, y)}")

    # Returns the k best candidates
    fitness_values = [fitness(f, x, y) for f in generation]
    best_candidates_indices = np.argpartition(fitness_values, options.k_best)
    best_candidates = generation[best_candidates_indices]
    return best_candidates

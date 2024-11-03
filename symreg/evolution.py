from dataclasses import dataclass
from symreg.formula import *
from symreg.crossover import *
from symreg.mutator import *
from symreg.generator import *
from symreg.complexity import *
import random
from copy import deepcopy

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

    loss_target: float = 0.0
    """When the loss reach this target, the algorithm stops."""

    iterations: int = 10
    """Count of iterations (generations) to simulate."""

    lambda_regularization: float = 0.1
    """Regularization parameter for the fitness function (coefficient of complexity impact on the loss)."""

    verbose: bool = True
    """Show the best candidate at every show_every step."""

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

    updater = None
    """Updater function to be called at each generation with the best candidates."""


def fitness(formula: Formula, x, y, lambda_coef: float) -> float:
    predicted_y = formula(x)
    predicted_y = np.nan_to_num(predicted_y, nan=np.inf)
    return np.mean((y - predicted_y) ** 2) / np.mean(y ** 2) * (1 + lambda_coef * np.log(1 + depth(formula)))


def run(x, y, options: Options):
    if options.generator is None:
        raise RuntimeError("A generator object must be provided. See the symreg.generator module.")
    if options.mutator is None:
        raise RuntimeError("A mutator object must be provided. See the symreg.mutator module.")

    # Generate initial population
    generation = [options.generator.generate() for _ in range(options.population_size)]

    crossover_count = int(options.crossover_rate * options.population_size)
    mutation_count = int(options.mutation_rate * options.population_size)

    try:
        for i in range(options.iterations):
            # Remove duplicates in the generation
            generation = list(set(generation))

            fitness_values = [fitness(f, x, y, options.lambda_regularization) for f in generation]
            best_candidates_indices = np.argpartition(fitness_values, [ 0, options.k_best ])[:options.k_best]
            best_candidates: List[Formula] = [ generation[i] for i in best_candidates_indices ]

            if options.updater is not None:
                options.updater(best_candidates)

            if i % options.show_every == 0:
                print(f"Iteration {i+1}: Loss: {np.min(fitness_values)}")
                if options.verbose:
                    print(f"  Best candidate: {best_candidates[0]}")

            if np.min(fitness_values) <= options.loss_target:
                break

            generation = [  ]
            generation += best_candidates

            # Generate formulas by recombining random parts of the previous best candidates (cross-over)
            for _ in range(crossover_count):
                formula = options.crossover.crossover(best_candidates)
                generation.append(formula)

            # Generate formulas by applying random mutations to the previous best candidates (mutations)
            for _ in range(mutation_count):
                formula = deepcopy(random.choice(best_candidates))
                if options.mutator.mutate(formula):
                    generation.append(formula)

            # Generate completely new random formulas
            for _ in range(options.population_size - len(generation)):
                formula = options.generator.generate()
                generation.append(formula)

    except KeyboardInterrupt:
        pass

    # Returns the k best candidates
    generation = list(set(generation)) # remove duplicates
    fitness_values = [fitness(f, x, y, options.lambda_regularization) for f in generation]
    best_candidates_indices = np.argpartition(fitness_values, list(range(options.k_best)))[:options.k_best]
    best_candidates = [ generation[i] for i in best_candidates_indices ]
    if options.updater is not None:
        options.updater(best_candidates)
    return best_candidates

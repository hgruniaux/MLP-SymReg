from symreg.formula import *
from symreg.generator import *
from symreg.mutator import *
from symreg.crossover import *
from symreg.random import *
from symreg.evolution import *
from symreg.derivative import derivate

import matplotlib.pyplot as plt

np.seterr(all='ignore')

target_x = np.linspace(0, 3, 100)
random_options = RandomOptions()
random_options.definition_set = target_x
random_options.max_depth = 4

generator = RandomGenerator(random_options)
mutator = SequentialMutator(
    [
        RandomMutator(
            [
                ConstantNoiseMutator(),
                SwapBinaryExpressionMutator(),
                BinaryOperatorMutator(),
                BinaryInserterMutator(),
                BinaryRemoverMutator(),
                UnaryOperatorMutator(),
                UnaryInserterMutator(),
                UnaryRemoverMutator(),
            ]
        ),
        SimplifyMutator(),
    ]
)

options = Options()
options.verbose = True
options.iterations = 10
options.population_size = 5000
options.crossover_rate = 0
options.mutation_rate = 0.90
options.k_best = 10
options.generator = generator
options.mutator = mutator

target_formula = random_formula(random_options)
print(f"Target: {target_formula}")
target_y = target_formula(target_x)
print(f"Mean = {np.mean(target_y)}, Std = {np.std(target_y)}")
target_y += 0.1 * np.random.normal(np.mean(target_y), np.std(target_y), np.shape(target_y))

plt.plot(target_x, target_y, label=f"Target function: {target_formula}", linestyle='dashed')

candidates = run(target_x, target_y, options)
for i, candidate in enumerate(candidates):
    y = candidate(target_x)
    if np.shape(y) != np.shape(target_x):
        y = np.array([y] * len(target_x))
    plt.plot(target_x, y, label=f"Candidate {i}: {candidate}")

plt.legend()
plt.savefig('output.png')

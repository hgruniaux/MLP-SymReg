from symreg.formula import *
from symreg.generator import *
from symreg.mutator import *
from symreg.crossover import *
from symreg.random import *
from symreg.evolution import *
from symreg.tokenizer import tokenize

import argparse

import matplotlib.pyplot as plt

np.seterr(all='ignore')

parser = argparse.ArgumentParser(
                    prog='SymReg Driver',
                    description='Test the evolutionary algorithm for symbolic regression',
                    epilog='Text at the bottom of help')
parser.add_argument('-v', '--verbose', action='store_true', help='Show intermediate information')
parser.add_argument('-n', '--noise', action='store_true', help='Add noise to the target function')
parser.add_argument('-i', '--iterations', type=int, default=10, help='Number of iterations')
parser.add_argument('-p', '--population', type=int, default=500, help='Population size')
parser.add_argument('-k', '--kbest', type=int, default=5, help='Number of best candidates')
parser.add_argument('-c', '--crossover', type=float, default=0.0, help='Crossover rate')
parser.add_argument('-m', '--mutation', type=float, default=0.95, help='Mutation rate')
parser.add_argument('-d', '--maxdepth', type=int, default=3, help='Maximum depth for expression generation')
parser.add_argument('-l', '--lambda_coef', type=float, default=0.1, help='Regularization parameter')
args = parser.parse_args()

target_x = np.linspace(0, 3, 100)
random_options = RandomOptions()
random_options.definition_set = target_x
random_options.max_depth = args.maxdepth

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
options.verbose = args.verbose
options.iterations = args.iterations
options.population_size = args.population
options.crossover_rate = args.crossover
options.mutation_rate = args.mutation
options.lambda_regularization = args.lambda_coef
options.k_best = args.kbest
options.generator = generator
options.mutator = mutator

target_formula = random_formula(random_options)
# target_formula = UnaryExpression(UnaryOp.SIN, VariableExpression() * 4.3)

print(f"Target: {target_formula}")
print(f"Tokens: {tokenize(target_formula)}")
target_y = target_formula(target_x)
print(f"Mean = {np.mean(target_y)}, Std = {np.std(target_y)}")
if args.noise:
    target_y += 0.1 * np.random.normal(np.mean(target_y), np.std(target_y), np.shape(target_y))

plt.plot(target_x, target_y, label=f"Target function: ${target_formula.latex()}$", linestyle='dashed')

candidates = run(target_x, target_y, options)
for i, candidate in enumerate(candidates):
    y = candidate(target_x)
    if np.shape(y) != np.shape(target_x):
        y = np.array([y] * len(target_x))
    plt.plot(target_x, y, label=f"Candidate {i}: ${candidate.latex()}$")

plt.legend()
plt.savefig('output.png')
